"""Turning end-to-end test results into a comparable form, storing them and comparing them.

An end-to-end test returns xarray objects, pandas objects, numpy arrays, plain
values or a recorded exception. ``encode`` turns each into a JSON structure
listing (``manifest.json.gz``) plus numpy arrays. That pair is what is stored as
the reference output and what the comparison reads, so the comparison never
depends on how xarray or pandas round-trip through a file format. The structure
listing records names, dimension order, data types, attributes and column order;
the arrays hold the values.

There are two comparison modes. ``exact`` demands identical values (NaN equal to
NaN, inf equal to inf, row and column order, dtypes and attributes). ``tolerance``
accepts float values within a relative 1e-12 provided the NaN and inf masks
agree; everything that is not a float must still match exactly.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

RTOL = 1e-12
# Absolute floor of the ``tolerance`` comparison mode, as a fraction of the largest finite |expected|
# in the same array (the value itself for a lone float).
ATOL_SCALE = 1e-9
MANIFEST = "manifest.json.gz"


@dataclasses.dataclass(frozen=True)
class Raised:
    """An exception recorded instead of a value."""

    type: str
    message: str


class Scrubber:
    """Replaces machine-specific path prefixes inside strings."""

    def __init__(self, *roots: str | Path) -> None:
        self.roots = sorted({str(root) for root in roots}, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for root in self.roots:
            text = text.replace(root, "<TMP>")
        return text


class ArrayStore:
    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self._seen: dict[tuple, str] = {}

    def add(self, array: np.ndarray) -> str:
        """Store ``array`` once; identical arrays (dtype, shape, bytes) share a key."""

        signature = (array.dtype.str, array.shape, np.ascontiguousarray(array).tobytes())
        if signature in self._seen:
            return self._seen[signature]
        key = f"a{len(self.arrays)}"
        self.arrays[key] = array
        self._seen[signature] = key
        return key


# --- encoding -------------------------------------------------------------


def jsonable(value: Any, scrub: Scrubber) -> Any:
    """Convert plain Python and numpy values to a tagged, JSON-safe structure."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, Path):
        return scrub(str(value))
    if value is pd.NA:
        return {"__na__": True}
    if value is pd.NaT:
        return {"__nat__": True}
    if isinstance(value, pd.Timestamp):
        return {"__timestamp__": value.isoformat()}
    if isinstance(value, np.datetime64):
        return {"__datetime64__": str(value)}
    if hasattr(value, "isoformat") and hasattr(value, "calendar"):
        return {"__cftime__": value.isoformat(), "calendar": str(value.calendar)}
    if hasattr(value, "isoformat"):
        return {"__datetime__": value.isoformat(), "type": type(value).__name__}
    if isinstance(value, np.ndarray):
        return {"__array__": [jsonable(item, scrub) for item in value.tolist()]}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": type(value).__name__,
            **{
                field.name: jsonable(getattr(value, field.name), scrub)
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, Mapping):
        return {str(key): jsonable(item, scrub) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"__tuple__": [jsonable(item, scrub) for item in value]}
    if isinstance(value, (set, frozenset)):
        return {"__set__": sorted((jsonable(item, scrub) for item in value), key=repr)}
    if isinstance(value, list):
        return [jsonable(item, scrub) for item in value]
    raise TypeError(f"cannot encode {type(value).__name__} in a regression artifact")


def _encode_array(array: np.ndarray, store: ArrayStore, scrub: Scrubber) -> dict:
    array = np.asarray(array)
    if array.dtype.kind == "O":
        flat = array.ravel().tolist()
        if all(isinstance(item, str) for item in flat):
            text = np.asarray([scrub(item) for item in flat], dtype=str)
            return {"mode": "objstr", "shape": list(array.shape), "key": store.add(text)}
        return {
            "mode": "objjson",
            "shape": list(array.shape),
            "values": [jsonable(item, scrub) for item in flat],
        }
    return {"mode": "array", "key": store.add(array)}


def _encode_variable(var: xr.Variable, store: ArrayStore, scrub: Scrubber) -> dict:
    return {
        "dims": list(var.dims),
        "dtype": str(var.dtype),
        "attrs": jsonable(dict(var.attrs), scrub),
        "data": _encode_array(np.asarray(var.values), store, scrub),
    }


def _encode_dataset(ds: xr.Dataset, store: ArrayStore, scrub: Scrubber) -> dict:
    variables = {}
    for name, var in ds.variables.items():
        entry = _encode_variable(var, store, scrub)
        entry["role"] = "coord" if name in ds.coords else "data"
        variables[str(name)] = entry
    return {
        "kind": "dataset",
        "attrs": jsonable(dict(ds.attrs), scrub),
        "variables": variables,
        "data_vars": [str(name) for name in ds.data_vars],
        "coords": [str(name) for name in ds.coords],
    }


def _encode_values(values: pd.Series | pd.Index, store: ArrayStore, scrub: Scrubber) -> dict:
    dtype = values.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        categories = _encode_values(pd.Index(dtype.categories), store, scrub)
        codes = np.asarray(values.cat.codes if isinstance(values, pd.Series) else values.codes)
        return {
            "dtype": str(dtype),
            "categorical": {"categories": categories, "ordered": bool(dtype.ordered)},
            "data": {"mode": "array", "key": store.add(codes)},
        }
    if isinstance(dtype, np.dtype):
        array = values.to_numpy()
    else:
        missing = np.asarray(values.isna())
        array = values.astype(object).to_numpy(dtype=object, copy=True)
        array[missing] = None
    return {"dtype": str(dtype), "data": _encode_array(array, store, scrub)}


def _encode_index(index: pd.Index, store: ArrayStore, scrub: Scrubber) -> dict:
    if isinstance(index, pd.RangeIndex):
        return {"type": "range", "start": index.start, "stop": index.stop, "step": index.step}
    if isinstance(index, pd.MultiIndex):
        return {
            "type": "multi",
            "names": [jsonable(name, scrub) for name in index.names],
            "levels": [
                _encode_values(index.get_level_values(i), store, scrub)
                for i in range(index.nlevels)
            ],
        }
    return {
        "type": "index",
        "name": jsonable(index.name, scrub),
        "values": _encode_values(index, store, scrub),
    }


def _encode_frame(frame: pd.DataFrame, store: ArrayStore, scrub: Scrubber) -> dict:
    return {
        "kind": "frame",
        "columns": [jsonable(column, scrub) for column in frame.columns],
        "column_names": jsonable(list(frame.columns.names), scrub),
        "index": _encode_index(frame.index, store, scrub),
        "values": [
            _encode_values(frame.iloc[:, position], store, scrub)
            for position in range(frame.shape[1])
        ],
    }


_ARTIFACTS = (xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series, np.ndarray, Raised)


def _holds_artifact(value: Any) -> bool:
    """True for a container that has a result object (frame, array, ...) among its items."""

    if isinstance(value, dict):
        items = list(value.values())
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return False
    return any(isinstance(item, _ARTIFACTS) or _holds_artifact(item) for item in items)


def encode(value: Any, store: ArrayStore, scrub: Scrubber) -> dict:
    """Encode one result object into a structure-listing entry (arrays go to ``store``)."""

    if isinstance(value, Raised):
        return {"kind": "raised", "type": value.type, "message": scrub(value.message)}
    if _holds_artifact(value):
        if isinstance(value, dict):
            items = {str(key): encode(item, store, scrub) for key, item in value.items()}
            return {"kind": "mapping", "items": items}
        return {
            "kind": "sequence",
            "type": type(value).__name__,
            "items": [encode(item, store, scrub) for item in value],
        }
    if isinstance(value, xr.Dataset):
        return _encode_dataset(value, store, scrub)
    if isinstance(value, xr.DataArray):
        inner = _encode_dataset(value.to_dataset(name="__value__"), store, scrub)
        return {"kind": "dataarray", "name": jsonable(value.name, scrub), "dataset": inner}
    if isinstance(value, pd.DataFrame):
        return _encode_frame(value, store, scrub)
    if isinstance(value, pd.Series):
        return {
            "kind": "series",
            "name": jsonable(value.name, scrub),
            "index": _encode_index(value.index, store, scrub),
            "values": _encode_values(value, store, scrub),
        }
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "dtype": str(value.dtype),
            "data": _encode_array(value, store, scrub),
        }
    return {"kind": "json", "value": jsonable(value, scrub)}


def encode_case(result: Any, scrub: Scrubber) -> tuple[dict, dict[str, np.ndarray]]:
    """Encode an end-to-end test result (a dict of named result objects, or a single one)."""

    store = ArrayStore()
    items = result if isinstance(result, dict) else {"result": result}
    manifest = {name: encode(value, store, scrub) for name, value in items.items()}
    manifest = json.loads(json.dumps(manifest))
    return manifest, store.arrays


# --- storage --------------------------------------------------------------


def save_golden(directory: Path, manifest: dict, arrays: dict[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    with gzip.GzipFile(directory / MANIFEST, "wb", mtime=0) as handle:
        handle.write(text.encode("utf-8"))
    arrays_path = directory / "arrays.npz"
    if arrays:
        np.savez_compressed(arrays_path, **arrays)
    elif arrays_path.exists():
        arrays_path.unlink()


def load_golden(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with gzip.open(directory / MANIFEST, "rb") as handle:
        manifest = json.loads(handle.read().decode("utf-8"))
    arrays_path = directory / "arrays.npz"
    arrays: dict[str, np.ndarray] = {}
    if arrays_path.exists():
        with np.load(arrays_path, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
    return manifest, arrays


# --- comparison -----------------------------------------------------------


class Mismatch(AssertionError):
    pass


def _fail(path: str, message: str) -> None:
    raise Mismatch(f"{path}: {message}")


def _floats_match(actual: np.ndarray, expected: np.ndarray, tier: str, path: str) -> None:
    a_nan, e_nan = np.isnan(actual), np.isnan(expected)
    if not np.array_equal(a_nan, e_nan):
        _fail(path, f"NaN mask differs at {int((a_nan != e_nan).sum())} positions")
    a_inf, e_inf = np.isinf(actual), np.isinf(expected)
    if not np.array_equal(a_inf, e_inf) or not np.array_equal(
        actual[a_inf], expected[e_inf]
    ):
        _fail(path, "inf mask or sign differs")
    finite = ~(a_nan | a_inf)
    a, e = actual[finite], expected[finite]
    if tier == "exact":
        bad = a != e
        if bad.any():
            index = int(np.flatnonzero(bad)[0])
            _fail(
                path,
                f"{int(bad.sum())} of {a.size} finite values differ (exact comparison); "
                f"first: actual={a[index]!r} expected={e[index]!r}",
            )
    else:
        atol = ATOL_SCALE * float(np.max(np.abs(e))) if e.size else 0.0
        bad = ~(np.abs(a - e) <= RTOL * np.abs(e) + atol)
        if bad.any():
            index = int(np.flatnonzero(bad)[0])
            _fail(
                path,
                f"{int(bad.sum())} of {a.size} finite values differ beyond rtol={RTOL:g}; "
                f"first: actual={a[index]!r} expected={e[index]!r}",
            )


def _arrays_match(actual: np.ndarray, expected: np.ndarray, tier: str, path: str) -> None:
    if actual.dtype != expected.dtype:
        _fail(path, f"array dtype {actual.dtype} != {expected.dtype}")
    if actual.shape != expected.shape:
        _fail(path, f"shape {actual.shape} != {expected.shape}")
    kind = actual.dtype.kind
    if kind == "f":
        _floats_match(actual, expected, tier, path)
    elif kind == "c":
        _floats_match(actual.real, expected.real, tier, path + ".real")
        _floats_match(actual.imag, expected.imag, tier, path + ".imag")
    elif kind in "Mm":
        both_nat = np.isnat(actual) & np.isnat(expected)
        if not np.array_equal(np.isnat(actual), np.isnat(expected)) or not np.array_equal(
            actual[~both_nat], expected[~both_nat]
        ):
            _fail(path, "datetime values differ")
    elif not np.array_equal(actual, expected):
        _fail(path, f"{int((actual != expected).sum())} values differ")


def _json_match(actual: Any, expected: Any, tier: str, path: str) -> None:
    if isinstance(expected, float) or isinstance(actual, float):
        if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
            _fail(path, f"{actual!r} != {expected!r}")
        if isinstance(actual, bool) or isinstance(expected, bool):
            _fail(path, f"{actual!r} != {expected!r}")
        if type(actual) is not type(expected):
            _fail(path, f"type {type(actual).__name__} != {type(expected).__name__}")
        if math.isnan(actual) and math.isnan(expected):
            return
        if tier == "exact":
            ok = actual == expected
        else:
            atol = ATOL_SCALE * abs(expected) if math.isfinite(expected) else 0.0
            ok = actual == expected or abs(actual - expected) <= RTOL * abs(expected) + atol
        if not ok:
            _fail(path, f"{actual!r} != {expected!r}")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            _fail(path, f"keys {sorted(actual) if isinstance(actual, dict) else actual!r} != {sorted(expected)}")
        for key in expected:
            _json_match(actual[key], expected[key], tier, f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            _fail(path, "list length differs")
        for position, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _json_match(a, e, tier, f"{path}[{position}]")
        return
    if actual != expected or type(actual) is not type(expected):
        _fail(path, f"{actual!r} != {expected!r}")


def _decode_array(entry: dict, arrays: dict[str, np.ndarray]) -> tuple[str, Any]:
    mode = entry["mode"]
    if mode == "array":
        return "array", arrays[entry["key"]]
    if mode == "objstr":
        text = arrays[entry["key"]]
        return "array", text.astype(object).reshape(entry["shape"])
    return "json", entry["values"]


def _data_match(actual, expected, a_arrays, e_arrays, tier, path) -> None:
    if actual["mode"] != expected["mode"]:
        _fail(path, f"storage mode {actual['mode']} != {expected['mode']}")
    mode = actual["mode"]
    if mode == "array":
        _arrays_match(a_arrays[actual["key"]], e_arrays[expected["key"]], tier, path)
    elif mode == "objstr":
        a = a_arrays[actual["key"]]
        e = e_arrays[expected["key"]]
        if actual["shape"] != expected["shape"] or not np.array_equal(a, e):
            _fail(path, "string values differ")
    else:
        if actual["shape"] != expected["shape"]:
            _fail(path, "shape differs")
        _json_match(actual["values"], expected["values"], tier, path)


def _values_match(actual, expected, a_arrays, e_arrays, tier, path) -> None:
    if actual["dtype"] != expected["dtype"]:
        _fail(path, f"dtype {actual['dtype']} != {expected['dtype']}")
    if ("categorical" in actual) != ("categorical" in expected):
        _fail(path, "categorical mismatch")
    if "categorical" in actual:
        if actual["categorical"]["ordered"] != expected["categorical"]["ordered"]:
            _fail(path, "category order flag differs")
        _values_match(
            actual["categorical"]["categories"],
            expected["categorical"]["categories"],
            a_arrays,
            e_arrays,
            tier,
            path + ".categories",
        )
    _data_match(actual["data"], expected["data"], a_arrays, e_arrays, tier, path)


def _index_match(actual, expected, a_arrays, e_arrays, tier, path) -> None:
    if actual["type"] != expected["type"]:
        _fail(path, f"index type {actual['type']} != {expected['type']}")
    if actual["type"] == "range":
        _json_match(actual, expected, "exact", path)
    elif actual["type"] == "multi":
        _json_match(actual["names"], expected["names"], "exact", path + ".names")
        for position, (a, e) in enumerate(zip(actual["levels"], expected["levels"], strict=True)):
            _values_match(a, e, a_arrays, e_arrays, tier, f"{path}.level{position}")
    else:
        _json_match(actual["name"], expected["name"], "exact", path + ".name")
        _values_match(actual["values"], expected["values"], a_arrays, e_arrays, tier, path)


def _variable_match(actual, expected, a_arrays, e_arrays, tier, path) -> None:
    for field in ("role", "dims", "dtype"):
        if actual.get(field) != expected.get(field):
            _fail(path, f"{field} {actual.get(field)!r} != {expected.get(field)!r}")
    _json_match(actual["attrs"], expected["attrs"], "exact", path + ".attrs")
    _data_match(actual["data"], expected["data"], a_arrays, e_arrays, tier, path)


def _dataset_match(actual, expected, a_arrays, e_arrays, tier, path) -> None:
    _json_match(actual["attrs"], expected["attrs"], "exact", path + ".attrs")
    a_vars, e_vars = actual["variables"], expected["variables"]
    if set(a_vars) != set(e_vars):
        _fail(
            path,
            f"variables differ: extra={sorted(set(a_vars) - set(e_vars))} "
            f"missing={sorted(set(e_vars) - set(a_vars))}",
        )
    # Stored structure listings sort their keys, so order lives in explicit lists. A
    # listing without them is checked by name only.
    for field in ("data_vars", "coords"):
        if field in expected and actual.get(field) != expected[field]:
            _fail(path, f"{field} order {actual.get(field)} != {expected[field]}")
    for name in e_vars:
        _variable_match(a_vars[name], e_vars[name], a_arrays, e_arrays, tier, f"{path}[{name}]")


def compare_entry(actual, expected, a_arrays, e_arrays, tier: str, path: str) -> None:
    if actual["kind"] != expected["kind"]:
        _fail(path, f"kind {actual['kind']} != {expected['kind']}")
    kind = actual["kind"]
    if kind == "raised":
        _json_match(actual, expected, "exact", path)
    elif kind == "mapping":
        if sorted(actual["items"]) != sorted(expected["items"]):
            _fail(path, f"keys {list(actual['items'])} != {list(expected['items'])}")
        for key, item in expected["items"].items():
            compare_entry(actual["items"][key], item, a_arrays, e_arrays, tier, f"{path}/{key}")
    elif kind == "sequence":
        _json_match(actual["type"], expected["type"], "exact", path + ".type")
        if len(actual["items"]) != len(expected["items"]):
            _fail(path, f"length {len(actual['items'])} != {len(expected['items'])}")
        for position, (a, e) in enumerate(zip(actual["items"], expected["items"], strict=True)):
            compare_entry(a, e, a_arrays, e_arrays, tier, f"{path}[{position}]")
    elif kind == "json":
        _json_match(actual["value"], expected["value"], tier, path)
    elif kind == "dataset":
        _dataset_match(actual, expected, a_arrays, e_arrays, tier, path)
    elif kind == "dataarray":
        _json_match(actual["name"], expected["name"], "exact", path + ".name")
        _dataset_match(actual["dataset"], expected["dataset"], a_arrays, e_arrays, tier, path)
    elif kind == "frame":
        _json_match(actual["columns"], expected["columns"], "exact", path + ".columns")
        _json_match(actual["column_names"], expected["column_names"], "exact", path)
        _index_match(actual["index"], expected["index"], a_arrays, e_arrays, tier, path + ".index")
        for position, (a, e) in enumerate(zip(actual["values"], expected["values"], strict=True)):
            name = expected["columns"][position]
            _values_match(a, e, a_arrays, e_arrays, tier, f"{path}[{name}]")
    elif kind == "series":
        _json_match(actual["name"], expected["name"], "exact", path + ".name")
        _index_match(actual["index"], expected["index"], a_arrays, e_arrays, tier, path + ".index")
        _values_match(actual["values"], expected["values"], a_arrays, e_arrays, tier, path)
    elif kind == "ndarray":
        if actual["dtype"] != expected["dtype"]:
            _fail(path, f"dtype {actual['dtype']} != {expected['dtype']}")
        _data_match(actual["data"], expected["data"], a_arrays, e_arrays, tier, path)
    else:
        _fail(path, f"unknown kind {kind!r}")


def compare_case(
    actual: tuple[dict, dict], expected: tuple[dict, dict], tier: str, case: str
) -> None:
    """Raise ``Mismatch`` when the actual encoding differs from the stored reference output."""

    if tier not in {"exact", "tolerance"}:
        raise ValueError(f"unknown comparison mode {tier!r}")
    a_manifest, a_arrays = actual
    e_manifest, e_arrays = expected
    if set(a_manifest) != set(e_manifest):
        _fail(case, f"artifact keys {sorted(a_manifest)} != {sorted(e_manifest)}")
    for key in e_manifest:
        compare_entry(a_manifest[key], e_manifest[key], a_arrays, e_arrays, tier, f"{case}/{key}")
