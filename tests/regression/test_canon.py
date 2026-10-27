"""The comparison code itself: what each comparison mode accepts and what it rejects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from .canon import Mismatch, Raised, Scrubber, compare_case, encode_case

SCRUB = Scrubber("/nowhere")


def _encoded(value):
    return encode_case({"value": value}, SCRUB)


def _passes(actual, expected, tier):
    compare_case(_encoded(actual), _encoded(expected), tier, "case")


def _fails(actual, expected, tier="exact"):
    with pytest.raises(Mismatch):
        _passes(actual, expected, tier)


def _frame():
    return pd.DataFrame({"a": [1.0, np.nan, np.inf], "b": ["x", "y", None], "c": [1, 2, 3]})


def test_identical_frames_pass_in_both_tiers():
    for tier in ("exact", "tolerance"):
        _passes(_frame(), _frame(), tier)


def test_one_ulp_fails_exact_and_passes_tolerance():
    base = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims="x", name="v")
    nudged = base.copy(data=np.nextafter(base.values, np.inf))
    _fails(nudged, base, "exact")
    _passes(nudged, base, "tolerance")


def test_beyond_tolerance_fails_both_tiers():
    base = pd.DataFrame({"a": [1.0, 2.0]})
    off = pd.DataFrame({"a": [1.0, 2.0 * (1 + 1e-6)]})
    _fails(off, base, "exact")
    _fails(off, base, "tolerance")


def test_nan_and_inf_masks_must_match_in_both_tiers():
    base = pd.DataFrame({"a": [1.0, np.nan, np.inf]})
    for tier in ("exact", "tolerance"):
        _fails(pd.DataFrame({"a": [1.0, 0.0, np.inf]}), base, tier)
        _fails(pd.DataFrame({"a": [1.0, np.nan, 5.0]}), base, tier)
        _fails(pd.DataFrame({"a": [1.0, np.nan, -np.inf]}), base, tier)


def test_dtype_column_order_and_row_order_are_strict():
    base = _frame()
    for tier in ("exact", "tolerance"):
        _fails(base.assign(c=base["c"].astype("float64")), base, tier)
        _fails(base[["b", "a", "c"]], base, tier)
        _fails(base.iloc[::-1], base, tier)
        _fails(base.rename(columns={"c": "d"}), base, tier)


def test_dataset_names_coords_attrs_and_dtypes_are_strict():
    base = xr.Dataset(
        {"v": (("x",), np.array([1.0, 2.0]), {"units": "m"})},
        coords={"x": [10, 20]},
        attrs={"title": "t"},
    )
    _passes(base, base.copy(deep=True), "exact")
    _fails(base.rename(v="w"), base)
    _fails(base.assign_coords(x=[10, 21]), base)
    _fails(base.assign_attrs(title="u"), base)
    _fails(base.assign(v=base["v"].assign_attrs(units="km")), base)
    _fails(base.assign(v=base["v"].astype("float32")), base)


def test_dataset_variable_order_and_integer_dtypes_are_strict():
    ds = xr.Dataset(
        {"a": (("x",), np.array([1.0, 2.0])), "n": (("x",), np.array([1, 2], dtype="int64"))},
        coords={"x": np.array([1, 2], dtype="int64")},
    )
    _passes(ds, ds.copy(deep=True), "exact")
    _fails(ds[["n", "a"]], ds)
    _fails(ds.assign(n=ds["n"].astype("int32")), ds)
    _fails(ds.assign_coords(x=ds["x"].astype("int32")), ds)


def test_golden_without_order_lists_is_compared_by_name():
    ds = xr.Dataset({"a": (("x",), [1.0]), "b": (("x",), [2.0])}, coords={"x": [1]})
    manifest, arrays = _encoded(ds)
    unordered = {"value": {k: v for k, v in manifest["value"].items() if k not in ("data_vars", "coords")}}
    compare_case(_encoded(ds[["b", "a"]]), (unordered, arrays), "exact", "case")


def test_raised_errors_compare_on_type_and_message():
    _passes(Raised("ValueError", "boom"), Raised("ValueError", "boom"), "exact")
    _fails(Raised("ValueError", "bang"), Raised("ValueError", "boom"))
    _fails(Raised("KeyError", "boom"), Raised("ValueError", "boom"))
    _fails(1.0, Raised("ValueError", "boom"))


def test_nested_artifacts_are_compared():
    nested = {"first": _frame(), "second": [xr.DataArray([1.0], dims="x", name="v"), 3]}
    _passes(nested, {"first": _frame(), "second": [xr.DataArray([1.0], dims="x", name="v"), 3]}, "exact")
    _fails({**nested, "first": _frame().iloc[:2]}, nested)


def test_plain_values_are_compared_with_type():
    _passes({"n": 1, "f": 0.5, "s": "x", "t": (1, 2)}, {"n": 1, "f": 0.5, "s": "x", "t": (1, 2)}, "exact")
    _fails({"n": 1.0}, {"n": 1})
    _fails({"t": [1, 2]}, {"t": (1, 2)})


def test_tolerance_tier_has_an_absolute_floor_scaled_to_the_array():
    base = pd.DataFrame({"a": [1.0, 4.7e-14, 2.0]})
    residual = pd.DataFrame({"a": [1.0, -3.0e-14, 2.0]})
    _fails(residual, base, "exact")
    _passes(residual, base, "tolerance")
    real = pd.DataFrame({"a": [1.0 * (1 + 1e-6), 4.7e-14, 2.0]})
    _fails(real, base, "tolerance")
