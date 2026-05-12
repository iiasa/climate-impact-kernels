from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np
import xarray as xr

ArrayLike: TypeAlias = xr.DataArray | xr.Dataset

DEFAULT_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
TARGET_LEVEL_DIM = "target_warming_level"


def match_warming_levels(
    gmt: xr.DataArray,
    target_levels: Sequence[float],
    dim: str = "year",
    bucket_width: float | None = None,
    accumulation_baseline_period: tuple[int, int] | slice | None = None,
) -> xr.Dataset:
    """Return all GMT years that fall inside each warming-level bucket."""

    _validate_gmt(gmt=gmt, dim=dim)
    levels = _normalize_target_levels(target_levels)
    half_width = _resolve_bucket_width(levels=levels, bucket_width=bucket_width) / 2

    gmt_values = np.asarray(gmt.values, dtype=float)
    years = np.asarray(gmt[dim].values)
    warming_rate = np.gradient(gmt_values, years) if gmt_values.size > 1 else np.full(1, np.nan)
    accumulated_warming = _compute_accumulated_warming(
        gmt=gmt,
        dim=dim,
        baseline_period=accumulation_baseline_period,
    )

    records: list[tuple[float, int, float, float, float, float]] = []
    for level in levels:
        matched = np.flatnonzero(
            np.isfinite(gmt_values)
            & (level - half_width <= gmt_values)
            & (gmt_values < level + half_width)
        )
        records.extend(
            (
                float(level),
                years[index],
                float(gmt_values[index]),
                float(abs(gmt_values[index] - level)),
                float(warming_rate[index]),
                float(accumulated_warming[index]),
            )
            for index in matched
        )

    return _records_to_dataset(records)


def assign_years_to_levels(
    gmt: xr.DataArray,
    target_levels: Sequence[float],
    dim: str = "year",
) -> xr.Dataset:
    """Assign each GMT year to the nearest target warming level."""

    _validate_gmt(gmt=gmt, dim=dim)
    levels = _normalize_target_levels(target_levels)
    level_index = np.asarray(levels)
    actual_warming = np.asarray(gmt.values, dtype=float)
    assigned_levels = np.asarray(
        [
            level_index[np.argmin(np.abs(level_index - value))]
            if np.isfinite(value)
            else np.nan
            for value in actual_warming
        ],
        dtype=float,
    )

    return xr.Dataset(
        data_vars={
            "warming_level": (dim, assigned_levels),
            "actual_warming": (dim, actual_warming),
            "absolute_error": (dim, np.abs(actual_warming - assigned_levels)),
        },
        coords={dim: gmt[dim]},
    )


def bin_by_warming_level(
    values: xr.DataArray,
    gmt: xr.DataArray,
    target_levels: Sequence[float],
    dim: str = "year",
    reducer: str = "mean",
) -> xr.DataArray:
    """Aggregate values by the nearest warming level."""

    values_aligned, gmt_aligned = xr.align(values, gmt, join="inner")
    assignments = assign_years_to_levels(gmt=gmt_aligned, target_levels=target_levels, dim=dim)
    labelled = values_aligned.assign_coords(
        warming_level=(dim, assignments["warming_level"].values)
    )
    grouped = labelled.groupby("warming_level")
    reduced = _reduce_grouped(grouped=grouped, dim=dim, reducer=reducer)
    return reduced.reindex(warming_level=np.asarray(target_levels, dtype=float))


def build_warming_level_assignments(
    gmt_by_path: Mapping[tuple, xr.DataArray],
    *,
    path_columns: Sequence[str],
    min_warming_level: float = 0.0,
    bin_width: float = 0.1,
    max_warming_level: float | None = None,
    year_dim: str = "year",
) -> xr.Dataset:
    """Assign each path/year pair to the nearest warming level.

    Returns an xarray Dataset indexed by the explicit path dimensions plus
    ``year_dim``. This is the shared assignment object consumed by
    ``attach_warming_levels``.
    """

    path_columns = tuple(path_columns)
    if not path_columns:
        raise ValueError("path_columns must not be empty")
    if min_warming_level < 0:
        raise ValueError("min_warming_level must be non-negative")
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")

    paths = list(gmt_by_path.items())
    if not paths:
        raise ValueError("gmt_by_path must not be empty")
    for key, gmt in paths:
        if len(key) != len(path_columns):
            raise ValueError(
                f"path key {key!r} does not match path_columns {path_columns!r}"
            )
        _validate_gmt(gmt=gmt, dim=year_dim)

    resolved_max = (
        max_warming_level
        if max_warming_level is not None
        else max(float(np.nanmax(gmt.values)) for _, gmt in paths)
    )
    upper = max(resolved_max, min_warming_level)
    level_count = int(np.ceil((upper - min_warming_level) / bin_width)) + 1
    target_levels = np.round(
        min_warming_level + np.arange(level_count, dtype=float) * bin_width,
        decimals=6,
    )

    per_path: list[xr.Dataset] = []
    for key, gmt in paths:
        assigned = assign_years_to_levels(
            gmt=gmt,
            target_levels=target_levels,
            dim=year_dim,
        )
        expanded = assigned.expand_dims(
            {
                column: xr.IndexVariable(column, [value])
                for column, value in zip(path_columns, key, strict=True)
            }
        )
        per_path.append(expanded)

    assignments = xr.combine_by_coords(
        per_path,
        combine_attrs="drop_conflicts",
        join="outer",
    ).sortby(year_dim)
    for column in reversed(path_columns):
        assignments = assignments.sortby(column)

    assignments = assignments.assign_coords(
        {TARGET_LEVEL_DIM: (TARGET_LEVEL_DIM, target_levels)}
    )
    assignments.attrs.update(
        {
            "description": "Nearest-neighbor warming-level assignments by GMT path and year",
            "path_dims": ",".join(path_columns),
            "year_dim": year_dim,
            "warming_level_units": "K",
        }
    )
    return assignments


def attach_warming_levels(
    data: ArrayLike,
    assignments: xr.Dataset,
    *,
    time_dim: str = "time",
) -> ArrayLike:
    """Attach warming-level diagnostics to a time- or year-indexed xarray object."""

    if "warming_level" not in assignments.data_vars:
        raise ValueError("assignments must define a 'warming_level' variable")

    year_dim = str(assignments.attrs.get("year_dim", "year"))
    path_dims = _assignment_path_dims(assignments, year_dim=year_dim)
    missing = set(path_dims).difference(data.dims)
    if missing:
        raise ValueError(
            f"data missing required assignment path dims: {', '.join(sorted(missing))}"
        )

    selection = assignments[["warming_level", "actual_warming", "absolute_error"]].sel(
        {dim: data[dim] for dim in path_dims}
    )

    if time_dim in data.dims:
        years = xr.DataArray(
            data[time_dim].dt.year.values,
            dims=(time_dim,),
            coords={time_dim: data[time_dim]},
            name=year_dim,
        )
        selection = selection.sel({year_dim: years})
    elif year_dim in data.dims:
        selection = selection.sel({year_dim: data[year_dim]})
    else:
        raise ValueError(
            f"data must include either {time_dim!r} or {year_dim!r} to attach warming levels"
        )

    if bool(selection["warming_level"].isnull().any()) and bool(
        _uncovered_nonnull_data(data, selection["warming_level"]).any()
    ):
        raise ValueError("assignments do not cover the non-null data domain")

    return data.assign_coords(
        warming_level=selection["warming_level"],
        actual_warming=selection["actual_warming"],
        absolute_error=selection["absolute_error"],
    )


def build_warming_level_quantile_surface(
    values: xr.DataArray,
    *,
    sample_dims: Sequence[str],
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    warming_level_coord: str = "warming_level",
) -> xr.Dataset:
    """Build a quantile surface from a GWL-labelled indicator array."""

    if values.name is None:
        raise ValueError("values must have a name")

    sample_dims = tuple(sample_dims)
    if not sample_dims:
        raise ValueError("sample_dims must not be empty")

    missing_dims = set(sample_dims).difference(values.dims)
    if missing_dims:
        raise ValueError(
            f"values missing sample dims: {', '.join(sorted(missing_dims))}"
        )
    if warming_level_coord not in values.coords:
        raise ValueError(f"values must define a {warming_level_coord!r} coordinate")

    warming_level = values.coords[warming_level_coord]
    invalid_dims = set(warming_level.dims).difference(sample_dims)
    if invalid_dims:
        raise ValueError(
            f"{warming_level_coord!r} dims must be a subset of sample_dims; got {warming_level.dims!r}"
        )

    quantiles = tuple(float(value) for value in quantiles)
    if not quantiles:
        raise ValueError("quantiles must not be empty")

    stacked = values.stack(__sample=sample_dims)
    valid_levels = stacked[warming_level_coord].notnull()
    if warming_level.dtype.kind in {"U", "S", "O"}:
        valid_levels = valid_levels & (stacked[warming_level_coord] != "")
    stacked = stacked.where(valid_levels, drop=True)

    quantile_surface = stacked.groupby(warming_level_coord).quantile(
        quantiles,
        dim="__sample",
    )
    sample_size = stacked.groupby(warming_level_coord).count(dim="__sample").rename("sample_size")

    return (
        xr.merge([quantile_surface.rename(values.name).to_dataset(), sample_size.to_dataset()])
        .sortby(warming_level_coord)
        .sortby("quantile")
        .assign_attrs(
            {
                "description": (
                    f"Quantile surface for {values.name} conditioned on annual-mean GMT"
                ),
                "warming_level_units": "K",
                "sample_dims": ",".join(sample_dims),
                "value_name": values.name,
            }
        )
    )


def _assignment_path_dims(assignments: xr.Dataset, *, year_dim: str) -> tuple[str, ...]:
    return tuple(dim for dim in assignments["warming_level"].dims if dim != year_dim)


def _uncovered_nonnull_data(data: ArrayLike, warming_level: xr.DataArray) -> xr.DataArray:
    if isinstance(data, xr.Dataset):
        present = data.to_array().notnull().any(dim="variable")
    else:
        present = data.notnull()
    return warming_level.isnull().broadcast_like(present) & present


def _validate_gmt(gmt: xr.DataArray, dim: str) -> None:
    if dim not in gmt.dims:
        raise ValueError(f"{dim!r} is not a dimension of the GMT series")
    if gmt.ndim != 1:
        raise ValueError("GMT input must be one-dimensional")


def _normalize_target_levels(target_levels: Sequence[float]) -> tuple[float, ...]:
    levels = tuple(float(level) for level in target_levels)
    if not levels:
        raise ValueError("target_levels must not be empty")
    return levels


def _resolve_bucket_width(levels: Sequence[float], bucket_width: float | None) -> float:
    if bucket_width is not None:
        if bucket_width <= 0:
            raise ValueError("bucket_width must be positive")
        return bucket_width

    if len(levels) == 1:
        raise ValueError("bucket_width is required when only one target level is supplied")

    steps = np.diff(np.asarray(levels, dtype=float))
    if np.any(steps <= 0):
        raise ValueError("target_levels must be strictly increasing")
    return float(steps.min())


def _compute_accumulated_warming(
    gmt: xr.DataArray,
    dim: str,
    baseline_period: tuple[int, int] | slice | None,
) -> np.ndarray:
    accumulated = gmt.cumsum(dim=dim)
    if baseline_period is None:
        return np.asarray(accumulated.values, dtype=float)

    baseline_slice = _normalize_period(baseline_period)
    baseline = accumulated.sel({dim: baseline_slice})
    if baseline.sizes.get(dim, 0) == 0:
        raise ValueError(f"baseline_period selects no entries along {dim!r}")

    return np.asarray((accumulated - baseline.mean(dim=dim)).values, dtype=float)


def _normalize_period(period: tuple[int, int] | slice) -> slice:
    match period:
        case slice():
            return period
        case (int(start), int(stop)):
            return slice(start, stop)
        case _:
            raise TypeError("period must be a slice or a two-year tuple")


def _records_to_dataset(
    records: list[tuple[float, int, float, float, float, float]],
) -> xr.Dataset:
    if not records:
        empty = np.asarray([], dtype=float)
        return xr.Dataset(
            data_vars={
                "warming_level": ("match", empty),
                "year": ("match", np.asarray([], dtype=int)),
                "actual_warming": ("match", empty),
                "absolute_error": ("match", empty),
                "warming_rate": ("match", empty),
                "accumulated_warming": ("match", empty),
            }
        )

    columns = list(zip(*records, strict=True))
    return xr.Dataset(
        data_vars={
            "warming_level": ("match", np.asarray(columns[0], dtype=float)),
            "year": ("match", np.asarray(columns[1])),
            "actual_warming": ("match", np.asarray(columns[2], dtype=float)),
            "absolute_error": ("match", np.asarray(columns[3], dtype=float)),
            "warming_rate": ("match", np.asarray(columns[4], dtype=float)),
            "accumulated_warming": ("match", np.asarray(columns[5], dtype=float)),
        }
    )


def _reduce_grouped(grouped: xr.core.groupby.DataArrayGroupBy, dim: str, reducer: str) -> xr.DataArray:
    match reducer:
        case "mean":
            return grouped.mean(dim=dim)
        case "sum":
            return grouped.sum(dim=dim)
        case _:
            raise ValueError("reducer must be one of {'mean', 'sum'}")
