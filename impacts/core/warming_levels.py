from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr


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
