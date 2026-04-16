from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


@dataclass(frozen=True)
class WarmingLevelAssignments:
    """Year-to-warming-level assignments across a set of GMT paths.

    Produced by ``build_warming_level_assignments`` and consumed by
    ``attach_warming_levels``. The design follows the two-dimensional
    GMT-binned lookup pattern used by rimeX
    (https://github.com/iiasa/rimeX) for probabilistic scenario exploration.
    """

    table: pd.DataFrame
    target_levels: tuple[float, ...]
    path_columns: tuple[str, ...]
    year_column: str


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
    year_column: str = "year",
) -> WarmingLevelAssignments:
    """Bin each GMT year to the nearest warming level across multiple paths.

    Each key in ``gmt_by_path`` is a tuple identifying a GMT path (for example
    ``(gcm, scenario)``), and ``path_columns`` gives the column names in the
    resulting table. The shared target-level grid is derived from the observed
    GMT range and ``bin_width``.

    This is the table-assembly counterpart of ``assign_years_to_levels``,
    intended for rimeX-compatible two-dimensional GMT lookups
    (https://github.com/iiasa/rimeX).
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
    for key, _ in paths:
        if len(key) != len(path_columns):
            raise ValueError(
                f"path key {key!r} does not match path_columns {path_columns!r}"
            )

    resolved_max = (
        max_warming_level
        if max_warming_level is not None
        else max(float(np.nanmax(gmt.values)) for _, gmt in paths)
    )
    upper = max(resolved_max, min_warming_level)
    level_count = int(np.ceil((upper - min_warming_level) / bin_width)) + 1
    target_levels = tuple(
        np.round(
            min_warming_level + np.arange(level_count, dtype=float) * bin_width,
            decimals=6,
        ).tolist()
    )

    frames: list[pd.DataFrame] = []
    for key, gmt in paths:
        assignments = assign_years_to_levels(gmt=gmt, target_levels=target_levels)
        frame = pd.DataFrame(
            {
                **{column: value for column, value in zip(path_columns, key, strict=True)},
                year_column: np.asarray(assignments["year"].values, dtype=int),
                "warming_level": np.asarray(
                    assignments["warming_level"].values, dtype=float
                ),
                "actual_warming": np.asarray(
                    assignments["actual_warming"].values, dtype=float
                ),
                "absolute_error": np.asarray(
                    assignments["absolute_error"].values, dtype=float
                ),
            }
        )
        frames.append(frame)

    return WarmingLevelAssignments(
        table=pd.concat(frames, ignore_index=True),
        target_levels=target_levels,
        path_columns=path_columns,
        year_column=year_column,
    )


def attach_warming_levels(
    frame: pd.DataFrame,
    assignments: WarmingLevelAssignments,
    *,
    time_column: str = "time",
) -> pd.DataFrame:
    """Join warming-level assignments onto a long-format impact table.

    Matches rows by ``assignments.path_columns`` and the year component of
    ``time_column``. Raises if any row cannot be assigned.
    """

    required = (*assignments.path_columns, time_column)
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(
            f"frame missing required columns: {', '.join(sorted(missing))}"
        )

    enriched = (
        frame.assign(**{time_column: lambda data: pd.to_datetime(data[time_column])})
        .assign(**{assignments.year_column: lambda data: data[time_column].dt.year})
        .merge(
            assignments.table,
            on=[*assignments.path_columns, assignments.year_column],
            how="left",
            validate="many_to_one",
        )
    )
    if enriched["warming_level"].isna().any():
        missing_rows = (
            enriched.loc[
                enriched["warming_level"].isna(),
                [*assignments.path_columns, assignments.year_column],
            ]
            .drop_duplicates()
            .sort_values([*assignments.path_columns, assignments.year_column])
        )
        raise ValueError(
            "missing warming-level assignments for "
            f"{missing_rows.to_dict(orient='records')[:10]}"
        )
    return enriched


def build_warming_level_quantile_map(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_columns: Sequence[str],
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    warming_level_column: str = "warming_level",
) -> xr.Dataset:
    """Quantile-map a value column over warming levels within each group.

    Returns an ``xr.Dataset`` indexed by ``(*group_columns, warming_level,
    quantile)`` with per-bin ``sample_size``. This is the shape rimeX
    (https://github.com/iiasa/rimeX) expects for two-dimensional GMT lookups.
    """

    quantiles = tuple(float(value) for value in quantiles)
    if not quantiles:
        raise ValueError("quantiles must not be empty")

    required = (*group_columns, warming_level_column, value_column)
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(
            f"frame missing required columns: {', '.join(sorted(missing))}"
        )

    grouping = [*group_columns, warming_level_column]
    quantile_frame = (
        frame.groupby(grouping, observed=True)[value_column]
        .quantile(quantiles)
        .rename(value_column)
        .reset_index()
        .rename(columns={f"level_{len(grouping)}": "quantile"})
    )
    count_frame = (
        frame.groupby(grouping, observed=True)
        .size()
        .rename("sample_size")
        .reset_index()
    )

    quantile_map = quantile_frame.set_index(
        [*group_columns, warming_level_column, "quantile"]
    ).to_xarray()
    sample_size = count_frame.set_index(
        [*group_columns, warming_level_column]
    ).to_xarray()

    return (
        xr.merge([quantile_map, sample_size])
        .sortby(warming_level_column)
        .sortby("quantile")
        .assign_attrs(
            {
                "description": (
                    f"Quantile map for {value_column} conditioned on annual-mean GMT"
                ),
                "warming_level_units": "K",
                "value_column": value_column,
                "group_columns": ",".join(group_columns),
            }
        )
    )


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
