"""Assigning years to warming levels by global mean temperature (GMT), pooling climate samples by warming level, and weighted group means."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

TARGET_LEVEL_DIM = "target_warming_level"


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
    ``year_dim``. ``pool_by_warming_level`` takes this
    Dataset as its input.
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


def pool_by_warming_level(
    values: xr.DataArray,
    assignments: xr.Dataset,
    target_levels: Sequence[float],
    *,
    name: str,
    count_name: str = "n_obs",
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    sample_dims: Sequence[str] = ("gcm", "scenario", "year"),
) -> xr.Dataset:
    """Pool (path, year) samples into warming-level bins and summarise each bin.

    Each sample of ``values`` (indexed by ``sample_dims``) is labelled with the
    ``warming_level`` that ``assignments`` gives its path and year, and samples
    sharing a level are pooled. The result holds, per level, the mean ``name``,
    its population standard deviation ``{name}_std``, the requested quantiles
    ``{name}_p10``, ``{name}_p50``, ``{name}_p90`` (linear interpolation), and
    the number of finite samples ``count_name``. Missing values are skipped.
    Samples without an assigned level are discarded. The levels are reindexed
    onto ``target_levels`` and the dimension is renamed ``gwl``; a target level
    with no samples gets NaN statistics and a count of 0 (no integer sentinel is
    written for empty bins).
    """

    sample_dims = tuple(sample_dims)
    quantiles = tuple(quantiles)
    aligned, level = xr.align(values, assignments["warming_level"], join="inner")
    aligned = aligned.assign_coords(warming_level=level)
    stacked = aligned.stack(sample=sample_dims)
    stacked = stacked.where(stacked["warming_level"].notnull(), drop=True)

    grouped = stacked.groupby("warming_level")
    mean = grouped.mean("sample", skipna=True)
    std = grouped.std("sample", skipna=True)
    quant = grouped.quantile(list(quantiles), dim="sample", skipna=True)
    count = grouped.count("sample")

    def on_axis(array: xr.DataArray, **kwargs) -> xr.DataArray:
        return array.reindex(warming_level=np.asarray(target_levels), **kwargs).rename(
            warming_level="gwl"
        )

    data = {
        name: on_axis(mean),
        f"{name}_std": on_axis(std),
        **{
            f"{name}_p{round(q * 100)}": on_axis(quant.sel(quantile=q, drop=True))
            for q in quantiles
        },
    }
    pooled = xr.Dataset(data)
    pooled[count_name] = on_axis(count, fill_value=0).astype("int64")
    return pooled


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


def _reduce_grouped(grouped: xr.core.groupby.DataArrayGroupBy, dim: str, reducer: str) -> xr.DataArray:
    match reducer:
        case "mean":
            return grouped.mean(dim=dim)
        case "sum":
            return grouped.sum(dim=dim)
        case _:
            raise ValueError("reducer must be one of {'mean', 'sum'}")


def weighted_group_mean(
    frame: pd.DataFrame,
    *,
    group_keys: Sequence[str],
    value_columns: Sequence[str],
    weights: pd.Series,
    weight_key_column: str = "country",
) -> pd.DataFrame:
    """Weighted mean of each value column within groups, ignoring missing values.

    Each row is weighted by ``weights[row[weight_key_column]]``; rows whose
    ``weight_key_column`` value has no positive weight are dropped. For every group and column the result is
    ``sum(w * v over finite v) / sum(w over finite v)``, so a group stays finite as
    long as one member carries data, and is NaN when none does.
    """

    group_keys = list(group_keys)
    value_columns = list(value_columns)
    work = frame.copy()
    work["_w"] = work[weight_key_column].map(weights).astype(float)
    work = work[work["_w"] > 0]
    parts = {}
    for column in value_columns:
        finite = work[column].notna()
        parts[f"num_{column}"] = work["_w"] * work[column].fillna(0.0)
        parts[f"den_{column}"] = work["_w"] * finite
    work = work.assign(**parts)
    summed = work.groupby(group_keys, observed=True)[
        [f"num_{c}" for c in value_columns] + [f"den_{c}" for c in value_columns]
    ].sum()
    out = pd.DataFrame(index=summed.index)
    for column in value_columns:
        den = summed[f"den_{column}"]
        out[column] = np.where(den > 0, summed[f"num_{column}"] / den, np.nan)
    return out.reset_index()
