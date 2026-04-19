from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np
import pandas as pd
import xarray as xr

ArrayLike: TypeAlias = xr.DataArray | xr.Dataset

CLIMATOLOGICAL_SEASONS: dict[str, tuple[int, ...]] = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}
_SEASON_START_MONTHS: dict[int, str] = {12: "DJF", 3: "MAM", 6: "JJA", 9: "SON"}


def aggregate_monthly(
    data: ArrayLike,
    time_dim: str = "time",
) -> ArrayLike:
    """Aggregate sub-monthly data into monthly means."""

    _validate_time_dim(data=data, time_dim=time_dim)
    monthly_labels = _month_start_labels(data=data, time_dim=time_dim)
    return data.groupby(monthly_labels).mean(dim=time_dim).rename({monthly_labels.name: time_dim})


def aggregate_annual(
    data: ArrayLike,
    time_dim: str = "time",
) -> ArrayLike:
    """Aggregate monthly data into annual means."""

    _validate_time_dim(data=data, time_dim=time_dim)
    return data.groupby(f"{time_dim}.year").mean(dim=time_dim)


def aggregate_seasonal(
    data: ArrayLike,
    seasons: Mapping[str, Sequence[int]] | None = None,
    time_dim: str = "time",
) -> ArrayLike:
    """Aggregate monthly data into season-by-year means.

    Standard climatological seasons use a quarterly resample path that keeps
    dask-backed execution streamable and consolidates the output season/year
    chunks. Non-standard season definitions fall back to the legacy
    implementation to preserve the broader API surface.
    """

    _validate_time_dim(data=data, time_dim=time_dim)
    season_definitions = dict(seasons) if seasons is not None else dict(CLIMATOLOGICAL_SEASONS)
    if not season_definitions:
        raise ValueError("no season definitions selected any months")

    season_names = list(season_definitions)
    if _uses_climatological_quarters(season_definitions):
        return _aggregate_climatological_seasons(
            data=data,
            season_names=season_names,
            time_dim=time_dim,
        )

    seasonal_arrays = [
        _aggregate_single_season(
            data=data,
            season=season,
            months=months,
            time_dim=time_dim,
        )
        for season, months in season_definitions.items()
    ]
    available = [array for array in seasonal_arrays if array is not None]
    if not available:
        raise ValueError("no season definitions selected any months")

    combined = xr.concat(available, dim="season", join="outer")
    return combined.reindex(season=season_names)


def _validate_time_dim(data: ArrayLike, time_dim: str) -> None:
    if time_dim not in data.dims:
        raise ValueError(f"{time_dim!r} is not a dimension of the input data")


def _aggregate_single_season(
    data: ArrayLike,
    season: str,
    months: Sequence[int],
    time_dim: str,
) -> ArrayLike | None:
    month = data[time_dim].dt.month
    selected = data.where(month.isin(months), drop=True)
    if selected.sizes.get(time_dim, 0) == 0:
        return None

    year = selected[time_dim].dt.year
    if 12 in months and any(month_ < 12 for month_ in months):
        year = year + xr.where(selected[time_dim].dt.month == 12, 1, 0)

    return selected.groupby(year.rename("year")).mean(dim=time_dim).expand_dims(season=[season])


def _uses_climatological_quarters(seasons: Mapping[str, Sequence[int]]) -> bool:
    return all(tuple(months) == CLIMATOLOGICAL_SEASONS.get(season) for season, months in seasons.items())


def _aggregate_climatological_seasons(
    data: ArrayLike,
    season_names: Sequence[str],
    time_dim: str,
) -> ArrayLike:
    resampled = data.resample({time_dim: "QS-DEC"}).mean()
    if resampled.sizes.get(time_dim, 0) == 0:
        raise ValueError("no season definitions selected any months")

    month_values = np.asarray(resampled[time_dim].dt.month.values, dtype=int)
    season_labels = np.asarray([_SEASON_START_MONTHS.get(month) for month in month_values], dtype=object)
    keep = np.isin(season_labels, season_names)
    if not bool(keep.any()):
        raise ValueError("no season definitions selected any months")

    filtered = resampled.isel({time_dim: keep})
    filtered_months = month_values[keep]
    filtered_years = np.asarray(filtered[time_dim].dt.year.values, dtype=int)

    seasonal = filtered.assign_coords(
        season=(time_dim, season_labels[keep].tolist()),
        year=(time_dim, (filtered_years + (filtered_months == 12).astype(int)).tolist()),
    )
    seasonal = seasonal.drop_vars(time_dim, errors="ignore")
    seasonal = seasonal.set_index({time_dim: ("season", "year")}).unstack(time_dim)
    seasonal = seasonal.transpose("season", "year", *[dim for dim in data.dims if dim != time_dim])
    seasonal = seasonal.reindex(season=list(season_names))
    return _consolidate_seasonal_chunks(seasonal)


def _consolidate_seasonal_chunks(data: ArrayLike) -> ArrayLike:
    if data.chunks is None:
        return data
    return data.chunk({dim: -1 for dim in ("season", "year") if dim in data.dims})


def _month_start_labels(data: ArrayLike, time_dim: str) -> xr.DataArray:
    """Return month-start labels that preserve the source calendar."""

    time_coord = data[time_dim]
    labels = [_month_start(value) for value in time_coord.values]
    return xr.DataArray(
        labels,
        dims=[time_dim],
        coords={time_dim: time_coord},
        name=f"{time_dim}_month",
    )


def _month_start(value: object) -> object:
    if hasattr(value, "year") and hasattr(value, "month"):
        return type(value)(value.year, value.month, 1)
    timestamp = pd.Timestamp(value)
    return pd.Timestamp(year=timestamp.year, month=timestamp.month, day=1)
