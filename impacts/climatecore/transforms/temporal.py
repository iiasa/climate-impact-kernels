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
_MONTH_CODES: dict[int, str] = {
    1: "JAN",
    2: "FEB",
    3: "MAR",
    4: "APR",
    5: "MAY",
    6: "JUN",
    7: "JUL",
    8: "AUG",
    9: "SEP",
    10: "OCT",
    11: "NOV",
    12: "DEC",
}


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
    """Aggregate time-indexed data into season-by-year means.

    `seasons=` accepts quarterly tilings of the calendar year and subsets of
    those tilings. Each season must be exactly three contiguous months, and all
    selected seasons must share the same quarterly phase.
    """

    _validate_time_dim(data=data, time_dim=time_dim)
    season_definitions = dict(seasons) if seasons is not None else dict(CLIMATOLOGICAL_SEASONS)
    if not season_definitions:
        raise ValueError("no season definitions selected any months")
    return _aggregate_quarterly_seasons(
        data=data,
        season_definitions=season_definitions,
        time_dim=time_dim,
    )


def _validate_time_dim(data: ArrayLike, time_dim: str) -> None:
    if time_dim not in data.dims:
        raise ValueError(f"{time_dim!r} is not a dimension of the input data")


def _aggregate_quarterly_seasons(
    data: ArrayLike,
    season_definitions: Mapping[str, Sequence[int]],
    time_dim: str,
) -> ArrayLike:
    season_names = list(season_definitions)
    season_start_months = _validate_quarterly_seasons(season_definitions)
    anchor_month = next(iter(season_start_months))
    resampled = data.resample({time_dim: _quarterly_frequency(anchor_month)}).mean()
    if resampled.sizes.get(time_dim, 0) == 0:
        raise ValueError("no season definitions selected any months")

    month_values = np.asarray(resampled[time_dim].dt.month.values, dtype=int)
    keep = np.isin(month_values, tuple(season_start_months))
    if not bool(np.any(keep)):
        raise ValueError("no season definitions selected any months")

    filtered = resampled.isel({time_dim: keep})
    filtered_months = month_values[keep]
    filtered_years = np.asarray(filtered[time_dim].dt.year.values, dtype=int)
    year_offsets = np.asarray(
        [_season_year_offset(start_month) for start_month in filtered_months],
        dtype=int,
    )

    seasonal = filtered.assign_coords(
        season=(time_dim, [season_start_months[start_month] for start_month in filtered_months]),
        year=(time_dim, filtered_years + year_offsets),
    )
    seasonal = seasonal.drop_vars(time_dim, errors="ignore")
    seasonal = seasonal.set_index({time_dim: ("season", "year")}).unstack(time_dim)
    seasonal = seasonal.transpose(
        "season",
        "year",
        *[dim for dim in data.dims if dim != time_dim],
        missing_dims="ignore",
    )
    seasonal = seasonal.reindex(season=season_names)
    return _consolidate_seasonal_chunks(seasonal)


def _validate_quarterly_seasons(seasons: Mapping[str, Sequence[int]]) -> dict[int, str]:
    season_start_months: dict[int, str] = {}
    quarterly_phase: int | None = None

    for season, months in seasons.items():
        normalized_months = _normalize_months(season=season, months=months)
        start_month = normalized_months[0]
        expected_months = _quarter_months(start_month)
        if normalized_months != expected_months:
            raise ValueError(
                f"{season!r} must be a contiguous three-month season in calendar order; "
                f"got {normalized_months!r}"
            )

        phase = start_month % 3
        if quarterly_phase is None:
            quarterly_phase = phase
        elif phase != quarterly_phase:
            raise ValueError("season definitions must share a quarterly phase")

        if start_month in season_start_months:
            raise ValueError(f"duplicate quarterly start month {start_month} in season definitions")

        season_start_months[start_month] = season

    return season_start_months


def _normalize_months(season: str, months: Sequence[int]) -> tuple[int, int, int]:
    normalized_months = tuple(int(month) for month in months)
    if len(normalized_months) != 3 or any(month < 1 or month > 12 for month in normalized_months):
        raise ValueError(f"{season!r} must define exactly three calendar months")
    return normalized_months


def _quarter_months(start_month: int) -> tuple[int, int, int]:
    return tuple(((start_month + offset - 1) % 12) + 1 for offset in range(3))


def _quarterly_frequency(start_month: int) -> str:
    return f"QS-{_MONTH_CODES[start_month]}"


def _season_year_offset(start_month: int) -> int:
    return int(_quarter_months(start_month)[-1] < start_month)


def _consolidate_seasonal_chunks(data: ArrayLike) -> ArrayLike:
    if not _is_dask_backed(data):
        return data
    return data.chunk({dim: -1 for dim in ("season", "year") if dim in data.dims})


def _is_dask_backed(data: ArrayLike) -> bool:
    if isinstance(data, xr.DataArray):
        return _is_dask_array(data.data)
    return any(_is_dask_array(variable.data) for variable in data.data_vars.values())


def _is_dask_array(values: object) -> bool:
    return hasattr(values, "__dask_graph__") and hasattr(values, "chunks")


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
