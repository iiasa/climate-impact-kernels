from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_MONTH_TO_SEASON = {12: "DJF", 3: "MAM", 6: "JJA", 9: "SON"}


def load_thermoelectric_ucr(source_path: str | Path) -> xr.DataArray:
    """Load the country-level thermoelectric parquet as an xarray UCR field."""

    monthly = pd.read_parquet(
        source_path,
        columns=[
            "cooling",
            "gcm",
            "scenario",
            "time",
            "country",
            "usable_capacity_mw",
            "total_mw",
        ],
    )
    if monthly["total_mw"].le(0).any():
        raise ValueError("total_mw must be strictly positive")

    monthly = monthly.assign(
        time=lambda data: pd.to_datetime(data["time"]),
        ucr=lambda data: data["usable_capacity_mw"] / data["total_mw"],
    )
    ucr = (
        monthly.set_index(["cooling", "gcm", "scenario", "time", "country"])["ucr"]
        .to_xarray()
        .rename("ucr")
        .transpose("cooling", "gcm", "scenario", "time", "country")
        .sortby("cooling")
        .sortby("gcm")
        .sortby("scenario")
        .sortby("time")
        .sortby("country")
    )
    ucr.attrs["definition"] = "usable_capacity_mw / total_mw"
    return ucr


def aggregate_thermoelectric_seasonal(ucr: xr.DataArray) -> xr.DataArray:
    """Aggregate monthly UCR to climatological seasons via QS-DEC.

    The thermoelectric country parquet is scenario-sparse along ``time``:
    each scenario only populates its own historical or future year range.
    Strict (``skipna=False``) resample after reindexing to a complete
    month axis ensures that boundary DJF buckets missing Jan or Feb fall
    out as NaN instead of leaking a partial single-month mean.
    """

    if "time" not in ucr.dims:
        raise ValueError("ucr must define a time dimension")

    time_min = pd.Timestamp(ucr["time"].min().item())
    time_max = pd.Timestamp(ucr["time"].max().item())
    complete_time = pd.date_range(
        f"{time_min.year}-01-01",
        f"{time_max.year + 1}-02-01",
        freq="MS",
    )
    complete = ucr.reindex(time=complete_time)
    resampled = complete.resample(time="QS-DEC").mean(skipna=False)

    months = np.asarray(resampled["time"].dt.month.values, dtype=int)
    years = np.asarray(resampled["time"].dt.year.values, dtype=int)
    keep = np.isin(months, tuple(_MONTH_TO_SEASON))
    resampled = resampled.isel(time=keep)
    months = months[keep]
    years = years[keep]
    season_names = [_MONTH_TO_SEASON[int(m)] for m in months]
    year_values = years + np.where(months == 12, 1, 0)

    seasonal = (
        resampled.assign_coords(
            season=("time", season_names),
            year=("time", year_values),
        )
        .drop_vars("time", errors="ignore")
        .set_index(time=("season", "year"))
        .unstack("time")
        .reindex(season=list(_MONTH_TO_SEASON.values()))
    )
    return seasonal.rename("ucr").transpose(
        "cooling",
        "gcm",
        "scenario",
        "season",
        "year",
        "country",
    )
