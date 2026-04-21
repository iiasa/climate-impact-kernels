from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr
from climateresponse import aggregate_seasonal


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
    """Aggregate monthly UCR to climatological seasons via QS-DEC."""

    if "time" not in ucr.dims:
        raise ValueError("ucr must define a time dimension")

    seasonal = aggregate_seasonal(ucr, time_dim="time").rename("ucr")
    return seasonal.transpose(
        "cooling",
        "gcm",
        "scenario",
        "season",
        "year",
        "country",
    )
