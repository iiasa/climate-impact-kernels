"""Thermoelectric usable capacity by global warming level: a warming-level summary.

The summary gives, for each cooling type, R12 region and climatological season,
the mean, standard deviation, 10th/50th/90th percentile and sample count of the
usable capacity ratio (usable capacity over nameplate capacity, 0 to 1) pooled
over every (GCM, scenario, year) sample whose global-mean temperature falls in a
0.1 K warming-level bin. Country ratios are aggregated to R12 by a
nameplate-capacity-weighted mean over the member countries that carry data.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from impacts.climatecore import pool_by_warming_level
from impacts.climatecore.regions import resolve_region_labels

logger = logging.getLogger(__name__)

SCENARIOS = ("historical", "ssp126", "ssp370", "ssp585")
SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")
WARMING_LEVEL_SUMMARY_FILENAME = "r12_thermoelectric_gwl_seasonal.nc"


def build_country_name_to_r12(names: Sequence[str]) -> dict[str, str]:
    """Map country names to bare R12 region codes (``NAM``, not ``R12_NAM``).

    Each name is resolved on its own through the R12 member-country aliases;
    names that do not resolve are dropped from the result (the warming-level summary builder logs the dropped names at INFO).
    """

    mapping: dict[str, str] = {}
    for name in names:
        resolved = resolve_region_labels([name], region_set="message_r12", strict=False)
        if resolved:
            mapping[name] = resolved[0][len("R12_") :]
    return mapping


def thermoelectric_warming_level_summary(
    country_monthly_parquet: Path,
    reference: xr.Dataset,
    assignments: xr.Dataset,
) -> xr.Dataset:
    """Build the seasonal usable-capacity warming-level summary by cooling type and R12 region.

    Parameters
    ----------
    country_monthly_parquet
        Country-by-month simulation output with columns ``cooling``, ``gcm``,
        ``scenario``, ``time``, ``country``, ``usable_capacity_mw`` and
        ``total_mw``. The ratio ``usable_capacity_mw / total_mw`` assumes both
        cover the same units: a ``total_mw`` that includes units of unknown
        usable capacity (see ``valid_mw`` from ``aggregate_thermoelectric_regions``)
        counts them as zero. Monthly ratios are collapsed to December-February,
        March-May, June-August and September-November seasons (a season
        with a missing month is missing) before aggregation.
    reference
        Reference warming-level summary; its ``gwl`` axis becomes the output bin axis and its
        ``cooling`` and ``region`` coordinates the output order.
    assignments
        Warming-level assignment per (gcm, scenario, year), as from
        ``isimip_warming_level_assignments`` over ``SCENARIOS``.

    Notes
    -----
    The ``source`` attribute is fixed text, not the path of ``country_monthly_parquet``.

    Returns
    -------
    xarray.Dataset
        ``capacity_factor``, ``capacity_factor_std``, ``capacity_factor_p10``,
        ``capacity_factor_p50``, ``capacity_factor_p90`` and ``n_observations``
        over (cooling, region, season, gwl). The country weight is the mean
        ``total_mw`` over time; countries that do not map to an R12 region are
        dropped.
    """

    usable_capacity_ratio = load_thermoelectric_usable_capacity_ratio(country_monthly_parquet)
    seasonal = aggregate_thermoelectric_seasonal(usable_capacity_ratio)
    seasonal = seasonal.sel(
        scenario=[s for s in seasonal["scenario"].values if s in SCENARIOS]
    )

    capacity = (
        pd.read_parquet(country_monthly_parquet, columns=["cooling", "country", "total_mw"])
        .groupby(["cooling", "country"])["total_mw"]
        .mean()
    )

    name_to_r12 = build_country_name_to_r12(seasonal["country"].values)
    countries = [c for c in seasonal["country"].values if c in name_to_r12]
    dropped = [c for c in seasonal["country"].values if c not in name_to_r12]
    logger.info("%d countries mapped to R12; dropped %s", len(countries), dropped)
    seasonal = seasonal.sel(country=countries)
    regions = np.array([name_to_r12[c] for c in countries])

    region_cube = _capacity_weighted_regions(seasonal, capacity, countries, regions)

    summary = pool_by_warming_level(
        region_cube,
        assignments,
        reference["gwl"].values,
        name="capacity_factor",
        count_name="n_observations",
    )
    summary = summary.reindex(
        cooling=reference["cooling"].values,
        region=reference["region"].values,
        season=list(SEASON_ORDER),
    ).transpose("cooling", "region", "season", "gwl")
    summary.attrs = {
        "title": "R12 Regional Capacity Factor by Global Warming Level and Season",
        "description": (
            "Capacity-weighted thermoelectric usable capacity ratio from plant-level simulation, aggregated "
            "to R12 regions and climatological season, binned by global warming level across the four-scenario "
            "basis (historical, ssp126/370/585)."
        ),
        "units": "fraction (0-1)",
        "source": "thermoelectric_country_monthly.parquet, capacity-weighted aggregation",
        "cooling_types": "dry = air-cooled, wet = once-through + recirculating freshwater",
        "season_convention": "QS-DEC DJF/MAM/JJA/SON",
    }
    return summary


def write_thermoelectric_warming_level_summary(
    summary: xr.Dataset, output_dir: Path
) -> Path:
    """Write the warming-level summary as a NetCDF file in ``output_dir`` and return its path."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / WARMING_LEVEL_SUMMARY_FILENAME
    summary.to_netcdf(path)
    return path


def _capacity_weighted_regions(
    seasonal: xr.DataArray,
    capacity: pd.Series,
    countries: Sequence[str],
    regions: np.ndarray,
) -> xr.DataArray:
    """Capacity-weighted mean over member countries with finite data, per cooling type."""

    finite = seasonal.notnull()
    parts = []
    for cooling in seasonal["cooling"].values:
        weight = xr.DataArray(
            [float(capacity.get((cooling, c), 0.0)) for c in countries],
            dims="country",
            coords={"country": countries},
        )
        block = seasonal.sel(cooling=cooling)
        mask = finite.sel(cooling=cooling)
        num = (block.fillna(0.0) * weight).assign_coords(region=("country", regions))
        den = (weight * mask).assign_coords(region=("country", regions))
        region_val = num.groupby("region").sum("country") / den.groupby("region").sum(
            "country"
        )
        region_val = region_val.where(den.groupby("region").sum("country") > 0)
        parts.append(region_val.expand_dims(cooling=[cooling]))
    return xr.concat(parts, dim="cooling")


_MONTH_TO_SEASON = {12: "DJF", 3: "MAM", 6: "JJA", 9: "SON"}


def load_thermoelectric_usable_capacity_ratio(source_path: str | Path) -> xr.DataArray:
    """Load the country-level thermoelectric parquet as an xarray field of the usable capacity ratio.

    The ratio ``usable_capacity_mw / total_mw`` assumes both columns cover the same
    units. ``aggregate_thermoelectric_regions`` reports ``valid_mw``, the MW of
    units whose usable capacity is known; a ``total_mw`` that includes units of
    unknown capacity counts them as zero usable capacity.
    """

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
    usable_capacity_ratio = (
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
    usable_capacity_ratio.attrs["definition"] = "usable_capacity_mw / total_mw"
    return usable_capacity_ratio


def aggregate_thermoelectric_seasonal(usable_capacity_ratio: xr.DataArray) -> xr.DataArray:
    """Aggregate the monthly usable capacity ratio to climatological seasons via QS-DEC.

    The thermoelectric country parquet has gaps along ``time`` for each scenario:
    each scenario only populates its own historical or future year range.
    The month axis runs from the December before the first year, and the
    resample is strict (``skipna=False``), so a DJF bucket missing any of its
    months is NaN rather than a partial mean.
    """

    if "time" not in usable_capacity_ratio.dims:
        raise ValueError("usable_capacity_ratio must define a time dimension")

    time_min = pd.Timestamp(usable_capacity_ratio["time"].min().item())
    time_max = pd.Timestamp(usable_capacity_ratio["time"].max().item())
    complete_time = pd.date_range(
        f"{time_min.year - 1}-12-01",
        f"{time_max.year + 1}-02-01",
        freq="MS",
    )
    complete = usable_capacity_ratio.reindex(time=complete_time)
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
