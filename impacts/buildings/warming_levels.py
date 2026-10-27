"""Buildings energy intensity summarised by global warming level (GWL).

A warming-level summary gives, for each R12 region, building archetype,
urban/rural class and season (annual summaries sum over seasons), the mean,
standard deviation, 10th/50th/90th percentile and sample count of the CHILLED
energy intensity pooled over every (GCM, scenario, year) sample whose global mean
temperature falls in a 0.1 K warming-level bin.

Intensities are per floor area: seasonal summaries hold the seasonal mean of the
daily rate (MJ m-2 day-1); annual summaries hold the sum over seasons of that rate
times the season length (MJ m-2 yr-1). Country values are aggregated to R12 by a
weighted mean over the member countries that carry data.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from impacts.climatecore import (
    grid_cell_area,
    pool_by_warming_level,
    weighted_group_mean,
)

SCENARIOS = ("historical", "ssp126", "ssp370", "ssp585")
SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")
SEASON_DAYS = {"DJF": 90.25, "MAM": 92.0, "JJA": 92.0, "SON": 91.0}
VINTAGE_TO_ARCHETYPE = {
    "s1": "s1",
    "s2": "s2",
    "s3": "s3",
    "s51": "s51_std",
    "s52": "s52_low",
}
URBAN_RURAL_NAMES = {"rur": "rural", "urb": "urban"}
ISO3_R12_PATH = Path(__file__).parent / "data" / "iso3_to_r12.csv"

WARMING_LEVEL_SUMMARY_FILENAMES = {
    "EI_cool_seasonal": "region_EI_cool_gwl_binned_seasonal.nc",
    "EI_heat_seasonal": "region_EI_heat_gwl_binned_seasonal.nc",
    "EI_cool_annual": "region_EI_cool_gwl_binned_v2.nc",
    "EI_heat_annual": "region_EI_heat_gwl_binned_v2.nc",
}

_COUNTRY_COLUMNS = [
    "gcm",
    "scenario",
    "building_type",
    "urt",
    "vintage",
    "season",
    "year",
    "country",
    "cooling_energy_total",
    "heating_energy",
]
_CPA_HANDLING = (
    "R12-native CPA: CHN and RCPA aggregated directly from member countries "
    "(no R11 CPA intensity split)"
)


def load_iso3_to_r12() -> dict[str, str]:
    """Load the packaged map of ISO3 country codes to bare R12 region codes (``NAM``).

    The table (``data/iso3_to_r12.csv``) was resolved once from the MESSAGE R12
    region definitions and is fixed on purpose: a change upstream cannot shift a
    country's region silently.
    """

    table = pd.read_csv(ISO3_R12_PATH, comment="#", keep_default_na=False)
    return dict(zip(table["iso3"], table["region"], strict=True))


def country_population(
    country_raster: xr.DataArray, population_density: xr.DataArray
) -> pd.Series:
    """Total population per country: density (persons km-2) times cell area, summed.

    ``country_raster`` holds a country code per grid cell (empty string for no
    country) and ``population_density`` is on the same latitude/longitude grid;
    missing density counts as zero.
    """

    density = population_density.sortby("lat")
    if not (
        np.allclose(country_raster["lat"], density["lat"])
        and np.allclose(country_raster["lon"], density["lon"])
    ):
        raise ValueError("population density grid does not match the country raster grid")
    area_km2 = grid_cell_area(country_raster["lat"], country_raster["lon"]) * 1e-6
    population = density.fillna(0.0) * area_km2
    frame = pd.DataFrame(
        {"country": country_raster.values.ravel(), "pop": population.values.ravel()}
    )
    frame = frame[frame["country"] != ""]
    return frame.groupby("country")["pop"].sum()


def country_land_area(country_raster: xr.DataArray) -> pd.Series:
    """Land area (m2) per country: grid-cell areas summed over the country's cells."""

    area = grid_cell_area(country_raster["lat"], country_raster["lon"]).broadcast_like(
        country_raster
    )
    frame = pd.DataFrame(
        {"country": country_raster.values.ravel(), "area": area.values.ravel()}
    )
    frame = frame[frame["country"] != ""]
    return frame.groupby("country")["area"].sum()


def buildings_warming_level_summaries(
    country_seasonal_parquet: Path,
    reference_cool: xr.Dataset,
    reference_heat: xr.Dataset,
    iso3_to_r12: Mapping[str, str],
    weights: pd.Series,
    assignments: xr.Dataset,
    *,
    matrix_version: str,
    weighting: str,
) -> dict[str, xr.Dataset]:
    """Build the seasonal and annual cooling and heating warming-level summaries.

    Parameters
    ----------
    country_seasonal_parquet
        Per-archetype country-by-season CHILLED output with columns ``gcm``,
        ``scenario``, ``building_type``, ``urt`` (``rur``/``urb``), ``vintage``,
        ``season``, ``year``, ``country`` (ISO3), ``cooling_energy_total`` and
        ``heating_energy`` (MJ m-2 day-1, seasonal mean of the daily rate).
    reference_cool, reference_heat
        Reference annual summaries; their ``gwl`` axis becomes the output bin axis
        and their ``region``, ``arch`` and ``urt`` coordinates the output order.
    iso3_to_r12
        Country-to-region map (see ``load_iso3_to_r12``); other countries are dropped.
    weights
        Country weights indexed by ISO3 (population or land area); countries
        without positive weight are dropped.
    assignments
        Warming-level assignment per (gcm, scenario, year), as from
        ``isimip_warming_level_assignments`` over ``SCENARIOS``. Its ``gwl``
        values are warming levels in K above the 1850-1900 global-mean
        temperature baseline, in 0.1 K bins.
    matrix_version, weighting
        Free text recorded as the source in the output attributes. The caller must
        keep ``weighting`` consistent with the ``weights`` passed; it is not
        checked.

    Returns
    -------
    dict
        ``EI_cool_seasonal``, ``EI_heat_seasonal``, ``EI_cool_annual`` and
        ``EI_heat_annual`` datasets. Seasonal summaries have dims
        (region, arch, urt, season, gwl); annual summaries drop ``season``. Urban
        class ``total`` is the unweighted mean of ``rural`` and ``urban``. The
        annual value is the sum over seasons of the seasonal mean daily rate times
        the season length in days, and is missing when any season is missing,
        before pooling.
    """

    df = pd.read_parquet(country_seasonal_parquet, columns=_COUNTRY_COLUMNS)
    df = df[df["scenario"].isin(SCENARIOS)]
    df = df[df["country"].isin(iso3_to_r12)]
    df["region"] = df["country"].map(iso3_to_r12)
    df["arch"] = df["building_type"] + "_" + df["vintage"].map(VINTAGE_TO_ARCHETYPE)
    df["urt"] = df["urt"].map(URBAN_RURAL_NAMES)

    keys = ["gcm", "scenario", "arch", "urt", "season", "year", "region"]
    region_df = weighted_group_mean(
        df,
        group_keys=keys,
        value_columns=["cooling_energy_total", "heating_energy"],
        weights=weights,
    )

    cube = region_df.set_index(keys).to_xarray()
    rural_urban = cube.sel(urt=["rural", "urban"])
    total = rural_urban.mean("urt", skipna=True).expand_dims(urt=["total"])
    cube = xr.concat([cube, total], dim="urt")

    season_days = xr.DataArray(
        [SEASON_DAYS[str(s)] for s in cube["season"].values],
        dims="season",
        coords={"season": cube["season"]},
    )
    annual_cube = (cube * season_days).sum("season", skipna=False)

    summaries = {}
    for variable, reference, out_var, mode in [
        ("cooling_energy_total", reference_cool, "EI_cool", "cool"),
        ("heating_energy", reference_heat, "EI_heat", "heat"),
    ]:
        target_gwl = reference["gwl"].values
        common = {
            "reference": reference,
            "out_var": out_var,
            "mode": mode,
            "source": Path(country_seasonal_parquet).name,
            "matrix_version": matrix_version,
            "weighting": weighting,
        }
        pooled = pool_by_warming_level(cube[variable], assignments, target_gwl, name=out_var)
        summaries[f"{out_var}_seasonal"] = _describe(
            pooled,
            seasonal=True,
            units="MJ m-2 day-1 (seasonal mean of daily rate)",
            **common,
        )
        pooled = pool_by_warming_level(
            annual_cube[variable], assignments, target_gwl, name=out_var
        )
        summaries[f"{out_var}_annual"] = _describe(
            pooled, seasonal=False, units="MJ/m2/yr", **common
        )
    return summaries


def write_buildings_warming_level_summaries(
    summaries: Mapping[str, xr.Dataset], output_dir: Path
) -> list[Path]:
    """Write each warming-level summary from ``buildings_warming_level_summaries`` as a NetCDF file."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, dataset in summaries.items():
        path = output_dir / WARMING_LEVEL_SUMMARY_FILENAMES[key]
        dataset.to_netcdf(path)
        paths.append(path)
    return paths


def _describe(
    summary: xr.Dataset,
    *,
    reference: xr.Dataset,
    out_var: str,
    mode: str,
    units: str,
    seasonal: bool,
    source: str,
    matrix_version: str,
    weighting: str,
) -> xr.Dataset:
    dims = ["region", "arch", "urt"] + (["season"] if seasonal else []) + ["gwl"]
    summary = summary.reindex(
        region=reference["region"].values,
        arch=reference["arch"].values,
        urt=reference["urt"].values,
    )
    if seasonal:
        summary = summary.reindex(season=list(SEASON_ORDER))
    summary = summary.transpose(*dims)
    long_name = f"Energy intensity ({mode}) response to GWL by archetype" + (
        " and season" if seasonal else ""
    )
    for name in summary.data_vars:
        if name != "n_obs":
            summary[name].attrs = {"units": units, "long_name": long_name}
    summary["n_obs"].attrs = {"long_name": "Number of finite observations in GWL bin"}
    kind = "seasonal" if seasonal else "annual"
    summary.attrs = {
        "title": f"RIME-compatible {out_var} {kind} response to GWL",
        "description": (
            f"R12 regional {mode} energy intensity by archetype"
            + (" and climatological season" if seasonal else "")
            + ", aggregated to global warming levels. "
            + (
                "Values are the seasonal MEAN of the daily intensity rate."
                if seasonal
                else "Annualized before GWL pooling as the sum over seasons of "
                "seasonal-mean daily rate x season length in days "
                "(DJF 90.25, MAM 92, JJA 92, SON 91)."
            )
        ),
        "mode": mode,
        "source": source,
        "matrix_version": matrix_version,
        "weighting": weighting,
        "cpa_handling": _CPA_HANDLING,
        "scenario_basis": "historical + ssp126/370/585 pooled by GWL (ssp245 dropped: no GMT series)",
        "rime_compatible": "RegionArray",
        "bin_width": "0.1°C",
        "region_version": "R12",
        "season_convention": "QS-DEC DJF/MAM/JJA/SON",
    }
    return summary
