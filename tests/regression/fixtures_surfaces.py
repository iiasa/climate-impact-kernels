"""Small made-up inputs for the functions that build warming-level summaries.

Everything is built from fixed seeds and literal tables, in the shape the library
functions read: a per-building-archetype country-by-season parquet, a
country-by-month thermoelectric parquet, monthly GMT tables, a country raster with
a population-density grid, and reference warming-level summaries that define the
output axes.

Edge regimes injected on purpose: a country with no cooling data at all, scattered
missing values, missing seasons (so some annual sums drop out), a country absent
from the weights, codes and names that map to no region or to a region the
reference lacks, a patched ISO3 code, a scenario without a GMT table, a reference
region with no data, and reference warming levels that skip populated bins.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

GCMS = ("GFDL-ESM4", "IPSL-CM6A-LR")
SCENARIOS_WITH_GMT = ("ssp126", "ssp370", "ssp585")
ALL_SCENARIOS = ("historical", "ssp126", "ssp245", "ssp370", "ssp585")

# ISO3 -> (lat block, lon block): each country covers a 2 x 2 block of a 6 x 10 grid;
# "GBR" is deliberately not on the raster, so it has no weight.
_RASTER_BLOCKS = {
    "NGA": (0, 0), "CMR": (0, 1), "COD": (0, 2), "KHM": (0, 3), "VNM": (0, 4),
    "CHN": (1, 0), "HKG": (1, 1), "CAN": (1, 2), "USA": (1, 3), "JPN": (1, 4),
    "AUS": (2, 0), "FRA": (2, 1), "TUR": (2, 2), "PSE": (2, 3), "ATA": (2, 4),
}  # fmt: skip
BUILDING_COUNTRIES = [*_RASTER_BLOCKS, "GBR", "XXA"]

# Warming levels of the reference warming-level summaries: populated bins, with some populated bins skipped.
BUILDINGS_GWL = [0.7, 0.8, 0.9, 1.0, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0]
THERMO_GWL = [0.6, 0.8, 0.9, 1.0, 1.1, 1.3, 1.4, 1.6, 1.7, 2.1]
REFERENCE_REGIONS = ["WEU", "NAM", "AFR", "CHN", "RCPA", "PAO", "SAS"]

THERMO_COUNTRIES = {
    # GAUL-style name: (has dry fleet, has wet fleet)
    "USA": (True, True), "Canada": (True, True), "China": (True, True),
    "France": (True, True), "ENGLAND & WALES": (True, True), "Japan": (False, True),
    "Nigeria": (True, True), "Cameroon": (True, True), "VIETNAM": (True, True),
    "CONGO DEM REP": (True, True), "Atlantis": (True, True),
}  # fmt: skip


@dataclasses.dataclass(frozen=True)
class SurfaceInputs:
    buildings_parquet: Path
    thermo_parquet: Path
    gmt_root: Path
    popdensity_path: Path
    annual_dir: Path
    country_raster: xr.DataArray
    popdensity: xr.DataArray
    gcms: tuple[str, ...]


def write_surface_inputs(root: Path) -> SurfaceInputs:
    root.mkdir(parents=True, exist_ok=True)
    raster, popdensity_path = _write_raster_and_density(root)
    gmt_root = _write_gmt(root / "gmt")
    annual_dir = _write_reference_surfaces(root / "annual")
    buildings = root / "chilled_country_seasonal_per_archetype.parquet"
    _buildings_frame().to_parquet(buildings, index=False)
    thermo = root / "thermoelectric_country_monthly.parquet"
    _thermo_frame().to_parquet(thermo, index=False)
    return SurfaceInputs(
        buildings_parquet=buildings,
        thermo_parquet=thermo,
        gmt_root=gmt_root,
        popdensity_path=popdensity_path,
        annual_dir=annual_dir,
        country_raster=raster,
        popdensity=xr.open_dataset(popdensity_path)[popdensity_path.stem],
        gcms=GCMS,
    )


# --- raster and population ------------------------------------------------------


def _write_raster_and_density(root: Path) -> tuple[xr.DataArray, Path]:
    lats = np.arange(6) * 0.5 + 10.25
    lons = np.arange(10) * 0.5 + 20.25
    codes = np.full((6, 10), "", dtype="U32")
    for code, (i, j) in _RASTER_BLOCKS.items():
        codes[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = code
    codes[0, 9] = ""
    codes[3, 3] = ""
    raster = xr.DataArray(codes, dims=("lat", "lon"), coords={"lat": lats, "lon": lons}, name="country")
    rng = np.random.default_rng(31)
    density = rng.uniform(1.0, 400.0, (6, 10))
    density[0, 9] = np.nan
    density[3, 2] = np.nan
    density[5, :] = np.nan
    # descending latitude, as the density files are stored
    path = root / "ssp2_2020.nc4"
    xr.Dataset(
        {"ssp2_2020": (("lat", "lon"), density[::-1])},
        coords={"lat": lats[::-1], "lon": lons},
    ).to_netcdf(path)
    return raster, path


# --- GMT -------------------------------------------------------------------------


def _write_gmt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    history = pd.date_range("2000-01-01", "2014-12-01", freq="MS")
    future = pd.date_range("2015-01-01", "2022-12-01", freq="MS")
    rates = {"ssp126": 0.03, "ssp370": 0.06, "ssp585": 0.09}
    for index, gcm in enumerate(GCMS):
        offset = (-0.15, 0.2)[index]
        years = history.year.to_numpy() + (history.month.to_numpy() - 1) / 12.0
        values = 0.6 + 0.045 * (years - 2000.0) + offset + rng.normal(0.0, 0.08, len(history))
        _write_gmt_csv(root, gcm, "historical", history, values)
        fyears = future.year.to_numpy() + (future.month.to_numpy() - 1) / 12.0
        for scenario, rate in rates.items():
            values = 1.25 + rate * (fyears - 2015.0) + offset + rng.normal(0.0, 0.08, len(future))
            _write_gmt_csv(root, gcm, scenario, future, values)
    return root


def _write_gmt_csv(root, gcm, scenario, dates, values) -> None:
    pd.DataFrame(
        {"date": dates.strftime("%Y-%m-%d"), "mean_anomaly_1850_1900": values}
    ).to_csv(root / f"{gcm.lower()}_{scenario}_monthly_temperature_gmt.csv", index=False)


# --- reference warming-level summaries ------------------------------------------------------------


def _write_reference_surfaces(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archs = [f"{b}_{v}" for b in ("mfh", "sfh") for v in ("s1", "s2", "s3", "s51_std", "s52_low")]
    for mode in ("cool", "heat"):
        xr.Dataset(
            {f"EI_{mode}": (("region", "arch", "urt", "gwl"), np.zeros((7, 10, 3, len(BUILDINGS_GWL))))},
            coords={
                "region": REFERENCE_REGIONS,
                "arch": archs,
                "urt": ["rural", "total", "urban"],
                "gwl": BUILDINGS_GWL,
            },
        ).to_netcdf(root / f"region_EI_{mode}_gwl_binned.nc")
    xr.Dataset(
        {"capacity_factor": (("cooling", "region", "gwl"), np.zeros((2, 7, len(THERMO_GWL))))},
        coords={"cooling": ["dry", "wet"], "region": REFERENCE_REGIONS, "gwl": THERMO_GWL},
    ).to_netcdf(root / "r12_thermoelectric_gwl.nc")
    return root


# --- buildings parquet -------------------------------------------------------------


def _buildings_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    scenario_years = {"historical": range(2002, 2015)}
    scenario_years |= {s: range(2015, 2023) for s in ALL_SCENARIOS[1:]}
    seasons = ("DJF", "MAM", "JJA", "SON")
    country_scale = {c: rng.uniform(0.5, 2.0) for c in BUILDING_COUNTRIES}
    rows = []
    for gcm_index, gcm in enumerate(GCMS):
        for scenario, years in scenario_years.items():
            warm = {"historical": 0.0, "ssp126": 0.3, "ssp245": 0.5, "ssp370": 0.7, "ssp585": 1.0}[scenario]
            for building_type in ("sfh", "mfh"):
                for urt in ("rur", "urb"):
                    for vintage in ("s1", "s2", "s51", "s52"):
                        shape = (len(list(years)), len(seasons), len(BUILDING_COUNTRIES))
                        base = rng.gamma(2.0, 0.5, shape)
                        for (yi, year), (si, season), (ci, country) in _product(years, seasons, BUILDING_COUNTRIES):
                            summer = season == "JJA"
                            cool = base[yi, si, ci] * country_scale[country] * (
                                (3.0 if summer else 0.4) + warm + 0.1 * gcm_index
                            )
                            heat = base[yi, si, ci] * country_scale[country] * (
                                0.3 if summer else 3.0 - 0.3 * warm
                            )
                            rows.append(
                                (gcm, scenario, building_type, urt, vintage, season, year, country, cool, heat)
                            )
    frame = pd.DataFrame(
        rows,
        columns=[
            "gcm", "scenario", "building_type", "urt", "vintage", "season", "year", "country",
            "cooling_energy_total", "heating_energy",
        ],
    )  # fmt: skip
    # no cooling data for one country; scattered missing values; whole missing seasons
    frame.loc[frame["country"] == "CAN", "cooling_energy_total"] = np.nan
    frame.loc[rng.random(len(frame)) < 0.04, "cooling_energy_total"] = np.nan
    frame.loc[rng.random(len(frame)) < 0.04, "heating_energy"] = np.nan
    missing = (
        (frame["country"].isin(["USA", "FRA", "NGA"]))
        & (frame["season"] == "DJF")
        & (frame["year"].isin([2002, 2022]))
    )
    frame = frame[~missing]
    frame = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    return frame


def _product(years, seasons, countries):
    for yi, year in enumerate(years):
        for si, season in enumerate(seasons):
            for ci, country in enumerate(countries):
                yield (yi, year), (si, season), (ci, country)


# --- thermoelectric parquet --------------------------------------------------------


def _thermo_frame() -> pd.DataFrame:
    rng = np.random.default_rng(23)
    spans = {"historical": ("2002-01-01", "2014-12-01")}
    spans |= {s: ("2015-01-01", "2021-12-01") for s in ALL_SCENARIOS[1:]}
    capacity = {
        (cooling, country): rng.uniform(50.0, 5000.0)
        for country in THERMO_COUNTRIES
        for cooling in ("dry", "wet")
    }
    rows = []
    for cooling in ("dry", "wet"):
        for gcm_index, gcm in enumerate(GCMS):
            for scenario, (start, end) in spans.items():
                times = pd.date_range(start, end, freq="MS")
                trend = np.linspace(0.0, 0.1 if scenario == "historical" else 0.25, len(times))
                for country, (has_dry, has_wet) in THERMO_COUNTRIES.items():
                    if not (has_dry if cooling == "dry" else has_wet):
                        continue
                    total = capacity[(cooling, country)]
                    sensitivity = 0.4 if cooling == "wet" else 0.15
                    summer = np.isin(times.month, (6, 7, 8))
                    ucr = np.clip(
                        0.98 - sensitivity * (trend + 0.05 * gcm_index) * np.where(summer, 1.5, 0.5)
                        + rng.normal(0.0, 0.02, len(times)),
                        0.2,
                        1.0,
                    )
                    for time, value in zip(times, ucr, strict=True):
                        rows.append((cooling, gcm, scenario, time, country, total * value, total))
    frame = pd.DataFrame(
        rows,
        columns=["cooling", "gcm", "scenario", "time", "country", "usable_capacity_mw", "total_mw"],
    )
    frame.loc[rng.random(len(frame)) < 0.01, "usable_capacity_mw"] = np.nan
    frame = frame[~((frame["country"] == "Cameroon") & (frame["cooling"] == "wet") & (frame["time"].dt.month == 7))]
    return frame.sample(frac=1.0, random_state=5).reset_index(drop=True)
