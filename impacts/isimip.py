"""ISIMIP3b file discovery and variable loading from pdrive.

Two file families on pdrive:

- **Bias-adjusted GCM climate** (``InputData/climate_updated/bias-adjusted/``):
  daily decadal NetCDFs for ``tas``, ``hurs``, ``ps``, etc.
- **WaterGAP2-2e hydrology** (``OutputData/WaterGAP2-2e/``):
  ``dis`` as daily decadal, ``triver`` as monthly single-file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import xarray as xr
from climateresponse import aggregate_monthly


@dataclass(frozen=True)
class ISIMIPVariable:
    """A single ISIMIP3b variable with its product and native frequency."""

    name: str
    product: str       # "climate" or "watergap"
    frequency: str     # "daily" or "monthly"


@dataclass(frozen=True)
class ISIMIPPaths:
    """Root paths for the two ISIMIP3b products on pdrive."""

    climate_root: Path
    watergap_root: Path


def isimip3b_paths(watxene_root: Path | None = None) -> ISIMIPPaths:
    """Resolve ISIMIP3b paths from an explicit root or ``$WATXENE_DATA_PATH``."""

    if watxene_root is None:
        env = os.environ.get("WATXENE_DATA_PATH")
        if env is None:
            raise EnvironmentError(
                "WATXENE_DATA_PATH is not set and no explicit watxene_root was provided"
            )
        watxene_root = Path(env)
    isimip3b = watxene_root / "ISIMIP" / "ISIMIP3b"
    return ISIMIPPaths(
        climate_root=isimip3b / "InputData" / "climate_updated" / "bias-adjusted",
        watergap_root=isimip3b / "OutputData" / "WaterGAP2-2e",
    )

# ── Pre-defined variable sets for cooling pipelines ──────────────────────────

WET_COOLING_VARIABLES = (
    ISIMIPVariable("tas", "climate", "daily"),
    ISIMIPVariable("hurs", "climate", "daily"),
    ISIMIPVariable("ps", "climate", "daily"),
    ISIMIPVariable("dis", "watergap", "daily"),
    ISIMIPVariable("triver", "watergap", "monthly"),
)

DRY_COOLING_VARIABLES = (
    ISIMIPVariable("tas", "climate", "daily"),
)

ISIMIP3B_GCMS = (
    "GFDL-ESM4",
    "IPSL-CM6A-LR",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0",
    "UKESM1-0-LL",
)

WATERGAP_SOCPATH = {
    "historical": "histsoc",
    "picontrol": "histsoc",
    "ssp126": "2015soc-from-histsoc",
    "ssp370": "2015soc-from-histsoc",
    "ssp585": "2015soc-from-histsoc",
}


# ── File discovery ───────────────────────────────────────────────────────────


def discover_isimip_files(
    variable: ISIMIPVariable,
    gcm: str,
    scenario: str,
    paths: ISIMIPPaths,
    *,
    year_range: tuple[int, int] | None = None,
) -> list[Path]:
    """Find NetCDF files for a variable/GCM/scenario combination.

    Returns files sorted by name (which sorts by time period).
    Filters to files overlapping year_range when provided.
    """

    directory = _variable_directory(variable, gcm, scenario, paths)
    pattern = _file_glob_pattern(variable, gcm, scenario)
    files = sorted(directory.glob(pattern))

    if year_range is not None:
        files = [f for f in files if _file_overlaps_years(f, year_range)]

    if not files:
        raise FileNotFoundError(
            f"no ISIMIP files for {variable.name}/{gcm}/{scenario} in {directory} "
            f"(pattern: {pattern}, year_range: {year_range})"
        )
    return files


def load_isimip_variable(
    variable: ISIMIPVariable,
    gcm: str,
    scenario: str,
    paths: ISIMIPPaths,
    *,
    year_range: tuple[int, int] | None = None,
    chunks: dict | None = None,
) -> xr.DataArray:
    """Load a single ISIMIP variable, resampling daily sources to monthly."""

    files = discover_isimip_files(variable, gcm, scenario, paths, year_range=year_range)
    ds = xr.open_mfdataset(files, combine="by_coords", chunks=chunks)
    da = ds[variable.name]

    if variable.frequency == "daily":
        da = aggregate_monthly(da)

    if year_range is not None:
        da = da.sel(time=slice(f"{year_range[0]}", f"{year_range[1]}"))

    return da


def load_isimip_dataset(
    variables: Sequence[ISIMIPVariable],
    gcm: str,
    scenario: str,
    paths: ISIMIPPaths,
    *,
    year_range: tuple[int, int] | None = None,
    chunks: dict | None = None,
) -> xr.Dataset:
    """Load multiple ISIMIP variables into a merged monthly Dataset.

    Each variable is loaded independently (possibly from different products),
    resampled to monthly if daily, calendar-standardized, and then merged
    on shared (time, lat, lon) coordinates.
    """

    arrays = {}
    for variable in variables:
        da = load_isimip_variable(
            variable, gcm, scenario, paths,
            year_range=year_range,
            chunks=chunks,
        )
        arrays[variable.name] = da

    merged = xr.merge(arrays.values(), join="inner")

    for var in merged.data_vars:
        source = next(v for v in variables if v.name == var)
        merged[var].attrs["isimip_product"] = source.product
        merged[var].attrs["isimip_native_frequency"] = source.frequency

    return merged


# ── Internal helpers ─────────────────────────────────────────────────────────


def _variable_directory(
    variable: ISIMIPVariable,
    gcm: str,
    scenario: str,
    paths: ISIMIPPaths,
) -> Path:
    if variable.product == "climate":
        return paths.climate_root / scenario / gcm
    return paths.watergap_root / scenario / gcm


def _file_glob_pattern(
    variable: ISIMIPVariable,
    gcm: str,
    scenario: str,
) -> str:
    gcm_lower = gcm.lower()
    if variable.product == "climate":
        return f"{gcm_lower}_*_{scenario}_{variable.name}_global_*.nc"
    socpath = WATERGAP_SOCPATH.get(scenario, "2015soc-from-histsoc")
    return f"watergap2-2e_{gcm_lower}_*_{scenario}_{socpath}_default_{variable.name}_global_*.nc"


def _file_overlaps_years(path: Path, year_range: tuple[int, int]) -> bool:
    """Check if a file's year span overlaps the requested range.

    ISIMIP filenames end with ``_{start_year}_{end_year}.nc``.
    """

    stem = path.stem
    parts = stem.rsplit("_", 2)
    if len(parts) < 3:
        return True
    try:
        file_start = int(parts[-2])
        file_end = int(parts[-1])
    except ValueError:
        return True
    return file_end >= year_range[0] and file_start <= year_range[1]
