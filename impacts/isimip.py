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
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from impacts.core import aggregate_monthly


@dataclass(frozen=True)
class ISIMIPVariable:
    """A single ISIMIP3b variable with its product and native frequency."""

    name: str
    product: str  # "climate" or "watergap"
    frequency: str  # "daily" or "monthly"
    output_name: str | None = (
        None  # rename to this in the output dataset (default: keep name)
    )


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


ISIMIP3B_GCMS = (
    "GFDL-ESM4",
    "IPSL-CM6A-LR",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0",
    "UKESM1-0-LL",
)

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
    target_frequency: Literal["native", "monthly"] = "monthly",
) -> xr.DataArray:
    """Load a single ISIMIP variable at the requested frequency.

    ``target_frequency="monthly"`` (default) resamples daily sources to monthly
    via ``aggregate_monthly`` and preserves the prior contract. ``"native"``
    returns the variable at its on-disk frequency, which daily CHILLED and
    similar pipelines need.
    """

    files = discover_isimip_files(variable, gcm, scenario, paths, year_range=year_range)
    ds = xr.open_mfdataset(files, combine="by_coords", chunks=chunks)
    da = ds[variable.name]

    if target_frequency == "monthly" and variable.frequency == "daily":
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
    target_frequency: Literal["native", "monthly"] = "monthly",
) -> xr.Dataset:
    """Load multiple ISIMIP variables into a merged Dataset at the target frequency.

    Each variable is loaded independently (possibly from different products),
    resampled if ``target_frequency="monthly"`` and the source is daily,
    calendar-standardized, and merged on shared coordinates. With
    ``target_frequency="native"`` the merge requires the variables to share a
    frequency on disk.
    """

    arrays = {}
    for variable in variables:
        da = load_isimip_variable(
            variable,
            gcm,
            scenario,
            paths,
            year_range=year_range,
            chunks=chunks,
            target_frequency=target_frequency,
        )
        da = _standardize_time_index(da)
        key = variable.output_name or variable.name
        da = da.rename(key)
        da.attrs["isimip_product"] = variable.product
        da.attrs["isimip_native_frequency"] = variable.frequency
        arrays[key] = da

    return xr.merge(arrays.values(), join="inner")


def load_isimip_gmt(
    gmt_root: str | Path,
    gcm: str,
    scenario: str,
    *,
    year_range: tuple[int, int] | None = None,
) -> xr.DataArray:
    """Load annual-mean GMT anomaly for one ISIMIP GCM/scenario path.

    Future scenarios are stitched onto the corresponding historical run so the
    returned series spans the full historical+future path.
    """

    root = Path(gmt_root)
    historical = _load_isimip_gmt_csv(root=root, gcm=gcm, scenario="historical")

    if scenario == "historical":
        combined = historical
    else:
        future = _load_isimip_gmt_csv(root=root, gcm=gcm, scenario=scenario)
        combined = xr.concat([historical, future], dim="year")
        combined = _deduplicate_years(combined)

    if year_range is not None:
        combined = combined.sel(year=slice(*year_range))

    return combined.rename("gmt_anomaly")


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
    return (
        f"watergap2-2e_{gcm_lower}_*_{scenario}_*_default_{variable.name}_global_*.nc"
    )


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


def _standardize_time_index(da: xr.DataArray, time_dim: str = "time") -> xr.DataArray:
    """Convert cftime time coordinates to pandas timestamps for cross-product merging."""

    time_values = da[time_dim].values
    if len(time_values) == 0 or isinstance(time_values[0], pd.Timestamp):
        return da
    pandas_times = pd.DatetimeIndex([pd.Timestamp(str(t)) for t in time_values])
    return da.assign_coords({time_dim: pandas_times})


def _load_isimip_gmt_csv(
    *,
    root: Path,
    gcm: str,
    scenario: str,
) -> xr.DataArray:
    path, anomaly_column = _discover_isimip_gmt_file(root=root, gcm=gcm, scenario=scenario)
    frame = pd.read_csv(path, usecols=["date", anomaly_column])
    annual = (
        frame.assign(date=lambda data: pd.to_datetime(data["date"]))
        .assign(year=lambda data: data["date"].dt.year)
        .groupby("year", as_index=False)[anomaly_column]
        .mean()
    )
    return xr.DataArray(
        annual[anomaly_column].to_numpy(dtype=float),
        dims=["year"],
        coords={"year": annual["year"].to_numpy(dtype=int)},
        attrs={"source_path": str(path), "anomaly_column": anomaly_column},
    )


def _discover_isimip_gmt_file(*, root: Path, gcm: str, scenario: str) -> tuple[Path, str]:
    matches = sorted(root.glob(f"{gcm.lower()}_{scenario}_monthly_temperature_*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"no GMT CSV found for {gcm}/{scenario} under {root}"
        )
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"multiple GMT CSVs found for {gcm}/{scenario}: {names}")
    path = matches[0]
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    anomaly_columns = [column for column in columns if column.startswith("mean_anomaly")]
    if not anomaly_columns:
        raise ValueError(f"no mean_anomaly column found in {path}")
    return path, anomaly_columns[0]


def _deduplicate_years(gmt: xr.DataArray) -> xr.DataArray:
    frame = pd.DataFrame(
        {
            "year": np.asarray(gmt["year"].values, dtype=int),
            "value": np.asarray(gmt.values, dtype=float),
        }
    )
    deduplicated = frame.drop_duplicates(subset="year", keep="last").sort_values("year")
    return xr.DataArray(
        deduplicated["value"].to_numpy(dtype=float),
        dims=["year"],
        coords={"year": deduplicated["year"].to_numpy(dtype=int)},
        attrs=gmt.attrs,
    )
