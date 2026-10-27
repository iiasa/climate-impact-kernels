"""ISIMIP file discovery and variable loading from a local copy of the archive.

Three file families, laid out under a data root as on the ISIMIP archive:

- **Bias-adjusted GCM climate** (``InputData/climate_updated/bias-adjusted/``):
  daily decadal NetCDFs for ``tas``, ``hurs``, ``ps``, etc.
- **WaterGAP2-2e hydrology** (``OutputData/WaterGAP2-2e/``):
  ``dis`` as daily decadal, ``triver`` as monthly single-file.
- **GCM ocean** (``InputData/climate/ocean/uncorrected/global/monthly/``):
  monthly sea-surface temperature ``tos`` on a 1-degree grid, not bias-adjusted.
  Its time axis counts months on a 365-day
  calendar, which is decoded to month starts here, and it is copied onto the
  finer grid of the other variables when loaded with them.

The same interface also resolves the ISIMIP3a GSWP3-W5E5 counterclim inputs
(observed weather with the warming trend removed) used by the thermoelectric
runs that sample years from a fixed reference period.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from impacts.climatecore import (
    aggregate_monthly,
    build_warming_level_assignments,
    refine_to_grid,
)


@dataclass(frozen=True)
class ISIMIPVariable:
    """A single ISIMIP3b variable with its product and native frequency."""

    name: str
    product: str  # "climate", "watergap" or "ocean"
    frequency: str  # "daily" or "monthly"
    output_name: str | None = (
        None  # rename to this in the output dataset (default: keep name)
    )
    resolution: str | None = None  # e.g. "60arcmin" where the archive carries several


DEFAULT_WATERGAP_SOC = "2015soc"
DEFAULT_WATERGAP_HISTORICAL_SOC = "histsoc"


@dataclass(frozen=True)
class ISIMIPPaths:
    """Root paths and directory layout for a pair of ISIMIP products.

    ``soc`` names the WaterGAP socio-economic forcing. The ``isimip3a_counterclim``
    layout reads it for every scenario, and the ``isimip3b`` layout reads it for
    every non-historical scenario, because an SSP directory holds several forcings
    covering the same years. ISIMIP3b historical runs sit under a single forcing,
    named by ``historical_soc``. The forcing changes what the result
    means, not only which files are read: ``2015soc`` holds human water use fixed at 2015 levels, so a
    counterclim reference built on it isolates climate change from water-use
    change instead of describing the actual past. Discovery never falls back to
    a forcing other than the one named.
    """

    climate_root: Path
    watergap_root: Path
    layout: Literal["isimip3b", "isimip3a_counterclim"] = "isimip3b"
    soc: str = DEFAULT_WATERGAP_SOC
    historical_soc: str = DEFAULT_WATERGAP_HISTORICAL_SOC
    ocean_root: Path | None = None


def isimip3b_paths(
    watxene_root: Path | None = None,
    *,
    soc: str = DEFAULT_WATERGAP_SOC,
    historical_soc: str = DEFAULT_WATERGAP_HISTORICAL_SOC,
) -> ISIMIPPaths:
    """Resolve ISIMIP3b paths from an explicit root or ``$WATXENE_DATA_PATH``.

    ``soc`` selects the WaterGAP forcing for SSP and other non-historical
    scenarios; ``historical_soc`` selects it for ``historical``, which the archive
    holds only under ``histsoc``. See :class:`ISIMIPPaths`.
    """
    isimip3b = _data_root(watxene_root) / "ISIMIP" / "ISIMIP3b"
    return ISIMIPPaths(
        climate_root=isimip3b / "InputData" / "climate_updated" / "bias-adjusted",
        watergap_root=isimip3b / "OutputData" / "WaterGAP2-2e",
        soc=soc,
        historical_soc=historical_soc,
        ocean_root=isimip3b / "InputData" / "climate" / "ocean" / "uncorrected" / "global" / "monthly",
    )


def isimip3a_counterclim_paths(
    watxene_root: Path | None = None,
    *,
    soc: str = DEFAULT_WATERGAP_SOC,
) -> ISIMIPPaths:
    """Resolve the ISIMIP3a GSWP3-W5E5 counterclim reference layout.

    Atmospheric drivers are mirrored under ``InputData``. WaterGAP
    files retain the ISIMIP archive layout under ``OutputData``. ``soc`` selects
    the WaterGAP water-use forcing (see :class:`ISIMIPPaths`).
    """

    isimip3a = _data_root(watxene_root) / "ISIMIP" / "ISIMIP3a"
    return ISIMIPPaths(
        climate_root=(
            isimip3a
            / "InputData"
            / "climate"
            / "counterclim"
            / "global"
            / "daily"
            / "historical"
            / "GSWP3-W5E5"
        ),
        watergap_root=(
            isimip3a
            / "OutputData"
            / "water_global"
            / "WaterGAP2-2e"
            / "gswp3-w5e5"
            / "historical"
        ),
        layout="isimip3a_counterclim",
        soc=soc,
    )


def _data_root(watxene_root: Path | None) -> Path:
    if watxene_root is not None:
        return Path(watxene_root)
    env = os.environ.get("WATXENE_DATA_PATH")
    if env is None:
        raise OSError("WATXENE_DATA_PATH is not set and no explicit watxene_root was provided")
    return Path(env)


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
    pattern = _file_glob_pattern(variable, gcm, scenario, paths)
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
    via ``aggregate_monthly``. ``"native"`` returns the variable at its on-disk
    frequency, which daily CHILLED and similar pipelines need.
    """

    files = discover_isimip_files(variable, gcm, scenario, paths, year_range=year_range)
    if variable.product == "ocean":
        ds = _decode_month_counts(
            xr.open_mfdataset(files, combine="by_coords", chunks=chunks, decode_times=False)
        )
    else:
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
    frequency on disk. A variable on a coarser grid that nests the finest grid
    present (the 1-degree ocean grid inside the 0.5-degree land grid) is copied
    onto that finest grid first; see ``refine_to_grid``.
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

    if arrays:
        finest = min(arrays.values(), key=lambda array: _grid_spacing(array["lat"]))
        for key, da in arrays.items():
            if da.sizes["lat"] > 1 and _grid_spacing(da["lat"]) > _grid_spacing(finest["lat"]):
                arrays[key] = refine_to_grid(da, lat=finest["lat"], lon=finest["lon"])

    return xr.merge(arrays.values(), join="exact")


def load_isimip_gmt(
    gmt_root: str | Path,
    gcm: str,
    scenario: str,
    *,
    year_range: tuple[int, int] | None = None,
) -> xr.DataArray:
    """Load the annual-mean global mean temperature (GMT) change for one ISIMIP GCM/scenario path.

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


def isimip_warming_level_assignments(
    gmt_root: str | Path,
    gcms: Sequence[str],
    scenarios: Sequence[str],
    *,
    bin_width: float = 0.1,
) -> xr.Dataset:
    """Assign every (gcm, scenario, year) to its nearest warming-level bin.

    The annual-mean GMT of each path comes from ``load_isimip_gmt`` and is
    matched to the nearest multiple of ``bin_width`` (default 0.1 K), in K above
    the 1850-1900 GMT baseline, as in ``build_warming_level_assignments``. The
    levels start at 0 K and stop at the highest GMT of any path, so every year
    goes to its nearest level, and a year below 0 K goes to 0 K.
    """

    gmt_by_path = {
        (gcm, scenario): load_isimip_gmt(gmt_root, gcm, scenario)
        for gcm in gcms
        for scenario in scenarios
    }
    return build_warming_level_assignments(
        gmt_by_path, path_columns=("gcm", "scenario"), bin_width=bin_width
    )


def _variable_directory(
    variable: ISIMIPVariable,
    gcm: str,
    scenario: str,
    paths: ISIMIPPaths,
) -> Path:
    if paths.layout == "isimip3a_counterclim" and variable.product == "climate":
        return paths.climate_root
    if paths.layout == "isimip3a_counterclim" and variable.product == "watergap":
        return paths.watergap_root
    if variable.product == "climate":
        return paths.climate_root / scenario / gcm
    if variable.product == "ocean":
        if paths.ocean_root is None:
            raise ValueError(f"no ocean root in this ISIMIP layout for {variable.name!r}")
        return paths.ocean_root / scenario / gcm
    return paths.watergap_root / scenario / gcm


def _file_glob_pattern(
    variable: ISIMIPVariable,
    gcm: str,
    scenario: str,
    paths: ISIMIPPaths,
) -> str:
    gcm_lower = gcm.lower()
    if variable.product == "ocean":
        resolution = variable.resolution or "*"
        return (
            f"{gcm_lower}_*_{scenario}_{variable.name}_{resolution}_global_"
            f"{variable.frequency}_*.nc"
        )
    if variable.product == "climate":
        if paths.layout == "isimip3a_counterclim":
            return (
                f"{gcm_lower}_{scenario}_{variable.name}_global_"
                f"{variable.frequency}_*.nc"
            )
        return f"{gcm_lower}_*_{scenario}_{variable.name}_global_*.nc"
    if paths.layout == "isimip3a_counterclim":
        return (
            f"watergap2-2e_{gcm_lower}_{scenario}_{paths.soc}_default_"
            f"{variable.name}_global_{variable.frequency}_*.nc"
        )
    soc = paths.historical_soc if scenario == "historical" else paths.soc
    return (
        f"watergap2-2e_{gcm_lower}_*_{scenario}_{soc}_default_"
        f"{variable.name}_global_{variable.frequency}_*.nc"
    )


def _file_overlaps_years(path: Path, year_range: tuple[int, int]) -> bool:
    """Check if a file's year span overlaps the requested range.

    ISIMIP filenames end with ``_{start_year}_{end_year}.nc``; a name that does
    not raises, since the file cannot be placed in or out of the range.
    """

    parts = path.stem.rsplit("_", 2)
    try:
        if len(parts) < 3:
            raise ValueError
        file_start = int(parts[-2])
        file_end = int(parts[-1])
    except ValueError:
        raise ValueError(
            f"cannot read the year span of ISIMIP file {path.name!r}; "
            "its name must end with _<start year>_<end year>.nc to be filtered by year_range"
        ) from None
    return file_end >= year_range[0] and file_start <= year_range[1]


def _decode_month_counts(ds: xr.Dataset, time_dim: str = "time") -> xr.Dataset:
    """Decode a ``months since <date>`` time axis to month-start timestamps.

    ISIMIP ocean files count months from 1601 on a 365-day calendar, which
    neither pandas nor cftime decodes. Values mark a point inside each month
    (often mid-month), so the month is the integer part of the count.
    """

    units = ds[time_dim].attrs.get("units", "")
    if not units.startswith("months since "):
        raise ValueError(f"expected a 'months since' time axis, got {units!r}")
    epoch_year, epoch_month = (int(part) for part in units.split()[2].split("-")[:2])
    counts = np.floor(np.asarray(ds[time_dim].values, dtype=float)).astype(int)
    if (np.diff(counts) != 1).any():
        raise ValueError("time axis is not one value per consecutive month")
    months = epoch_year * 12 + (epoch_month - 1) + counts
    starts = pd.to_datetime(
        pd.DataFrame({"year": months // 12, "month": months % 12 + 1, "day": 1})
    )
    starts = pd.DatetimeIndex(starts).as_unit("ns")
    bounds = [name for name in ds.data_vars if time_dim in ds[name].dims and name.endswith("_bnds")]
    return ds.drop_vars(bounds).assign_coords({time_dim: starts})


def _grid_spacing(coordinate: xr.DataArray) -> float:
    values = np.asarray(coordinate.values, dtype=float)
    return float(np.abs(np.diff(values)).max()) if values.size > 1 else float("inf")


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
    if len(anomaly_columns) > 1:
        # The archive's GMT files repeat the anomaly block, so pandas reads one column
        # as mean_anomaly, mean_anomaly.1, ...; copies equal to rounding (the archive's
        # differ by at most 1e-15 K) are one column.
        values = pd.read_csv(path, usecols=anomaly_columns)[anomaly_columns].to_numpy(dtype=float)
        copies = all(
            np.allclose(values[:, 0], values[:, i], rtol=0.0, atol=1e-9, equal_nan=True)
            for i in range(1, values.shape[1])
        )
        if not copies:
            raise ValueError(
                f"{path} has several mean_anomaly columns ({', '.join(anomaly_columns)}) "
                "with different values; keep only the column to use"
            )
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
