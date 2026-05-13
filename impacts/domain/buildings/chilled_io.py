"""Input loaders, validators, and grid/mapping builders for CHILLED.

This module owns everything that touches the filesystem or external file
contracts for the CHILLED workflow. The kernel in
``impacts.domain.buildings.chilled`` consumes the artifacts these
functions produce and never reads from disk itself.

Path defaults live in ``impacts.domain.buildings.chilled_config``; callers
pass explicit paths to override them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from impacts.climatecore.transforms.spatial import attach_gridded_dataset

from .chilled import ARCHETYPE_PARAMETER_COLUMNS, MONTHS
from .chilled_config import (
    DEFAULT_ARCHETYPE_INPUT_PATH,
    DEFAULT_GAUL_CSV_PATH,
    DEFAULT_GAUL_RASTER_PATH,
    DEFAULT_PARAMETRIC_INPUT_PATH,
    DEFAULT_R12_YAML_PATH,
    DEFAULT_SOLAR_HORIZONTAL_PATH,
    DEFAULT_SOLAR_VERTICAL_PATH,
)

REQUIRED_SPATIAL_UNIT_COLUMNS = frozenset({"grid_lat", "grid_lon", "region"})
REQUIRED_ARCHETYPE_COLUMNS = frozenset(
    {
        "name",
        "building_type",
        "urt",
        "arch",
        "region",
        "area_env",
        "gl_perc",
        "vol",
        "u_val",
        "ach_cl",
        "ach_op",
        "gn_int",
        "gl_g",
        "gl_sh",
        "roof_area",
        "roof_abs",
        "u_roof",
    }
)
REQUIRED_PARAMETRIC_COLUMNS = frozenset(
    {
        "id_run",
        "name_run",
        "cop",
        "eff",
        "t_sp_c",
        "t_sp_c_max",
        "t_sp_h",
        "f_c",
        "f_f",
        "f_h",
    }
)
ALPS2025_BUILDING_TYPES = frozenset({"sfh", "mfh"})
ALPS2025_URT_VALUES = frozenset({"rur", "urb"})
ALPS2025_ARCH_VALUES = frozenset({"s1", "s2", "s3", "s51", "s52"})


def load_message_ix_buildings_archetypes(path: Path | str = DEFAULT_ARCHETYPE_INPUT_PATH) -> pd.DataFrame:
    """Load and validate the regional CHILLED archetype table."""

    frame = pd.read_csv(path)
    return validate_archetype_table(frame)


def load_message_ix_buildings_parametric_runs(path: Path | str | None = DEFAULT_PARAMETRIC_INPUT_PATH) -> pd.DataFrame:
    """Load and validate the CHILLED parametric-run table."""

    if path is None:
        raise FileNotFoundError(
            "ALPS2025-only CHILLED support does not infer a parametric-run table; "
            "pass CHILLED_PARAMETRIC_PATH explicitly when upstream orchestration provides one"
        )

    frame = pd.read_csv(path)
    missing = REQUIRED_PARAMETRIC_COLUMNS.difference(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"parametric run table missing required columns: {missing_list}")
    return frame.copy()


def validate_archetype_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the ALPS2025 archetype table contract."""

    missing = REQUIRED_ARCHETYPE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"archetype table missing required columns: {', '.join(sorted(missing))}")

    validated = frame.copy()

    invalid_building_types = set(validated["building_type"].dropna()) - ALPS2025_BUILDING_TYPES
    if invalid_building_types:
        raise ValueError(
            "archetype table has unsupported building_type values: "
            f"{', '.join(sorted(map(str, invalid_building_types)))}"
        )

    invalid_urt = set(validated["urt"].dropna()) - ALPS2025_URT_VALUES
    if invalid_urt:
        raise ValueError(
            f"archetype table has unsupported urt values: {', '.join(sorted(map(str, invalid_urt)))}"
        )

    invalid_arch = set(validated["arch"].dropna()) - ALPS2025_ARCH_VALUES
    if invalid_arch:
        raise ValueError(
            f"archetype table has unsupported arch values: {', '.join(sorted(map(str, invalid_arch)))}"
        )

    duplicate_mask = validated.duplicated(["region", "building_type", "urt", "arch"], keep=False)
    if duplicate_mask.any():
        duplicates = (
            validated.loc[duplicate_mask, ["region", "building_type", "urt", "arch"]]
            .drop_duplicates()
            .sort_values(["region", "building_type", "urt", "arch"])
        )
        raise ValueError(
            "archetype table must be unique on (region, building_type, urt, arch); duplicates found for "
            f"{duplicates.to_dict(orient='records')}"
        )

    return validated


def normalize_chilled_solar_climatology(
    dataset: xr.Dataset,
    *,
    vertical_name: str = "solar_vertical",
    horizontal_name: str = "solar_horizontal",
) -> xr.Dataset:
    """Normalize the fixed monthly solar climatology into the production contract."""

    missing = [name for name in (vertical_name, horizontal_name) if name not in dataset.data_vars]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"solar climatology missing required variables: {missing_list}")
    if "month" not in dataset.coords:
        raise ValueError("solar climatology must define a month coordinate")
    if set(dataset["month"].values.tolist()) != set(MONTHS):
        raise ValueError("solar climatology month coordinate must contain 1..12")

    normalized = dataset[[vertical_name, horizontal_name]].rename(
        {
            vertical_name: "solar_vertical",
            horizontal_name: "solar_horizontal",
        }
    )
    if "lat" not in normalized.coords or "lon" not in normalized.coords:
        raise ValueError("solar climatology must provide lat/lon coordinates")
    return normalized.sortby("month")


def load_ewembi_solar_climatology(
    *,
    vertical_path: Path | str = DEFAULT_SOLAR_VERTICAL_PATH,
    horizontal_path: Path | str = DEFAULT_SOLAR_HORIZONTAL_PATH,
) -> xr.Dataset:
    """Load EWEMBI solar irradiance climatologies into the production contract.

    Normalizes dimension order to (month, lat, lon), lat to ascending (S→N),
    and dtype to float64. Both files use variable name 'irrad'.
    """

    vert_raw = xr.open_dataset(vertical_path)
    horiz_raw = xr.open_dataset(horizontal_path)

    vert = vert_raw["irrad"].transpose("month", "lat", "lon").sortby("lat").astype(np.float64)
    horiz = horiz_raw["irrad"].transpose("month", "lat", "lon").sortby("lat").astype(np.float64)

    return xr.Dataset(
        data_vars={
            "solar_vertical": vert,
            "solar_horizontal": horiz,
        },
    )


def load_r12_region_raster(
    *,
    gaul_raster_path: Path | str = DEFAULT_GAUL_RASTER_PATH,
    gaul_csv_path: Path | str = DEFAULT_GAUL_CSV_PATH,
    r12_yaml_path: Path | str = DEFAULT_R12_YAML_PATH,
) -> xr.DataArray:
    """Build an R12 region raster from the GAUL country raster and MESSAGE R12 node definitions."""

    with open(r12_yaml_path) as fh:
        r12_spec = yaml.safe_load(fh)

    iso_to_r12: dict[str, str] = {}
    for node_id, definition in r12_spec.items():
        if node_id == "World" or "child" not in definition:
            continue
        region = node_id.removeprefix("R12_")
        for iso3 in definition["child"]:
            iso_to_r12[iso3] = region

    gaul_csv = pd.read_csv(gaul_csv_path)
    gaul_to_r12: dict[int, str] = {}
    for _, row in gaul_csv.iterrows():
        iso3 = str(row["ISO3"])
        raster_id = int(row["ID"])
        if iso3 in iso_to_r12:
            gaul_to_r12[raster_id] = iso_to_r12[iso3]

    gaul_raster = xr.open_dataarray(gaul_raster_path).sortby("lat")
    gaul_values = gaul_raster.values
    region_array = np.full(gaul_values.shape, "", dtype="U8")
    for raster_id, region in gaul_to_r12.items():
        region_array[gaul_values == raster_id] = region

    return xr.DataArray(
        region_array,
        dims=gaul_raster.dims,
        coords=gaul_raster.coords,
        name="region",
    )


def build_chilled_parameter_grids(
    region_raster: xr.DataArray,
    archetypes: pd.DataFrame,
    solar: xr.Dataset,
    *,
    building_type: str,
    urt: str,
    vintage: str,
) -> xr.Dataset:
    """Build gridded archetype parameters for one ALPS2025 source-table combination.

    Maps region_raster values to archetype parameter values for the selected
    building_type × urt × arch rows exactly as provided by the ALPS2025 table,
    and flattens monthly solar into 24 per-cell columns matching the
    kernel's _monthly_static_array contract.
    """

    row = archetypes.loc[
        (archetypes["building_type"] == building_type)
        & (archetypes["urt"] == urt)
        & (archetypes["arch"] == vintage)
    ]
    if row.empty:
        raise ValueError(f"no archetype found for building_type={building_type}, urt={urt}, vintage={vintage}")

    region_to_params: dict[str, dict[str, float]] = {}
    for _, record in row.iterrows():
        region_to_params[str(record["region"])] = {
            col: float(record[col]) for col in ARCHETYPE_PARAMETER_COLUMNS
        }

    template = xr.DataArray(
        np.full(region_raster.shape, np.nan),
        dims=region_raster.dims,
        coords=region_raster.coords,
    )
    param_arrays: dict[str, xr.DataArray] = {}
    for col in ARCHETYPE_PARAMETER_COLUMNS:
        grid = template.copy()
        for region, params in region_to_params.items():
            grid = xr.where(region_raster == region, params[col], grid)
        param_arrays[col] = grid

    solar_aligned = solar.sel(lat=region_raster.lat, lon=region_raster.lon)
    for month in MONTHS:
        param_arrays[f"solar_vertical_m{month:02d}"] = solar_aligned["solar_vertical"].sel(month=month).drop_vars("month")
        param_arrays[f"solar_horizontal_m{month:02d}"] = solar_aligned["solar_horizontal"].sel(month=month).drop_vars("month")

    return xr.Dataset(data_vars=param_arrays)


def build_chilled_archetype_mapping(
    spatial_units: pd.DataFrame,
    archetypes: pd.DataFrame,
    solar_climatology: xr.Dataset,
) -> pd.DataFrame:
    """Explode spatial units to region-matched ALPS2025 archetypes and attach solar climatology.

    The ALPS2025 source table already includes the sfh×urban and mfh×rural
    combinations. Those rows are consumed as upstream modeling assumptions,
    not repaired into older urban/rural proxy semantics inside impacts.
    """

    missing_units = REQUIRED_SPATIAL_UNIT_COLUMNS.difference(spatial_units.columns)
    if missing_units:
        missing_list = ", ".join(sorted(missing_units))
        raise ValueError(f"spatial_units missing required columns: {missing_list}")

    archetype_table = validate_archetype_table(archetypes.copy())
    solar = normalize_chilled_solar_climatology(solar_climatology)

    base = spatial_units.reset_index(drop=True).copy()
    if "spatial_unit_id" not in base.columns:
        base["spatial_unit_id"] = np.arange(len(base), dtype=int)

    mapping = base.merge(
        archetype_table,
        on="region",
        how="inner",
        validate="many_to_many",
        suffixes=("", "_archetype"),
    ).reset_index(drop=True)
    if mapping.empty:
        raise ValueError("no spatial units matched the archetype regions")

    sampling_index = mapping[["grid_lat", "grid_lon"]].copy()
    sampling_index["entity_id"] = np.arange(len(mapping), dtype=int)
    sampled = attach_gridded_dataset(
        sampling_index,
        solar,
        entity_id_column="entity_id",
        metadata_columns=("entity_id",),
    )

    for month in MONTHS:
        mapping[f"solar_vertical_m{month:02d}"] = sampled["solar_vertical"].sel(month=month).to_numpy()
        mapping[f"solar_horizontal_m{month:02d}"] = sampled["solar_horizontal"].sel(month=month).to_numpy()

    if "permanent" not in mapping.columns:
        mapping["permanent"] = np.nan

    return mapping
