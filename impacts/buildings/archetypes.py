"""Input loaders, validators, and grid and mapping builders for CHILLED.

This module holds everything that reads files or checks their required columns
for the CHILLED workflow. The kernel in ``impacts.buildings.chilled`` uses the
tables and grids these functions produce and never reads from disk itself.

Each loader takes an explicit path; without one it reads the environment
variable documented in ``impacts.buildings._config`` and raises when that is
unset.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from impacts.engine import (
    SITE_VALUES_BY_KEY,
    SITE_VALUES_DIGEST,
    SITE_VALUES_ORDER,
    site_values_digest,
)

from ._config import (
    ALPS_DATA_VARIABLE,
    ARCHETYPE_PATH_VARIABLE,
    GAUL_CSV_FILE,
    GAUL_RASTER_FILE,
    PARAMETRIC_PATH_VARIABLE,
    R12_YAML_PATH_VARIABLE,
    SETPOINT_PATH_VARIABLE,
    SOLAR_HORIZONTAL_FILE,
    SOLAR_VERTICAL_FILE,
    resolve_input_path,
)
from .chilled import ARCHETYPE_PARAMETER_COLUMNS, CHILLED_SETPOINT_COLUMNS, MONTHS

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
REQUIRED_SETPOINT_COLUMNS = frozenset({"region", *CHILLED_SETPOINT_COLUMNS})
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


def load_message_ix_buildings_archetypes(path: Path | str | None = None) -> pd.DataFrame:
    """Load and validate the regional CHILLED building archetype table (``CHILLED_ARCHETYPE_PATH`` by default)."""

    frame = pd.read_csv(resolve_input_path(path, ARCHETYPE_PATH_VARIABLE))
    return validate_archetype_table(frame)


def load_message_ix_buildings_parametric_runs(path: Path | str | None = None) -> pd.DataFrame:
    """Load and validate the CHILLED parametric-run table (``CHILLED_PARAMETRIC_PATH`` by default)."""

    if path is None:
        path = os.environ.get(PARAMETRIC_PATH_VARIABLE) or None
    if path is None:
        raise FileNotFoundError(
            "ALPS2025-only CHILLED support does not infer a parametric-run table; "
            "pass CHILLED_PARAMETRIC_PATH explicitly when an earlier step provides one"
        )

    frame = pd.read_csv(path)
    missing = REQUIRED_PARAMETRIC_COLUMNS.difference(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"parametric run table missing required columns: {missing_list}")
    return frame.copy()


def load_region_cooling_setpoints(path: Path | str | None = None) -> pd.DataFrame:
    """Load a region-conditioned cooling setpoint table (``CHILLED_SETPOINT_PATH`` by default).

    The CSV needs ``region``, ``cooling_setpoint_c`` and ``cooling_max_c``; other
    columns are ignored. No table is bundled with the package. The returned frame
    is keyed on ``region``, and its ``attrs`` record which table it came from
    (``description``, ``source``), so the grid and mapping builders can pass
    them on and evaluation outputs can name the setpoints they used.
    """

    path = resolve_input_path(path, SETPOINT_PATH_VARIABLE)
    frame = pd.read_csv(path)
    missing = REQUIRED_SETPOINT_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"setpoint table missing required columns: {', '.join(sorted(missing))}")
    if frame["region"].duplicated().any():
        duplicates = sorted(frame.loc[frame["region"].duplicated(), "region"].unique())
        raise ValueError(f"setpoint table must be unique on region; duplicates: {duplicates}")
    table = frame[["region", *CHILLED_SETPOINT_COLUMNS]].copy()
    table.attrs = {
        "description": f"per-region setpoints from the setpoint table {path.name}",
        "source": f"{path.name}, a region cooling setpoint table supplied by the caller",
    }
    # pandas carries attrs through edits, so tie them to the region-to-value
    # pairs they describe, whatever the row order.
    table.attrs[SITE_VALUES_BY_KEY] = _setpoints_by_region(table)
    return table


def validate_archetype_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Check that the ALPS2025 archetype table has the required columns and allowed values, with no duplicate archetypes."""

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
    """Normalize the fixed monthly solar climatology to ``solar_vertical``/``solar_horizontal`` on (month, lat, lon)."""

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
    vertical_path: Path | str | None = None,
    horizontal_path: Path | str | None = None,
) -> xr.Dataset:
    """Load EWEMBI solar irradiance climatologies.

    Without explicit paths the files are looked up in the ``ALPS_DATA_PATH``
    directory. Normalizes dimension order to (month, lat, lon), lat to ascending
    (S→N), and dtype to float64. Both files use variable name 'irrad'.
    """

    vert_raw = xr.open_dataset(resolve_input_path(vertical_path, ALPS_DATA_VARIABLE, SOLAR_VERTICAL_FILE))
    horiz_raw = xr.open_dataset(resolve_input_path(horizontal_path, ALPS_DATA_VARIABLE, SOLAR_HORIZONTAL_FILE))

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
    gaul_raster_path: Path | str | None = None,
    gaul_csv_path: Path | str | None = None,
    r12_yaml_path: Path | str | None = None,
) -> xr.DataArray:
    """Build an R12 region raster from the GAUL country raster and MESSAGE R12 node definitions.

    Unset paths come from ``ALPS_DATA_PATH`` (the GAUL files) and
    ``MESSAGE_R12_YAML_PATH``.
    """

    gaul_raster_path = resolve_input_path(gaul_raster_path, ALPS_DATA_VARIABLE, GAUL_RASTER_FILE)
    gaul_csv_path = resolve_input_path(gaul_csv_path, ALPS_DATA_VARIABLE, GAUL_CSV_FILE)
    with open(resolve_input_path(r12_yaml_path, R12_YAML_PATH_VARIABLE)) as fh:
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
    setpoints: pd.DataFrame | None = None,
) -> xr.Dataset:
    """Build gridded building archetype parameters for one ALPS2025 source-table combination.

    Maps region_raster values to archetype parameter values for the selected
    building_type × urt × arch rows exactly as provided by the ALPS2025 table,
    and carries monthly solar as (month, lat, lon) fields for the kernel.

    When ``setpoints`` is provided, ``cooling_setpoint_c``/``cooling_max_c`` are
    rasterized per region like the envelope parameters; cells whose region is
    absent from the table stay NaN.
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

    if setpoints is not None:
        missing_cols = REQUIRED_SETPOINT_COLUMNS.difference(setpoints.columns)
        if missing_cols:
            raise ValueError(f"setpoints missing required columns: {', '.join(sorted(missing_cols))}")
        region_to_setpoint = setpoints.set_index("region")[list(CHILLED_SETPOINT_COLUMNS)].to_dict("index")
        for col in CHILLED_SETPOINT_COLUMNS:
            grid = template.copy()
            for region, values in region_to_setpoint.items():
                grid = xr.where(region_raster == str(region), float(values[col]), grid)
            provenance = _setpoint_source(setpoints)
            if provenance:
                # Sorted by lat and lon, so a grid evaluated against climate in
                # the other latitude order still matches.
                provenance[SITE_VALUES_DIGEST] = site_values_digest(grid, ("lat", "lon"))
                provenance[SITE_VALUES_ORDER] = "lat lon"
            param_arrays[col] = grid.assign_attrs(provenance)

    solar_aligned = solar.sel(lat=region_raster.lat, lon=region_raster.lon)
    param_arrays["solar_vertical"] = solar_aligned["solar_vertical"]
    param_arrays["solar_horizontal"] = solar_aligned["solar_horizontal"]

    return xr.Dataset(data_vars=param_arrays)


def build_chilled_archetype_mapping(
    spatial_units: pd.DataFrame,
    archetypes: pd.DataFrame,
    solar_climatology: xr.Dataset,
    *,
    setpoints: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, xr.Dataset]:
    """Pair each spatial unit with every ALPS2025 building archetype of its region.

    Returns the mapping table plus the normalized (month, lat, lon) solar
    climatology to hand to ``evaluate_mapped(..., gridded_coords=...)``;
    the kernel reads solar as a (month, entity) coordinate rather than from
    flattened per-month columns.

    The ALPS2025 source table already includes the sfh×urban and mfh×rural
    combinations; those rows are used as given.

    When ``setpoints`` is provided (a region-keyed table from
    ``load_region_cooling_setpoints``), its ``cooling_setpoint_c``/
    ``cooling_max_c`` columns are joined on ``region`` for
    ``CHILLED_REGION_SETPOINT_KERNEL`` to read per site; regions absent from
    the table carry NaN. Without it the mapping omits the columns.
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

    if "permanent" not in mapping.columns:
        mapping["permanent"] = np.nan

    if setpoints is not None:
        missing_cols = REQUIRED_SETPOINT_COLUMNS.difference(setpoints.columns)
        if missing_cols:
            raise ValueError(f"setpoints missing required columns: {', '.join(sorted(missing_cols))}")
        mapping = mapping.merge(
            setpoints[["region", *CHILLED_SETPOINT_COLUMNS]],
            on="region",
            how="left",
            validate="many_to_one",
        ).reset_index(drop=True)
        # The engine reads the recorded source of site values from here. Each
        # column's source carries its region-to-value map, which the engine checks
        # site by site, so a subset or reordering of the mapping keeps it and an
        # edited value does not.
        provenance = _setpoint_source(setpoints)
        if provenance:
            by_region = _setpoints_by_region(setpoints)
            mapping.attrs["site_value_attrs"] = {
                **mapping.attrs.get("site_value_attrs", {}),
                **{
                    col: {**provenance, SITE_VALUES_BY_KEY: {"key": "region", "values": by_region[col]}}
                    for col in CHILLED_SETPOINT_COLUMNS
                },
            }

    return mapping, solar


def _setpoint_source(setpoints: pd.DataFrame) -> dict[str, str]:
    """The ``description`` and ``source`` a setpoint table records about itself.

    Empty unless the table still holds the region-to-value pairs they were
    recorded for: sorting keeps them, editing or dropping a row does not.
    """

    if setpoints.attrs.get(SITE_VALUES_BY_KEY) != _setpoints_by_region(setpoints):
        return {}
    return {key: str(setpoints.attrs[key]) for key in ("description", "source") if key in setpoints.attrs}


def _setpoints_by_region(setpoints: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Each setpoint column as a region-to-value map, sorted by region."""

    ordered = setpoints.sort_values("region", kind="stable")
    return {
        col: {
            # None stands for a missing value, which compares equal to itself.
            str(region): None if pd.isna(value) else float(value)
            for region, value in zip(ordered["region"], ordered[col], strict=True)
        }
        for col in CHILLED_SETPOINT_COLUMNS
    }
