from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from climateresponse import EvaluationResult, normalize_temperature
from climateresponse.spatial import attach_gridded_dataset

from impacts.kernel import (
    Duration,
    ImpactKernel,
    evaluate_gridded_kernel,
    evaluate_kernel,
    normalize_climate_inputs,
)

DEFAULT_CHILLED_VARIABLE_NAMES = {
    "air_temperature_c": "tas",
}

REQUIRED_CHILLED_CLIMATE_VARIABLES = frozenset({"air_temperature_c"})
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
ARCHETYPE_PARAMETER_COLUMNS = (
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
)
MONTHS = tuple(range(1, 13))
_KERNEL_ENTITY_CHUNK = 10000
SOLAR_VERTICAL_COLUMNS = tuple(f"solar_vertical_m{month:02d}" for month in MONTHS)
SOLAR_HORIZONTAL_COLUMNS = tuple(f"solar_horizontal_m{month:02d}" for month in MONTHS)
SOLAR_COLUMNS = SOLAR_VERTICAL_COLUMNS + SOLAR_HORIZONTAL_COLUMNS
CHILLED_METADATA_COLUMNS = (
    "spatial_unit_id",
    "region",
    "name",
    "urt",
    "arch",
    "permanent",
    *ARCHETYPE_PARAMETER_COLUMNS,
    *SOLAR_COLUMNS,
)

WATXENE_PATH = Path(os.environ.get("WATXENE_PATH", "/mnt/p/watxene"))
ALPS_DATA_PATH = WATXENE_PATH / "ISIMIP_postprocessed" / "data_for_vignesh" / "ALPS_2024"

DEFAULT_ARCHETYPE_INPUT_PATH = Path(
    os.environ.get(
        "CHILLED_ARCHETYPE_PATH",
        str(Path.home() / "message-ix-buildings/message_ix_buildings/data/chilled/version/ALPS2025/arch_input_reg.csv"),
    )
)
DEFAULT_PARAMETRIC_INPUT_PATH = Path(
    os.environ.get("CHILLED_PARAMETRIC_PATH", "")
) if os.environ.get("CHILLED_PARAMETRIC_PATH") else None
DEFAULT_R12_YAML_PATH = Path(
    os.environ.get(
        "MESSAGE_R12_YAML_PATH",
        str(Path.home() / "message-ix-models/message_ix_models/data/node/R12.yaml"),
    )
)
DEFAULT_GAUL_RASTER_PATH = ALPS_DATA_PATH / "gaul_lvl0_hybrid_05_3.nc"
DEFAULT_GAUL_CSV_PATH = ALPS_DATA_PATH / "GAUL_lvl0_raster0.5.csv"
DEFAULT_SOLAR_VERTICAL_PATH = ALPS_DATA_PATH / "EWEMBI_vert_irrad_1980-2009_avg.nc"
DEFAULT_SOLAR_HORIZONTAL_PATH = ALPS_DATA_PATH / "EWEMBI_horiz_irrad_1980-2009_avg.nc"


@dataclass(frozen=True)
class ChilledEnergyAssumptions:
    """Equipment and setpoint parameters for the CHILLED kernel."""

    cooling_setpoint_c: float = 26.0
    cooling_max_c: float = 28.0
    heating_setpoint_c: float = 20.0
    cooling_hours_fraction: float = 0.333
    fan_hours_fraction: float = 0.2
    heating_hours_fraction: float = 1.0
    ac_cop: float = 3.2
    heating_efficiency: float = 0.9
    fan_power_w: float = 55.0
    area_per_fan_m2: float = 25.0

    @classmethod
    def from_parametric_row(cls, row: pd.Series | Mapping[str, object]) -> ChilledEnergyAssumptions:
        values = dict(row)
        return cls(
            cooling_setpoint_c=float(values["t_sp_c"]),
            cooling_max_c=float(values["t_sp_c_max"]),
            heating_setpoint_c=float(values["t_sp_h"]),
            cooling_hours_fraction=float(values["f_c"]),
            fan_hours_fraction=float(values["f_f"]),
            heating_hours_fraction=float(values["f_h"]),
            ac_cop=float(values["cop"]),
            heating_efficiency=float(values["eff"]),
        )

    def as_kernel_params(self) -> dict[str, float]:
        return {
            "cooling_setpoint_c": self.cooling_setpoint_c,
            "cooling_max_c": self.cooling_max_c,
            "heating_setpoint_c": self.heating_setpoint_c,
            "cooling_hours_fraction": self.cooling_hours_fraction,
            "fan_hours_fraction": self.fan_hours_fraction,
            "heating_hours_fraction": self.heating_hours_fraction,
            "ac_cop": self.ac_cop,
            "heating_efficiency": self.heating_efficiency,
            "fan_power_w": self.fan_power_w,
            "area_per_fan_m2": self.area_per_fan_m2,
        }


CHILLED_NORMALIZATIONS = (
    ("air_temperature_c", "C", normalize_temperature),
)


def normalize_chilled_climate_inputs(
    dataset: xr.Dataset,
    *,
    variable_names: Mapping[str, str] | None = None,
    variable_units: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Select, rename, and normalize the daily climate inputs required by CHILLED."""

    return normalize_climate_inputs(
        dataset,
        default_variable_names=DEFAULT_CHILLED_VARIABLE_NAMES,
        normalizations=CHILLED_NORMALIZATIONS,
        variable_names=variable_names,
        variable_units=variable_units,
    )


def load_message_ix_buildings_archetypes(path: Path | str = DEFAULT_ARCHETYPE_INPUT_PATH) -> pd.DataFrame:
    """Load and validate the regional CHILLED archetype table."""

    frame = pd.read_csv(path)
    return validate_archetype_table(frame)


def load_message_ix_buildings_parametric_runs(path: Path | str = DEFAULT_PARAMETRIC_INPUT_PATH) -> pd.DataFrame:
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


def run_chilled_gridded_ei(
    climate: xr.Dataset,
    parameters: xr.Dataset,
    *,
    assumptions: ChilledEnergyAssumptions | None = None,
    duration: Duration = "daily",
    spatial_dims: tuple[str, ...] = ("lat", "lon"),
) -> xr.Dataset:
    """Evaluate CHILLED EI on the full grid via evaluate_gridded_kernel."""

    return evaluate_gridded_kernel(
        create_chilled_kernel(assumptions=assumptions),
        climate,
        parameters,
        duration=duration,
        spatial_dims=spatial_dims,
    )


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


def create_chilled_kernel(
    assumptions: ChilledEnergyAssumptions | None = None,
) -> ImpactKernel:
    """Build the CHILLED production kernel on the shared mapped-kernel contract."""

    params = assumptions or ChilledEnergyAssumptions()
    return ImpactKernel(
        name="chilled_buildings",
        normalize_inputs=normalize_chilled_climate_inputs,
        physics_kernel=partial(_evaluate_chilled_kernel, assumptions=params),
        metadata_columns=CHILLED_METADATA_COLUMNS,
    )


def run_chilled_archetype_ei(
    spatial_units: pd.DataFrame,
    climate: xr.Dataset,
    archetypes: pd.DataFrame,
    solar_climatology: xr.Dataset,
    *,
    assumptions: ChilledEnergyAssumptions | None = None,
    duration: Duration = "daily",
) -> xr.Dataset:
    """Evaluate CHILLED energy intensity at native spatial units."""

    mapping = build_chilled_archetype_mapping(
        spatial_units=spatial_units,
        archetypes=archetypes,
        solar_climatology=solar_climatology,
    )
    return evaluate_kernel(
        create_chilled_kernel(assumptions=assumptions),
        mapping,
        climate,
        duration=duration,
    )


def _evaluate_chilled_kernel(
    attached: xr.Dataset,
    *,
    assumptions: ChilledEnergyAssumptions,
) -> EvaluationResult:
    if "time" not in attached.sizes:
        raise ValueError("attached climate inputs must include a time dimension")

    time_index = attached.indexes["time"]
    month_index = np.asarray(time_index.month - 1, dtype=np.int32)

    air_temperature = attached["air_temperature_c"].transpose("time", "entity")
    solar_vertical = _monthly_static_array(attached, prefix="solar_vertical")
    solar_horizontal = _monthly_static_array(attached, prefix="solar_horizontal")

    gl_perc = _entity_coord(attached, "gl_perc")
    gl_g = _entity_coord(attached, "gl_g")
    gl_sh = _entity_coord(attached, "gl_sh")
    roof_area = _entity_coord(attached, "roof_area")
    roof_abs = _entity_coord(attached, "roof_abs")
    u_roof = _entity_coord(attached, "u_roof")
    u_val = _entity_coord(attached, "u_val")
    area_env = _entity_coord(attached, "area_env")
    vol = _entity_coord(attached, "vol")
    ach_cl = _entity_coord(attached, "ach_cl")
    ach_op = _entity_coord(attached, "ach_op")
    gn_int = _entity_coord(attached, "gn_int")

    gn_sol = solar_vertical * gl_perc * gl_g * gl_sh + solar_horizontal * 0.04 * u_roof * roof_area * roof_abs
    h_tr = u_val * area_env
    h_v_cl = 1200.0 * vol * ach_cl / 3600.0
    h_v_op = 1200.0 * vol * ach_op / 3600.0
    t_bal_h = assumptions.heating_setpoint_c - ((gn_int + gn_sol) / (h_tr + h_v_cl))
    t_bal_c = assumptions.cooling_setpoint_c - ((gn_int + gn_sol) / (h_tr + h_v_cl))
    t_max_c = assumptions.cooling_max_c - ((gn_int + gn_sol) / (h_tr + h_v_op))

    # Chunk on entity before the month→time broadcast so the (time, entity)
    # result stays dask-backed instead of eager-materializing as a single
    # float64 array (~4.5 GB at 6-year × global-0.5° scale).
    month_indexer = xr.DataArray(month_index, dims=("time",))
    entity_chunk = min(_KERNEL_ENTITY_CHUNK, t_bal_h.sizes["entity"])
    t_bal_h_daily = t_bal_h.chunk({"entity": entity_chunk}).isel(month=month_indexer)
    t_bal_c_daily = t_bal_c.chunk({"entity": entity_chunk}).isel(month=month_indexer)
    t_max_c_daily = t_max_c.chunk({"entity": entity_chunk}).isel(month=month_indexer)

    vdd_h = xr.where(t_bal_h_daily >= air_temperature, t_bal_h_daily - air_temperature, 0.0)
    vdd_tmax_c = xr.where(air_temperature >= t_max_c_daily, air_temperature - t_max_c_daily, 0.0)
    nd = (air_temperature - t_max_c_daily > 0.0).astype(float)
    nf = (air_temperature > t_bal_c_daily).astype(float)

    q_c = (h_tr + h_v_cl) * assumptions.cooling_hours_fraction * (vdd_tmax_c + ((t_max_c_daily - t_bal_c_daily) * nd)) * 86400.0 / 1e6
    cooling_energy_ac = q_c / assumptions.ac_cop
    cooling_energy_fan = (
        assumptions.fan_hours_fraction
        * assumptions.fan_power_w
        * nf
        * 86400.0
        / (1e6 * assumptions.area_per_fan_m2)
    )
    q_h = (h_tr + h_v_cl) * assumptions.heating_hours_fraction * vdd_h * 86400.0 / 1e6
    heating_energy = q_h / assumptions.heating_efficiency

    return EvaluationResult(
        data_vars={
            "vdd_h": vdd_h,
            "vdd_tmax_c": vdd_tmax_c,
            "Nd": nd,
            "Nf": nf,
            "cooling_energy_ac": cooling_energy_ac,
            "cooling_energy_fan": cooling_energy_fan,
            "cooling_energy_total": cooling_energy_ac + cooling_energy_fan,
            "heating_energy": heating_energy,
        },
        attrs={
            "cooling_model": "CHILLED",
            "cooling_model_citation": (
                "Mastrucci, A. et al. Heating and cooling degree-day sensitivity "
                "with CHILLED archetype energy-intensity equations."
            ),
            "climate_contract": "daily tas only; solar climatology carried as static monthly metadata",
            "energy_units": "MJ/m2/day",
        },
    )


def _monthly_static_array(attached: xr.Dataset, *, prefix: str) -> xr.DataArray:
    arrays = [_entity_coord(attached, f"{prefix}_m{month:02d}") for month in MONTHS]
    return xr.concat(arrays, dim=xr.IndexVariable("month", np.asarray(MONTHS, dtype=int)))


def _entity_coord(attached: xr.Dataset, name: str) -> xr.DataArray:
    return xr.DataArray(attached.coords[name].data, dims=("entity",), coords={"entity": attached.coords["entity"]})

