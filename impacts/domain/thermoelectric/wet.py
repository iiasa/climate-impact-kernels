from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr
from impacts.climatecore import (
    EvaluationResult,
    normalize_pressure,
    normalize_temperature,
)
from impacts.isimip import ISIMIPVariable
from impacts.kernel import ImpactKernel, normalize_climate_inputs
from impacts.domain.thermoelectric._common import (
    join_country_from_cooling,
    validate_unique_units,
)

REQUIRED_WET_COOLING_VARIABLES = frozenset(
    {
        "water_temperature",
        "discharge",
        "air_temperature",
        "relative_humidity",
        "surface_pressure",
    }
)

DEFAULT_ISIMIP_VARIABLE_NAMES = {
    "water_temperature": "tw",
    "discharge": "dis",
    "air_temperature": "tas",
    "relative_humidity": "hurs",
    "surface_pressure": "ps",
}

WET_COOLING_VARIABLES = (
    ISIMIPVariable("tas", "climate", "daily"),
    ISIMIPVariable("hurs", "climate", "daily"),
    ISIMIPVariable("ps", "climate", "daily"),
    ISIMIPVariable("dis", "watergap", "daily"),
    ISIMIPVariable("triver", "watergap", "monthly", output_name="tw"),
)

REQUIRED_WET_MAPPING_COLUMNS = frozenset(
    {
        "UNITID",
        "LOCATIONID",
        "grid_lat",
        "grid_lon",
        "mw",
    }
)

REQUIRED_COOLING_JOIN_COLUMNS = frozenset(
    {
        "UNITID",
        "LOCATIONID",
        "fuel",
        "technology_family",
        "cooling_category",
        "cooling_physics",
        "requires_water",
    }
)

WET_METADATA_COLUMNS = (
    "UNITID",
    "LOCATIONID",
    "country",
    "fuel",
    "wet_fuel_type",
    "mw",
    "cooling_category",
    "cooling_physics",
    "thermal_efficiency",
    "heat_rejection_coefficient",
    "evaporation_coefficient",
    "cell_total_mw",
    "unit_share_ratio",
)

COMBUSTIBLE_GAS_MARKERS = ("GAS", "LNG", "NG")
COMBUSTIBLE_OIL_MARKERS = ("DIESEL", "FO", "HFO", "KERO", "LFO", "NAPHTHA", "OIL")
COMBUSTIBLE_BIOMASS_MARKERS = ("BIO", "WOOD", "BAGASSE")
COMBUSTIBLE_COAL_MARKERS = ("COAL", "COKE", "LIGNITE", "PEAT", "PETCOKE", "SHALE")
NUCLEAR_MARKERS = ("NUC", "URANIUM", "UR")

FUEL_EFFICIENCY_CURVE_PARAMETERS = {
    "coal": (24.14675, 44.89252, 198.43242, 1.63882),
    "gas": (38.59261, 64.74196, 269.6733, 1.28565),
    "oil": (35.96246, 50.84005, 224.66888, 1.50589),
    "biomass": (16.04344, 36.14518, 24.37557, 1.87955),
}

HEAT_REJECTION_COEFFICIENTS = {
    "coal": 0.12,
    "gas": 0.12,
    "oil": 0.12,
    "biomass": 0.12,
    "nuclear": 0.0,
}

EVAPORATION_COEFFICIENTS = {
    "coal": 0.0042,
    "gas": 0.0088,
    "oil": 0.0088,
    "biomass": 0.0086,
    "nuclear": 0.0061,
}


@dataclass(frozen=True)
class WetCoolingAssumptions:
    """Default constants for the Li et al. wet-cooling withdrawal equations."""

    water_density_kg_per_m3: float = 1000.0
    water_specific_heat_j_per_g_k: float = 4.184
    environmental_flow_fraction: float = 0.8
    concentration_cycles: float = 5.0
    tower_approach_c: float = 6.0
    max_discharge_temp_c: float = 32.0
    max_temp_rise_c: float = 10.0


WET_COOLING_NORMALIZATIONS = (
    ("water_temperature", "C", normalize_temperature),
    ("air_temperature", "C", normalize_temperature),
    ("surface_pressure", "kPa", normalize_pressure),
)


def normalize_wet_cooling_climate_inputs(
    dataset: xr.Dataset,
    *,
    variable_names: Mapping[str, str] | None = None,
    variable_units: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Select, rename, and normalize the climate inputs required by wet-cooling physics."""

    return normalize_climate_inputs(
        dataset,
        default_variable_names=DEFAULT_ISIMIP_VARIABLE_NAMES,
        normalizations=WET_COOLING_NORMALIZATIONS,
        variable_names=variable_names,
        variable_units=variable_units,
    )


def build_wet_cooling_mapping(
    unit_cell_mapping: pd.DataFrame,
    cooling_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Join plant coordinates to Li-style wet-cooling coefficients."""

    missing_mapping = REQUIRED_WET_MAPPING_COLUMNS.difference(unit_cell_mapping.columns)
    if missing_mapping:
        missing_list = ", ".join(sorted(missing_mapping))
        raise ValueError(f"unit_cell_mapping missing required columns: {missing_list}")

    missing_cooling = REQUIRED_COOLING_JOIN_COLUMNS.difference(cooling_mapping.columns)
    if missing_cooling:
        missing_list = ", ".join(sorted(missing_cooling))
        raise ValueError(f"cooling_mapping missing required columns: {missing_list}")

    validate_unique_units(unit_cell_mapping, frame_name="unit_cell_mapping")
    validate_unique_units(cooling_mapping, frame_name="cooling_mapping")

    joined = join_country_from_cooling(unit_cell_mapping, cooling_mapping).merge(
        cooling_mapping[
            [
                "UNITID",
                "LOCATIONID",
                "fuel",
                "technology_family",
                "cooling_category",
                "cooling_physics",
                "requires_water",
            ]
        ],
        on=["UNITID", "LOCATIONID"],
        how="inner",
        validate="one_to_one",
    )

    wet = joined.loc[joined["requires_water"].fillna(False)].copy()
    supported = wet["cooling_physics"].isin({"once_through", "recirculating"})
    if not supported.all():
        unknown = sorted(wet.loc[~supported, "cooling_physics"].dropna().unique().tolist())
        raise ValueError(f"unsupported wet cooling physics: {unknown}")

    wet["wet_fuel_type"] = [
        _classify_wet_fuel_type(fuel=fuel, technology_family=technology_family)
        for fuel, technology_family in zip(
            wet["fuel"],
            wet["technology_family"],
            strict=True,
        )
    ]
    wet["thermal_efficiency"] = [
        _fuel_efficiency(wet_fuel_type, mw)
        for wet_fuel_type, mw in zip(
            wet["wet_fuel_type"],
            wet["mw"],
            strict=True,
        )
    ]
    wet["heat_rejection_coefficient"] = wet["wet_fuel_type"].map(HEAT_REJECTION_COEFFICIENTS)
    wet["evaporation_coefficient"] = wet["wet_fuel_type"].map(EVAPORATION_COEFFICIENTS)
    wet["cell_total_mw"] = wet.groupby(["grid_lat", "grid_lon"])["mw"].transform("sum")
    wet["unit_share_ratio"] = wet["mw"] / wet["cell_total_mw"]

    return wet.sort_values(["country", "LOCATIONID", "UNITID"], kind="stable").reset_index(drop=True)


def create_wet_cooling_kernel(
    assumptions: WetCoolingAssumptions | None = None,
) -> ImpactKernel:
    """Build the Li et al. wet-cooling mapped-kernel contract."""

    params = assumptions or WetCoolingAssumptions()
    return ImpactKernel(
        name="wet_cooling",
        normalize_inputs=normalize_wet_cooling_climate_inputs,
        physics_kernel=partial(_evaluate_wet_cooling_kernel, params=params),
        metadata_columns=WET_METADATA_COLUMNS,
        clips={
            "capacity_factor": (0.0, 1.0),
            "capacity_loss": (0.0, 1.0),
            "usable_capacity_mw": (0.0, None),
        },
    )


def _evaluate_wet_cooling_kernel(
    attached: xr.Dataset,
    *,
    params: WetCoolingAssumptions,
) -> EvaluationResult:
    wet_bulb = _compute_wet_bulb_temperature(
        air_temperature_c=attached["air_temperature"],
        relative_humidity=attached["relative_humidity"],
    ).rename("wet_bulb_temperature")
    saturation_vapor_pressure = _compute_saturation_vapor_pressure(attached["air_temperature"]).rename(
        "saturation_vapor_pressure_pa"
    )
    actual_vapor_pressure = (
        attached["relative_humidity"] / 100.0 * saturation_vapor_pressure
    ).rename("actual_vapor_pressure_pa")
    humidity_ratio_out = _humidity_ratio(
        vapor_pressure_pa=saturation_vapor_pressure,
        surface_pressure_kpa=attached["surface_pressure"],
    ).rename("humidity_ratio_out_g_per_kg")
    humidity_ratio_in = _humidity_ratio(
        vapor_pressure_pa=actual_vapor_pressure,
        surface_pressure_kpa=attached["surface_pressure"],
    ).rename("humidity_ratio_in_g_per_kg")
    enthalpy_out = _enthalpy_kj_per_kg(
        air_temperature_c=attached["air_temperature"],
        humidity_ratio_g_per_kg=humidity_ratio_out,
    ).rename("enthalpy_out_kj_per_kg")
    enthalpy_in = _enthalpy_kj_per_kg(
        air_temperature_c=attached["air_temperature"],
        humidity_ratio_g_per_kg=humidity_ratio_in,
    ).rename("enthalpy_in_kj_per_kg")
    temperature_headroom = _once_through_temperature_headroom(
        water_temperature_c=attached["water_temperature"],
        max_discharge_temp_c=params.max_discharge_temp_c,
        max_temp_rise_c=params.max_temp_rise_c,
    ).rename("temperature_headroom_c")
    available_discharge = (
        attached["discharge"] * (1.0 - params.environmental_flow_fraction)
    ).rename("available_discharge_m3_per_s")
    unit_available_discharge = (
        available_discharge * attached.coords["unit_share_ratio"]
    ).rename("unit_available_discharge_m3_per_s")
    discharge_constraint_applies = (attached.coords["cooling_category"] != "ot_saline").rename(
        "discharge_constraint_applies"
    )

    once_through_withdrawal = _once_through_withdrawal_intensity(
        thermal_efficiency=attached.coords["thermal_efficiency"],
        heat_rejection_coefficient=attached.coords["heat_rejection_coefficient"],
        temperature_headroom_c=temperature_headroom,
        params=params,
    )
    recirculating_withdrawal = _recirculating_withdrawal_intensity(
        thermal_efficiency=attached.coords["thermal_efficiency"],
        heat_rejection_coefficient=attached.coords["heat_rejection_coefficient"],
        humidity_ratio_out=humidity_ratio_out,
        humidity_ratio_in=humidity_ratio_in,
        enthalpy_out=enthalpy_out,
        enthalpy_in=enthalpy_in,
        wet_bulb_temperature_c=wet_bulb,
        water_temperature_c=attached["water_temperature"],
        params=params,
    )
    cooling_physics = attached.coords["cooling_physics"]
    withdrawal_intensity = xr.where(
        cooling_physics == "once_through",
        once_through_withdrawal,
        recirculating_withdrawal,
    ).rename("withdrawal_intensity_m3_per_mwh")

    discharge_limited_capacity = np.minimum(
        3600.0 * unit_available_discharge / withdrawal_intensity,
        attached.coords["mw"],
    )
    thermal_only_capacity = xr.where(
        temperature_headroom > 0.0,
        attached.coords["mw"],
        0.0,
    )
    usable_capacity = xr.where(
        (cooling_physics == "once_through") & (~discharge_constraint_applies),
        thermal_only_capacity,
        discharge_limited_capacity,
    ).rename("usable_capacity_mw")
    capacity_factor = (usable_capacity / attached.coords["mw"]).clip(min=0.0, max=1.0).rename("capacity_factor")
    capacity_loss = (1.0 - capacity_factor).rename("capacity_loss")

    return EvaluationResult(
        data_vars={
            "water_temperature": attached["water_temperature"],
            "discharge": attached["discharge"],
            "air_temperature": attached["air_temperature"],
            "relative_humidity": attached["relative_humidity"],
            "surface_pressure": attached["surface_pressure"],
            "wet_bulb_temperature": wet_bulb,
            "saturation_vapor_pressure_pa": saturation_vapor_pressure,
            "actual_vapor_pressure_pa": actual_vapor_pressure,
            "humidity_ratio_out_g_per_kg": humidity_ratio_out,
            "humidity_ratio_in_g_per_kg": humidity_ratio_in,
            "enthalpy_out_kj_per_kg": enthalpy_out,
            "enthalpy_in_kj_per_kg": enthalpy_in,
            "temperature_headroom_c": temperature_headroom,
            "available_discharge_m3_per_s": available_discharge,
            "unit_available_discharge_m3_per_s": unit_available_discharge,
            "discharge_constraint_applies": discharge_constraint_applies,
            "thermal_efficiency_fraction": attached.coords["thermal_efficiency"].rename(
                "thermal_efficiency_fraction"
            ),
            "heat_rejection_coefficient_fraction": attached.coords["heat_rejection_coefficient"].rename(
                "heat_rejection_coefficient_fraction"
            ),
            "evaporation_coefficient_fraction": attached.coords["evaporation_coefficient"].rename(
                "evaporation_coefficient_fraction"
            ),
            "withdrawal_intensity_m3_per_mwh": withdrawal_intensity,
            "capacity_factor": capacity_factor,
            "capacity_loss": capacity_loss,
            "usable_capacity_mw": usable_capacity,
        },
        attrs={
            "water_density_kg_per_m3": params.water_density_kg_per_m3,
            "water_specific_heat_j_per_g_k": params.water_specific_heat_j_per_g_k,
            "environmental_flow_fraction": params.environmental_flow_fraction,
            "concentration_cycles": params.concentration_cycles,
            "tower_approach_c": params.tower_approach_c,
            "max_discharge_temp_c": params.max_discharge_temp_c,
            "max_temp_rise_c": params.max_temp_rise_c,
            "cooling_model_citation": (
                "Li, S. et al. Global hydroclimatic risks and strategic "
                "decommissioning pathways for thermal power units. Nature "
                "Sustainability. https://doi.org/10.1038/s41893-025-01692-9"
            ),
            "ot_saline_policy": (
                "ot_saline once-through units are exempt from river-discharge limits and use "
                "thermal headroom only; freshwater once-through units retain Li's discharge constraint."
            ),
        },
    )


def _classify_wet_fuel_type(
    fuel: str,
    technology_family: str | None = None,
) -> str:
    normalized = fuel.upper()
    technology = (technology_family or "").lower()
    if any(marker in normalized for marker in NUCLEAR_MARKERS):
        return "nuclear"
    if normalized == "WSTH":
        if technology in {"combined_cycle", "gas_turbine", "internal_combustion"}:
            return "gas"
        if technology in {"steam", "other", ""}:
            return "coal"
    if any(marker in normalized for marker in COMBUSTIBLE_BIOMASS_MARKERS):
        return "biomass"
    if any(marker in normalized for marker in COMBUSTIBLE_COAL_MARKERS):
        return "coal"
    if any(marker in normalized for marker in COMBUSTIBLE_OIL_MARKERS):
        return "oil"
    if any(marker in normalized for marker in COMBUSTIBLE_GAS_MARKERS):
        return "gas"
    raise ValueError(f"unsupported wet-cooling fuel category: {fuel!r}")


def _fuel_efficiency(fuel_type: str, mw: float) -> float:
    if fuel_type == "nuclear":
        return 0.35

    a1, a2, x0, power = FUEL_EFFICIENCY_CURVE_PARAMETERS[fuel_type]
    return (a2 + (a1 - a2) / (1.0 + (float(mw) / x0) ** power)) * 0.01


def _once_through_temperature_headroom(
    *,
    water_temperature_c: xr.DataArray,
    max_discharge_temp_c: float,
    max_temp_rise_c: float,
) -> xr.DataArray:
    return np.maximum(
        np.minimum(max_discharge_temp_c - water_temperature_c, max_temp_rise_c),
        0.0,
    )


def _once_through_withdrawal_intensity(
    *,
    thermal_efficiency: xr.DataArray,
    heat_rejection_coefficient: xr.DataArray,
    temperature_headroom_c: xr.DataArray,
    params: WetCoolingAssumptions,
) -> xr.DataArray:
    numerator = 3.6e6 * (1.0 - thermal_efficiency - heat_rejection_coefficient) / thermal_efficiency
    denominator = (
        params.water_density_kg_per_m3
        * params.water_specific_heat_j_per_g_k
        * temperature_headroom_c
    )
    return xr.where(temperature_headroom_c > 0.0, numerator / denominator, np.inf)


def _recirculating_withdrawal_intensity(
    *,
    thermal_efficiency: xr.DataArray,
    heat_rejection_coefficient: xr.DataArray,
    humidity_ratio_out: xr.DataArray,
    humidity_ratio_in: xr.DataArray,
    enthalpy_out: xr.DataArray,
    enthalpy_in: xr.DataArray,
    wet_bulb_temperature_c: xr.DataArray,
    water_temperature_c: xr.DataArray,
    params: WetCoolingAssumptions,
) -> xr.DataArray:
    humidity_delta = humidity_ratio_out - humidity_ratio_in
    numerator = (
        3.6e6
        * (1.0 - thermal_efficiency - heat_rejection_coefficient)
        / thermal_efficiency
        * humidity_delta
        / 1000.0
    )
    denominator = params.water_density_kg_per_m3 * (
        (enthalpy_out - enthalpy_in) * ((params.concentration_cycles - 1.0) / params.concentration_cycles)
        + (
            ((wet_bulb_temperature_c + params.tower_approach_c) / params.concentration_cycles)
            - water_temperature_c
        )
        * params.water_specific_heat_j_per_g_k
        * humidity_delta
        / 1000.0
    )
    return numerator / denominator


def _compute_saturation_vapor_pressure(air_temperature_c: xr.DataArray) -> xr.DataArray:
    exponent = 7.5 * air_temperature_c / (237.7 + air_temperature_c)
    return 6.11 * np.power(10.0, exponent) * 100.0


def _humidity_ratio(
    *,
    vapor_pressure_pa: xr.DataArray,
    surface_pressure_kpa: xr.DataArray,
) -> xr.DataArray:
    surface_pressure_pa = surface_pressure_kpa * 1000.0
    return 621.9907 * vapor_pressure_pa / (surface_pressure_pa - vapor_pressure_pa)


def _enthalpy_kj_per_kg(
    *,
    air_temperature_c: xr.DataArray,
    humidity_ratio_g_per_kg: xr.DataArray,
) -> xr.DataArray:
    air_temperature_k = air_temperature_c + 273.15
    return air_temperature_k * (1.01 + 0.00189 * humidity_ratio_g_per_kg) + 2.5 * humidity_ratio_g_per_kg


def _compute_wet_bulb_temperature(
    *,
    air_temperature_c: xr.DataArray,
    relative_humidity: xr.DataArray,
) -> xr.DataArray:
    rh = relative_humidity.clip(min=1.0, max=100.0)
    temperature = air_temperature_c
    return (
        temperature * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(temperature + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * np.power(rh, 1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    )
