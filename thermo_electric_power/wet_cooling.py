from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from climateresponse import aggregate_seasonal

from thermo_electric_power.climate import attach_gridded_dataset

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
        "cooling_category",
        "cooling_physics",
        "requires_water",
    }
)


@dataclass(frozen=True)
class WetCoolingAssumptions:
    """Default screening assumptions for plant-level wet-cooling penalties."""

    once_through_design_intake_temp_c: float = 15.0
    once_through_condenser_delta_c: float = 10.0
    once_through_max_discharge_temp_c: float = 32.0
    once_through_min_discharge_m3_per_s_per_mw: float = 0.05
    recirculating_design_wet_bulb_temp_c: float = 16.0
    recirculating_tower_approach_c: float = 7.0
    recirculating_penalty_per_c: float = 0.015
    recirculating_reference_pressure_kpa: float = 101.325
    recirculating_penalty_per_kpa: float = 0.002


def normalize_wet_cooling_climate_inputs(
    dataset: xr.Dataset,
    *,
    variable_names: Mapping[str, str] | None = None,
    variable_units: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Select, rename, and normalize the climate inputs required by wet-cooling physics."""

    variables = dict(DEFAULT_ISIMIP_VARIABLE_NAMES)
    if variable_names is not None:
        variables.update(variable_names)

    missing = [source_name for source_name in variables.values() if source_name not in dataset.data_vars]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"dataset missing required climate variables: {missing_list}")

    normalized = dataset[list(variables.values())].rename(
        {source_name: target_name for target_name, source_name in variables.items()}
    )
    normalized["water_temperature"] = _normalize_temperature(
        normalized["water_temperature"],
        unit=_resolve_unit(normalized["water_temperature"], variable_units=variable_units),
    )
    normalized["air_temperature"] = _normalize_temperature(
        normalized["air_temperature"],
        unit=_resolve_unit(normalized["air_temperature"], variable_units=variable_units),
    )
    normalized["surface_pressure"] = _normalize_pressure(
        normalized["surface_pressure"],
        unit=_resolve_unit(normalized["surface_pressure"], variable_units=variable_units),
    )
    return normalized


def build_wet_cooling_mapping(
    unit_cell_mapping: pd.DataFrame,
    cooling_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Join plant coordinates to water-dependent cooling assignments."""

    missing_mapping = REQUIRED_WET_MAPPING_COLUMNS.difference(unit_cell_mapping.columns)
    if missing_mapping:
        missing_list = ", ".join(sorted(missing_mapping))
        raise ValueError(f"unit_cell_mapping missing required columns: {missing_list}")

    missing_cooling = REQUIRED_COOLING_JOIN_COLUMNS.difference(cooling_mapping.columns)
    if missing_cooling:
        missing_list = ", ".join(sorted(missing_cooling))
        raise ValueError(f"cooling_mapping missing required columns: {missing_list}")

    _validate_unique_units(unit_cell_mapping, frame_name="unit_cell_mapping")
    _validate_unique_units(cooling_mapping, frame_name="cooling_mapping")

    joined = _join_country_from_cooling(unit_cell_mapping, cooling_mapping).merge(
        cooling_mapping[
            [column for column in ["UNITID", "LOCATIONID", "cooling_category", "cooling_physics", "requires_water"]]
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

    return wet.sort_values(["country", "LOCATIONID", "UNITID"], kind="stable").reset_index(drop=True)


def attach_wet_cooling_climate(
    wet_cooling_mapping: pd.DataFrame,
    climate_inputs: xr.Dataset,
) -> xr.Dataset:
    """Attach normalized wet-cooling climate series to plant-level mappings."""

    return attach_gridded_dataset(
        wet_cooling_mapping,
        climate_inputs,
        metadata_columns=[
            "UNITID",
            "LOCATIONID",
            "country",
            "mw",
            "cooling_category",
            "cooling_physics",
        ],
    )


def compute_monthly_wet_cooling_impacts(
    wet_cooling_mapping: pd.DataFrame,
    climate_inputs: xr.Dataset,
    *,
    assumptions: WetCoolingAssumptions | None = None,
) -> xr.Dataset:
    """Compute plant-level monthly wet-cooling capacity outcomes."""

    params = assumptions or WetCoolingAssumptions()
    normalized_inputs = normalize_wet_cooling_climate_inputs(climate_inputs)
    attached = attach_wet_cooling_climate(wet_cooling_mapping, normalized_inputs)

    once_through_capacity = _once_through_capacity_factor(attached, params=params)
    recirculating_capacity = _recirculating_capacity_factor(attached, params=params)
    physics = attached.coords["cooling_physics"]
    capacity_factor = xr.where(physics == "once_through", once_through_capacity, recirculating_capacity)
    capacity_factor = capacity_factor.clip(min=0.0, max=1.0).rename("capacity_factor")
    capacity_loss = (1.0 - capacity_factor).rename("capacity_loss")
    usable_capacity = (capacity_factor * attached.coords["mw"]).rename("usable_capacity_mw")
    wet_bulb = _compute_wet_bulb_temperature(
        air_temperature_c=attached["air_temperature"],
        relative_humidity=attached["relative_humidity"],
    ).rename("wet_bulb_temperature")

    monthly = xr.Dataset(
        data_vars={
            "water_temperature": attached["water_temperature"],
            "discharge": attached["discharge"],
            "air_temperature": attached["air_temperature"],
            "relative_humidity": attached["relative_humidity"],
            "surface_pressure": attached["surface_pressure"],
            "wet_bulb_temperature": wet_bulb,
            "capacity_factor": capacity_factor,
            "capacity_loss": capacity_loss,
            "usable_capacity_mw": usable_capacity,
        },
        coords=attached.coords,
        attrs={
            "once_through_design_intake_temp_c": params.once_through_design_intake_temp_c,
            "once_through_condenser_delta_c": params.once_through_condenser_delta_c,
            "once_through_max_discharge_temp_c": params.once_through_max_discharge_temp_c,
            "once_through_min_discharge_m3_per_s_per_mw": (
                params.once_through_min_discharge_m3_per_s_per_mw
            ),
            "recirculating_design_wet_bulb_temp_c": params.recirculating_design_wet_bulb_temp_c,
            "recirculating_tower_approach_c": params.recirculating_tower_approach_c,
            "recirculating_penalty_per_c": params.recirculating_penalty_per_c,
            "recirculating_reference_pressure_kpa": params.recirculating_reference_pressure_kpa,
            "recirculating_penalty_per_kpa": params.recirculating_penalty_per_kpa,
        },
    )
    return monthly.transpose("time", "entity")


def aggregate_seasonal_wet_cooling_impacts(
    monthly_impacts: xr.Dataset,
    *,
    seasons: Mapping[str, tuple[int, ...]] | None = None,
) -> xr.Dataset:
    """Aggregate monthly wet-cooling outputs into seasonal means."""

    return aggregate_seasonal(monthly_impacts, seasons=seasons)


def _validate_unique_units(frame: pd.DataFrame, *, frame_name: str) -> None:
    duplicates = frame["UNITID"].duplicated(keep=False)
    if duplicates.any():
        duplicate_units = frame.loc[duplicates, "UNITID"].tolist()[:5]
        raise ValueError(f"{frame_name} must have unique UNITID values; duplicates include {duplicate_units}")


def _join_country_from_cooling(
    unit_cell_mapping: pd.DataFrame,
    cooling_mapping: pd.DataFrame,
) -> pd.DataFrame:
    if "country" in unit_cell_mapping.columns:
        return unit_cell_mapping
    if "country" not in cooling_mapping.columns:
        raise ValueError("country must be present in unit_cell_mapping or cooling_mapping")
    return unit_cell_mapping.merge(
        cooling_mapping[["UNITID", "LOCATIONID", "country"]],
        on=["UNITID", "LOCATIONID"],
        how="left",
        validate="one_to_one",
    )


def _resolve_unit(
    data: xr.DataArray,
    *,
    variable_units: Mapping[str, str] | None,
) -> str:
    explicit_unit = None if variable_units is None else variable_units.get(str(data.name))
    unit = explicit_unit or data.attrs.get("units")
    if unit is None:
        raise ValueError(f"{data.name!r} is missing units metadata; pass variable_units explicitly")
    return str(unit).strip()


def _normalize_temperature(data: xr.DataArray, *, unit: str) -> xr.DataArray:
    normalized_unit = unit.lower()
    if normalized_unit in {"k", "kelvin"}:
        converted = data - 273.15
        converted.attrs = data.attrs
        converted.attrs["units"] = "C"
        return converted
    if normalized_unit in {"c", "degc", "degree_celsius", "degrees_celsius", "celsius"}:
        return data
    raise ValueError(f"unsupported temperature unit for {data.name!r}: {unit}")


def _normalize_pressure(data: xr.DataArray, *, unit: str) -> xr.DataArray:
    normalized_unit = unit.lower()
    if normalized_unit in {"pa", "pascal", "pascals"}:
        converted = data / 1_000.0
        converted.attrs = data.attrs
        converted.attrs["units"] = "kPa"
        return converted
    if normalized_unit in {"kpa", "kilopascal", "kilopascals"}:
        return data
    raise ValueError(f"unsupported pressure unit for {data.name!r}: {unit}")


def _once_through_capacity_factor(
    attached: xr.Dataset,
    *,
    params: WetCoolingAssumptions,
) -> xr.DataArray:
    intake_limit = params.once_through_max_discharge_temp_c - params.once_through_condenser_delta_c
    thermal_headroom = (
        (intake_limit - attached["water_temperature"])
        / (intake_limit - params.once_through_design_intake_temp_c)
    ).clip(min=0.0, max=1.0)
    water_headroom = (
        attached["discharge"] / (attached.coords["mw"] * params.once_through_min_discharge_m3_per_s_per_mw)
    ).clip(max=1.0)
    return xr.apply_ufunc(np.minimum, thermal_headroom, water_headroom)


def _recirculating_capacity_factor(
    attached: xr.Dataset,
    *,
    params: WetCoolingAssumptions,
) -> xr.DataArray:
    wet_bulb = _compute_wet_bulb_temperature(
        air_temperature_c=attached["air_temperature"],
        relative_humidity=attached["relative_humidity"],
    )
    wet_bulb_excess = (
        wet_bulb + params.recirculating_tower_approach_c - params.recirculating_design_wet_bulb_temp_c
    ).clip(min=0.0)
    low_pressure = (params.recirculating_reference_pressure_kpa - attached["surface_pressure"]).clip(min=0.0)
    penalty = (
        wet_bulb_excess * params.recirculating_penalty_per_c
        + low_pressure * params.recirculating_penalty_per_kpa
    )
    return (1.0 - penalty).clip(min=0.0, max=1.0)


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
