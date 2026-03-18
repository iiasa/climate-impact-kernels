from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from climateresponse import aggregate_seasonal

from thermo_electric_power.climate import attach_gridded_dataset

REQUIRED_DRY_COOLING_VARIABLES = frozenset({"air_temperature_k"})

DEFAULT_DRY_COOLING_VARIABLE_NAMES = {
    "air_temperature_k": "tas",
}

REQUIRED_DRY_MAPPING_COLUMNS = frozenset(
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
        "utype",
        "technology_family",
        "cooling_category",
        "cooling_physics",
    }
)

SUPPORTED_DRY_COOLING_ENGINE_TYPES = frozenset(
    {
        "combustible_steam",
        "combined_cycle",
        "non_combustible_steam",
        "nuclear_steam",
    }
)

COMBUSTIBLE_FUEL_MARKERS = (
    "BIO",
    "COAL",
    "COKE",
    "DIESEL",
    "FO",
    "GAS",
    "HFO",
    "LFO",
    "LIGNITE",
    "LNG",
    "NAPHTHA",
    "NG",
    "OIL",
    "PEAT",
    "PETCOKE",
    "SHALE",
    "WASTE",
)

NON_COMBUSTIBLE_FUEL_MARKERS = (
    "GEOTHERMAL",
    "SOLAR",
    "WASTE HEAT",
)


@dataclass(frozen=True)
class DryCoolingAssumptions:
    """Screening assumptions for plant-level dry-cooling penalties."""

    combustible_steam_parasitic_load_fraction: float = 0.0243
    combined_cycle_parasitic_load_fraction: float = 0.0081
    non_combustible_steam_parasitic_load_fraction: float = 0.0056
    unsupported_capacity_factor: float = 1.0


def normalize_dry_cooling_climate_inputs(
    dataset: xr.Dataset,
    *,
    variable_names: Mapping[str, str] | None = None,
    variable_units: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Select, rename, and normalize the climate inputs required by dry cooling."""

    variables = dict(DEFAULT_DRY_COOLING_VARIABLE_NAMES)
    if variable_names is not None:
        variables.update(variable_names)

    missing = [source_name for source_name in variables.values() if source_name not in dataset.data_vars]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"dataset missing required climate variables: {missing_list}")

    normalized = dataset[list(variables.values())].rename(
        {source_name: target_name for target_name, source_name in variables.items()}
    )
    normalized["air_temperature_k"] = _normalize_temperature_kelvin(
        normalized["air_temperature_k"],
        unit=_resolve_unit(normalized["air_temperature_k"], variable_units=variable_units),
    )
    return normalized


def classify_dry_cooling_engine_type(
    *,
    technology_family: str,
    fuel: str,
) -> tuple[str, bool, str]:
    """Map the local technology contract onto Qin dry-cooling equation families."""

    match technology_family:
        case "combined_cycle":
            return "combined_cycle", True, "supported_qin_family"
        case "steam":
            fuel_group = _classify_fuel_group(fuel)
            match fuel_group:
                case "nuclear":
                    return "nuclear_steam", True, "supported_qin_family"
                case "combustible":
                    return "combustible_steam", True, "supported_qin_family"
                case _:
                    return "non_combustible_steam", True, "supported_qin_family"
        case "gas_turbine" | "internal_combustion":
            return "no_steam_cycle", False, "excluded_no_steam_cycle"
        case _:
            return "unsupported_other", False, "excluded_unsupported_technology_family"


def build_dry_cooling_mapping(
    unit_cell_mapping: pd.DataFrame,
    cooling_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Join plant coordinates to dry-cooling engine classifications."""

    missing_mapping = REQUIRED_DRY_MAPPING_COLUMNS.difference(unit_cell_mapping.columns)
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
            [
                "UNITID",
                "LOCATIONID",
                "fuel",
                "utype",
                "technology_family",
                "cooling_category",
                "cooling_physics",
            ]
        ],
        on=["UNITID", "LOCATIONID"],
        how="inner",
        validate="one_to_one",
    )

    dry = joined.loc[joined["cooling_physics"] == "dry"].copy()
    engine_mapping = dry.apply(
        lambda row: classify_dry_cooling_engine_type(
            technology_family=row["technology_family"],
            fuel=row["fuel"],
        ),
        axis=1,
    )
    engine_frame = pd.DataFrame(
        engine_mapping.tolist(),
        columns=[
            "dry_cooling_engine_type",
            "dry_cooling_supported",
            "dry_cooling_support_reason",
        ],
        index=dry.index,
    )
    return (
        dry.join(engine_frame)
        .sort_values(["country", "LOCATIONID", "UNITID"], kind="stable")
        .reset_index(drop=True)
    )


def attach_dry_cooling_climate(
    dry_cooling_mapping: pd.DataFrame,
    climate_inputs: xr.Dataset,
) -> xr.Dataset:
    """Attach normalized dry-cooling climate series to plant-level mappings."""

    return attach_gridded_dataset(
        dry_cooling_mapping,
        climate_inputs,
        metadata_columns=[
            "UNITID",
            "LOCATIONID",
            "country",
            "fuel",
            "utype",
            "technology_family",
            "mw",
            "cooling_category",
            "cooling_physics",
            "dry_cooling_engine_type",
            "dry_cooling_supported",
            "dry_cooling_support_reason",
        ],
    )


def compute_monthly_dry_cooling_impacts(
    dry_cooling_mapping: pd.DataFrame,
    climate_inputs: xr.Dataset,
    *,
    assumptions: DryCoolingAssumptions | None = None,
) -> xr.Dataset:
    """Compute plant-level monthly dry-cooling capacity outcomes."""

    params = assumptions or DryCoolingAssumptions()
    normalized_inputs = normalize_dry_cooling_climate_inputs(climate_inputs)
    attached = attach_dry_cooling_climate(dry_cooling_mapping, normalized_inputs)

    engine_type = attached.coords["dry_cooling_engine_type"]
    supported = attached.coords["dry_cooling_supported"]
    air_temperature = attached["air_temperature_k"]

    efficiency_loss = xr.where(
        engine_type == "combustible_steam",
        _combustible_steam_efficiency_loss(air_temperature),
        xr.where(
            engine_type == "combined_cycle",
            _combined_cycle_efficiency_loss(air_temperature),
            xr.where(
                engine_type == "nuclear_steam",
                _nuclear_steam_efficiency_loss(air_temperature),
                xr.where(
                    engine_type == "non_combustible_steam",
                    _non_combustible_steam_efficiency_loss(air_temperature),
                    0.0,
                ),
            ),
        ),
    ).clip(min=0.0)

    parasitic_load = xr.where(
        engine_type == "combustible_steam",
        params.combustible_steam_parasitic_load_fraction,
        xr.where(
            engine_type == "combined_cycle",
            params.combined_cycle_parasitic_load_fraction,
            xr.where(
                engine_type.isin(["nuclear_steam", "non_combustible_steam"]),
                params.non_combustible_steam_parasitic_load_fraction,
                0.0,
            ),
        ),
    )

    capacity_factor = xr.where(
        supported,
        1.0 - efficiency_loss - parasitic_load,
        params.unsupported_capacity_factor,
    ).clip(min=0.0, max=1.0)
    capacity_loss = (1.0 - capacity_factor).rename("capacity_loss")
    usable_capacity = (capacity_factor * attached.coords["mw"]).rename("usable_capacity_mw")
    backpressure = _compute_backpressure(air_temperature).rename("backpressure_inhg")

    monthly = xr.Dataset(
        data_vars={
            "air_temperature_k": air_temperature,
            "backpressure_inhg": backpressure,
            "efficiency_loss": efficiency_loss.rename("efficiency_loss"),
            "parasitic_load": parasitic_load.rename("parasitic_load"),
            "capacity_factor": capacity_factor.rename("capacity_factor"),
            "capacity_loss": capacity_loss,
            "usable_capacity_mw": usable_capacity,
        },
        coords=attached.coords,
        attrs={
            "combustible_steam_parasitic_load_fraction": (
                params.combustible_steam_parasitic_load_fraction
            ),
            "combined_cycle_parasitic_load_fraction": params.combined_cycle_parasitic_load_fraction,
            "non_combustible_steam_parasitic_load_fraction": (
                params.non_combustible_steam_parasitic_load_fraction
            ),
            "unsupported_capacity_factor": params.unsupported_capacity_factor,
            "unsupported_handling": "Units without a Qin steam-cycle family keep capacity_factor=1.0.",
        },
    )
    return monthly.transpose("time", "entity")


def aggregate_seasonal_dry_cooling_impacts(
    monthly_impacts: xr.Dataset,
    *,
    seasons: Mapping[str, tuple[int, ...]] | None = None,
) -> xr.Dataset:
    """Aggregate monthly dry-cooling outputs into seasonal means."""

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


def _normalize_temperature_kelvin(data: xr.DataArray, *, unit: str) -> xr.DataArray:
    normalized_unit = unit.lower()
    if normalized_unit in {"k", "kelvin"}:
        return data
    if normalized_unit in {"c", "degc", "degree_celsius", "degrees_celsius", "celsius"}:
        converted = data + 273.15
        converted.attrs = data.attrs
        converted.attrs["units"] = "K"
        return converted
    raise ValueError(f"unsupported temperature unit for {data.name!r}: {unit}")


def _classify_fuel_group(fuel: str) -> str:
    normalized = fuel.upper()
    if "NUC" in normalized or "URANIUM" in normalized:
        return "nuclear"
    if any(marker in normalized for marker in NON_COMBUSTIBLE_FUEL_MARKERS):
        return "non_combustible"
    if any(marker in normalized for marker in COMBUSTIBLE_FUEL_MARKERS):
        return "combustible"
    return "non_combustible"


def _compute_backpressure(air_temperature_k: xr.DataArray) -> xr.DataArray:
    return 1.031 * xr.apply_ufunc(np.exp, 0.019 * air_temperature_k)


def _combustible_steam_efficiency_loss(air_temperature_k: xr.DataArray) -> xr.DataArray:
    return (
        1.73e-10 * xr.apply_ufunc(np.exp, 0.0684 * air_temperature_k)
        - 6.64e-7 * xr.apply_ufunc(np.exp, 0.0342 * air_temperature_k)
        - 0.0062
    )


def _combined_cycle_efficiency_loss(air_temperature_k: xr.DataArray) -> xr.DataArray:
    return (
        -1.83e-15 * xr.apply_ufunc(np.exp, 0.1026 * air_temperature_k)
        + 2.26e-10 * xr.apply_ufunc(np.exp, 0.0684 * air_temperature_k)
        - 2.66e-6 * xr.apply_ufunc(np.exp, 0.0342 * air_temperature_k)
        + 0.0033
    )


def _non_combustible_steam_efficiency_loss(air_temperature_k: xr.DataArray) -> xr.DataArray:
    return (
        -5.95e-15 * xr.apply_ufunc(np.exp, 0.1026 * air_temperature_k)
        + 4.66e-10 * xr.apply_ufunc(np.exp, 0.0684 * air_temperature_k)
        - 4.75e-6 * xr.apply_ufunc(np.exp, 0.0342 * air_temperature_k)
        + 0.0098
    )


def _nuclear_steam_efficiency_loss(air_temperature_k: xr.DataArray) -> xr.DataArray:
    return (
        -2.75e-15 * xr.apply_ufunc(np.exp, 0.1026 * air_temperature_k)
        + 2.73e-10 * xr.apply_ufunc(np.exp, 0.0684 * air_temperature_k)
        - 3.45e-6 * xr.apply_ufunc(np.exp, 0.0342 * air_temperature_k)
        + 0.0111
    )
