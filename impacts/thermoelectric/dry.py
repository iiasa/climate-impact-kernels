from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr
from impacts.core import (
    EvaluationResult,
    normalize_temperature,
)
from impacts.isimip import ISIMIPVariable
from impacts.kernel import ImpactKernel, normalize_climate_inputs
from impacts.thermoelectric._common import (
    join_country_from_cooling,
    validate_unique_units,
)

REQUIRED_DRY_COOLING_VARIABLES = frozenset({"air_temperature_k"})

DEFAULT_DRY_COOLING_VARIABLE_NAMES = {
    "air_temperature_k": "tas",
}

DRY_COOLING_VARIABLES = (
    ISIMIPVariable("tas", "climate", "daily"),
)

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

DRY_METADATA_COLUMNS = (
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
    """Default dry-cooling penalty parameters following Qin et al. (2023)."""

    combustible_steam_parasitic_load_fraction: float = 0.0243
    combined_cycle_parasitic_load_fraction: float = 0.0081
    non_combustible_steam_parasitic_load_fraction: float = 0.0056
    nuclear_steam_parasitic_load_fraction: float = 0.0304
    unsupported_capacity_factor: float = 1.0


DRY_COOLING_NORMALIZATIONS = (
    ("air_temperature_k", "K", normalize_temperature),
)


def normalize_dry_cooling_climate_inputs(
    dataset: xr.Dataset,
    *,
    variable_names: Mapping[str, str] | None = None,
    variable_units: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Select, rename, and normalize the climate inputs required by dry cooling."""

    return normalize_climate_inputs(
        dataset,
        default_variable_names=DEFAULT_DRY_COOLING_VARIABLE_NAMES,
        normalizations=DRY_COOLING_NORMALIZATIONS,
        variable_names=variable_names,
        variable_units=variable_units,
    )


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

    validate_unique_units(unit_cell_mapping, frame_name="unit_cell_mapping")
    validate_unique_units(cooling_mapping, frame_name="cooling_mapping")

    joined = join_country_from_cooling(unit_cell_mapping, cooling_mapping).merge(
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
    engine_mapping = [
        classify_dry_cooling_engine_type(
            technology_family=technology_family,
            fuel=fuel,
        )
        for technology_family, fuel in zip(
            dry["technology_family"],
            dry["fuel"],
            strict=True,
        )
    ]
    engine_frame = pd.DataFrame(
        engine_mapping,
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


def create_dry_cooling_kernel(
    assumptions: DryCoolingAssumptions | None = None,
) -> ImpactKernel:
    """Build the Qin et al. dry-cooling mapped-kernel contract."""

    params = assumptions or DryCoolingAssumptions()
    return ImpactKernel(
        name="dry_cooling",
        normalize_inputs=normalize_dry_cooling_climate_inputs,
        physics_kernel=partial(_evaluate_dry_cooling_kernel, params=params),
        metadata_columns=DRY_METADATA_COLUMNS,
        clips={
            "efficiency_loss": (0.0, None),
            "parasitic_load": (0.0, None),
            "capacity_factor": (0.0, 1.0),
            "capacity_loss": (0.0, 1.0),
        },
    )


def _evaluate_dry_cooling_kernel(
    attached: xr.Dataset,
    *,
    params: DryCoolingAssumptions,
) -> EvaluationResult:
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
                engine_type == "nuclear_steam",
                params.nuclear_steam_parasitic_load_fraction,
                xr.where(
                    engine_type == "non_combustible_steam",
                    params.non_combustible_steam_parasitic_load_fraction,
                    0.0,
                ),
            ),
        ),
    ).rename("parasitic_load")

    capacity_factor = xr.where(
        supported,
        1.0 - efficiency_loss - parasitic_load,
        params.unsupported_capacity_factor,
    ).clip(min=0.0, max=1.0)

    return EvaluationResult(
        data_vars={
            "air_temperature_k": air_temperature,
            "backpressure_inhg": _compute_backpressure(air_temperature).rename("backpressure_inhg"),
            "efficiency_loss": efficiency_loss.rename("efficiency_loss"),
            "parasitic_load": parasitic_load,
            "capacity_factor": capacity_factor.rename("capacity_factor"),
            "capacity_loss": (1.0 - capacity_factor).rename("capacity_loss"),
            "usable_capacity_mw": (capacity_factor * attached.coords["mw"]).rename("usable_capacity_mw"),
        },
        attrs={
            "combustible_steam_parasitic_load_fraction": (
                params.combustible_steam_parasitic_load_fraction
            ),
            "combined_cycle_parasitic_load_fraction": params.combined_cycle_parasitic_load_fraction,
            "non_combustible_steam_parasitic_load_fraction": (
                params.non_combustible_steam_parasitic_load_fraction
            ),
            "nuclear_steam_parasitic_load_fraction": params.nuclear_steam_parasitic_load_fraction,
            "unsupported_capacity_factor": params.unsupported_capacity_factor,
            "backpressure_source_note": (
                "EPA 2001 condenser backpressure fit evaluated with condenser inlet temperature in Fahrenheit."
            ),
            "parasitic_load_source_note": (
                "Nuclear dry-tower fan penalty uses EPA 2001 Table 3-19 (3.04%); "
                "non-combustible steam retains the Qin reproduced-table value (0.56%)."
            ),
            "cooling_model_citation": (
                "Qin, Y. et al. Global assessment of the carbon-water tradeoff "
                "of dry cooling for thermal power generation. Nature Water 1, "
                "682-693. https://doi.org/10.1038/s44221-023-00120-6"
            ),
            "unsupported_handling": "Units without a Qin steam-cycle family keep capacity_factor=1.0.",
        },
    )


def _classify_fuel_group(fuel: str) -> str:
    normalized = fuel.upper()
    if "NUC" in normalized or "URANIUM" in normalized or normalized == "UR":
        return "nuclear"
    if any(marker in normalized for marker in NON_COMBUSTIBLE_FUEL_MARKERS):
        return "non_combustible"
    if any(marker in normalized for marker in COMBUSTIBLE_FUEL_MARKERS):
        return "combustible"
    return "non_combustible"


def _compute_backpressure(air_temperature_k: xr.DataArray) -> xr.DataArray:
    air_temperature_f = (air_temperature_k - 273.15) * 9.0 / 5.0 + 32.0
    return 1.031 * np.exp(0.019 * air_temperature_f)


def _combustible_steam_efficiency_loss(air_temperature_k: xr.DataArray) -> xr.DataArray:
    return (
        1.73e-10 * np.exp(0.0684 * air_temperature_k)
        - 6.64e-7 * np.exp(0.0342 * air_temperature_k)
        - 0.0062
    )


def _combined_cycle_efficiency_loss(air_temperature_k: xr.DataArray) -> xr.DataArray:
    return (
        -1.83e-15 * np.exp(0.1026 * air_temperature_k)
        + 2.26e-10 * np.exp(0.0684 * air_temperature_k)
        - 2.66e-6 * np.exp(0.0342 * air_temperature_k)
        + 0.0033
    )


def _non_combustible_steam_efficiency_loss(air_temperature_k: xr.DataArray) -> xr.DataArray:
    return (
        -5.95e-15 * np.exp(0.1026 * air_temperature_k)
        + 4.66e-10 * np.exp(0.0684 * air_temperature_k)
        - 4.75e-6 * np.exp(0.0342 * air_temperature_k)
        + 0.0098
    )


def _nuclear_steam_efficiency_loss(air_temperature_k: xr.DataArray) -> xr.DataArray:
    return (
        -2.75e-15 * np.exp(0.1026 * air_temperature_k)
        + 2.73e-10 * np.exp(0.0684 * air_temperature_k)
        - 3.45e-6 * np.exp(0.0342 * air_temperature_k)
        + 0.0111
    )
