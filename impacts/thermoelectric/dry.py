"""Dry-cooling capacity kernel.

The efficiency-loss fits and fan loads follow Qin et al. (2023), Nature Water,
https://doi.org/10.1038/s44221-023-00120-6; the nuclear fan load and the
condenser backpressure fit follow EPA (2001).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from impacts.engine import ClimateInput, Duration, Kernel, evaluate_mapped
from impacts.isimip import ISIMIPVariable
from impacts.thermoelectric.fleet import (
    join_country_from_cooling,
    validate_unique_units,
)
from impacts.thermoelectric.fuels import classify_fuel

QIN_2023_DOI = "doi:10.1038/s44221-023-00120-6"

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

DRY_SITE_COLUMNS = (
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


@dataclass(frozen=True)
class DryCoolingAssumptions:
    """Dry-tower fan loads per steam-cycle family, following Qin et al. (2023)."""

    combustible_steam_parasitic_load_fraction: float = field(
        default=0.0243, metadata={"unit": "1", "source": f"Qin et al. 2023, {QIN_2023_DOI}"}
    )
    combined_cycle_parasitic_load_fraction: float = field(
        default=0.0081, metadata={"unit": "1", "source": f"Qin et al. 2023, {QIN_2023_DOI}"}
    )
    non_combustible_steam_parasitic_load_fraction: float = field(
        default=0.0056,
        metadata={"unit": "1", "source": f"Qin et al. 2023, {QIN_2023_DOI}, fan and pumping energy table"},
    )
    nuclear_steam_parasitic_load_fraction: float = field(
        default=0.0304, metadata={"unit": "1", "source": "EPA 2001, Table 3-19 (dry-tower fan penalty)"}
    )
    unsupported_capacity_factor: float = field(
        default=1.0,
        metadata={"unit": "1", "source": "assumption: units without a Qin steam-cycle family keep full capacity"},
    )


def classify_dry_cooling_engine_type(
    *,
    technology_family: str,
    fuel: str,
) -> tuple[str, bool, str]:
    """Map a unit's technology family and fuel to the steam-cycle family (Qin et al.) whose dry-cooling equations apply.

    Returns the family name, whether Qin et al. cover it, and the reason.
    """

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
    """Join plant coordinates to the steam-cycle family of each dry-cooled unit."""

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


# Dry-cooling efficiency loss as a function of air temperature T (K): Qin et al.
# (2023) eqs. 1-4, one fit per steam-cycle family, with eq. 5 substituted to give
# a sum of exponentials in T:
#     loss(T) = c3 e^(0.1026 T) + c2 e^(0.0684 T) + c1 e^(0.0342 T) + c0
# Coefficients (c3, c2, c1, c0), dimensionless. A family without a term has a zero
# coefficient there.
EFFICIENCY_LOSS_COEFFICIENTS = {
    "combustible_steam": (0.0, 1.73e-10, -6.64e-7, -0.0062),
    "combined_cycle": (-1.83e-15, 2.26e-10, -2.66e-6, 0.0033),
    "nuclear_steam": (-2.75e-15, 2.73e-10, -3.45e-6, 0.0111),
    "non_combustible_steam": (-5.95e-15, 4.66e-10, -4.75e-6, 0.0098),
}
# Exponents of the three terms, K-1.
EFFICIENCY_LOSS_EXPONENTS_PER_K = (0.1026, 0.0684, 0.0342)
# Fan (parasitic) load of the dry tower per family, as a field of DryCoolingAssumptions.
PARASITIC_LOAD_ASSUMPTION = {
    "combustible_steam": "combustible_steam_parasitic_load_fraction",
    "combined_cycle": "combined_cycle_parasitic_load_fraction",
    "nuclear_steam": "nuclear_steam_parasitic_load_fraction",
    "non_combustible_steam": "non_combustible_steam_parasitic_load_fraction",
}
# Condenser backpressure, EPA (2001) fit: P = 1.031 inHg x e^(0.019 T_F), with the
# condenser inlet at air temperature T_F in Fahrenheit. Reported, not used.
BACKPRESSURE_INHG = 1.031
BACKPRESSURE_EXPONENT_PER_F = 0.019


def dry_cooling_physics(
    climate: xr.Dataset,
    sites: xr.Dataset,
    assumptions: DryCoolingAssumptions,
) -> xr.Dataset:
    """Capacity factor of dry-cooled steam cycles (Qin et al. 2023).

    ``climate`` holds air temperature (K). Units of a family without a Qin fit
    keep ``unsupported_capacity_factor``.
    """

    air_temperature = climate["air_temperature_k"]
    engine_type = sites["dry_cooling_engine_type"]

    # Efficiency loss and fan load of each unit's family; zero for other units.
    efficiency_loss = _by_engine_type(
        engine_type,
        {
            family: _efficiency_loss(air_temperature, coefficients)
            for family, coefficients in EFFICIENCY_LOSS_COEFFICIENTS.items()
        },
    ).clip(min=0.0)
    parasitic_load = _by_engine_type(
        engine_type,
        {family: getattr(assumptions, name) for family, name in PARASITIC_LOAD_ASSUMPTION.items()},
    ).clip(min=0.0)

    # A capacity factor lies in [0, 1].
    capacity_factor = xr.where(
        sites["dry_cooling_supported"],
        1.0 - efficiency_loss - parasitic_load,
        assumptions.unsupported_capacity_factor,
    ).clip(min=0.0, max=1.0)

    return xr.Dataset(
        {
            "air_temperature_k": air_temperature,
            "backpressure_inhg": _backpressure_inhg(air_temperature),
            "efficiency_loss": efficiency_loss,
            "parasitic_load": parasitic_load,
            "capacity_factor": capacity_factor,
            "capacity_loss": 1.0 - capacity_factor,
            "usable_capacity_mw": capacity_factor * sites["mw"],
        }
    )


def _by_engine_type(engine_type: xr.DataArray, values: dict, default: float = 0.0) -> xr.DataArray:
    """Pick each unit's value by its engine type; ``default`` for types not listed."""

    selected = default
    for family, value in reversed(values.items()):
        selected = xr.where(engine_type == family, value, selected)
    return selected


def _efficiency_loss(air_temperature_k: xr.DataArray, coefficients: tuple[float, float, float, float]) -> xr.DataArray:
    c3, c2, c1, c0 = coefficients
    k3, k2, k1 = EFFICIENCY_LOSS_EXPONENTS_PER_K
    return (
        c3 * np.exp(k3 * air_temperature_k)
        + c2 * np.exp(k2 * air_temperature_k)
        + c1 * np.exp(k1 * air_temperature_k)
        + c0
    )


def _backpressure_inhg(air_temperature_k: xr.DataArray) -> xr.DataArray:
    air_temperature_f = (air_temperature_k - 273.15) * 9.0 / 5.0 + 32.0
    return BACKPRESSURE_INHG * np.exp(BACKPRESSURE_EXPONENT_PER_F * air_temperature_f)


DRY_COOLING_KERNEL = Kernel(
    name="dry_cooling",
    inputs={"air_temperature_k": ClimateInput("tas", unit="K")},
    physics=dry_cooling_physics,
    site_columns=DRY_SITE_COLUMNS,
    assumptions=DryCoolingAssumptions(),
    attrs={
        "backpressure_source_note": (
            "EPA 2001 condenser backpressure fit evaluated with condenser inlet temperature in Fahrenheit."
        ),
        "parasitic_load_source_note": (
            "Nuclear dry-tower fan penalty uses EPA 2001 Table 3-19 (3.04%); "
            "non-combustible steam uses the Qin et al. 2023 fan and pumping energy table (0.56%)."
        ),
        "cooling_model_citation": (
            "Qin, Y. et al. Global assessment of the carbon-water tradeoff "
            "of dry cooling for thermal power generation. Nature Water 1, "
            "682-693. https://doi.org/10.1038/s44221-023-00120-6"
        ),
        "unsupported_handling": "Units without a Qin steam-cycle family keep capacity_factor=1.0.",
    },
)


def dry_cooling(
    unit_cell_mapping: pd.DataFrame,
    cooling_mapping: pd.DataFrame,
    climate: xr.Dataset,
    duration: Duration,
    *,
    assumptions: DryCoolingAssumptions | None = None,
    seasons: Mapping[str, tuple[int, ...]] | None = None,
    chunks: Mapping[str, int] | None = None,
    variable_names: Mapping[str, str] | None = None,
    variable_units: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Dry-cooling capacity of a fleet, in one call.

    Joins the fleet's cells to its cooling assignment (``build_dry_cooling_mapping``)
    and evaluates ``DRY_COOLING_KERNEL`` at ``duration``. For the steps one at a
    time, call that function and ``impacts.engine.evaluate_mapped``.
    """

    return evaluate_mapped(
        DRY_COOLING_KERNEL,
        build_dry_cooling_mapping(unit_cell_mapping, cooling_mapping),
        climate,
        duration,
        site_id_column="UNITID",
        assumptions=assumptions,
        seasons=seasons,
        chunks=chunks,
        variable_names=variable_names,
        variable_units=variable_units,
    )


def _classify_fuel_group(fuel: str) -> str:
    fuel_class = classify_fuel(fuel)
    if fuel_class == "nuclear":
        return "nuclear"
    if fuel_class in {"non_combustible", "waste_heat"}:
        return "non_combustible"
    return "combustible"
