"""Wet-cooling (once-through and recirculating) withdrawal kernel.

The equations follow Li et al. (2025), https://doi.org/10.1038/s41893-025-01692-9,
and are adapted from the authors' released analysis code, Li, S. & Qin, Y.,
https://doi.org/10.5281/zenodo.17369544, licensed CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Changes from that code: the
equations are vectorised over xarray; moist-air enthalpy uses air temperature
in degrees Celsius, as its constants require, where the released code uses
kelvin (see ``_moist_air_enthalpy_kj_per_kg``); and saline once-through units
take their allowed temperature rise (how much the cooling water may warm) from
sea-surface temperature at an intake point instead of river temperature (see ``assign_saline_intakes``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from impacts.climatecore import nearest_valid_cells
from impacts.engine import ClimateInput, Duration, Kernel, evaluate_mapped
from impacts.isimip import ISIMIPVariable
from impacts.thermoelectric.fleet import (
    join_country_from_cooling,
    validate_unique_units,
)
from impacts.thermoelectric.fuels import classify_fuel

LI_2025_DOI = "doi:10.1038/s41893-025-01692-9"

WET_COOLING_VARIABLES = (
    ISIMIPVariable("tas", "climate", "daily"),
    ISIMIPVariable("hurs", "climate", "daily"),
    ISIMIPVariable("ps", "climate", "daily"),
    ISIMIPVariable("dis", "watergap", "daily"),
    ISIMIPVariable("triver", "watergap", "monthly", output_name="tw"),
)

# Saline once-through units draw seawater, so they read sea-surface temperature
# rather than river temperature. Load these alongside WET_COOLING_VARIABLES when
# the fleet holds ot_saline units.
SEA_SURFACE_TEMPERATURE_NAME = "tos"
SALINE_INTAKE_VARIABLES = (
    ISIMIPVariable("tos", "ocean", "monthly", resolution="60arcmin"),
)
DEFAULT_SALINE_INTAKE_MAX_DISTANCE_KM = 150.0

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

WET_SITE_COLUMNS = (
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

FUEL_EFFICIENCY_CURVE_PARAMETERS = {
    "coal": (24.14675, 44.89252, 198.43242, 1.63882),
    "gas": (38.59261, 64.74196, 269.6733, 1.28565),
    "oil": (35.96246, 50.84005, 224.66888, 1.50589),
    "biomass": (16.04344, 36.14518, 24.37557, 1.87955),
}
# WEPP splits a combined cycle into a gas-turbine unit and a steam unit, and only
# the steam unit rejects heat to cooling water. Its Rankine steam cycle is given
# the coal curve and coefficients; the class keeps its own name so the
# assumption stays visible.
FUEL_EFFICIENCY_CURVE_PARAMETERS["steam_bottoming"] = FUEL_EFFICIENCY_CURVE_PARAMETERS["coal"]

HEAT_REJECTION_COEFFICIENTS = {
    "coal": 0.12,
    "gas": 0.12,
    "oil": 0.12,
    "biomass": 0.12,
    "nuclear": 0.0,
}
HEAT_REJECTION_COEFFICIENTS["steam_bottoming"] = HEAT_REJECTION_COEFFICIENTS["coal"]

EVAPORATION_COEFFICIENTS = {
    "coal": 0.0042,
    "gas": 0.0088,
    "oil": 0.0088,
    "biomass": 0.0086,
    "nuclear": 0.0061,
}
EVAPORATION_COEFFICIENTS["steam_bottoming"] = EVAPORATION_COEFFICIENTS["coal"]


@dataclass(frozen=True)
class WetCoolingAssumptions:
    """Constants of the Li et al. (2025) wet-cooling withdrawal and capacity equations."""

    water_density_kg_per_m3: float = field(
        default=1000.0,
        metadata={"unit": "kg m-3", "source": f"Li et al. 2025, {LI_2025_DOI}, Methods (rho_w in eqs. 1-2)"},
    )
    water_specific_heat_j_per_g_k: float = field(
        default=4.184,
        metadata={"unit": "J g-1 K-1", "source": f"Li et al. 2025, {LI_2025_DOI}, Methods (C_p in eqs. 1-2)"},
    )
    environmental_flow_fraction: float = field(
        default=0.8,
        metadata={"unit": "1", "source": f"Li et al. 2025, {LI_2025_DOI}, Methods: share of discharge reserved for ecosystems (Q_env in eq. 4)"},
    )
    concentration_cycles: float = field(
        default=5.0,
        metadata={"unit": "1", "source": f"Li et al. 2025, {LI_2025_DOI}, Methods: cooling-tower cycles of concentration (n_cc in eq. 2)"},
    )
    tower_approach_c: float = field(
        default=6.0,
        metadata={"unit": "K", "source": f"Li et al. 2025, {LI_2025_DOI}, Methods: cooling-tower approach temperature (T_app in eq. 2)"},
    )
    max_discharge_temp_c: float = field(
        default=32.0,
        metadata={"unit": "degC", "source": f"Li et al. 2025, {LI_2025_DOI}, Methods: maximum discharge temperature (T_lmax in eq. 1)"},
    )
    max_temp_rise_c: float = field(
        default=10.0,
        metadata={"unit": "K", "source": f"Li et al. 2025, {LI_2025_DOI}, Methods: maximum water temperature rise (dT_lmax in eq. 1)"},
    )


def build_wet_cooling_mapping(
    unit_cell_mapping: pd.DataFrame,
    cooling_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Join plant coordinates to Li-style wet-cooling coefficients.

    ``cell_total_mw`` and ``unit_share_ratio`` divide a cell's river discharge among
    the discharge-constrained units only: ``ot_saline`` units are excluded from the
    total and get a share of 0.
    """

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

    undecided = joined["requires_water"].isna()
    if undecided.any():
        units = joined.loc[undecided, "UNITID"].tolist()[:5]
        raise ValueError(
            f"cooling_mapping has null requires_water for {int(undecided.sum())} units "
            f"(first UNITIDs: {units}); set it to True or False for every unit"
        )
    wet = joined.loc[joined["requires_water"].astype(bool)].copy()
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
        fuel_efficiency(wet_fuel_type, mw)
        for wet_fuel_type, mw in zip(
            wet["wet_fuel_type"],
            wet["mw"],
            strict=True,
        )
    ]
    wet["heat_rejection_coefficient"] = wet["wet_fuel_type"].map(HEAT_REJECTION_COEFFICIENTS)
    wet["evaporation_coefficient"] = wet["wet_fuel_type"].map(EVAPORATION_COEFFICIENTS)
    # Only discharge-constrained units divide the river: ot_saline units take their
    # cooling water from the sea, so they neither count in the cell total nor
    # receive a share.
    draws_on_river = wet["cooling_category"] != "ot_saline"
    wet["cell_total_mw"] = (
        wet["mw"]
        .where(draws_on_river, 0.0)
        .groupby([wet["grid_lat"], wet["grid_lon"]])
        .transform("sum")
    )
    wet["unit_share_ratio"] = (wet["mw"] / wet["cell_total_mw"]).where(draws_on_river, 0.0)

    return wet.sort_values(["country", "LOCATIONID", "UNITID"], kind="stable").reset_index(drop=True)


def assign_saline_intakes(
    wet_mapping: pd.DataFrame,
    sea_surface_temperature: xr.DataArray,
    *,
    max_distance_km: float = DEFAULT_SALINE_INTAKE_MAX_DISTANCE_KM,
) -> pd.DataFrame:
    """Give each saline once-through unit the nearest ocean cell as its intake.

    Adds ``intake_lat``, ``intake_lon`` and ``intake_distance_km``. A unit's
    intake is the nearest cell, by great-circle distance from the unit's cell
    centre, where ``sea_surface_temperature`` is finite at every time step. Pass
    the field on the grid the kernel will evaluate on (``load_isimip_dataset``
    copies the 1-degree ocean grid onto the 0.5-degree land grid). Coarse ocean
    grids leave most coastlines a cell or two offshore, so the distance is
    usually tens of kilometres. Beyond ``max_distance_km`` the intake is left
    empty and the unit's capacity comes out unknown (NaN): such units are
    typically inland plants labelled saline, or on a sea the model's ocean
    omits (the Caspian, in GFDL-ESM4). Other units get empty intake columns.
    """

    valid = sea_surface_temperature.notnull()
    if "time" in valid.dims:
        valid = valid.all("time")
    saline = (wet_mapping["cooling_category"] == "ot_saline").to_numpy()
    assigned = wet_mapping.assign(intake_lat=np.nan, intake_lon=np.nan, intake_distance_km=np.nan)
    if saline.any():
        intakes = nearest_valid_cells(
            wet_mapping.loc[saline, "grid_lat"].to_numpy(),
            wet_mapping.loc[saline, "grid_lon"].to_numpy(),
            valid,
            max_distance_km=max_distance_km,
        )
        assigned.loc[saline, ["intake_lat", "intake_lon", "intake_distance_km"]] = intakes[
            ["cell_lat", "cell_lon", "distance_km"]
        ].to_numpy()
    return assigned


# Physical constants. Equation numbers below are those of the Methods of
# Li et al. (2025). The kernel implements eqs. 1-5 (withdrawal intensity, usable
# capacity, available water, capacity ratio); eq. 6, the relative change in
# capacity ratio between periods, is an analysis step outside it. The paper names
# but does not print the psychrometric steps (humidity ratios, enthalpies,
# wet-bulb temperature), and its supplement does not either, so those carry no
# equation number.

# Saturation vapour pressure, Magnus form as in Li & Qin's released code:
# e_s = 6.11 hPa x 10^(7.5 T / (237.7 + T)), T in C.
MAGNUS_E0_HPA = 6.11
MAGNUS_A = 7.5
MAGNUS_B_C = 237.7
PA_PER_HPA = 100.0
PA_PER_KPA = 1000.0
# Humidity ratio W = 621.9907 e / (p - e) in g per kg dry air: 1000 times the ratio
# of the molar masses of water vapour and dry air (Li & Qin's released code).
WATER_TO_DRY_AIR_G_PER_KG = 621.9907
# Moist-air enthalpy h = T (c_pa + c_pv W) + L W, referenced to 0 C, kJ per kg dry air.
DRY_AIR_HEAT_CAPACITY_KJ_PER_KG_K = 1.01  # c_pa
WATER_VAPOUR_HEAT_CAPACITY_KJ_PER_G_K = 0.00189  # c_pv
LATENT_HEAT_KJ_PER_G = 2.5  # L, vaporisation at 0 C
# Unit conversions.
KJ_PER_MWH = 3.6e6
SECONDS_PER_HOUR = 3600.0
G_PER_KG = 1000.0
# Stull (2011), https://doi.org/10.1175/JAMC-D-11-0143.1, fits wet-bulb temperature
# for relative humidity 5-99% and air temperature -20 to 50 C. The kernel clips
# relative humidity to [1, 100] % before applying it and does not limit temperature.
STULL_RELATIVE_HUMIDITY_CLIP_PERCENT = (1.0, 100.0)


def water_limited_capacity_mw(
    discharge: Any,
    withdrawal_intensity: Any,
    *,
    nameplate_mw: Any = None,
    zero_intensity: float | None = None,
    negative_intensity: float | None = None,
    zero_discharge: float | None = None,
    finite_positive_only: bool = False,
) -> Any:
    """Capacity the available water can cool, MW: Li et al. (2025) eq. 3 with eq. 4 substituted.

    ``3600 * discharge / withdrawal_intensity`` for discharge in m3 s-1 and
    withdrawal intensity in m3 MWh-1 (the D x 24 hours of a month cancel). With
    ``discharge=1.0`` it is the water-limited capacity per m3 s-1 of discharge,
    ``3600 / withdrawal_intensity`` (MW per m3 s-1): the capacity per unit flow that
    the siting methods use to split the wet capacity loss into a thermal part and a
    water-availability part. Inputs are numpy arrays,
    scalars or xarray objects (lazy ones stay lazy).

    The callers treat the edges of the quotient differently; each rule is an
    argument, and with every argument at its default the result is the plain
    IEEE quotient (a zero intensity gives +inf for positive discharge and NaN
    for zero discharge; a negative intensity gives a negative capacity). The
    rules apply in the order listed.

    Parameters:
        discharge: Available discharge, m3 s-1.
        withdrawal_intensity: Withdrawal intensity, m3 MWh-1.
        zero_intensity: Result where the intensity is zero; ``None`` keeps the
            quotient.
        negative_intensity: Result where the intensity is negative; ``None``
            keeps the quotient.
        zero_discharge: Result where the discharge is zero, whatever the
            intensity, including an unknown one; ``None`` keeps the quotient.
        finite_positive_only: Make the result NaN wherever it is not finite and
            positive (after the rules above), so an infinite intensity (zero
            capacity) or an infinite capacity counts as unknown.
        nameplate_mw: Cap the result at this nameplate capacity (MW) with
            ``numpy.minimum``, which keeps NaN; ``None`` leaves it uncapped.

    An unknown (NaN) discharge or intensity gives NaN, except where
    ``zero_discharge`` applies.
    """

    with np.errstate(divide="ignore", invalid="ignore"):
        capacity = SECONDS_PER_HOUR * discharge / withdrawal_intensity
        if zero_intensity is not None:
            capacity = xr.where(withdrawal_intensity == 0.0, zero_intensity, capacity)
        if negative_intensity is not None:
            capacity = xr.where(withdrawal_intensity < 0.0, negative_intensity, capacity)
        if zero_discharge is not None:
            capacity = xr.where(discharge == 0.0, zero_discharge, capacity)
        if finite_positive_only:
            capacity = xr.where(np.isfinite(capacity) & (capacity > 0.0), capacity, np.nan)
        if nameplate_mw is not None:
            capacity = np.minimum(capacity, nameplate_mw)
    return capacity


def wet_cooling_physics(
    climate: xr.Dataset,
    sites: xr.Dataset,
    assumptions: WetCoolingAssumptions,
) -> xr.Dataset:
    """Withdrawal intensity and usable capacity of wet-cooled units (Li et al. 2025).

    ``climate`` holds air temperature (C), relative humidity (%), surface pressure
    (kPa), river temperature (C), river discharge (m3 s-1) and, optionally, the
    sea-surface temperature (C) at each saline unit's intake.
    """

    a = assumptions
    air_temperature = climate["air_temperature"]
    relative_humidity = climate["relative_humidity"]
    surface_pressure = climate["surface_pressure"]
    river_temperature = climate["water_temperature"]
    efficiency = sites["thermal_efficiency"]
    heat_rejection = sites["heat_rejection_coefficient"]
    mw = sites["mw"]

    # Psychrometrics of the air entering (ambient) and leaving (saturated) a tower:
    # w_in, w_out, h_a,in and h_a,out of eq. 2. The paper computes w_out from tas and
    # "pr"; the code uses surface pressure ps.
    saturation_vapour_pressure = _saturation_vapour_pressure_pa(air_temperature)
    vapour_pressure = relative_humidity / 100.0 * saturation_vapour_pressure
    humidity_ratio_out = _humidity_ratio_g_per_kg(saturation_vapour_pressure, surface_pressure)
    humidity_ratio_in = _humidity_ratio_g_per_kg(vapour_pressure, surface_pressure)
    enthalpy_out = _moist_air_enthalpy_kj_per_kg(air_temperature, humidity_ratio_out)
    enthalpy_in = _moist_air_enthalpy_kj_per_kg(air_temperature, humidity_ratio_in)
    wet_bulb = _stull_wet_bulb_c(air_temperature, relative_humidity)

    # Once-through units discharge back to their source, so the temperature limits
    # apply to the water they draw: the river for freshwater units, the sea at the
    # intake point for saline units.
    saline = sites["cooling_category"] == "ot_saline"
    if "sea_surface_temperature" in climate.data_vars:
        intake_temperature = xr.where(
            saline, climate["sea_surface_temperature"], river_temperature, keep_attrs=False
        ).assign_attrs(river_temperature.attrs)
    elif bool(saline.any()):
        raise ValueError(
            "ot_saline units draw seawater: the climate inputs need sea-surface temperature "
            f"({SEA_SURFACE_TEMPERATURE_NAME!r}) and the mapping needs intake points "
            "(assign_saline_intakes)"
        )
    else:
        intake_temperature = river_temperature
    # Allowed temperature rise ("headroom"), the denominator term of eq. 1:
    # max(min(T_lmax - T_w, dT_lmax), 0). Departure: saline units take T_w from the
    # sea at their intake, not the river.
    headroom = np.maximum(
        np.minimum(a.max_discharge_temp_c - intake_temperature, a.max_temp_rise_c), 0.0
    )

    # River water a unit may withdraw, eq. 4 per second rather than per month:
    # (Q - Q_env) x NC / NC_total, with Q_env = 80% of Q. Departure: NC_total counts
    # only the cell's river-drawing units, and saline units get no share.
    available_discharge = climate["discharge"] * (1.0 - a.environmental_flow_fraction)
    unit_available_discharge = available_discharge * sites["unit_share_ratio"]
    draws_on_river = sites["cooling_category"] != "ot_saline"

    # Heat rejected to cooling water per MWh generated, kJ MWh-1: the factor
    # (1 - eta_net - k_os) / eta_net of eqs. 1 and 2, with the MWh-to-kJ conversion
    # the printed equations leave implicit.
    rejected_heat = KJ_PER_MWH * (1.0 - efficiency - heat_rejection) / efficiency

    # Once-through withdrawal intensity WI_ot, m3 MWh-1, eq. 1: rejected heat over the
    # heat one m3 carries at the allowed temperature rise. Eq. 1 is undefined when that
    # rise is zero; the code makes it infinite there (zero capacity) and NaN where the
    # rise is unknown.
    once_through = xr.where(
        headroom > 0.0,
        rejected_heat / (a.water_density_kg_per_m3 * a.water_specific_heat_j_per_g_k * headroom),
        np.inf,
    ).where(headroom.notnull())

    # Recirculating withdrawal intensity WI_rc, m3 MWh-1, eq. 2, with (1 - 1/n_cc)
    # written as (n_cc - 1)/n_cc and the humidity ratios converted from g to kg.
    cycles = a.concentration_cycles
    humidity_delta = humidity_ratio_out - humidity_ratio_in
    blowdown_temperature_term = (wet_bulb + a.tower_approach_c) / cycles - river_temperature
    recirculating = (
        rejected_heat
        * humidity_delta
        / G_PER_KG
        / (
            a.water_density_kg_per_m3
            * (
                (enthalpy_out - enthalpy_in) * ((cycles - 1.0) / cycles)
                + blowdown_temperature_term * a.water_specific_heat_j_per_g_k * humidity_delta / G_PER_KG
            )
        )
    )
    # Departure from eq. 2, which is 0/0 at saturation: at relative humidity 100%
    # the humidity-ratio difference is zero. It has a finite limit, since the enthalpy
    # difference is proportional to the humidity-ratio difference
    # (dh/dW = c_pv T + L at fixed T), so cancelling it gives the value there.
    enthalpy_per_humidity_ratio = WATER_VAPOUR_HEAT_CAPACITY_KJ_PER_G_K * air_temperature + LATENT_HEAT_KJ_PER_G
    recirculating_saturated = (
        rejected_heat
        / G_PER_KG
        / (
            a.water_density_kg_per_m3
            * (
                enthalpy_per_humidity_ratio * ((cycles - 1.0) / cycles)
                + blowdown_temperature_term * a.water_specific_heat_j_per_g_k / G_PER_KG
            )
        )
    )
    recirculating = xr.where(humidity_delta == 0.0, recirculating_saturated, recirculating)
    withdrawal_intensity = xr.where(
        sites["cooling_physics"] == "once_through", once_through, recirculating
    )

    # Usable capacity UC, MW, eq. 3 with eq. 4 substituted (the D x 24 hours of a
    # month cancel): min(3600 x unit discharge / WI, NC), the plain quotient at
    # every edge. Departure: saline once-through units are limited by
    # temperature alone.
    discharge_limited = water_limited_capacity_mw(
        unit_available_discharge, withdrawal_intensity, nameplate_mw=mw
    )
    # Unknown intake temperature gives an unknown allowed temperature rise, hence unknown capacity
    # rather than zero.
    temperature_limited = xr.where(headroom > 0.0, mw, 0.0).where(headroom.notnull())
    # A dry river carries no cooling water whatever its temperature, so
    # river-drawing units have no capacity at zero discharge even where the
    # temperature-dependent intensity is undefined (where eq. 3 gives NaN).
    dry_river = draws_on_river & (available_discharge == 0.0)
    usable_capacity = xr.where(
        dry_river,
        0.0,
        xr.where(
            (sites["cooling_physics"] == "once_through") & (~draws_on_river),
            temperature_limited,
            discharge_limited,
        ),
    )
    # Usable capacity ratio UCR = UC / NC, eq. 5, which states 0 <= UCR <= 1;
    # usable capacity is not negative.
    capacity_factor = (usable_capacity / mw).clip(0.0, 1.0)

    return xr.Dataset(
        {
            "water_temperature": river_temperature,
            "discharge": climate["discharge"],
            "air_temperature": air_temperature,
            "relative_humidity": relative_humidity,
            "surface_pressure": surface_pressure,
            "intake_temperature": intake_temperature,
            "wet_bulb_temperature": wet_bulb,
            "saturation_vapor_pressure_pa": saturation_vapour_pressure,
            "actual_vapor_pressure_pa": vapour_pressure,
            "humidity_ratio_out_g_per_kg": humidity_ratio_out,
            "humidity_ratio_in_g_per_kg": humidity_ratio_in,
            "enthalpy_out_kj_per_kg": enthalpy_out,
            "enthalpy_in_kj_per_kg": enthalpy_in,
            "temperature_headroom_c": headroom,
            "available_discharge_m3_per_s": available_discharge,
            "unit_available_discharge_m3_per_s": unit_available_discharge,
            "discharge_constraint_applies": draws_on_river,
            "thermal_efficiency_fraction": efficiency,
            "heat_rejection_coefficient_fraction": heat_rejection,
            "evaporation_coefficient_fraction": sites["evaporation_coefficient"],
            "withdrawal_intensity_m3_per_mwh": withdrawal_intensity,
            "capacity_factor": capacity_factor,
            "capacity_loss": 1.0 - capacity_factor,
            "usable_capacity_mw": usable_capacity.clip(min=0.0),
        }
    )


def _saturation_vapour_pressure_pa(air_temperature_c: xr.DataArray) -> xr.DataArray:
    """Saturation vapour pressure in Pa (Magnus form, Tetens coefficients)."""
    exponent = MAGNUS_A * air_temperature_c / (MAGNUS_B_C + air_temperature_c)
    return MAGNUS_E0_HPA * np.power(10.0, exponent) * PA_PER_HPA


def _humidity_ratio_g_per_kg(vapour_pressure_pa: xr.DataArray, surface_pressure_kpa: xr.DataArray) -> xr.DataArray:
    """Humidity ratio W = 621.9907 e / (p - e), g water vapour per kg dry air."""
    surface_pressure_pa = surface_pressure_kpa * PA_PER_KPA
    return WATER_TO_DRY_AIR_G_PER_KG * vapour_pressure_pa / (surface_pressure_pa - vapour_pressure_pa)


def _moist_air_enthalpy_kj_per_kg(air_temperature_c: xr.DataArray, humidity_ratio_g_per_kg: xr.DataArray) -> xr.DataArray:
    """Moist-air enthalpy h = T (1.01 + 0.00189 W) + 2.5 W, kJ per kg dry air.

    T is in degrees Celsius, as the constants require: they are referenced to 0 C
    (2.5 kJ/g is the latent heat of vaporisation at 0 C). The authors' released
    code uses kelvin. The paper does not print the formula (it cites earlier
    studies for the derivation).
    """
    return (
        air_temperature_c
        * (DRY_AIR_HEAT_CAPACITY_KJ_PER_KG_K + WATER_VAPOUR_HEAT_CAPACITY_KJ_PER_G_K * humidity_ratio_g_per_kg)
        + LATENT_HEAT_KJ_PER_G * humidity_ratio_g_per_kg
    )


def _stull_wet_bulb_c(air_temperature_c: xr.DataArray, relative_humidity: xr.DataArray) -> xr.DataArray:
    """Wet-bulb temperature in C, Stull (2011) eq. 1."""
    low, high = STULL_RELATIVE_HUMIDITY_CLIP_PERCENT
    rh = relative_humidity.clip(min=low, max=high)
    t = air_temperature_c
    return (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * np.power(rh, 1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    )


WET_COOLING_KERNEL = Kernel(
    name="wet_cooling",
    inputs={
        "water_temperature": ClimateInput("tw", unit="C"),
        "discharge": ClimateInput("dis"),
        "air_temperature": ClimateInput("tas", unit="C"),
        "relative_humidity": ClimateInput("hurs"),
        "surface_pressure": ClimateInput("ps", unit="kPa"),
        # Read only by ot_saline units, at their sea intake (see assign_saline_intakes).
        "sea_surface_temperature": ClimateInput(
            SEA_SURFACE_TEMPERATURE_NAME, unit="C", point=("intake_lat", "intake_lon"), optional=True
        ),
    },
    physics=wet_cooling_physics,
    site_columns=WET_SITE_COLUMNS,
    assumptions=WetCoolingAssumptions(),
    attrs={
        "cooling_model_citation": (
            "Li, S. et al. Global hydroclimatic risks and strategic "
            "decommissioning pathways for thermal power units. Nature "
            "Sustainability. https://doi.org/10.1038/s41893-025-01692-9"
        ),
        "ot_saline_policy": (
            "ot_saline once-through units draw seawater: they are exempt from river-discharge "
            "limits, and their allowed temperature rise uses sea-surface temperature at the nearest "
            "ocean cell under the same discharge-temperature limits as river units; freshwater "
            "once-through units keep Li's river discharge and temperature."
        ),
    },
)

def wet_cooling(
    unit_cell_mapping: pd.DataFrame,
    cooling_mapping: pd.DataFrame,
    climate: xr.Dataset,
    duration: Duration,
    *,
    assumptions: WetCoolingAssumptions | None = None,
    seasons: Mapping[str, tuple[int, ...]] | None = None,
    chunks: Mapping[str, int] | None = None,
    saline_intake_max_distance_km: float = DEFAULT_SALINE_INTAKE_MAX_DISTANCE_KM,
    variable_names: Mapping[str, str] | None = None,
    variable_units: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Wet-cooling withdrawal and usable capacity of a fleet, in one call.

    Joins the fleet's cells to its cooling assignment (``build_wet_cooling_mapping``),
    gives saline once-through units their sea intakes when ``climate`` holds
    sea-surface temperature (``assign_saline_intakes``), and evaluates
    ``WET_COOLING_KERNEL`` at ``duration``. For the steps one at a time, call those
    functions and ``impacts.engine.evaluate_mapped``.
    """

    mapping = build_wet_cooling_mapping(unit_cell_mapping, cooling_mapping)
    sea_source = (variable_names or {}).get("sea_surface_temperature", SEA_SURFACE_TEMPERATURE_NAME)
    if sea_source in climate.data_vars:
        mapping = assign_saline_intakes(
            mapping, climate[sea_source], max_distance_km=saline_intake_max_distance_km
        )
    return evaluate_mapped(
        WET_COOLING_KERNEL,
        mapping,
        climate,
        duration,
        site_id_column="UNITID",
        assumptions=assumptions,
        seasons=seasons,
        chunks=chunks,
        variable_names=variable_names,
        variable_units=variable_units,
    )


def _classify_wet_fuel_type(
    fuel: str,
    technology_family: str | None = None,
) -> str:
    """Map a fuel code and technology family to a wet-cooling class.

    ``WSTH`` is the steam unit that WEPP splits off a combined cycle. A steam unit
    is classed ``steam_bottoming`` and rejects heat like a Rankine steam cycle; the
    gas-turbine, combined-cycle and internal-combustion rows are classed ``gas``.
    """

    normalized = fuel.upper()
    technology = (technology_family or "").lower()
    if normalized == "WSTH":
        if technology in {"combined_cycle", "gas_turbine", "internal_combustion"}:
            return "gas"
        if technology in {"steam", "other", ""}:
            return "steam_bottoming"
    fuel_class = classify_fuel(fuel)
    if fuel_class in HEAT_REJECTION_COEFFICIENTS:
        return fuel_class
    raise ValueError(f"unsupported wet-cooling fuel category: {fuel!r}")


def fuel_efficiency(fuel_type: str, mw: float) -> float:
    """Return the Li-style fuel and unit-size thermal efficiency fraction."""
    if fuel_type == "nuclear":
        return 0.35

    a1, a2, x0, power = FUEL_EFFICIENCY_CURVE_PARAMETERS[fuel_type]
    return (a2 + (a1 - a2) / (1.0 + (float(mw) / x0) ** power)) * 0.01
