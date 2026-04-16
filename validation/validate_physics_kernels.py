"""
Equation-fidelity validation of wet and dry cooling kernels against primary sources.

Wet cooling: Li et al. 2025 (Core_analysis_hydroclimatic_risks.py)
Dry cooling: Qin et al. 2023 efficiency-loss regression equations

This script constructs representative plant configurations across climate regimes,
computes expected outputs from the original published equations (raw numpy), and
compares them against the impacts kernel outputs. All intermediate values are
verified, not just final capacity factors.

Run:
    uv run --no-sync python validation/validate_physics_kernels.py
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
import xarray as xr

from impacts.kernel import evaluate_kernel
from impacts.thermoelectric.dry import (
    DryCoolingAssumptions,
    build_dry_cooling_mapping,
    create_dry_cooling_kernel,
)
from impacts.thermoelectric.wet import (
    WetCoolingAssumptions,
    build_wet_cooling_mapping,
    create_wet_cooling_kernel,
)

# ── Physical constants (Li script lines 19-29) ──────────────────────────────

WATER_DENSITY = 1000.0
WATER_SPECIFIC_HEAT = 4.184
CONCENTRATION_CYCLES = 5
TOWER_APPROACH = 6.0
MAX_DISCHARGE_TEMP = 32.0
MAX_TEMP_RISE = 10.0
ENVIRONMENTAL_FLOW_FRACTION = 0.8

EFFICIENCY_PARAMS = {
    "COAL": (24.14675, 44.89252, 198.43242, 1.63882),
    "GAS": (38.59261, 64.74196, 269.6733, 1.28565),
    "OIL": (35.96246, 50.84005, 224.66888, 1.50589),
    "BIOMASS": (16.04344, 36.14518, 24.37557, 1.87955),
}
HEAT_REJECTION = {"COAL": 0.12, "GAS": 0.12, "OIL": 0.12, "BIOMASS": 0.12, "UR": 0.0}
EVAPORATION = {"COAL": 0.0042, "GAS": 0.0088, "OIL": 0.0088, "BIOMASS": 0.0086, "UR": 0.0061}


# ── Li et al. original equations ─────────────────────────────────────────────


def li_efficiency(fuel: str, mw: float) -> float:
    if fuel == "UR":
        return 0.35
    a1, a2, x0, p = EFFICIENCY_PARAMS[fuel]
    return (a2 + (a1 - a2) / (1 + (mw / x0) ** p)) * 0.01


def li_wet_cooling(
    fuel: str,
    cooling: str,
    mw: float,
    tas_k: float,
    hurs: float,
    ps_pa: float,
    tw_k: float,
    dis: float,
) -> dict:
    """Reproduce Li's wet-cooling equations step by step."""
    eta = li_efficiency(fuel, mw)
    kos = HEAT_REJECTION[fuel]
    tas_c = tas_k - 273.15
    tw_c = tw_k - 273.15

    es = 6.11 * (10 ** (7.5 * tas_c / (237.7 + tas_c))) * 100.0
    e = hurs / 100.0 * es
    wout = 621.9907 * es / (ps_pa - es)
    win = 621.9907 * e / (ps_pa - e)
    hout = tas_k * (1.01 + 0.00189 * wout) + 2.5 * wout
    hin = tas_k * (1.01 + 0.00189 * win) + 2.5 * win

    wb_c = (
        tas_c * np.arctan(0.151977 * np.sqrt(hurs + 8.313659))
        + np.arctan(tas_c + hurs)
        - np.arctan(hurs - 1.676331)
        + 0.00391838 * hurs**1.5 * np.arctan(0.023101 * hurs)
        - 4.686035
    )

    if cooling == "recirculating":
        withdrawal = (
            3.6e6
            * (1 - eta - kos)
            / eta
            * (wout - win)
            / 1000.0
            / (
                WATER_DENSITY
                * (
                    (hout - hin) * ((CONCENTRATION_CYCLES - 1) / CONCENTRATION_CYCLES)
                    + (
                        ((wb_c + TOWER_APPROACH) / CONCENTRATION_CYCLES) - tw_c
                    )
                    * WATER_SPECIFIC_HEAT
                    * (wout - win)
                    / 1000.0
                )
            )
        )
    else:
        headroom = max(min(MAX_DISCHARGE_TEMP - tw_c, MAX_TEMP_RISE), 0)
        withdrawal = (
            3.6e6 * (1 - eta - kos) / eta / (WATER_DENSITY * WATER_SPECIFIC_HEAT * headroom)
            if headroom > 0
            else math.inf
        )

    dis_avail = dis * (1 - ENVIRONMENTAL_FLOW_FRACTION)
    uc = 0.0 if math.isinf(withdrawal) else min(3600.0 * dis_avail / withdrawal, mw)

    return {
        "eta": eta,
        "es_pa": es,
        "e_pa": e,
        "wout": wout,
        "win": win,
        "hout": hout,
        "hin": hin,
        "wb_c": wb_c,
        "temp_headroom": max(min(MAX_DISCHARGE_TEMP - tw_c, MAX_TEMP_RISE), 0) if cooling == "once_through" else None,
        "withdrawal": withdrawal,
        "usable_capacity_mw": uc,
        "ucr": uc / mw,
    }


# ── Qin et al. original equations ───────────────────────────────────────────


def qin_combustible_steam_loss(t_k: float) -> float:
    return max(1.73e-10 * np.exp(0.0684 * t_k) - 6.64e-7 * np.exp(0.0342 * t_k) - 0.0062, 0.0)


def qin_combined_cycle_loss(t_k: float) -> float:
    return max(
        -1.83e-15 * np.exp(0.1026 * t_k)
        + 2.26e-10 * np.exp(0.0684 * t_k)
        - 2.66e-6 * np.exp(0.0342 * t_k)
        + 0.0033,
        0.0,
    )


def qin_nuclear_steam_loss(t_k: float) -> float:
    return max(
        -2.75e-15 * np.exp(0.1026 * t_k)
        + 2.73e-10 * np.exp(0.0684 * t_k)
        - 3.45e-6 * np.exp(0.0342 * t_k)
        + 0.0111,
        0.0,
    )


def qin_non_combustible_steam_loss(t_k: float) -> float:
    return max(
        -5.95e-15 * np.exp(0.1026 * t_k)
        + 4.66e-10 * np.exp(0.0684 * t_k)
        - 4.75e-6 * np.exp(0.0342 * t_k)
        + 0.0098,
        0.0,
    )


PARASITIC_LOAD = {
    "combustible_steam": 0.0243,
    "combined_cycle": 0.0081,
    "nuclear_steam": 0.0304,
    "non_combustible_steam": 0.0056,
}
LOSS_FN = {
    "combustible_steam": qin_combustible_steam_loss,
    "combined_cycle": qin_combined_cycle_loss,
    "nuclear_steam": qin_nuclear_steam_loss,
    "non_combustible_steam": qin_non_combustible_steam_loss,
}


def qin_dry_cooling(engine_type: str, t_c: float, mw: float) -> dict:
    t_k = t_c + 273.15
    t_f = t_c * 9.0 / 5.0 + 32.0
    bp = 1.031 * np.exp(0.019 * t_f)
    if engine_type == "no_steam_cycle":
        return {"loss": 0.0, "parasitic": 0.0, "cf": 1.0, "bp": bp, "uc": mw}
    loss = LOSS_FN[engine_type](t_k)
    parasitic = PARASITIC_LOAD[engine_type]
    cf = max(1.0 - loss - parasitic, 0.0)
    return {"loss": loss, "parasitic": parasitic, "cf": cf, "bp": bp, "uc": cf * mw}


# ── Test case definitions ────────────────────────────────────────────────────

WET_CASES = [
    # (label, fuel, cooling, mw, tas_K, hurs_%, ps_Pa, tw_K, dis_m3s)
    ("coal_ot_warm", "COAL", "once_through", 500.0, 303.15, 70.0, 101325.0, 298.15, 50.0),
    ("coal_ot_hot", "COAL", "once_through", 500.0, 308.15, 80.0, 101325.0, 304.15, 30.0),
    ("gas_tower", "GAS", "recirculating", 300.0, 313.15, 50.0, 101325.0, 295.15, 20.0),
    ("nuc_tower", "UR", "recirculating", 1000.0, 293.15, 65.0, 101325.0, 288.15, 100.0),
    ("coal_ot_zero_headroom", "COAL", "once_through", 500.0, 308.15, 80.0, 101325.0, 305.15, 50.0),
]

DRY_CASES = [
    # (label, fuel, utype, tech_family, engine_type, mw, T_air_C)
    ("coal_10C", "COAL", "ST", "steam", "combustible_steam", 600.0, 10.0),
    ("coal_20C", "COAL", "ST", "steam", "combustible_steam", 600.0, 20.0),
    ("coal_30C", "COAL", "ST", "steam", "combustible_steam", 600.0, 30.0),
    ("coal_40C", "COAL", "ST", "steam", "combustible_steam", 600.0, 40.0),
    ("gas_cc_10C", "GAS", "CC", "combined_cycle", "combined_cycle", 400.0, 10.0),
    ("gas_cc_30C", "GAS", "CC", "combined_cycle", "combined_cycle", 400.0, 30.0),
    ("gas_cc_40C", "GAS", "CC", "combined_cycle", "combined_cycle", 400.0, 40.0),
    ("nuclear_30C", "UR", "ST", "steam", "nuclear_steam", 1000.0, 30.0),
    ("nuclear_40C", "UR", "ST", "steam", "nuclear_steam", 1000.0, 40.0),
    ("gt_excluded", "DIESEL", "GT", "gas_turbine", "no_steam_cycle", 50.0, 35.0),
]


# ── Kernel runners ───────────────────────────────────────────────────────────


def _build_wet_kernel_outputs(cases: list) -> xr.Dataset:
    """Build mapping + climate dataset and run the wet-cooling kernel."""
    map_rows, cool_rows = [], []
    for i, (_, fuel, cooling, mw, *_rest) in enumerate(cases):
        cat = "ot_fresh" if cooling == "once_through" else "cl_fresh"
        map_rows.append({"UNITID": i, "LOCATIONID": i, "grid_lat": 10.25 + i, "grid_lon": 20.25 + i, "mw": mw})
        cool_rows.append({
            "UNITID": i, "LOCATIONID": i, "country": "TEST", "fuel": fuel,
            "technology_family": "steam", "cooling_category": cat,
            "cooling_physics": cooling, "requires_water": True,
        })

    wet_map = build_wet_cooling_mapping(pd.DataFrame(map_rows), pd.DataFrame(cool_rows))

    n = len(cases)
    lats = [10.25 + i for i in range(n)]
    lons = [20.25 + i for i in range(n)]
    shape = (1, n, n)
    tw = np.full(shape, np.nan)
    dis = np.full(shape, np.nan)
    tas = np.full(shape, np.nan)
    hurs = np.full(shape, np.nan)
    ps = np.full(shape, np.nan)
    for i, (*_, mw, tas_v, hurs_v, ps_v, tw_v, dis_v) in enumerate(cases):
        tw[0, i, i] = tw_v
        dis[0, i, i] = dis_v
        tas[0, i, i] = tas_v
        hurs[0, i, i] = hurs_v
        ps[0, i, i] = ps_v

    climate = xr.Dataset(
        {
            "tw": (["time", "lat", "lon"], tw, {"units": "K"}),
            "dis": (["time", "lat", "lon"], dis, {"units": "m3/s"}),
            "tas": (["time", "lat", "lon"], tas, {"units": "K"}),
            "hurs": (["time", "lat", "lon"], hurs, {"units": "%"}),
            "ps": (["time", "lat", "lon"], ps, {"units": "Pa"}),
        },
        coords={"time": pd.date_range("2001-01-01", periods=1, freq="MS"), "lat": lats, "lon": lons},
    )
    return evaluate_kernel(create_wet_cooling_kernel(), wet_map, climate, duration="native")


def _build_dry_kernel_outputs(cases: list) -> xr.Dataset:
    """Build mapping + climate dataset and run the dry-cooling kernel."""
    map_rows, cool_rows = [], []
    for i, (_, fuel, utype, tech, _etype, mw, _tc) in enumerate(cases):
        map_rows.append({"UNITID": i, "LOCATIONID": i, "grid_lat": 30.25 + i, "grid_lon": 40.25 + i, "mw": mw})
        cool_rows.append({
            "UNITID": i, "LOCATIONID": i, "country": "TEST", "fuel": fuel, "utype": utype,
            "technology_family": tech, "cooling_category": "air", "cooling_physics": "dry",
        })

    dry_map = build_dry_cooling_mapping(pd.DataFrame(map_rows), pd.DataFrame(cool_rows))

    n = len(cases)
    lats = [30.25 + i for i in range(n)]
    lons = [40.25 + i for i in range(n)]
    tas = np.full((1, n, n), np.nan)
    for i, (*_, tc) in enumerate(cases):
        tas[0, i, i] = tc

    climate = xr.Dataset(
        {"tas": (["time", "lat", "lon"], tas, {"units": "C"})},
        coords={"time": pd.date_range("2001-01-01", periods=1, freq="MS"), "lat": lats, "lon": lons},
    )
    return evaluate_kernel(create_dry_cooling_kernel(), dry_map, climate, duration="native")


# ── Comparison logic ─────────────────────────────────────────────────────────


def _close(a: float, b: float, rtol: float = 1e-6) -> bool:
    if a == 0.0 and b == 0.0:
        return True
    if math.isinf(a) and math.isinf(b):
        return True
    return np.isclose(a, b, rtol=rtol)


def validate_wet_cooling() -> list[str]:
    """Compare Li et al. equations against the impacts wet-cooling kernel."""
    kernel = _build_wet_kernel_outputs(WET_CASES)
    failures = []

    for i, (label, fuel, cooling, mw, tas, hurs, ps, tw, dis) in enumerate(WET_CASES):
        ref = li_wet_cooling(fuel, cooling, mw, tas, hurs, ps, tw, dis)
        k = kernel.isel(entity=i, time=0)

        checks = [
            ("saturation_vapor_pressure_pa", ref["es_pa"]),
            ("actual_vapor_pressure_pa", ref["e_pa"]),
            ("humidity_ratio_out_g_per_kg", ref["wout"]),
            ("humidity_ratio_in_g_per_kg", ref["win"]),
            ("enthalpy_out_kj_per_kg", ref["hout"]),
            ("enthalpy_in_kj_per_kg", ref["hin"]),
            ("wet_bulb_temperature", ref["wb_c"]),
            ("withdrawal_intensity_m3_per_mwh", ref["withdrawal"]),
            ("usable_capacity_mw", ref["usable_capacity_mw"]),
            ("capacity_factor", ref["ucr"]),
        ]
        if ref["temp_headroom"] is not None:
            checks.append(("temperature_headroom_c", ref["temp_headroom"]))

        for var, expected in checks:
            actual = float(k[var])
            if not _close(expected, actual):
                failures.append(f"WET {label}.{var}: expected={expected}, got={actual}")

    return failures


def validate_dry_cooling() -> list[str]:
    """Compare Qin et al. equations against the impacts dry-cooling kernel."""
    kernel = _build_dry_kernel_outputs(DRY_CASES)
    failures = []

    for i, (label, _fuel, _utype, _tech, engine_type, mw, t_c) in enumerate(DRY_CASES):
        ref = qin_dry_cooling(engine_type, t_c, mw)
        k = kernel.isel(entity=i, time=0)

        checks = [
            ("backpressure_inhg", ref["bp"]),
            ("efficiency_loss", ref["loss"]),
            ("parasitic_load", ref["parasitic"]),
            ("capacity_factor", ref["cf"]),
            ("usable_capacity_mw", ref["uc"]),
        ]
        for var, expected in checks:
            actual = float(k[var])
            if not _close(expected, actual):
                failures.append(f"DRY {label}.{var}: expected={expected}, got={actual}")

    return failures


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    print("Wet cooling: Li et al. 2025 equations vs impacts kernel")
    print(f"  Cases: {len(WET_CASES)}")
    wet_failures = validate_wet_cooling()
    for f in wet_failures:
        print(f"  FAIL: {f}")
    print(f"  {'PASSED' if not wet_failures else f'{len(wet_failures)} failures'}")

    print()
    print("Dry cooling: Qin et al. 2023 equations vs impacts kernel")
    print(f"  Cases: {len(DRY_CASES)}")
    dry_failures = validate_dry_cooling()
    for f in dry_failures:
        print(f"  FAIL: {f}")
    print(f"  {'PASSED' if not dry_failures else f'{len(dry_failures)} failures'}")

    all_failures = wet_failures + dry_failures
    if all_failures:
        print(f"\n{len(all_failures)} total failures")
        return 1
    print("\nAll physics kernel equations match primary sources.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
