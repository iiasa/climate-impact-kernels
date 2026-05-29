from __future__ import annotations

import math
import unittest

import pandas as pd
import xarray as xr

from impacts.kernel import evaluate_mapped_kernel
from impacts.domain.thermoelectric.wet import (
    WetCoolingAssumptions,
    build_wet_cooling_mapping,
    create_wet_cooling_kernel,
    normalize_wet_cooling_climate_inputs,
)


class WetCoolingContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.unit_cell_mapping = pd.DataFrame(
            {
                "UNITID": [101, 102, 202, 303, 404, 505],
                "LOCATIONID": [10, 11, 20, 30, 40, 50],
                "country": ["Taiwan", "Taiwan", "Korea", "USA", "Japan", "Japan"],
                "grid_lat": [23.25, 23.25, 37.25, 37.25, 37.25, 23.25],
                "grid_lon": [121.25, 121.25, 127.25, 127.25, 127.25, 121.25],
                "mw": [100.0, 200.0, 150.0, 80.0, 120.0, 70.0],
            }
        )
        self.cooling_mapping = pd.DataFrame(
            {
                "UNITID": [101, 102, 202, 303, 404, 505],
                "LOCATIONID": [10, 11, 20, 30, 40, 50],
                "country": ["Taiwan", "Taiwan", "Korea", "USA", "Japan", "Japan"],
                "fuel": ["COAL", "UR", "LNG", "COAL", "WSTH", "KERO"],
                "technology_family": [
                    "steam",
                    "steam",
                    "combined_cycle",
                    "steam",
                    "steam",
                    "combined_cycle",
                ],
                "cooling_category": ["ot_fresh", "ot_saline", "cl_fresh", "air", "cl_fresh", "ot_saline"],
                "cooling_physics": [
                    "once_through",
                    "once_through",
                    "recirculating",
                    "dry",
                    "recirculating",
                    "once_through",
                ],
                "requires_water": [True, True, True, False, True, True],
            }
        )
        self.climate = xr.Dataset(
            data_vars={
                "tw": (
                    ("time", "lat", "lon"),
                    [
                        [[288.15, 293.15], [293.15, 294.15]],
                        [[292.15, 295.15], [295.15, 296.15]],
                        [[297.15, 297.15], [297.15, 298.15]],
                    ],
                ),
                "dis": (
                    ("time", "lat", "lon"),
                    [
                        [[1.0, 0.30], [0.30, 0.50]],
                        [[0.6, 0.20], [0.20, 0.40]],
                        [[0.3, 0.10], [0.10, 0.20]],
                    ],
                ),
                "tas": (
                    ("time", "lat", "lon"),
                    [
                        [[298.15, 300.15], [300.15, 301.15]],
                        [[300.15, 302.15], [302.15, 303.15]],
                        [[303.15, 304.15], [304.15, 305.15]],
                    ],
                ),
                "hurs": (
                    ("time", "lat", "lon"),
                    [
                        [[60.0, 65.0], [65.0, 70.0]],
                        [[70.0, 75.0], [75.0, 80.0]],
                        [[80.0, 82.0], [82.0, 85.0]],
                    ],
                ),
                "ps": (
                    ("time", "lat", "lon"),
                    [
                        [[101325.0, 100500.0], [100500.0, 100000.0]],
                        [[101325.0, 100000.0], [100000.0, 99500.0]],
                        [[101325.0, 99500.0], [99500.0, 99000.0]],
                    ],
                ),
            },
            coords={
                "time": pd.to_datetime(["2001-01-01", "2001-02-01", "2001-06-01"]),
                "lat": [23.25, 37.25],
                "lon": [121.25, 127.25],
            },
        )
        self.climate["tw"].attrs["units"] = "K"
        self.climate["tas"].attrs["units"] = "K"
        self.climate["ps"].attrs["units"] = "Pa"

    def test_normalize_wet_cooling_inputs_renames_and_converts_units(self) -> None:
        normalized = normalize_wet_cooling_climate_inputs(self.climate)

        self.assertEqual(
            set(normalized.data_vars),
            {
                "water_temperature",
                "discharge",
                "air_temperature",
                "relative_humidity",
                "surface_pressure",
            },
        )
        self.assertAlmostEqual(normalized["water_temperature"].isel(time=0, lat=0, lon=0).item(), 15.0)
        self.assertAlmostEqual(normalized["surface_pressure"].isel(time=0, lat=1, lon=0).item(), 100.5)

    def test_build_wet_cooling_mapping_adds_fuel_coefficients_and_cell_shares(self) -> None:
        mapping = build_wet_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)

        self.assertEqual(mapping["UNITID"].tolist(), [404, 505, 202, 101, 102])
        self.assertEqual(mapping["wet_fuel_type"].tolist(), ["coal", "oil", "gas", "coal", "nuclear"])
        self.assertAlmostEqual(mapping.loc[mapping["UNITID"] == 101, "unit_share_ratio"].item(), 100.0 / 370.0)
        self.assertAlmostEqual(mapping.loc[mapping["UNITID"] == 102, "unit_share_ratio"].item(), 200.0 / 370.0)
        self.assertAlmostEqual(mapping.loc[mapping["UNITID"] == 102, "thermal_efficiency"].item(), 0.35)
        self.assertAlmostEqual(mapping.loc[mapping["UNITID"] == 404, "thermal_efficiency"].item(), _fuel_efficiency("COAL", 120.0))
        self.assertAlmostEqual(mapping.loc[mapping["UNITID"] == 505, "thermal_efficiency"].item(), _fuel_efficiency("OIL", 70.0))
        self.assertAlmostEqual(mapping.loc[mapping["UNITID"] == 101, "heat_rejection_coefficient"].item(), 0.12)
        self.assertAlmostEqual(mapping.loc[mapping["UNITID"] == 202, "evaporation_coefficient"].item(), 0.0088)
        self.assertAlmostEqual(mapping.loc[mapping["UNITID"] == 404, "evaporation_coefficient"].item(), 0.0042)
        self.assertAlmostEqual(mapping.loc[mapping["UNITID"] == 505, "evaporation_coefficient"].item(), 0.0088)

    def test_build_wet_cooling_mapping_accepts_artifact_schema_without_country(self) -> None:
        mapping = build_wet_cooling_mapping(
            self.unit_cell_mapping.drop(columns=["country"]),
            self.cooling_mapping,
        )

        self.assertEqual(mapping["country"].tolist(), ["Japan", "Japan", "Korea", "Taiwan", "Taiwan"])

    def test_compute_monthly_wet_cooling_impacts_matches_li_equations(self) -> None:
        mapping = build_wet_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)
        monthly = evaluate_mapped_kernel(create_wet_cooling_kernel(), mapping, self.climate, duration="native")
        params = WetCoolingAssumptions()

        self.assertEqual(monthly.sizes["entity"], 5)
        self.assertEqual(monthly.coords["UNITID"].values.tolist(), [404, 505, 202, 101, 102])
        self.assertEqual(monthly["capacity_factor"].dims, ("time", "entity"))

        coal_withdrawal = _once_through_withdrawal_intensity(
            fuel="COAL",
            mw=100.0,
            water_temperature_c=15.0,
            params=params,
        )
        coal_unit_discharge = 1.0 * (1.0 - params.environmental_flow_fraction) * (100.0 / 370.0)
        coal_usable_capacity = min(3600.0 * coal_unit_discharge / coal_withdrawal, 100.0)
        self.assertAlmostEqual(
            monthly["withdrawal_intensity_m3_per_mwh"].sel(time="2001-01-01", entity=101).item(),
            coal_withdrawal,
            places=8,
        )
        self.assertAlmostEqual(
            monthly["unit_available_discharge_m3_per_s"].sel(time="2001-01-01", entity=101).item(),
            coal_unit_discharge,
            places=8,
        )
        self.assertAlmostEqual(
            monthly["usable_capacity_mw"].sel(time="2001-01-01", entity=101).item(),
            coal_usable_capacity,
            places=8,
        )
        self.assertFalse(monthly["discharge_constraint_applies"].sel(entity=102).item())
        self.assertAlmostEqual(
            monthly["usable_capacity_mw"].sel(time="2001-06-01", entity=102).item(),
            200.0,
            places=8,
        )
        self.assertAlmostEqual(
            monthly["capacity_factor"].sel(time="2001-06-01", entity=102).item(),
            1.0,
            places=8,
        )
        self.assertLess(
            monthly["capacity_factor"].sel(time="2001-06-01", entity=101).item(),
            1.0,
        )

        recirculating_withdrawal = _recirculating_withdrawal_intensity(
            fuel="LNG",
            mw=150.0,
            water_temperature_c=21.0,
            air_temperature_c=28.0,
            relative_humidity=70.0,
            surface_pressure_kpa=100.0,
            params=params,
        )
        recirculating_usable_capacity = min(
            3600.0 * (0.50 * (1.0 - params.environmental_flow_fraction)) / recirculating_withdrawal,
            150.0,
        )
        self.assertAlmostEqual(
            monthly["withdrawal_intensity_m3_per_mwh"].sel(time="2001-01-01", entity=202).item(),
            recirculating_withdrawal,
            places=8,
        )
        self.assertAlmostEqual(
            monthly["usable_capacity_mw"].sel(time="2001-01-01", entity=202).item(),
            recirculating_usable_capacity,
            places=8,
        )
        self.assertAlmostEqual(
            monthly["capacity_factor"].sel(time="2001-06-01", entity=102).item(),
            monthly["usable_capacity_mw"].sel(time="2001-06-01", entity=102).item() / 200.0,
            places=8,
        )
        self.assertIn("ot_saline once-through units are exempt", monthly.attrs["ot_saline_policy"])

    def test_aggregate_seasonal_wet_cooling_impacts_uses_climateresponse(self) -> None:
        mapping = build_wet_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)
        seasonal = evaluate_mapped_kernel(
            create_wet_cooling_kernel(), mapping, self.climate, duration="seasonal"
        ).compute()

        self.assertEqual(seasonal["capacity_factor"].dims, ("season", "year", "entity"))
        self.assertEqual(seasonal.coords["season"].values.tolist(), ["DJF", "MAM", "JJA", "SON"])
        monthly = evaluate_mapped_kernel(create_wet_cooling_kernel(), mapping, self.climate, duration="native")
        expected_djf = monthly["capacity_factor"].sel(time=["2001-01-01", "2001-02-01"], entity=101).mean().item()
        self.assertAlmostEqual(
            seasonal["capacity_factor"].sel(season="DJF", year=2001, entity=101).item(),
            expected_djf,
        )
        self.assertAlmostEqual(
            seasonal["capacity_factor"].sel(season="JJA", year=2001, entity=202).item(),
            monthly["capacity_factor"].sel(time="2001-06-01", entity=202).item(),
        )


class WetCoolingDaskTest(WetCoolingContractTest):
    """Re-run all wet-cooling tests with dask-backed climate arrays.

    Kernels must produce correct results from chunked inputs.
    Tests that extract scalars override the parent to .compute() first.
    """

    def setUp(self) -> None:
        super().setUp()
        self.climate = self.climate.chunk({"time": 1})

    def test_normalize_wet_cooling_inputs_renames_and_converts_units(self) -> None:
        normalized = normalize_wet_cooling_climate_inputs(self.climate).compute()
        self.assertEqual(
            set(normalized.data_vars),
            {"water_temperature", "discharge", "air_temperature", "relative_humidity", "surface_pressure"},
        )
        self.assertAlmostEqual(normalized["water_temperature"].isel(time=0, lat=0, lon=0).item(), 15.0)
        self.assertAlmostEqual(normalized["surface_pressure"].isel(time=0, lat=1, lon=0).item(), 100.5)

    def test_compute_monthly_wet_cooling_impacts_matches_li_equations(self) -> None:
        mapping = build_wet_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)
        monthly = evaluate_mapped_kernel(create_wet_cooling_kernel(), mapping, self.climate, duration="native").compute()
        params = WetCoolingAssumptions()

        self.assertEqual(monthly.sizes["entity"], 5)
        self.assertEqual(monthly["capacity_factor"].dims, ("time", "entity"))

        coal_withdrawal = _once_through_withdrawal_intensity(fuel="COAL", mw=100.0, water_temperature_c=15.0, params=params)
        coal_unit_discharge = 1.0 * (1.0 - params.environmental_flow_fraction) * (100.0 / 370.0)
        coal_usable_capacity = min(3600.0 * coal_unit_discharge / coal_withdrawal, 100.0)
        self.assertAlmostEqual(
            monthly["withdrawal_intensity_m3_per_mwh"].sel(time="2001-01-01", entity=101).item(),
            coal_withdrawal, places=8,
        )
        self.assertAlmostEqual(
            monthly["usable_capacity_mw"].sel(time="2001-01-01", entity=101).item(),
            coal_usable_capacity, places=8,
        )
        self.assertFalse(monthly["discharge_constraint_applies"].sel(entity=102).item())

    def test_aggregate_seasonal_wet_cooling_impacts_uses_climateresponse(self) -> None:
        mapping = build_wet_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)
        seasonal = evaluate_mapped_kernel(create_wet_cooling_kernel(), mapping, self.climate, duration="seasonal").compute()
        self.assertEqual(seasonal["capacity_factor"].dims, ("season", "year", "entity"))


def _classify_fuel(fuel: str) -> str:
    normalized = fuel.upper()
    if any(marker in normalized for marker in ("NUC", "URANIUM", "UR")):
        return "nuclear"
    if normalized == "WSTH":
        return "coal"
    if any(marker in normalized for marker in ("BIO", "WOOD", "BAGASSE")):
        return "biomass"
    if any(marker in normalized for marker in ("COAL", "COKE", "LIGNITE", "PEAT", "PETCOKE", "SHALE")):
        return "coal"
    if any(marker in normalized for marker in ("DIESEL", "FO", "HFO", "KERO", "LFO", "NAPHTHA", "OIL")):
        return "oil"
    if any(marker in normalized for marker in ("GAS", "LNG", "NG")):
        return "gas"
    raise AssertionError(f"unexpected test fuel: {fuel}")


def _fuel_efficiency(fuel: str, mw: float) -> float:
    fuel_type = _classify_fuel(fuel)
    if fuel_type == "nuclear":
        return 0.35

    parameters = {
        "coal": (24.14675, 44.89252, 198.43242, 1.63882),
        "gas": (38.59261, 64.74196, 269.6733, 1.28565),
        "oil": (35.96246, 50.84005, 224.66888, 1.50589),
        "biomass": (16.04344, 36.14518, 24.37557, 1.87955),
    }
    a1, a2, x0, power = parameters[fuel_type]
    return (a2 + (a1 - a2) / (1.0 + (mw / x0) ** power)) * 0.01


def _heat_rejection_coefficient(fuel: str) -> float:
    coefficients = {
        "coal": 0.12,
        "gas": 0.12,
        "oil": 0.12,
        "biomass": 0.12,
        "nuclear": 0.0,
    }
    return coefficients[_classify_fuel(fuel)]


def _once_through_withdrawal_intensity(
    *,
    fuel: str,
    mw: float,
    water_temperature_c: float,
    params: WetCoolingAssumptions,
) -> float:
    efficiency = _fuel_efficiency(fuel, mw)
    kos = _heat_rejection_coefficient(fuel)
    temperature_headroom = max(min(params.max_discharge_temp_c - water_temperature_c, params.max_temp_rise_c), 0.0)
    if temperature_headroom == 0.0:
        return math.inf
    return (
        3.6e6
        * (1.0 - efficiency - kos)
        / efficiency
        / (params.water_density_kg_per_m3 * params.water_specific_heat_j_per_g_k * temperature_headroom)
    )


def _recirculating_withdrawal_intensity(
    *,
    fuel: str,
    mw: float,
    water_temperature_c: float,
    air_temperature_c: float,
    relative_humidity: float,
    surface_pressure_kpa: float,
    params: WetCoolingAssumptions,
) -> float:
    efficiency = _fuel_efficiency(fuel, mw)
    kos = _heat_rejection_coefficient(fuel)
    saturation_vapor_pressure = 6.11 * (10 ** (7.5 * air_temperature_c / (237.7 + air_temperature_c))) * 100.0
    actual_vapor_pressure = relative_humidity / 100.0 * saturation_vapor_pressure
    surface_pressure_pa = surface_pressure_kpa * 1000.0
    humidity_ratio_out = 621.9907 * saturation_vapor_pressure / (surface_pressure_pa - saturation_vapor_pressure)
    humidity_ratio_in = 621.9907 * actual_vapor_pressure / (surface_pressure_pa - actual_vapor_pressure)
    air_temperature_k = air_temperature_c + 273.15
    enthalpy_out = air_temperature_k * (1.01 + 0.00189 * humidity_ratio_out) + 2.5 * humidity_ratio_out
    enthalpy_in = air_temperature_k * (1.01 + 0.00189 * humidity_ratio_in) + 2.5 * humidity_ratio_in
    wet_bulb = (
        air_temperature_c * math.atan(0.151977 * math.sqrt(relative_humidity + 8.313659))
        + math.atan(air_temperature_c + relative_humidity)
        - math.atan(relative_humidity - 1.676331)
        + 0.00391838 * (relative_humidity**1.5) * math.atan(0.023101 * relative_humidity)
        - 4.686035
    )
    humidity_delta = humidity_ratio_out - humidity_ratio_in
    return (
        3.6e6
        * (1.0 - efficiency - kos)
        / efficiency
        * humidity_delta
        / 1000.0
        / (
            params.water_density_kg_per_m3
            * (
                (enthalpy_out - enthalpy_in) * ((params.concentration_cycles - 1.0) / params.concentration_cycles)
                + ((wet_bulb + params.tower_approach_c) / params.concentration_cycles - water_temperature_c)
                * params.water_specific_heat_j_per_g_k
                * humidity_delta
                / 1000.0
            )
        )
    )


if __name__ == "__main__":
    unittest.main()
