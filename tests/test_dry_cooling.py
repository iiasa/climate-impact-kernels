from __future__ import annotations

import unittest

import pandas as pd
import xarray as xr

from thermo_electric_power.dry_cooling import (
    aggregate_seasonal_dry_cooling_impacts,
    build_dry_cooling_mapping,
    compute_monthly_dry_cooling_impacts,
    normalize_dry_cooling_climate_inputs,
)


class DryCoolingContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.unit_cell_mapping = pd.DataFrame(
            {
                "UNITID": [101, 202, 303, 404],
                "LOCATIONID": [10, 20, 30, 40],
                "country": ["Taiwan", "Korea", "USA", "France"],
                "grid_lat": [23.25, 37.25, 41.25, 48.25],
                "grid_lon": [121.25, 127.25, -87.25, 2.25],
                "mw": [100.0, 200.0, 80.0, 900.0],
            }
        )
        self.cooling_mapping = pd.DataFrame(
            {
                "UNITID": [101, 202, 303, 404, 505],
                "LOCATIONID": [10, 20, 30, 40, 50],
                "fuel": ["COAL", "LNG", "DIESEL", "NUCLEAR", "COAL"],
                "utype": ["ST", "GT/C", "GT", "ST", "ST"],
                "technology_family": [
                    "steam",
                    "combined_cycle",
                    "gas_turbine",
                    "steam",
                    "steam",
                ],
                "cooling_category": ["air", "air", "air", "air", "cl_fresh"],
                "cooling_physics": ["dry", "dry", "dry", "dry", "recirculating"],
            }
        )
        self.climate = xr.Dataset(
            data_vars={
                "tas": (
                    ("time", "lat", "lon"),
                    [
                        [
                            [10.0, 10.5, 11.0, 11.5],
                            [12.0, 12.5, 13.0, 13.5],
                            [14.0, 14.5, 15.0, 15.5],
                            [16.0, 16.5, 17.0, 17.5],
                        ],
                        [
                            [20.0, 20.5, 21.0, 21.5],
                            [22.0, 22.5, 23.0, 23.5],
                            [24.0, 24.5, 25.0, 25.5],
                            [26.0, 26.5, 27.0, 27.5],
                        ],
                        [
                            [30.0, 30.5, 31.0, 31.5],
                            [32.0, 32.5, 33.0, 33.5],
                            [34.0, 34.5, 35.0, 35.5],
                            [36.0, 36.5, 37.0, 37.5],
                        ],
                    ],
                )
            },
            coords={
                "time": pd.to_datetime(["2001-01-01", "2001-02-01", "2001-06-01"]),
                "lat": [23.25, 37.25, 41.25, 48.25],
                "lon": [-87.25, 2.25, 121.25, 127.25],
            },
        )

    def test_normalize_dry_cooling_inputs_renames_and_converts_units(self) -> None:
        normalized = normalize_dry_cooling_climate_inputs(self.climate)

        self.assertEqual(set(normalized.data_vars), {"air_temperature_k"})
        self.assertAlmostEqual(normalized["air_temperature_k"].isel(time=0, lat=0, lon=0).item(), 283.15)

    def test_build_dry_cooling_mapping_classifies_qin_families_and_exclusions(self) -> None:
        mapping = build_dry_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)

        self.assertEqual(mapping["UNITID"].tolist(), [404, 202, 101, 303])
        self.assertEqual(
            mapping["dry_cooling_engine_type"].tolist(),
            ["nuclear_steam", "combined_cycle", "combustible_steam", "no_steam_cycle"],
        )
        self.assertEqual(mapping["dry_cooling_supported"].tolist(), [True, True, True, False])
        self.assertEqual(
            mapping["dry_cooling_support_reason"].tolist(),
            [
                "supported_qin_family",
                "supported_qin_family",
                "supported_qin_family",
                "excluded_no_steam_cycle",
            ],
        )

    def test_compute_monthly_dry_cooling_impacts_returns_capacity_outputs(self) -> None:
        mapping = build_dry_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)
        monthly = compute_monthly_dry_cooling_impacts(mapping, self.climate)

        self.assertEqual(monthly.sizes["entity"], 4)
        self.assertEqual(monthly["capacity_factor"].dims, ("time", "entity"))

        combustible_steam = monthly["capacity_factor"].sel(entity=101)
        combined_cycle = monthly["capacity_factor"].sel(entity=202)
        unsupported = monthly["capacity_factor"].sel(entity=303)
        nuclear = monthly["capacity_factor"].sel(entity=404)

        self.assertGreater(combustible_steam.isel(time=0).item(), combustible_steam.isel(time=2).item())
        self.assertGreater(combined_cycle.isel(time=0).item(), combined_cycle.isel(time=2).item())
        self.assertGreater(nuclear.isel(time=0).item(), nuclear.isel(time=2).item())
        self.assertAlmostEqual(unsupported.isel(time=0).item(), 1.0)
        self.assertAlmostEqual(
            monthly["usable_capacity_mw"].sel(time="2001-01-01", entity=303).item(),
            80.0,
        )

    def test_aggregate_seasonal_dry_cooling_impacts_uses_climateresponse(self) -> None:
        mapping = build_dry_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)
        monthly = compute_monthly_dry_cooling_impacts(mapping, self.climate)

        seasonal = aggregate_seasonal_dry_cooling_impacts(monthly)

        self.assertEqual(seasonal["capacity_factor"].dims, ("season", "year", "entity"))
        self.assertEqual(seasonal.coords["season"].values.tolist(), ["DJF", "MAM", "JJA", "SON"])
        self.assertAlmostEqual(
            seasonal["capacity_factor"].sel(season="DJF", year=2001, entity=101).item(),
            monthly["capacity_factor"].sel(entity=101).isel(time=slice(0, 2)).mean().item(),
        )
        self.assertAlmostEqual(
            seasonal["capacity_factor"].sel(season="JJA", year=2001, entity=303).item(),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
