from __future__ import annotations

import unittest

import pandas as pd
import xarray as xr

from thermo_electric_power.wet_cooling import (
    aggregate_seasonal_wet_cooling_impacts,
    build_wet_cooling_mapping,
    compute_monthly_wet_cooling_impacts,
    normalize_wet_cooling_climate_inputs,
)


class WetCoolingContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.unit_cell_mapping = pd.DataFrame(
            {
                "UNITID": [101, 202, 303],
                "LOCATIONID": [10, 20, 30],
                "country": ["Taiwan", "Korea", "USA"],
                "grid_lat": [23.25, 37.25, 41.25],
                "grid_lon": [121.25, 127.25, -87.25],
                "mw": [100.0, 200.0, 80.0],
            }
        )
        self.cooling_mapping = pd.DataFrame(
            {
                "UNITID": [101, 202, 303],
                "LOCATIONID": [10, 20, 30],
                "country": ["Taiwan", "Korea", "USA"],
                "cooling_category": ["ot_fresh", "cl_fresh", "air"],
                "cooling_physics": ["once_through", "recirculating", "dry"],
                "requires_water": [True, True, False],
            }
        )
        self.climate = xr.Dataset(
            data_vars={
                "tw": (
                    ("time", "lat", "lon"),
                    [
                        [[288.15, 289.15], [294.15, 296.15]],
                        [[292.15, 293.15], [296.15, 299.15]],
                        [[294.15, 295.15], [298.15, 301.15]],
                    ],
                ),
                "dis": (
                    ("time", "lat", "lon"),
                    [
                        [[8.0, 8.0], [15.0, 15.0]],
                        [[4.0, 4.0], [15.0, 15.0]],
                        [[2.0, 2.0], [15.0, 15.0]],
                    ],
                ),
                "tas": (
                    ("time", "lat", "lon"),
                    [
                        [[298.15, 299.15], [301.15, 302.15]],
                        [[300.15, 301.15], [304.15, 305.15]],
                        [[301.15, 302.15], [306.15, 307.15]],
                    ],
                ),
                "hurs": (
                    ("time", "lat", "lon"),
                    [
                        [[60.0, 60.0], [70.0, 70.0]],
                        [[60.0, 60.0], [80.0, 80.0]],
                        [[60.0, 60.0], [85.0, 85.0]],
                    ],
                ),
                "ps": (
                    ("time", "lat", "lon"),
                    [
                        [[101325.0, 101325.0], [100000.0, 100000.0]],
                        [[101325.0, 101325.0], [99000.0, 99000.0]],
                        [[101325.0, 101325.0], [98000.0, 98000.0]],
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
        self.assertAlmostEqual(
            normalized["water_temperature"].isel(time=0, lat=0, lon=0).item(),
            15.0,
        )
        self.assertAlmostEqual(
            normalized["surface_pressure"].isel(time=0, lat=1, lon=0).item(),
            100.0,
        )

    def test_build_wet_cooling_mapping_filters_to_water_dependent_units(self) -> None:
        mapping = build_wet_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)

        self.assertEqual(mapping["UNITID"].tolist(), [202, 101])
        self.assertEqual(mapping["cooling_physics"].tolist(), ["recirculating", "once_through"])

    def test_build_wet_cooling_mapping_accepts_artifact_schema_without_country(self) -> None:
        mapping = build_wet_cooling_mapping(
            self.unit_cell_mapping.drop(columns=["country"]),
            self.cooling_mapping,
        )

        self.assertEqual(mapping["country"].tolist(), ["Korea", "Taiwan"])

    def test_compute_monthly_wet_cooling_impacts_returns_capacity_outputs(self) -> None:
        mapping = build_wet_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)

        monthly = compute_monthly_wet_cooling_impacts(mapping, self.climate)

        self.assertEqual(monthly.sizes["entity"], 2)
        self.assertEqual(monthly.coords["UNITID"].values.tolist(), [202, 101])
        self.assertEqual(monthly["capacity_factor"].dims, ("time", "entity"))

        once_through = monthly["capacity_factor"].sel(entity=101)
        recirculating = monthly["capacity_factor"].sel(entity=202)

        self.assertAlmostEqual(once_through.isel(time=0).item(), 1.0)
        self.assertLess(once_through.isel(time=1).item(), once_through.isel(time=0).item())
        self.assertLess(once_through.isel(time=2).item(), once_through.isel(time=1).item())
        self.assertLess(recirculating.isel(time=2).item(), recirculating.isel(time=0).item())
        self.assertAlmostEqual(
            monthly["usable_capacity_mw"].sel(time="2001-01-01", entity=101).item(),
            100.0,
        )

    def test_aggregate_seasonal_wet_cooling_impacts_uses_climateresponse(self) -> None:
        mapping = build_wet_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)
        monthly = compute_monthly_wet_cooling_impacts(mapping, self.climate)

        seasonal = aggregate_seasonal_wet_cooling_impacts(monthly)

        self.assertEqual(seasonal["capacity_factor"].dims, ("season", "year", "entity"))
        self.assertEqual(seasonal.coords["season"].values.tolist(), ["DJF", "MAM", "JJA", "SON"])
        self.assertAlmostEqual(
            seasonal["capacity_factor"].sel(season="DJF", year=2001, entity=101).item(),
            0.7142857142857143,
        )
        self.assertAlmostEqual(
            seasonal["capacity_factor"].sel(season="JJA", year=2001, entity=202).item(),
            monthly["capacity_factor"].sel(time="2001-06-01", entity=202).item(),
        )


if __name__ == "__main__":
    unittest.main()
