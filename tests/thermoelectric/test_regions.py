from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import xarray as xr

from impacts.thermoelectric.regions import (
    aggregate_thermoelectric_file,
    aggregate_thermoelectric_regions,
    write_aggregated_thermoelectric_file,
)
from tests.regression import fixtures


class ThermoelectricAggregationTest(unittest.TestCase):
    def setUp(self) -> None:
        # The R12 aggregations read the regional definitions; use the small definitions folder the end-to-end tests use.
        root = Path(self.enterContext(tempfile.TemporaryDirectory())) / "definitions"
        self.enterContext(fixtures.common_definitions(fixtures.write_common_definitions(root)))
        self.dataset = xr.Dataset(
            data_vars={
                "capacity_factor": (("time", "entity"), [[0.5, 1.0, 0.25, 0.75]]),
                "usable_capacity_mw": (("time", "entity"), [[50.0, 300.0, 50.0, 300.0]]),
                "withdrawal_intensity_m3_per_mwh": (("time", "entity"), [[2.0, 4.0, 6.0, 8.0]]),
            },
            coords={
                "time": [0],
                "entity": [101, 202, 303, 404],
                "country": ("entity", ["USA", "USA", "VIETNAM", "JAPAN"]),
                "mw": ("entity", [100.0, 300.0, 200.0, 400.0]),
            },
        )

    def test_aggregate_to_countries(self) -> None:
        reduced = aggregate_thermoelectric_regions(self.dataset, region_set="country")

        self.assertEqual(reduced["capacity_factor"].dims, ("time", "region"))
        self.assertAlmostEqual(reduced["capacity_factor"].sel(region="USA").item(), 0.875)
        self.assertAlmostEqual(reduced["usable_capacity_mw"].sel(region="USA").item(), 350.0)
        self.assertEqual(reduced.coords["total_mw"].sel(region="USA").item(), 400.0)

    def test_valid_mw_excludes_units_without_usable_capacity(self) -> None:
        dataset = self.dataset.copy(deep=True)
        dataset["usable_capacity_mw"] = (
            ("time", "entity"),
            [[50.0, float("nan"), float("nan"), float("nan")]],
        )

        reduced = aggregate_thermoelectric_regions(dataset, region_set="country")

        self.assertEqual(reduced["valid_mw"].dims, ("time", "region"))
        self.assertEqual(reduced["valid_mw"].sel(region="USA").item(), 100.0)
        self.assertEqual(reduced.coords["total_mw"].sel(region="USA").item(), 400.0)
        # Regions with no known unit have no valid MW, and an unknown (not zero) capacity.
        self.assertEqual(reduced["valid_mw"].sel(region="VIETNAM").item(), 0.0)
        self.assertTrue(math.isnan(reduced["usable_capacity_mw"].sel(region="VIETNAM").item()))

    def test_regional_capacity_factor_ignores_units_with_unknown_value(self) -> None:
        dataset = self.dataset.copy(deep=True)
        dataset["capacity_factor"] = (
            ("time", "entity"),
            [[float("nan"), 1.0, 0.25, 0.75]],
        )

        reduced = aggregate_thermoelectric_regions(dataset, region_set="country")

        # USA: the unit without a value (100 MW) carries no weight, so the mean is 1.0.
        self.assertAlmostEqual(reduced["capacity_factor"].sel(region="USA").item(), 1.0)

    def test_cell_discharge_is_summed_over_distinct_cells(self) -> None:
        # Units 101 and 202 share a cell, so they carry the same cell-level discharge.
        dataset = self.dataset.copy(deep=True)
        dataset["available_discharge_m3_per_s"] = (("time", "entity"), [[5.0, 5.0, 7.0, 9.0]])
        dataset["unit_available_discharge_m3_per_s"] = (("time", "entity"), [[1.0, 4.0, 7.0, 9.0]])
        dataset = dataset.assign_coords(
            grid_lat=("entity", [10.25, 10.25, 20.25, 30.25]),
            grid_lon=("entity", [100.25, 100.25, 110.25, 120.25]),
        )

        reduced = aggregate_thermoelectric_regions(dataset, region_set="country")

        # USA holds the shared cell (5, counted once) only: not 10.
        self.assertEqual(reduced["available_discharge_m3_per_s"].sel(region="USA").item(), 5.0)
        self.assertEqual(reduced["available_discharge_m3_per_s"].sel(region="VIETNAM").item(), 7.0)
        # The unit-level share still sums over units to the cell total.
        self.assertEqual(reduced["unit_available_discharge_m3_per_s"].sel(region="USA").item(), 5.0)

    def test_cell_discharge_needs_cell_coordinates(self) -> None:
        dataset = self.dataset.copy(deep=True)
        dataset["available_discharge_m3_per_s"] = (("time", "entity"), [[5.0, 5.0, 7.0, 9.0]])

        with self.assertRaisesRegex(ValueError, "grid_lat"):
            aggregate_thermoelectric_regions(dataset, region_set="country")

    def test_entity_variable_without_a_reduction_is_refused(self) -> None:
        dataset = self.dataset.copy(deep=True)
        dataset["cooling_water_cost"] = (("time", "entity"), [[1.0, 2.0, 3.0, 4.0]])
        dataset["another_unknown"] = (("time", "entity"), [[1.0, 2.0, 3.0, 4.0]])
        dataset["fleet_note"] = ("time", [1.0])

        with self.assertRaisesRegex(
            ValueError,
            r"no regional reduction is defined for entity variables "
            r"\['another_unknown', 'cooling_water_cost'\]; add each to the sum or "
            "weighted-mean tables",
        ):
            aggregate_thermoelectric_regions(dataset, region_set="country")

    def test_aggregate_to_message_r12(self) -> None:
        reduced = aggregate_thermoelectric_regions(self.dataset, region_set="message_r12")

        self.assertEqual(
            set(reduced.coords["region"].values.tolist()),
            {"R12_NAM", "R12_RCPA", "R12_PAO"},
        )
        self.assertEqual(
            reduced.coords["region_name"].sel(region="R12_NAM").item(),
            "North America",
        )
        self.assertAlmostEqual(reduced["usable_capacity_mw"].sel(region="R12_NAM").item(), 350.0)

    def test_file_helpers_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = f"{tmpdir}/source.nc"
            target = f"{tmpdir}/country.nc"
            self.dataset.to_netcdf(source)

            reduced = aggregate_thermoelectric_file(source, region_set="country")
            written = write_aggregated_thermoelectric_file(
                source,
                target,
                region_set="message_r12",
            )

            self.assertEqual(reduced.coords["total_mw"].sel(region="USA").item(), 400.0)
            self.assertEqual(written.name, "country.nc")
            reopened = xr.open_dataset(written)
            self.assertIn("R12_NAM", reopened.coords["region"].values.tolist())
            reopened.close()

