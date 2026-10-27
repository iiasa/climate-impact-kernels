import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import xarray as xr

from impacts.climatecore.regions import (
    aggregate_site_dataarray,
    aggregate_site_dataset,
    common_definitions_root,
    message_r12_country_to_region,
    message_r12_region_labels,
    resolve_region_labels,
)
from tests.regression import fixtures


class RegionRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        # The small definitions folder the end-to-end tests use: the two R12 files the
        # registry reads.
        root = Path(self.enterContext(tempfile.TemporaryDirectory())) / "definitions"
        self.enterContext(fixtures.common_definitions(fixtures.write_common_definitions(root)))

    def test_message_r12_resolution_uses_upstream_mapping_and_aliases(self) -> None:
        # USA and VIETNAM are aliases of country names in the mapping, ENGLAND & WALES
        # an alias of United Kingdom, and CHANNEL ISLANDS an alias of a region.
        resolved = resolve_region_labels(
            ["USA", "VIETNAM", "ENGLAND & WALES", "CHANNEL ISLANDS"],
            region_set="message_r12",
        )

        self.assertEqual(resolved, ("R12_NAM", "R12_RCPA", "R12_WEU", "R12_WEU"))
        self.assertEqual(message_r12_region_labels()["R12_NAM"], "North America")
        self.assertEqual(message_r12_country_to_region()["United States"], "R12_NAM")

    def test_a_country_outside_the_mapping_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            resolve_region_labels(["Atlantis"], region_set="message_r12")

    def test_the_registry_needs_a_definitions_checkout(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("COMMON_DEFINITIONS_ROOT", None)
            with self.assertRaisesRegex(FileNotFoundError, "COMMON_DEFINITIONS_ROOT is not set"):
                common_definitions_root()


class EntityAggregationTest(unittest.TestCase):
    def test_aggregate_entity_dataarray_weighted_mean(self) -> None:
        values = xr.DataArray(
            [[0.5, 1.0, 0.25]],
            dims=["time", "entity"],
            coords={"time": [0], "entity": [101, 202, 303]},
        )

        reduced = aggregate_site_dataarray(
            values,
            region_labels=["A", "A", "B"],
            reduction="weighted_mean",
            weights=xr.DataArray(
                [100.0, 300.0, 200.0],
                dims=["entity"],
                coords={"entity": [101, 202, 303]},
            ),
        )

        self.assertEqual(reduced.dims, ("time", "region"))
        self.assertAlmostEqual(reduced.sel(region="A").item(), 0.875)
        self.assertAlmostEqual(reduced.sel(region="B").item(), 0.25)

    def test_weighted_mean_ignores_entities_without_a_value(self) -> None:
        values = xr.DataArray(
            [[float("nan"), 1.0, 0.5, float("nan")]],
            dims=["time", "entity"],
            coords={"time": [0], "entity": [101, 202, 303, 404]},
        )

        reduced = aggregate_site_dataarray(
            values,
            region_labels=["A", "A", "B", "C"],
            reduction="weighted_mean",
            weights=xr.DataArray(
                [100.0, 300.0, 200.0, 50.0],
                dims=["entity"],
                coords={"entity": [101, 202, 303, 404]},
            ),
        )

        # The missing site carries no weight: A is the one valid value, not 300/400.
        self.assertAlmostEqual(reduced.sel(region="A").item(), 1.0)
        self.assertAlmostEqual(reduced.sel(region="B").item(), 0.5)
        # A region with no valid value has no mean.
        self.assertTrue(math.isnan(reduced.sel(region="C").item()))

    def test_sum_of_an_all_missing_region_is_missing_not_zero(self) -> None:
        values = xr.DataArray(
            [[float("nan"), 2.0, float("nan"), 3.0]],
            dims=["time", "entity"],
            coords={"time": [0], "entity": [101, 202, 303, 404]},
        )

        reduced = aggregate_site_dataarray(
            values, region_labels=["A", "A", "B", "C"], reduction="sum"
        )

        self.assertEqual(reduced.sel(region="A").item(), 2.0)
        self.assertTrue(math.isnan(reduced.sel(region="B").item()))
        self.assertEqual(reduced.sel(region="C").item(), 3.0)

    def test_aggregate_entity_dataset_rejects_variable_without_reduction(self) -> None:
        dataset = xr.Dataset(
            data_vars={
                "capacity_factor": (("time", "entity"), [[0.5, 1.0]]),
                "forgotten": (("time", "entity"), [[1.0, 2.0]]),
            },
            coords={"time": [0], "entity": [101, 202]},
        )

        with self.assertRaisesRegex(ValueError, "forgotten"):
            aggregate_site_dataset(
                dataset,
                region_labels=["A", "A"],
                variable_reductions={"capacity_factor": "sum"},
            )

    def test_aggregate_entity_dataset_mixes_sum_and_weighted_mean(self) -> None:
        dataset = xr.Dataset(
            data_vars={
                "capacity_factor": (("time", "entity"), [[0.5, 1.0, 0.25]]),
                "usable_capacity_mw": (("time", "entity"), [[50.0, 300.0, 50.0]]),
            },
            coords={
                "time": [0],
                "entity": [101, 202, 303],
                "mw": ("entity", [100.0, 300.0, 200.0]),
            },
        )

        reduced = aggregate_site_dataset(
            dataset,
            region_labels=["A", "A", "B"],
            variable_reductions={
                "capacity_factor": "weighted_mean",
                "usable_capacity_mw": "sum",
            },
            weights=dataset.coords["mw"],
        )

        self.assertAlmostEqual(reduced["capacity_factor"].sel(region="A").item(), 0.875)
        self.assertAlmostEqual(reduced["usable_capacity_mw"].sel(region="A").item(), 350.0)


if __name__ == "__main__":
    unittest.main()
