from __future__ import annotations

import tempfile
import unittest

import xarray as xr

from impacts.thermoelectric.aggregation import (
    aggregate_thermoelectric_file,
    aggregate_thermoelectric_regions,
    write_aggregated_thermoelectric_file,
)


class ThermoelectricAggregationTest(unittest.TestCase):
    def setUp(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
