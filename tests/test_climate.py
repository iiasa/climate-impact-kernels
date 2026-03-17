from __future__ import annotations

import unittest

import pandas as pd
import xarray as xr

from thermo_electric_power.climate import attach_gridded_dataarray, attach_gridded_dataset


class ClimateJoinTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = xr.Dataset(
            data_vars={
                "tas": (
                    ("time", "lat", "lon"),
                    [
                        [[280.0, 281.0], [282.0, 283.0]],
                        [[284.0, 285.0], [286.0, 287.0]],
                    ],
                ),
                "dis": (
                    ("time", "lat", "lon"),
                    [
                        [[10.0, 20.0], [30.0, 40.0]],
                        [[50.0, 60.0], [70.0, 80.0]],
                    ],
                ),
            },
            coords={
                "time": [2000, 2001],
                "lat": [23.25, 23.75],
                "lon": [121.25, 121.75],
            },
        )

    def test_attach_dataset_samples_repeated_cells(self) -> None:
        mapping = pd.DataFrame(
            {
                "UNITID": [101, 102, 103],
                "grid_lat": [23.25, 23.25, 23.75],
                "grid_lon": [121.25, 121.25, 121.75],
                "mw": [100.0, 150.0, 75.0],
            }
        )

        attached = attach_gridded_dataset(mapping, self.dataset)

        self.assertEqual(attached.sizes["entity"], 3)
        self.assertEqual(attached.coords["UNITID"].values.tolist(), [101, 102, 103])
        self.assertEqual(attached["tas"].isel(time=0).values.tolist(), [280.0, 280.0, 283.0])
        self.assertEqual(attached["dis"].isel(time=1).values.tolist(), [50.0, 50.0, 80.0])

    def test_attach_dataset_uses_requested_entity_id_column(self) -> None:
        mapping = pd.DataFrame(
            {
                "LOCATIONID": [10, 10, 20],
                "cell_id": ["23.25,121.25", "23.75,121.75", "23.25,121.25"],
                "grid_lat": [23.25, 23.75, 23.25],
                "grid_lon": [121.25, 121.75, 121.25],
            }
        )

        attached = attach_gridded_dataset(
            mapping,
            self.dataset,
            entity_id_column="cell_id",
            metadata_columns=["LOCATIONID"],
        )

        self.assertEqual(attached.coords["entity"].values.tolist(), ["23.25,121.25", "23.75,121.75", "23.25,121.25"])
        self.assertEqual(attached.coords["LOCATIONID"].values.tolist(), [10, 10, 20])

    def test_attach_dataarray_preserves_dataarray_interface(self) -> None:
        mapping = pd.DataFrame(
            {
                "UNITID": [101, 102],
                "grid_lat": [23.25, 23.75],
                "grid_lon": [121.75, 121.25],
            }
        )

        attached = attach_gridded_dataarray(mapping, self.dataset["tas"])

        self.assertEqual(attached.name, "tas")
        self.assertEqual(attached.dims, ("time", "entity"))
        self.assertEqual(attached.isel(time=0).values.tolist(), [281.0, 282.0])

    def test_attach_dataset_rejects_missing_grid_cells(self) -> None:
        mapping = pd.DataFrame(
            {
                "UNITID": [101],
                "grid_lat": [99.25],
                "grid_lon": [121.25],
            }
        )

        with self.assertRaisesRegex(ValueError, "missing lat values"):
            attach_gridded_dataset(mapping, self.dataset)

    def test_attach_dataset_accepts_360_longitude_grid(self) -> None:
        dataset = xr.Dataset(
            data_vars={
                "tas": (
                    ("time", "lat", "lon"),
                    [[[280.0]], [[284.0]]],
                ),
            },
            coords={
                "time": [2000, 2001],
                "lat": [23.25],
                "lon": [359.75],
            },
        )
        mapping = pd.DataFrame(
            {
                "UNITID": [101],
                "grid_lat": [23.25],
                "grid_lon": [-0.25],
            }
        )

        attached = attach_gridded_dataset(mapping, dataset)

        self.assertEqual(attached["tas"].isel(time=0, entity=0).item(), 280.0)
        self.assertEqual(attached.coords["grid_lon"].item(), -0.25)


if __name__ == "__main__":
    unittest.main()
