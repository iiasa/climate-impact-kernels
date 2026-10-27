import unittest

import numpy as np
import pandas as pd
import xarray as xr

from impacts.climatecore import (
    attach_gridded_dataarray,
    attach_gridded_dataset,
    grid_cell_area,
    nearest_valid_cells,
    refine_to_grid,
)


class SpatialTest(unittest.TestCase):
    def test_attach_gridded_dataset_preserves_requested_metadata(self) -> None:
        attached = attach_gridded_dataset(
            mapped_sites=pd.DataFrame(
                {
                    "UNITID": [101, 202],
                    "country": ["A", "B"],
                    "grid_lat": [0.5, 1.5],
                    "grid_lon": [-0.5, 0.5],
                }
            ),
            dataset=xr.Dataset(
                data_vars={
                    "tas": (
                        ("time", "lat", "lon"),
                        [[[1.0, 2.0], [3.0, 4.0]]],
                    )
                },
                coords={"time": [0], "lat": [0.5, 1.5], "lon": [-0.5, 0.5]},
            ),
            site_id_column="UNITID",
            metadata_columns=["UNITID", "country"],
        )

        self.assertEqual(attached["tas"].dims, ("time", "entity"))
        self.assertEqual(attached.coords["UNITID"].values.tolist(), [101, 202])
        self.assertEqual(attached.coords["country"].values.tolist(), ["A", "B"])

    def test_attach_gridded_dataarray_normalizes_longitude_to_360_grid(self) -> None:
        attached = attach_gridded_dataarray(
            mapped_sites=pd.DataFrame(
                {
                    "UNITID": [101],
                    "grid_lat": [0.5],
                    "grid_lon": [-0.5],
                }
            ),
            data=xr.DataArray(
                [[[9.0]]],
                dims=["time", "lat", "lon"],
                coords={"time": [0], "lat": [0.5], "lon": [359.5]},
                name="tas",
            ),
            site_id_column="UNITID",
        )

        self.assertEqual(attached.sel(entity=101).item(), 9.0)

    def test_attach_gridded_dataset_requires_a_named_entity_id_column(self) -> None:
        mapped = pd.DataFrame({"UNITID": [101], "grid_lat": [0.5], "grid_lon": [0.5]})
        grid = xr.Dataset(
            data_vars={"tas": (("lat", "lon"), [[1.0]])}, coords={"lat": [0.5], "lon": [0.5]}
        )

        with self.assertRaisesRegex(TypeError, "site_id_column"):
            attach_gridded_dataset(mapped, grid)  # type: ignore[call-arg]
        with self.assertRaisesRegex(
            ValueError,
            "'unit_id' is not a column in mapped_sites; pass site_id_column naming the column",
        ):
            attach_gridded_dataset(mapped, grid, site_id_column="unit_id")

    def test_attach_gridded_dataset_rejects_duplicate_entity_ids(self) -> None:
        mapped = pd.DataFrame(
            {"UNITID": [101, 101, 202], "grid_lat": [0.5] * 3, "grid_lon": [0.5] * 3}
        )
        grid = xr.Dataset(
            data_vars={"tas": (("lat", "lon"), [[1.0]])}, coords={"lat": [0.5], "lon": [0.5]}
        )

        with self.assertRaisesRegex(
            ValueError, r"site_id_column 'UNITID' must hold unique values; duplicated: \[101\]"
        ):
            attach_gridded_dataset(mapped, grid, site_id_column="UNITID")

    def test_grid_cell_areas_of_a_global_grid_sum_to_the_surface_of_the_sphere(self) -> None:
        # 4 pi R^2 with R = 6371 km is 5.1006e14 m2.
        lat = xr.DataArray(np.arange(-89.5, 90.0, 1.0), dims=["lat"])
        lon = xr.DataArray(np.arange(0.5, 360.0, 1.0), dims=["lon"])

        weights = grid_cell_area(lat, lon)

        self.assertEqual(weights.dims, ("lat", "lon"))
        self.assertAlmostEqual(weights.sum().item() / 5.1006e14, 1.0, places=4)

    def test_a_one_degree_cell_on_the_equator_is_111_km_square(self) -> None:
        # One degree of arc on a sphere of radius 6371 km is 111.19 km, so a cell
        # one degree on each side just north of the equator has an area of about 12,364 km2.
        weights = grid_cell_area(
            xr.DataArray([-0.5, 0.5], dims=["lat"]), xr.DataArray([0.5, 1.5], dims=["lon"])
        )

        self.assertAlmostEqual(weights.sel(lat=0.5, lon=0.5).item() / 1.2364e10, 1.0, places=3)

    def test_grid_cell_area_needs_a_resolution_for_a_single_cell(self) -> None:
        lat = xr.DataArray([0.0, 1.0], dims=["lat"], coords={"lat": [0.0, 1.0]})
        single = xr.DataArray([10.0], dims=["lon"], coords={"lon": [10.0]})

        with self.assertRaisesRegex(
            ValueError,
            "'lon' has a single cell, whose width cannot be inferred; "
            "pass resolution_deg with the cell width in degrees",
        ):
            grid_cell_area(lat, single)
        with self.assertRaisesRegex(ValueError, "'lat' has a single cell"):
            grid_cell_area(single.rename(lon="lat"), single)
        with self.assertRaisesRegex(ValueError, "resolution_deg must be positive"):
            grid_cell_area(lat, single, resolution_deg=0.0)

        explicit = grid_cell_area(lat, single, resolution_deg=2.0)
        two_wide = grid_cell_area(
            lat, xr.DataArray([9.0, 11.0], dims=["lon"], coords={"lon": [9.0, 11.0]})
        )
        np.testing.assert_allclose(explicit.values[:, 0], two_wide.values[:, 0])


class GridRefinementTest(unittest.TestCase):
    def test_coarse_cells_repeat_over_the_fine_cells_inside_them(self) -> None:
        coarse = xr.DataArray(
            [[[1.0, 2.0], [3.0, 4.0]]],
            dims=("time", "lat", "lon"),
            coords={"time": [0], "lat": [0.5, -0.5], "lon": [-0.5, 0.5]},
        )
        fine_lat = xr.DataArray([0.75, 0.25, -0.25, -0.75], dims="lat")
        fine_lon = xr.DataArray([-0.75, -0.25, 0.25, 0.75], dims="lon")
        refined = refine_to_grid(coarse, lat=fine_lat, lon=fine_lon)

        self.assertEqual(refined.sizes, {"time": 1, "lat": 4, "lon": 4})
        self.assertEqual(refined["lat"].values.tolist(), [0.75, 0.25, -0.25, -0.75])
        self.assertEqual(
            refined.isel(time=0).values.tolist(),
            [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0], [3.0, 3.0, 4.0, 4.0]],
        )

    def test_refuses_a_target_centre_on_a_source_boundary(self) -> None:
        coarse = xr.DataArray(
            [[1.0, 2.0]], dims=("lat", "lon"), coords={"lat": [0.5], "lon": [-0.5, 0.5]}
        )
        with self.assertRaisesRegex(ValueError, "lat"):
            refine_to_grid(coarse, lat=xr.DataArray([0.0], dims="lat"), lon=xr.DataArray([0.25], dims="lon"))

    def test_refuses_a_target_centre_outside_the_source_grid(self) -> None:
        coarse = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0]], dims=("lat", "lon"), coords={"lat": [0.5, -0.5], "lon": [-0.5, 0.5]}
        )
        with self.assertRaisesRegex(ValueError, "lon"):
            refine_to_grid(coarse, lat=xr.DataArray([0.25], dims="lat"), lon=xr.DataArray([1.25], dims="lon"))


class NearestValidCellTest(unittest.TestCase):
    def setUp(self) -> None:
        # Valid cells at (0, 1) and (0, 3) degrees; one degree of longitude on the
        # equator is 111.19 km.
        self.valid = xr.DataArray(
            [[False, True, False, True]],
            dims=("lat", "lon"),
            coords={"lat": [0.0], "lon": [0.0, 1.0, 2.0, 3.0]},
        )

    def test_finds_the_nearest_valid_cell_and_its_distance(self) -> None:
        found = nearest_valid_cells(
            np.array([0.0, 0.0]), np.array([0.0, 2.6]), self.valid, max_distance_km=500.0
        )
        self.assertEqual(found["cell_lon"].tolist(), [1.0, 3.0])
        self.assertAlmostEqual(found["distance_km"].iloc[0], 111.19, delta=0.05)
        self.assertAlmostEqual(found["distance_km"].iloc[1], 0.4 * 111.19, delta=0.05)

    def test_leaves_cells_empty_beyond_the_reach_but_reports_distance(self) -> None:
        found = nearest_valid_cells(np.array([0.0]), np.array([-1.0]), self.valid, max_distance_km=100.0)
        self.assertTrue(np.isnan(found["cell_lat"].iloc[0]))
        self.assertTrue(np.isnan(found["cell_lon"].iloc[0]))
        self.assertAlmostEqual(found["distance_km"].iloc[0], 2 * 111.19, delta=0.1)

    def test_crosses_the_dateline(self) -> None:
        valid = xr.DataArray(
            [[True, False]], dims=("lat", "lon"), coords={"lat": [0.0], "lon": [-179.5, 0.0]}
        )
        found = nearest_valid_cells(np.array([0.0]), np.array([179.5]), valid, max_distance_km=200.0)
        self.assertEqual(found["cell_lon"].iloc[0], -179.5)
        self.assertAlmostEqual(found["distance_km"].iloc[0], 111.19, delta=0.05)


if __name__ == "__main__":
    unittest.main()
