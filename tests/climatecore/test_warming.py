import unittest

import numpy as np
import pandas as pd
import xarray as xr

from impacts.climatecore import (
    assign_years_to_levels,
    attach_warming_levels,
    bin_by_warming_level,
    build_warming_level_assignments,
    build_warming_level_quantile_surface,
    compute_running_mean,
    interpolate_along_warming_level,
    match_warming_levels,
)


class GMTTest(unittest.TestCase):
    def test_centered_running_mean(self) -> None:
        running = compute_running_mean(
            data=xr.DataArray(
                [0.0, 1.0, 2.0, 3.0, 4.0],
                dims=["year"],
                coords={"year": [2000, 2001, 2002, 2003, 2004]},
            ),
            window=3,
        )

        self.assertTrue(np.isnan(running.sel(year=2000)))
        self.assertEqual(running.sel(year=2002).item(), 2.0)


class WarmingLevelTest(unittest.TestCase):
    def test_match_warming_levels_uses_buckets_and_diagnostics(self) -> None:
        matches = match_warming_levels(
            gmt=xr.DataArray(
                [0.7, 1.1, 1.6],
                dims=["year"],
                coords={"year": [2000, 2001, 2002]},
            ),
            target_levels=[1.0, 1.5],
            bucket_width=0.5,
        )

        self.assertEqual(matches["year"].values.tolist(), [2001, 2002])
        self.assertEqual(matches["warming_level"].values.tolist(), [1.0, 1.5])
        self.assertTrue("accumulated_warming" in matches.data_vars)

    def test_assign_years_to_levels(self) -> None:
        assignments = assign_years_to_levels(
            gmt=xr.DataArray(
                [0.9, 1.2, 1.8],
                dims=["year"],
                coords={"year": [2000, 2001, 2002]},
            ),
            target_levels=[1.0, 2.0],
        )

        self.assertEqual(assignments["warming_level"].values.tolist(), [1.0, 1.0, 2.0])

    def test_bin_by_warming_level_groups_values(self) -> None:
        bins = bin_by_warming_level(
            values=xr.DataArray(
                [10.0, 11.0, 12.0],
                dims=["year"],
                coords={"year": [2000, 2001, 2002]},
            ),
            gmt=xr.DataArray(
                [0.9, 1.2, 1.8],
                dims=["year"],
                coords={"year": [2000, 2001, 2002]},
            ),
            target_levels=[1.0, 2.0],
        )

        self.assertTrue(np.allclose(bins.values, [10.5, 12.0]))


class ResponseTest(unittest.TestCase):
    def test_linear_interpolation(self) -> None:
        interpolated = interpolate_along_warming_level(
            values=xr.DataArray(
                [10.0, 30.0],
                dims=["warming_level"],
                coords={"warming_level": [1.0, 2.0]},
            ),
            target_levels=1.5,
        )

        self.assertEqual(interpolated.item(), 20.0)


class WarmingLevelLookupTest(unittest.TestCase):
    def test_assignments_attach_and_quantile_surface_round_trip(self) -> None:
        gmt_by_path = {
            ("GFDL-ESM4", "historical"): xr.DataArray(
                [0.1, 0.6],
                dims=["year"],
                coords={"year": [1850, 1851]},
            ),
            ("GFDL-ESM4", "ssp126"): xr.DataArray(
                [1.0, 1.0],
                dims=["year"],
                coords={"year": [2015, 2016]},
            ),
        }
        values = xr.DataArray(
            [[
                [
                    [[0.4], [0.6], [np.nan], [np.nan]],
                    [[np.nan], [np.nan], [0.7], [0.9]],
                ]
            ]],
            dims=["impact_type", "gcm", "scenario", "time", "region"],
            coords={
                "impact_type": ["wet"],
                "gcm": ["GFDL-ESM4"],
                "scenario": ["historical", "ssp126"],
                "time": pd.to_datetime(["1850-01-01", "1851-01-01", "2015-01-01", "2016-01-01"]),
                "region": ["A"],
            },
            name="value",
        )

        assignments = build_warming_level_assignments(
            gmt_by_path,
            path_columns=("gcm", "scenario"),
            bin_width=0.5,
        )
        enriched = attach_warming_levels(values, assignments)
        quantile_surface = build_warming_level_quantile_surface(
            enriched,
            sample_dims=("gcm", "scenario", "time"),
            quantiles=(0.5,),
        )

        self.assertEqual(assignments["target_warming_level"].values.tolist(), [0.0, 0.5, 1.0])
        historical_levels = enriched["warming_level"].sel(
            gcm="GFDL-ESM4",
            scenario="historical",
        ).values
        self.assertEqual(historical_levels[:2].tolist(), [0.0, 0.5])
        self.assertTrue(np.isnan(historical_levels[2:]).all())
        self.assertEqual(
            quantile_surface["sample_size"]
            .sel(impact_type="wet", region="A", warming_level=1.0)
            .item(),
            2,
        )
        self.assertAlmostEqual(
            quantile_surface["value"]
            .sel(impact_type="wet", region="A", warming_level=1.0, quantile=0.5)
            .item(),
            0.8,
        )


if __name__ == "__main__":
    unittest.main()
