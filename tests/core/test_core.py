import unittest

import numpy as np
import xarray as xr

from climateresponse import (
    aggregate_annual,
    aggregate_seasonal,
    assign_years_to_levels,
    bin_by_warming_level,
    compute_anomaly,
    compute_running_mean,
    deterministic_resampling,
    interpolate_along_warming_level,
    match_warming_levels,
    weighted_average,
    weighted_quantiles,
)


class BaselineTest(unittest.TestCase):
    def test_compute_anomaly_uses_baseline_mean(self) -> None:
        anomalies = compute_anomaly(
            data=xr.DataArray([1.0, 3.0, 5.0], dims=["year"], coords={"year": [2000, 2001, 2002]}),
            baseline_period=(2000, 2001),
        )

        self.assertTrue(np.allclose(anomalies.values, [-1.0, 1.0, 3.0]))


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


class TemporalTest(unittest.TestCase):
    def test_aggregate_annual(self) -> None:
        annual = aggregate_annual(
            xr.DataArray(
                [1.0, 3.0, 5.0],
                dims=["time"],
                coords={
                    "time": np.array(
                        ["2000-01-01", "2000-02-01", "2001-01-01"],
                        dtype="datetime64[ns]",
                    )
                },
            )
        )

        self.assertEqual(annual.sel(year=2000).item(), 2.0)
        self.assertEqual(annual.sel(year=2001).item(), 5.0)

    def test_aggregate_seasonal_rolls_december_forward(self) -> None:
        seasonal = aggregate_seasonal(
            xr.DataArray(
                [6.0, 3.0, 0.0],
                dims=["time"],
                coords={
                    "time": np.array(
                        ["2000-12-01", "2001-01-01", "2001-02-01"],
                        dtype="datetime64[ns]",
                    )
                },
            )
        )

        self.assertEqual(seasonal.sel(season="DJF", year=2001).item(), 3.0)


class SpatialTest(unittest.TestCase):
    def test_weighted_average_respects_mask(self) -> None:
        value = weighted_average(
            values=xr.DataArray(
                [[10.0, 20.0], [30.0, 40.0]],
                dims=["lat", "lon"],
                coords={"lat": [0, 1], "lon": [0, 1]},
            ),
            weights=xr.DataArray(
                [[1.0, 3.0], [1.0, 1.0]],
                dims=["lat", "lon"],
                coords={"lat": [0, 1], "lon": [0, 1]},
            ),
            mask=xr.DataArray(
                [[True, False], [False, False]],
                dims=["lat", "lon"],
                coords={"lat": [0, 1], "lon": [0, 1]},
            ),
            dims=("lat", "lon"),
        )

        self.assertEqual(value.item(), 10.0)


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


class StatsTest(unittest.TestCase):
    def test_weighted_quantiles(self) -> None:
        quantiles = weighted_quantiles(
            values=[1.0, 2.0, 3.0],
            weights=[1.0, 1.0, 2.0],
            quantiles=[0.25, 0.75],
        )

        self.assertEqual(quantiles, (1.0, 3.0))

    def test_deterministic_resampling(self) -> None:
        samples = deterministic_resampling(values=[1.0, 2.0, 3.0, 4.0], n_samples=2)

        self.assertEqual(samples, (1.0, 3.0))


if __name__ == "__main__":
    unittest.main()
