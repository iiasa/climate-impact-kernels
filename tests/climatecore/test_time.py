import unittest
from datetime import datetime

import cftime
import numpy as np
import pandas as pd
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from impacts.climatecore import aggregate_monthly, aggregate_seasonal


class TemporalTest(unittest.TestCase):
    def test_aggregate_monthly_preserves_pandas_month_starts(self) -> None:
        monthly = aggregate_monthly(
            xr.DataArray(
                [1.0, 3.0, 5.0],
                dims=["time"],
                coords={
                    "time": np.array(
                        ["2000-01-01", "2000-01-15", "2000-02-01"],
                        dtype="datetime64[ns]",
                    )
                },
            )
        )

        self.assertEqual(
            monthly.time.values.tolist(),
            [datetime(2000, 1, 1), datetime(2000, 2, 1)],
        )
        self.assertEqual(monthly.sel(time="2000-01-01").item(), 2.0)
        self.assertEqual(monthly.sel(time="2000-02-01").item(), 5.0)

    def test_aggregate_monthly_preserves_360_day_calendar(self) -> None:
        time = [
            cftime.Datetime360Day(2001, 2, 1),
            cftime.Datetime360Day(2001, 2, 30),
            cftime.Datetime360Day(2001, 3, 1),
        ]
        monthly = aggregate_monthly(
            xr.DataArray(
                [10.0, 14.0, 20.0],
                dims=["time"],
                coords={"time": time},
            )
        )

        self.assertEqual(
            monthly.time.values.tolist(),
            [
                cftime.Datetime360Day(2001, 2, 1),
                cftime.Datetime360Day(2001, 3, 1),
            ],
        )
        self.assertEqual(monthly.sel(time=cftime.Datetime360Day(2001, 2, 1)).item(), 12.0)
        self.assertEqual(monthly.sel(time=cftime.Datetime360Day(2001, 3, 1)).item(), 20.0)

    def test_aggregate_monthly_preserves_monthly_constants_across_lengths(self) -> None:
        day_counts = [31, 28, 31, 30]
        values = []
        time = []

        for month, day_count in enumerate(day_counts, start=1):
            values.extend([float(month)] * day_count)
            time.extend(cftime.DatetimeNoLeap(2001, month, day) for day in range(1, day_count + 1))

        monthly = aggregate_monthly(
            xr.DataArray(
                values,
                dims=["time"],
                coords={"time": time},
            )
        )

        self.assertEqual(monthly.values.tolist(), [1.0, 2.0, 3.0, 4.0])

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

    def test_aggregate_seasonal_needs_all_three_months(self) -> None:
        seasonal = aggregate_seasonal(
            xr.DataArray(
                np.arange(1.0, 13.0),
                dims=["time"],
                coords={"time": pd.date_range("2001-01-01", periods=12, freq="MS")},
            )
        )

        # Jan and Feb 2001 lack their December, and Dec 2001 lacks Jan and Feb 2002.
        self.assertTrue(np.isnan(seasonal.sel(season="DJF", year=2001).item()))
        self.assertTrue(np.isnan(seasonal.sel(season="DJF", year=2002).item()))
        self.assertEqual(seasonal.sel(season="MAM", year=2001).item(), 4.0)
        self.assertEqual(seasonal.sel(season="SON", year=2001).item(), 10.0)

    def test_aggregate_seasonal_needs_all_three_months_of_daily_data(self) -> None:
        daily = pd.date_range("2001-01-01", "2001-02-28", freq="D")
        short = aggregate_seasonal(
            xr.DataArray(np.ones(len(daily)), dims=["time"], coords={"time": daily})
        )
        full_index = pd.date_range("2000-12-01", "2001-02-28", freq="D")
        full = aggregate_seasonal(
            xr.DataArray(np.ones(len(full_index)), dims=["time"], coords={"time": full_index})
        )

        self.assertTrue(np.isnan(short.sel(season="DJF", year=2001).item()))
        self.assertEqual(full.sel(season="DJF", year=2001).item(), 1.0)

    def test_aggregate_seasonal_still_skips_missing_values_within_a_full_season(self) -> None:
        seasonal = aggregate_seasonal(
            xr.DataArray(
                [6.0, np.nan, 0.0],
                dims=["time"],
                coords={"time": pd.date_range("2000-12-01", periods=3, freq="MS")},
            )
        )

        self.assertEqual(seasonal.sel(season="DJF", year=2001).item(), 3.0)

    def test_aggregate_seasonal_preserves_standard_subset_order(self) -> None:
        seasonal = aggregate_seasonal(
            xr.DataArray(
                [12.0, 1.0, 2.0, 6.0, 7.0, 8.0],
                dims=["time"],
                coords={
                    "time": np.array(
                        [
                            "2000-12-01",
                            "2001-01-01",
                            "2001-02-01",
                            "2001-06-01",
                            "2001-07-01",
                            "2001-08-01",
                        ],
                        dtype="datetime64[ns]",
                    )
                },
            ),
            seasons={"JJA": (6, 7, 8), "DJF": (12, 1, 2)},
        )

        self.assertEqual(seasonal.coords["season"].values.tolist(), ["JJA", "DJF"])
        self.assertEqual(seasonal.sel(season="JJA", year=2001).item(), 7.0)
        self.assertEqual(seasonal.sel(season="DJF", year=2001).item(), 5.0)

    def test_aggregate_seasonal_supports_phase_shifted_quarters(self) -> None:
        seasonal = aggregate_seasonal(
            xr.DataArray(
                np.arange(1.0, 13.0),
                dims=["time"],
                coords={
                    "time": np.array(
                        [
                            "2001-01-01",
                            "2001-02-01",
                            "2001-03-01",
                            "2001-04-01",
                            "2001-05-01",
                            "2001-06-01",
                            "2001-07-01",
                            "2001-08-01",
                            "2001-09-01",
                            "2001-10-01",
                            "2001-11-01",
                            "2001-12-01",
                        ],
                        dtype="datetime64[ns]",
                    )
                },
            ),
            seasons={
                "JFM": (1, 2, 3),
                "AMJ": (4, 5, 6),
                "JAS": (7, 8, 9),
                "OND": (10, 11, 12),
            },
        )

        self.assertEqual(seasonal.coords["season"].values.tolist(), ["JFM", "AMJ", "JAS", "OND"])
        self.assertEqual(seasonal.sel(season="JFM", year=2001).item(), 2.0)
        self.assertEqual(seasonal.sel(season="OND", year=2001).item(), 11.0)

    def test_aggregate_seasonal_preserves_360_day_calendar(self) -> None:
        seasonal = aggregate_seasonal(
            xr.DataArray(
                [6.0, 3.0, 0.0],
                dims=["time"],
                coords={
                    "time": [
                        cftime.Datetime360Day(2000, 12, 1),
                        cftime.Datetime360Day(2001, 1, 1),
                        cftime.Datetime360Day(2001, 2, 1),
                    ]
                },
            )
        )

        self.assertEqual(seasonal.sel(season="DJF", year=2001).item(), 3.0)

    def test_aggregate_seasonal_rejects_unsupported_seasons(self) -> None:
        data = xr.DataArray(
            [4.0, 5.0, 6.0],
            dims=["time"],
            coords={
                "time": np.array(
                    ["2001-04-01", "2001-05-01", "2001-06-01"],
                    dtype="datetime64[ns]",
                )
            },
        )

        with self.assertRaisesRegex(ValueError, "quarterly phase"):
            aggregate_seasonal(
                data,
                seasons={
                    "DJF": (12, 1, 2),
                    "JFM": (1, 2, 3),
                },
            )

    def test_aggregate_seasonal_supports_datasets_with_heterogeneous_dims(self) -> None:
        seasonal = aggregate_seasonal(
            xr.Dataset(
                data_vars={
                    "capacity_factor": (
                        ("time", "entity"),
                        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    ),
                    "mean_temperature": (
                        ("time",),
                        [10.0, 12.0, 14.0],
                    ),
                },
                coords={
                    "time": np.array(
                        ["2001-01-01", "2001-02-01", "2001-03-01"],
                        dtype="datetime64[ns]",
                    ),
                    "entity": [101, 202],
                },
            ),
            seasons={"JFM": (1, 2, 3)},
        )

        self.assertEqual(seasonal["capacity_factor"].dims, ("season", "year", "entity"))
        self.assertEqual(seasonal["mean_temperature"].dims, ("season", "year"))
        self.assertEqual(seasonal["mean_temperature"].sel(season="JFM", year=2001).item(), 12.0)

    @unittest.skipIf(da is None, "dask is not installed")
    def test_aggregate_seasonal_consolidates_standard_dask_chunks(self) -> None:
        time = pd.date_range("2000-01-01", "2001-12-31", freq="D")
        data = xr.DataArray(
            np.arange(len(time) * 4, dtype=float).reshape(len(time), 4),
            dims=["time", "entity"],
            coords={"time": time, "entity": [101, 202, 303, 404]},
        ).chunk({"time": 90, "entity": 2})

        seasonal = aggregate_seasonal(data)

        self.assertEqual(seasonal.dims, ("season", "year", "entity"))
        self.assertIsInstance(seasonal.data, da.Array)
        self.assertEqual(seasonal.chunks[0], (4,))
        self.assertEqual(seasonal.chunks[1], (3,))
        self.assertEqual(seasonal.chunks[2], (2, 2))

    @unittest.skipIf(da is None, "dask is not installed")
    def test_aggregate_seasonal_keeps_datasets_lazy(self) -> None:
        seasonal = aggregate_seasonal(
            xr.DataArray(
                np.arange(12.0),
                dims=["time"],
                coords={"time": pd.date_range("2001-01-01", periods=12, freq="MS")},
            ).chunk({"time": 3}).to_dataset(name="tas")
        )

        self.assertTrue(hasattr(seasonal["tas"].data, "__dask_graph__"))


if __name__ == "__main__":
    unittest.main()
