from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from impacts.thermoelectric.warming_levels import (
    aggregate_thermoelectric_seasonal,
    load_thermoelectric_usable_capacity_ratio,
)


class ThermoelectricRimeXTest(unittest.TestCase):
    def test_load_thermoelectric_ucr_returns_sparse_xarray_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "thermoelectric.parquet"
            pd.DataFrame(
                {
                    "cooling": ["wet", "wet", "wet", "wet"],
                    "gcm": ["GFDL-ESM4"] * 4,
                    "scenario": ["historical", "historical", "ssp126", "ssp126"],
                    "time": pd.to_datetime(
                        ["1850-01-01", "1850-02-01", "2015-01-01", "2015-02-01"]
                    ),
                    "country": ["A"] * 4,
                    "usable_capacity_mw": [50.0, 60.0, 70.0, 80.0],
                    "total_mw": [100.0, 100.0, 100.0, 100.0],
                }
            ).to_parquet(source, index=False)

            ucr = load_thermoelectric_usable_capacity_ratio(source)

            self.assertEqual(
                ucr.dims,
                ("cooling", "gcm", "scenario", "time", "country"),
            )
            self.assertAlmostEqual(
                ucr.sel(
                    cooling="wet",
                    gcm="GFDL-ESM4",
                    scenario="historical",
                    time="1850-01-01",
                    country="A",
                ).item(),
                0.5,
            )
            self.assertTrue(
                np.isnan(
                    ucr.sel(
                        cooling="wet",
                        gcm="GFDL-ESM4",
                        scenario="historical",
                        time="2015-01-01",
                        country="A",
                    ).item()
                )
            )

    def test_the_first_djf_of_a_record_starting_in_january_is_missing(self) -> None:
        times = pd.date_range("2002-01-01", "2003-02-01", freq="MS")
        values = np.where(times.month == 12, 0.0, 1.0)
        monthly = xr.DataArray(
            values.reshape(1, 1, 1, -1, 1),
            dims=["cooling", "gcm", "scenario", "time", "country"],
            coords={
                "cooling": ["wet"],
                "gcm": ["GFDL-ESM4"],
                "scenario": ["historical"],
                "time": times,
                "country": ["A"],
            },
            name="ucr",
        )

        djf = aggregate_thermoelectric_seasonal(monthly).sel(
            cooling="wet", gcm="GFDL-ESM4", scenario="historical", season="DJF", country="A"
        )

        # January and February 2002 have no December before them, so that season is
        # missing; December 2002 to February 2003 is complete.
        self.assertTrue(np.isnan(djf.sel(year=2002).item()))
        self.assertAlmostEqual(djf.sel(year=2003).item(), 2.0 / 3.0)

    def test_aggregate_thermoelectric_seasonal_uses_strict_means(self) -> None:
        monthly = xr.DataArray(
            [[[[[1.0], [2.0], [3.0], [np.nan], [np.nan], [4.0]]]]],
            dims=["cooling", "gcm", "scenario", "time", "country"],
            coords={
                "cooling": ["wet"],
                "gcm": ["GFDL-ESM4"],
                "scenario": ["historical"],
                "time": pd.to_datetime(
                    [
                        "2001-12-01",
                        "2002-01-01",
                        "2002-02-01",
                        "2002-03-01",
                        "2002-04-01",
                        "2002-05-01",
                    ]
                ),
                "country": ["A"],
            },
            name="ucr",
        )

        seasonal = aggregate_thermoelectric_seasonal(monthly)

        self.assertEqual(
            seasonal.dims,
            ("cooling", "gcm", "scenario", "season", "year", "country"),
        )
        self.assertAlmostEqual(
            seasonal.sel(
                cooling="wet",
                gcm="GFDL-ESM4",
                scenario="historical",
                season="DJF",
                year=2002,
                country="A",
            ).item(),
            2.0,
        )
        self.assertTrue(
            np.isnan(
                seasonal.sel(
                    cooling="wet",
                    gcm="GFDL-ESM4",
                    scenario="historical",
                    season="MAM",
                    year=2002,
                    country="A",
                ).item()
            )
        )


if __name__ == "__main__":
    unittest.main()
