from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import xarray as xr

from impacts.isimip import (
    ISIMIPPaths,
    ISIMIPVariable,
    _file_overlaps_years,
    _standardize_time_index,
    discover_isimip_files,
    load_isimip_gmt,
)


class FileDiscoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)

        self.paths = ISIMIPPaths(
            climate_root=root / "climate",
            watergap_root=root / "watergap",
        )

        # Create mock file structure for climate/historical/GFDL-ESM4
        climate_dir = root / "climate" / "historical" / "GFDL-ESM4"
        climate_dir.mkdir(parents=True)
        for years in ["1991_2000", "2001_2010", "2011_2014"]:
            (climate_dir / f"gfdl-esm4_r1i1p1f1_w5e5_historical_tas_global_daily_{years}.nc").touch()
            (climate_dir / f"gfdl-esm4_r1i1p1f1_w5e5_historical_hurs_global_daily_{years}.nc").touch()

        # Create mock WaterGAP files
        watergap_dir = root / "watergap" / "historical" / "GFDL-ESM4"
        watergap_dir.mkdir(parents=True)
        for years in ["2001_2010", "2011_2014"]:
            (watergap_dir / f"watergap2-2e_gfdl-esm4_w5e5_historical_histsoc_default_dis_global_daily_{years}.nc").touch()
        (watergap_dir / "watergap2-2e_gfdl-esm4_w5e5_historical_histsoc_default_triver_global_monthly_1850_2014.nc").touch()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_discover_climate_files_returns_sorted(self) -> None:
        tas = ISIMIPVariable("tas", "climate", "daily")
        files = discover_isimip_files(tas, "GFDL-ESM4", "historical", self.paths)
        self.assertEqual(len(files), 3)
        self.assertTrue(all(f.exists() for f in files))
        years_in_names = [f.stem.split("_")[-2] for f in files]
        self.assertEqual(years_in_names, ["1991", "2001", "2011"])

    def test_discover_with_year_range_filters(self) -> None:
        tas = ISIMIPVariable("tas", "climate", "daily")
        files = discover_isimip_files(
            tas, "GFDL-ESM4", "historical", self.paths,
            year_range=(2001, 2010),
        )
        self.assertEqual(len(files), 1)
        self.assertIn("2001_2010", files[0].name)

    def test_discover_watergap_daily_files(self) -> None:
        dis = ISIMIPVariable("dis", "watergap", "daily")
        files = discover_isimip_files(dis, "GFDL-ESM4", "historical", self.paths)
        self.assertEqual(len(files), 2)

    def test_discover_watergap_monthly_file(self) -> None:
        triver = ISIMIPVariable("triver", "watergap", "monthly")
        files = discover_isimip_files(triver, "GFDL-ESM4", "historical", self.paths)
        self.assertEqual(len(files), 1)
        self.assertIn("monthly", files[0].name)

    def test_discover_raises_on_missing(self) -> None:
        tas = ISIMIPVariable("tas", "climate", "daily")
        with self.assertRaises(FileNotFoundError):
            discover_isimip_files(tas, "NONEXISTENT-GCM", "historical", self.paths)

class YearOverlapTest(unittest.TestCase):
    def test_file_within_range(self) -> None:
        path = Path("foo_global_daily_2001_2010.nc")
        self.assertTrue(_file_overlaps_years(path, (2005, 2008)))

    def test_file_before_range(self) -> None:
        path = Path("foo_global_daily_1991_2000.nc")
        self.assertFalse(_file_overlaps_years(path, (2001, 2010)))

    def test_file_after_range(self) -> None:
        path = Path("foo_global_daily_2011_2020.nc")
        self.assertFalse(_file_overlaps_years(path, (2001, 2010)))

    def test_file_overlaps_start(self) -> None:
        path = Path("foo_global_daily_1991_2005.nc")
        self.assertTrue(_file_overlaps_years(path, (2001, 2010)))

    def test_unparseable_returns_true(self) -> None:
        path = Path("some_weird_file.nc")
        self.assertTrue(_file_overlaps_years(path, (2001, 2010)))


class CalendarStandardizationTest(unittest.TestCase):
    def test_cftime_converted_to_pandas_timestamps(self) -> None:
        import datetime
        cftime_times = [cftime.DatetimeNoLeap(2015, m, 1) for m in range(1, 4)]
        da = xr.DataArray(np.ones(3), dims=["time"], coords={"time": cftime_times})
        result = _standardize_time_index(da)
        self.assertIsInstance(result.time.values[0], np.datetime64)

    def test_pandas_timestamps_pass_through(self) -> None:
        times = pd.date_range("2015-01-01", periods=3, freq="MS")
        da = xr.DataArray(np.ones(3), dims=["time"], coords={"time": times})
        result = _standardize_time_index(da)
        np.testing.assert_array_equal(result.time.values, da.time.values)

    def test_mixed_calendar_merge_produces_nonempty_time(self) -> None:
        """The bug: cftime and pandas month-starts don't merge via inner join."""
        pandas_times = pd.date_range("2015-01-01", periods=3, freq="MS")
        cftime_times = [cftime.DatetimeNoLeap(2015, m, 1) for m in range(1, 4)]

        a = xr.DataArray(np.ones(3), dims=["time"], coords={"time": pandas_times}, name="tas")
        b = xr.DataArray(np.ones(3), dims=["time"], coords={"time": cftime_times}, name="tw")

        # Without standardization: inner join produces empty time
        bad_merge = xr.merge([a, b], join="inner")
        self.assertEqual(bad_merge.sizes["time"], 0)

        # With standardization: works
        b_fixed = _standardize_time_index(b)
        good_merge = xr.merge([a, b_fixed], join="inner")
        self.assertEqual(good_merge.sizes["time"], 3)


class GMTLoadingTest(unittest.TestCase):
    def test_load_isimip_gmt_stitches_historical_and_future(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            historical = pd.DataFrame(
                {
                    "date": ["1850-01-15", "1850-02-15", "1851-01-15", "1851-02-15"],
                    "mean_anomaly": [0.1, 0.3, 0.5, 0.7],
                }
            )
            future = pd.DataFrame(
                {
                    "date": ["2015-01-15", "2015-02-15", "2016-01-15", "2016-02-15"],
                    "mean_anomaly": [1.0, 1.2, 1.4, 1.6],
                }
            )
            historical.to_csv(
                root / "gfdl-esm4_historical_monthly_temperature_1850_2014.csv",
                index=False,
            )
            future.to_csv(
                root / "gfdl-esm4_ssp126_monthly_temperature_2015_2100.csv",
                index=False,
            )

            gmt = load_isimip_gmt(root, "GFDL-ESM4", "ssp126")

            self.assertEqual(gmt["year"].values.tolist(), [1850, 1851, 2015, 2016])
            self.assertEqual(gmt.values.tolist(), [0.2, 0.6, 1.1, 1.5])


if __name__ == "__main__":
    unittest.main()
