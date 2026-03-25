from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from impacts.isimip import (
    ISIMIPPaths,
    ISIMIPVariable,
    WATERGAP_SOCPATH,
    _file_overlaps_years,
    discover_isimip_files,
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

    def test_socpath_mapping_covers_standard_scenarios(self) -> None:
        for scenario in ["historical", "ssp126", "ssp370", "ssp585"]:
            self.assertIn(scenario, WATERGAP_SOCPATH)


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


if __name__ == "__main__":
    unittest.main()
