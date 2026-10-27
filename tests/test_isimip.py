"""The isimip3b WaterGAP glob must select a single forcing and frequency.

An SSP directory holds several human-forcing experiments covering the same
years, and `dis` ships as both a decadal daily series and a single monthly file.
A glob that fixes neither selects several at once and fails on a non-monotonic time axis.
These tests build a fake archive tree and assert on file selection alone.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from impacts.isimip import (
    ISIMIPPaths,
    ISIMIPVariable,
    _file_overlaps_years,
    _standardize_time_index,
    discover_isimip_files,
    isimip3a_counterclim_paths,
    isimip3b_paths,
    load_isimip_dataset,
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


class CounterclimFileDiscoveryTest(unittest.TestCase):
    def test_counterclim_resolver_matches_local_atmosphere_and_watergap_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = isimip3a_counterclim_paths(root)
            climate_dir = paths.climate_root
            watergap_dir = paths.watergap_root
            climate_dir.mkdir(parents=True)
            watergap_dir.mkdir(parents=True)

            for years in ["1901_1910", "1911_1920"]:
                (
                    climate_dir
                    / f"gswp3-w5e5_counterclim_tas_global_daily_{years}.nc"
                ).touch()
            (
                watergap_dir
                / "watergap2-2e_gswp3-w5e5_counterclim_2015soc_default_"
                "dis_global_daily_1901_2019.nc"
            ).touch()
            (
                watergap_dir
                / "watergap2-2e_gswp3-w5e5_counterclim_2015soc_default_"
                "dis_global_monthly_1901_2019.nc"
            ).touch()
            (
                watergap_dir
                / "watergap2-2e_gswp3-w5e5_counterclim_2015soc_default_"
                "triver_global_monthly_1901_2019.nc"
            ).touch()

            tas = ISIMIPVariable("tas", "climate", "daily")
            dis = ISIMIPVariable("dis", "watergap", "daily")
            triver = ISIMIPVariable("triver", "watergap", "monthly")

            self.assertEqual(
                len(
                    discover_isimip_files(
                        tas, "GSWP3-W5E5", "counterclim", paths
                    )
                ),
                2,
            )
            self.assertEqual(
                len(
                    discover_isimip_files(
                        dis, "GSWP3-W5E5", "counterclim", paths
                    )
                ),
                1,
            )
            self.assertEqual(
                len(
                    discover_isimip_files(
                        triver, "GSWP3-W5E5", "counterclim", paths
                    )
                ),
                1,
            )

    def test_watergap_discovery_follows_the_configured_water_use_forcing(self) -> None:
        """`soc` picks the water-use forcing; it must not stay fixed to 2015soc."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            watergap_dir = isimip3a_counterclim_paths(root).watergap_root
            watergap_dir.mkdir(parents=True)
            for soc in ["2015soc", "histsoc"]:
                (
                    watergap_dir
                    / f"watergap2-2e_gswp3-w5e5_counterclim_{soc}_default_"
                    "dis_global_daily_1901_2019.nc"
                ).touch()

            dis = ISIMIPVariable("dis", "watergap", "daily")

            default_files = discover_isimip_files(
                dis, "GSWP3-W5E5", "counterclim", isimip3a_counterclim_paths(root)
            )
            histsoc_files = discover_isimip_files(
                dis,
                "GSWP3-W5E5",
                "counterclim",
                isimip3a_counterclim_paths(root, soc="histsoc"),
            )

        self.assertEqual(len(default_files), 1)
        self.assertIn("2015soc", default_files[0].name)
        self.assertEqual(len(histsoc_files), 1)
        self.assertIn("histsoc", histsoc_files[0].name)


class OceanLoadingTest(unittest.TestCase):
    """Sea-surface temperature: archive layout, month-count time axis, nesting onto the land grid."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)
        self.paths = isimip3b_paths(root)

        months = pd.date_range("2001-01-01", periods=3, freq="MS")
        land_lat = [0.75, 0.25, -0.25, -0.75]
        land_lon = [10.25, 10.75]
        watergap_dir = self.paths.watergap_root / "historical" / "GFDL-ESM4"
        watergap_dir.mkdir(parents=True)
        xr.Dataset(
            {"triver": (("time", "lat", "lon"), np.full((3, 4, 2), 290.0))},
            coords={"time": months, "lat": land_lat, "lon": land_lon},
        ).to_netcdf(
            watergap_dir
            / "watergap2-2e_gfdl-esm4_w5e5_historical_histsoc_default_triver_global_monthly_2001_2001.nc"
        )

        # 1-degree ocean grid; time counts months from 1601 at mid-month on a
        # 365-day calendar, as in the ISIMIP archive.
        ocean_dir = self.paths.ocean_root / "historical" / "GFDL-ESM4"
        ocean_dir.mkdir(parents=True)
        tos = np.array([[[20.0, 0.0], [21.0, 0.0]], [[22.0, 0.0], [23.0, 0.0]], [[24.0, 0.0], [25.0, 0.0]]])
        counts = (2001 - 1601) * 12 + np.array([0.5, 1.4, 2.5])
        ocean = xr.Dataset(
            {
                "tos": (("time", "lat", "lon"), tos),
                "time_bnds": (("time", "bnds"), np.stack([np.floor(counts), np.floor(counts) + 1], axis=1)),
            },
            coords={"time": counts, "lat": [0.5, -0.5], "lon": [10.5, 11.5]},
        )
        ocean["time"].attrs.update({"units": "months since 1601-1-1 00:00:00", "calendar": "365_day"})
        ocean["tos"].attrs["units"] = "degC"
        for resolution in ("60arcmin", "120arcmin"):
            ocean.to_netcdf(
                ocean_dir / f"gfdl-esm4_r1i1p1f1_historical_tos_{resolution}_global_monthly_2001_2001.nc",
                encoding={"time": {"dtype": "float64"}},
            )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_discovery_picks_the_requested_resolution(self) -> None:
        tos = ISIMIPVariable("tos", "ocean", "monthly", resolution="60arcmin")
        files = discover_isimip_files(tos, "GFDL-ESM4", "historical", self.paths)
        self.assertEqual([f.name for f in files], ["gfdl-esm4_r1i1p1f1_historical_tos_60arcmin_global_monthly_2001_2001.nc"])

    def test_ocean_variable_joins_the_land_grid_at_month_starts(self) -> None:
        loaded = load_isimip_dataset(
            [
                ISIMIPVariable("triver", "watergap", "monthly", output_name="tw"),
                ISIMIPVariable("tos", "ocean", "monthly", resolution="60arcmin"),
            ],
            "GFDL-ESM4",
            "historical",
            self.paths,
        )
        self.assertEqual(loaded["time"].values.tolist(), loaded["tw"]["time"].values.tolist())
        self.assertEqual(pd.DatetimeIndex(loaded["time"].values).strftime("%Y-%m-%d").tolist(), ["2001-01-01", "2001-02-01", "2001-03-01"])
        self.assertEqual(loaded["tos"].sizes, {"time": 3, "lat": 4, "lon": 2})
        # Each 1-degree value covers the four 0.5-degree cells inside it.
        self.assertEqual(loaded["tos"].isel(time=1).values.tolist(), [[22.0, 22.0], [22.0, 22.0], [23.0, 23.0], [23.0, 23.0]])

    def test_refuses_a_time_axis_with_a_gap(self) -> None:
        from impacts.isimip import _decode_month_counts

        ds = xr.Dataset(coords={"time": ("time", [4800.5, 4802.5], {"units": "months since 1601-1-1"})})
        with self.assertRaisesRegex(ValueError, "consecutive"):
            _decode_month_counts(ds)


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

    def test_unparseable_name_is_refused_and_named(self) -> None:
        for name in ("some_weird_file.nc", "single.nc", "foo_global_daily_2001_latest.nc"):
            with self.assertRaisesRegex(
                ValueError,
                f"cannot read the year span of ISIMIP file '{name}'; "
                r"its name must end with _<start year>_<end year>\.nc",
            ):
                _file_overlaps_years(Path("dir") / name, (2001, 2010))


class CalendarStandardizationTest(unittest.TestCase):
    def test_cftime_converted_to_pandas_timestamps(self) -> None:
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
        """Month-starts in a cftime calendar and in pandas merge only once standardized."""
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


class DatasetMergeTest(unittest.TestCase):
    """Variables on different grids or time spans are an error, not a silent truncation."""

    @staticmethod
    def _load(arrays: dict[str, xr.DataArray]) -> xr.Dataset:
        variables = [ISIMIPVariable(name, "climate", "monthly") for name in arrays]

        def fake_load(variable, *_args, **_kwargs):
            return arrays[variable.name].copy()

        with mock.patch("impacts.isimip.load_isimip_variable", side_effect=fake_load):
            return load_isimip_dataset(variables, "GFDL-ESM4", "ssp585", None)

    @staticmethod
    def _field(months: int, lats: list[float], name: str) -> xr.DataArray:
        return xr.DataArray(
            np.ones((months, len(lats), 1)),
            dims=["time", "lat", "lon"],
            coords={
                "time": pd.date_range("2015-01-01", periods=months, freq="MS"),
                "lat": lats,
                "lon": [0.25],
            },
            name=name,
        )

    def test_matching_variables_merge(self) -> None:
        merged = self._load(
            {"tas": self._field(3, [0.25], "tas"), "hurs": self._field(3, [0.25], "hurs")}
        )

        self.assertEqual(merged.sizes["time"], 3)

    def test_different_time_spans_raise_instead_of_truncating(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact"):
            self._load({"tas": self._field(3, [0.25], "tas"), "hurs": self._field(2, [0.25], "hurs")})

    def test_different_grids_raise_instead_of_truncating(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact"):
            self._load({"tas": self._field(3, [0.25, 0.75], "tas"), "hurs": self._field(3, [0.25], "hurs")})


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

    def test_repeated_identical_anomaly_columns_read_as_one(self) -> None:
        # The archive's GMT files repeat the anomaly block under the same header.
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "gfdl-esm4_historical_monthly_temperature_1850_2014.csv").write_text(
                "date,mean_anomaly,std_anomaly,mean_anomaly,std_anomaly\n"
                "1850-01-15,0.1,1.0,0.1,1.0\n"
                "1851-01-15,0.5,1.0,0.5,1.0\n"
            )

            gmt = load_isimip_gmt(root, "GFDL-ESM4", "historical")

            self.assertEqual(gmt.values.tolist(), [0.1, 0.5])
            self.assertEqual(gmt.attrs["anomaly_column"], "mean_anomaly")

    def test_several_anomaly_columns_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pd.DataFrame(
                {
                    "date": ["1850-01-15", "1851-01-15"],
                    "mean_anomaly_1850_1900": [0.1, 0.5],
                    "mean_anomaly_1995_2014": [-0.6, -0.2],
                }
            ).to_csv(root / "gfdl-esm4_historical_monthly_temperature_1850_2014.csv", index=False)

            with self.assertRaisesRegex(
                ValueError,
                r"has several mean_anomaly columns \(mean_anomaly_1850_1900, mean_anomaly_1995_2014\) "
                "with different values; keep only the column to use",
            ):
                load_isimip_gmt(root, "GFDL-ESM4", "historical")


GCM = "GFDL-ESM4"
SCENARIO = "ssp370"
PREFIX = "watergap2-2e_gfdl-esm4_w5e5"

SSP_ARCHIVE = (
    f"{PREFIX}_ssp370_1850soc_default_dis_global_monthly_2015_2100.nc",
    f"{PREFIX}_ssp370_2015soc_default_dis_global_daily_2041_2050.nc",
    f"{PREFIX}_ssp370_2015soc_default_dis_global_daily_2051_2060.nc",
    f"{PREFIX}_ssp370_2015soc_default_dis_global_monthly_2015_2100.nc",
    f"{PREFIX}_ssp370_1850soc_default_triver_global_monthly_2015_2100.nc",
    f"{PREFIX}_ssp370_2015soc-from-histsoc_default_triver_global_monthly_2015_2100.nc",
    f"{PREFIX}_ssp370_2015soc_default_triver_global_monthly_2015_2100.nc",
)
HISTORICAL_FILE = f"{PREFIX}_historical_histsoc_default_triver_global_monthly_1850_2014.nc"

DIS_DAILY = ("dis", "watergap", "daily")
TRIVER = ("triver", "watergap", "monthly")


@pytest.fixture
def root(tmp_path):
    ssp = isimip3b_paths(tmp_path).watergap_root / SCENARIO / GCM
    ssp.mkdir(parents=True)
    for name in SSP_ARCHIVE:
        (ssp / name).touch()
    hist = isimip3b_paths(tmp_path).watergap_root / "historical" / GCM
    hist.mkdir(parents=True)
    (hist / HISTORICAL_FILE).touch()
    return tmp_path


def _names(variable, scenario, paths, **kwargs):
    return [
        p.name
        for p in discover_isimip_files(
            ISIMIPVariable(*variable), GCM, scenario, paths, **kwargs
        )
    ]


def test_daily_discharge_selects_only_daily_2015soc_files(root):
    names = _names(DIS_DAILY, SCENARIO, isimip3b_paths(root), year_range=(2041, 2060))
    assert names == [
        f"{PREFIX}_ssp370_2015soc_default_dis_global_daily_2041_2050.nc",
        f"{PREFIX}_ssp370_2015soc_default_dis_global_daily_2051_2060.nc",
    ]


def test_river_temperature_selects_only_the_2015soc_monthly_file(root):
    assert _names(TRIVER, SCENARIO, isimip3b_paths(root)) == [
        f"{PREFIX}_ssp370_2015soc_default_triver_global_monthly_2015_2100.nc"
    ]


def test_soc_override_selects_the_naturalized_proxy(root):
    paths = isimip3b_paths(root, soc="1850soc")
    assert _names(TRIVER, SCENARIO, paths) == [
        f"{PREFIX}_ssp370_1850soc_default_triver_global_monthly_2015_2100.nc"
    ]


def test_historical_reads_histsoc_not_the_ssp_forcing(root):
    assert _names(TRIVER, "historical", isimip3b_paths(root)) == [HISTORICAL_FILE]


def test_forcing_without_files_raises_instead_of_falling_back(root):
    with pytest.raises(FileNotFoundError, match="ssp370soc"):
        _names(TRIVER, SCENARIO, isimip3b_paths(root, soc="ssp370soc"))
    with pytest.raises(FileNotFoundError, match="2015soc"):
        _names(TRIVER, "historical", isimip3b_paths(root, historical_soc="2015soc"))
