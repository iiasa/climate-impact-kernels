from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from impacts.buildings.rimex import (
    aggregate_chilled_to_country,
    expand_chilled_metadata,
    load_country_raster,
    parse_chilled_matrix_filename,
)
from impacts.thermoelectric.rimex import (
    aggregate_thermoelectric_seasonal,
    load_thermoelectric_ucr,
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

            ucr = load_thermoelectric_ucr(source)

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


class ChilledRimeXTest(unittest.TestCase):
    def test_parse_chilled_matrix_filename(self) -> None:
        parsed = parse_chilled_matrix_filename(
            "seasonal_mpi_esm1_2_hr_ssp585_sfh_rur_s1_2015_2100.nc"
        )

        self.assertEqual(parsed["gcm"], "MPI-ESM1-2-HR")
        self.assertEqual(parsed["scenario"], "ssp585")
        self.assertEqual(parsed["building_type"], "sfh")

    def test_load_country_raster_uses_lookup_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raster_path = root / "gaul.nc"
            csv_path = root / "gaul.csv"

            xr.DataArray(
                [[1, 2]],
                dims=["lat", "lon"],
                coords={"lat": [0.5], "lon": [0.5, 1.5]},
            ).to_netcdf(raster_path)
            pd.DataFrame({"ID": [1, 2], "ISO3": ["AAA", "BBB"]}).to_csv(csv_path, index=False)

            country_raster = load_country_raster(
                gaul_raster_path=raster_path,
                gaul_csv_path=csv_path,
            )

            self.assertEqual(country_raster.values.tolist(), [["AAA", "BBB"]])

    def test_aggregate_chilled_to_country_applies_sum_and_weighted_mean(self) -> None:
        country_raster = xr.DataArray(
            [["A", "A", "B", "B"]],
            dims=["lat", "lon"],
            coords={"lat": [0.5], "lon": [0.5, 1.5, 2.5, 3.5]},
            name="country",
        )
        dataset = xr.Dataset(
            data_vars={
                "cooling_energy_total": (
                    ("season", "year", "entity"),
                    [[[1.0, 2.0, 3.0, 4.0]]],
                ),
                "heating_energy": (
                    ("season", "year", "entity"),
                    [[[5.0, 6.0, 7.0, 8.0]]],
                ),
                "Nd": (
                    ("season", "year", "entity"),
                    [[[10.0, 14.0, 20.0, 22.0]]],
                ),
                "Nf": (
                    ("season", "year", "entity"),
                    [[[1.0, 3.0, 5.0, 7.0]]],
                ),
            },
            coords={
                "season": ["DJF"],
                "year": [2001],
                "entity": [0, 1, 2, 3],
                "lat": ("entity", [0.5, 0.5, 0.5, 0.5]),
                "lon": ("entity", [0.5, 1.5, 2.5, 3.5]),
            },
        )

        aggregated = aggregate_chilled_to_country(dataset, country_raster)
        expanded = expand_chilled_metadata(
            aggregated,
            {
                "gcm": "GFDL-ESM4",
                "scenario": "ssp585",
                "building_type": "sfh",
                "urt": "rur",
                "vintage": "s1",
                "year_start": 2015,
                "year_end": 2100,
            },
        )

        self.assertAlmostEqual(
            aggregated["cooling_energy_total"].sel(country="A", season="DJF", year=2001).item(),
            3.0,
        )
        self.assertAlmostEqual(
            aggregated["heating_energy"].sel(country="B", season="DJF", year=2001).item(),
            15.0,
        )
        self.assertAlmostEqual(
            aggregated["Nd"].sel(country="A", season="DJF", year=2001).item(),
            12.0,
        )
        self.assertAlmostEqual(
            aggregated["Nf"].sel(country="B", season="DJF", year=2001).item(),
            6.0,
        )
        self.assertEqual(expanded.coords["building_type"].values.tolist(), ["sfh"])


if __name__ == "__main__":
    unittest.main()
