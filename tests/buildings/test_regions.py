from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from impacts.buildings.regions import (
    aggregate_chilled_to_country,
    expand_chilled_metadata,
    load_country_raster,
    parse_chilled_matrix_filename,
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

    def _gaul_files(self, root: Path, table: pd.DataFrame) -> dict[str, Path]:
        raster_path = root / "gaul.nc"
        csv_path = root / "gaul.csv"
        xr.DataArray(
            [[1, 2]], dims=["lat", "lon"], coords={"lat": [0.5], "lon": [0.5, 1.5]}
        ).to_netcdf(raster_path)
        table.to_csv(csv_path, index=False)
        return {"gaul_raster_path": raster_path, "gaul_csv_path": csv_path}

    def test_load_country_raster_requires_the_named_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._gaul_files(
                Path(tmpdir), pd.DataFrame({"ID": [1, 2], "iso3": ["AAA", "BBB"], "NAME": ["a", "b"]})
            )

            with self.assertRaisesRegex(
                ValueError,
                r"country column 'ISO3' is not in the GAUL lookup CSV \(columns: ID, iso3, NAME\); "
                "pass country_column naming the column",
            ):
                load_country_raster(**files)
            self.assertEqual(
                load_country_raster(**files, country_column="iso3").values.tolist(), [["AAA", "BBB"]]
            )

    def test_load_country_raster_refuses_null_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._gaul_files(
                Path(tmpdir), pd.DataFrame({"ID": [1, 2], "ISO3": ["AAA", None]})
            )

            with self.assertRaisesRegex(
                ValueError,
                r"column 'ISO3' is null for 1 rows \(first IDs: \[2\]\); "
                "fill the labels or pass another country_column",
            ):
                load_country_raster(**files)

    def test_aggregate_chilled_to_country_takes_area_weighted_means(self) -> None:
        country_raster = xr.DataArray(
            [["A", "A", "B", "B"], ["A", "A", "B", "B"]],
            dims=["lat", "lon"],
            coords={"lat": [0.5, 1.5], "lon": [0.5, 1.5, 2.5, 3.5]},
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
            1.5,
        )
        self.assertAlmostEqual(
            aggregated["heating_energy"].sel(country="B", season="DJF", year=2001).item(),
            7.5,
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

    def test_country_energy_intensity_is_not_summed_over_cells(self) -> None:
        # Three latitude rows near the pole have very different cell areas; the
        # area of a 1-degree row is proportional to cos(latitude).
        lats = [87.5, 88.5, 89.5]
        lons = [0.5, 1.5]
        country_raster = xr.DataArray(
            [["A", "A"]] * 3,
            dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
            name="country",
        )
        row_value = {87.5: 10.0, 88.5: 20.0, 89.5: 60.0}
        cell_lats = [lat for lat in lats for _ in lons]
        values = [row_value[lat] for lat in cell_lats]
        dataset = xr.Dataset(
            data_vars={
                name: (("season", "year", "entity"), [[values]])
                for name in ("cooling_energy_total", "heating_energy")
            },
            coords={
                "season": ["JJA"],
                "year": [2001],
                "entity": list(range(len(cell_lats))),
                "lat": ("entity", cell_lats),
                "lon": ("entity", lons * 3),
            },
        )

        aggregated = aggregate_chilled_to_country(dataset, country_raster)

        weights = [np.cos(np.deg2rad(lat)) for lat in lats]
        expected = sum(w * row_value[lat] for w, lat in zip(weights, lats, strict=True)) / sum(weights)
        for name in ("cooling_energy_total", "heating_energy"):
            self.assertAlmostEqual(
                aggregated[name].sel(country="A", season="JJA", year=2001).item(), expected, places=3
            )


if __name__ == "__main__":
    unittest.main()
