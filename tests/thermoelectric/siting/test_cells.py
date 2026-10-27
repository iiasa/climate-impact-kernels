from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

from impacts.engine import evaluate_mapped
from impacts.thermoelectric.siting.cells import (
    build_synthetic_sites,
    build_valid_cells,
    default_wet_synthetic_plants,
    evaluate_wet_potential_sites,
    evaluate_wet_potential_sites_seasonal_and_monthly,
)
from impacts.thermoelectric.siting.site_evaluation import SampledYear, _to_long
from impacts.thermoelectric.wet import WET_COOLING_KERNEL

from .fixtures import COAL_ONCE_THROUGH, djf_climate, one_cell


class WetCeilingAggregationTest(unittest.TestCase):
    def test_ceiling_is_mean_of_monthly_ratios_not_ratio_of_means(self) -> None:
        sites = build_synthetic_sites(one_cell(), default_wet_synthetic_plants())
        climate = djf_climate()

        seasonal = evaluate_wet_potential_sites(sites, climate).compute()
        ceiling = seasonal["water_limited_capacity_ceiling_mw"].sel(
            season="DJF", year=2001
        )
        unit_id = sites.loc[
            sites["archetype"] == COAL_ONCE_THROUGH, "UNITID"
        ].item()
        got = float(ceiling.sel(entity=unit_id).item())

        # Reconstruct the monthly numerator/denominator the kernel produced.
        monthly = evaluate_mapped(
            WET_COOLING_KERNEL, sites, climate, duration="native",
            site_id_column="UNITID",
        ).compute()
        discharge = monthly["available_discharge_m3_per_s"].sel(entity=unit_id)
        intensity = monthly["withdrawal_intensity_m3_per_mwh"].sel(entity=unit_id)
        monthly_ceiling = 3600.0 * discharge / intensity

        mean_of_ratios = float(monthly_ceiling.mean().item())
        ratio_of_means = float(
            3600.0 * discharge.mean().item() / intensity.mean().item()
        )

        self.assertAlmostEqual(got, mean_of_ratios, places=6)
        # The fixture makes the two orders differ materially, so the test tells them apart.
        relative_gap = abs(mean_of_ratios - ratio_of_means) / mean_of_ratios
        self.assertGreater(relative_gap, 0.05)


class WetCeilingMissingDriverTest(unittest.TestCase):
    """Dry months count as zero and months with unknown drivers drop out of the season."""

    def test_dry_month_is_zero_and_unknown_month_is_skipped(self) -> None:
        sites = build_synthetic_sites(one_cell(), default_wet_synthetic_plants())
        climate = xr.Dataset(
            data_vars={
                # Dec: known. Jan: river temperature unknown with water in the river.
                # Feb: river temperature unknown and the river dry.
                "tw": (("time", "lat", "lon"), [[[283.15]], [[np.nan]], [[np.nan]]]),
                "dis": (("time", "lat", "lon"), [[[1.0]], [[1.0]], [[0.0]]]),
                "tas": (("time", "lat", "lon"), [[[290.15]], [[291.15]], [[292.15]]]),
                "hurs": (("time", "lat", "lon"), [[[70.0]], [[70.0]], [[70.0]]]),
                "ps": (("time", "lat", "lon"), [[[101325.0]], [[101325.0]], [[101325.0]]]),
            },
            coords={
                "time": pd.to_datetime(["2000-12-01", "2001-01-01", "2001-02-01"]),
                "lat": [0.25],
                "lon": [0.25],
            },
        )
        for name in ("tw", "tas"):
            climate[name].attrs["units"] = "K"
        climate["ps"].attrs["units"] = "Pa"

        seasonal = evaluate_wet_potential_sites(sites, climate).compute()
        monthly = evaluate_mapped(
            WET_COOLING_KERNEL, sites, climate, duration="native",
            site_id_column="UNITID",
        ).compute()

        for archetype in default_wet_synthetic_plants():
            unit_id = sites.loc[sites["archetype"] == archetype.name, "UNITID"].item()
            december = monthly.isel(time=0).sel(entity=unit_id)
            december_ceiling = (
                3600.0
                * december["available_discharge_m3_per_s"].item()
                / december["withdrawal_intensity_m3_per_mwh"].item()
            )
            got = seasonal["water_limited_capacity_ceiling_mw"].sel(
                season="DJF", year=2001, entity=unit_id
            ).item()
            # Mean of December and the dry February; the unknown January is not in it.
            self.assertAlmostEqual(got, december_ceiling / 2.0, places=6, msg=archetype.name)


class MonthlyDriverEmissionTest(unittest.TestCase):
    """The monthly raw drivers are written without disturbing the seasonal water limit."""

    def test_seasonal_output_matches_seasonal_only_path(self) -> None:
        sites = build_synthetic_sites(one_cell(), default_wet_synthetic_plants())
        climate = djf_climate()

        seasonal_only = evaluate_wet_potential_sites(sites, climate).compute()
        seasonal, _monthly = evaluate_wet_potential_sites_seasonal_and_monthly(
            sites, climate
        )
        seasonal = seasonal.compute()

        xr.testing.assert_identical(
            seasonal["water_limited_capacity_ceiling_mw"],
            seasonal_only["water_limited_capacity_ceiling_mw"],
        )

    def test_december_rolls_into_next_year_djf(self) -> None:
        sites = build_synthetic_sites(one_cell(), default_wet_synthetic_plants())
        # December 2000 belongs to DJF 2001; the monthly labelling must match the
        # seasonal water limit's year convention or the two cannot be joined later.
        climate = djf_climate()

        _seasonal, monthly = evaluate_wet_potential_sites_seasonal_and_monthly(
            sites, climate
        )
        monthly = monthly.compute()

        by_month = {
            int(m): (s, int(y))
            for m, s, y in zip(
                monthly["month"].values,
                monthly["season"].values,
                monthly["year"].values,
                strict=True,
            )
        }
        self.assertEqual(by_month[12], ("DJF", 2001))
        self.assertEqual(by_month[1], ("DJF", 2001))

    def test_monthly_rows_reconstruct_the_seasonal_ceiling(self) -> None:
        cells = one_cell().assign(population="candidate")
        sites = build_synthetic_sites(cells, default_wet_synthetic_plants())
        climate = djf_climate()
        sample = SampledYear(scenario="ssp585", year=2001, warming_level=3.5)
        responses = {
            "available_discharge_m3_per_s": "m3/s",
            "withdrawal_intensity_m3_per_mwh": "m3/MWh",
        }

        seasonal, monthly = evaluate_wet_potential_sites_seasonal_and_monthly(
            sites, climate
        )
        long = _to_long(monthly, sites, sample, "GFDL-ESM4", responses, monthly=True)

        # Every monthly row carries a month; the response set matches the request.
        self.assertTrue(long["month"].notna().all())
        self.assertEqual(set(long["response"]), set(responses))

        unit_id = sites.loc[
            sites["archetype"] == COAL_ONCE_THROUGH, "UNITID"
        ].item()
        rows = long[
            (long["archetype"] == COAL_ONCE_THROUGH) & (long["season"] == "DJF")
        ]
        wide = rows.pivot_table(index="month", columns="response", values="value")
        monthly_ceiling = (
            3600.0
            * wide["available_discharge_m3_per_s"]
            / wide["withdrawal_intensity_m3_per_mwh"]
        )
        got = float(
            seasonal["water_limited_capacity_ceiling_mw"]
            .sel(season="DJF", year=2001, entity=unit_id)
            .item()
        )
        self.assertAlmostEqual(float(monthly_ceiling.mean()), got, places=6)


class UniverseSplitTest(unittest.TestCase):
    """Cells considered: for dry cooling every land cell in an R12 region, for wet
    cooling only the cells with river temperature data."""

    @staticmethod
    def _region_rasters(
        lat: list[float], lon: list[float]
    ) -> tuple[xr.DataArray, xr.DataArray]:
        region = xr.DataArray(
            [["AFR", "AFR"], ["", "AFR"]],
            dims=("lat", "lon"),
            coords={"lat": lat, "lon": lon},
        )
        country = xr.DataArray(
            [["NGA", "NGA"], ["", "NGA"]],
            dims=("lat", "lon"),
            coords={"lat": lat, "lon": lon},
        )
        return region, country

    def _patched_universe(
        self,
        *,
        require_hydrology: bool,
        triver: xr.DataArray | None = None,
        scenario: str | list[str] = "ssp585",
    ) -> pd.DataFrame:
        lat = [0.25, 0.75]
        lon = [0.25, 0.75]
        # (0.25,0.25) river land; (0.25,0.75) arid land (no triver); (0.75,0.25)
        # ocean (no triver, no region); (0.75,0.75) river land.
        if triver is None:
            triver = xr.DataArray(
                [[[1.0, np.nan], [np.nan, 2.0]]],
                dims=("time", "lat", "lon"),
                coords={"time": [0], "lat": lat, "lon": lon},
            )
        region, country = self._region_rasters(lat, lon)
        with (
            mock.patch(
                "impacts.thermoelectric.siting.cells.load_isimip_variable",
                return_value=triver,
            ),
            mock.patch(
                "impacts.thermoelectric.siting.cells.build_region_country_rasters",
                return_value=(region, country),
            ),
        ):
            return build_valid_cells(
                paths=None,
                gcm="GFDL-ESM4",
                scenario=scenario,
                region_paths=None,
                require_hydrology=require_hydrology,
            )

    def test_dry_is_superset_of_wet_by_arid_land(self) -> None:
        wet = self._patched_universe(require_hydrology=True)
        dry = self._patched_universe(require_hydrology=False)

        wet_cells = set(wet["cell_id"])
        dry_cells = set(dry["cell_id"])

        self.assertEqual(len(wet_cells), 2)  # the two river-land cells
        self.assertEqual(len(dry_cells), 3)  # river land + arid land (ocean dropped)
        self.assertTrue(wet_cells.issubset(dry_cells))
        # The extra dry cell is the arid land the hydrology mask drops.
        self.assertEqual(len(dry_cells - wet_cells), 1)

    def test_wet_universe_keeps_cell_finite_at_a_later_time_step(self) -> None:
        lat = [0.25, 0.75]
        lon = [0.25, 0.75]
        triver = xr.DataArray(
            [
                [[np.nan, np.nan], [np.nan, 2.0]],
                [[1.0, np.nan], [np.nan, np.nan]],
            ],
            dims=("time", "lat", "lon"),
            coords={"time": [0, 1], "lat": lat, "lon": lon},
        )

        wet = self._patched_universe(require_hydrology=True, triver=triver)

        self.assertEqual(
            set(zip(wet["grid_lat"], wet["grid_lon"], strict=True)),
            {(0.25, 0.25), (0.75, 0.75)},
        )

    def test_wet_universe_unions_hydrology_across_scenarios(self) -> None:
        lat = [0.25, 0.75]
        lon = [0.25, 0.75]
        by_scenario = {
            "ssp126": xr.DataArray(
                [[[1.0, np.nan], [np.nan, np.nan]]],
                dims=("time", "lat", "lon"),
                coords={"time": [0], "lat": lat, "lon": lon},
            ),
            "ssp370": xr.DataArray(
                [[[np.nan, np.nan], [np.nan, 2.0]]],
                dims=("time", "lat", "lon"),
                coords={"time": [0], "lat": lat, "lon": lon},
            ),
        }
        region, country = self._region_rasters(lat, lon)

        def _load_isimip_variable(_variable, _gcm, scenario, *_args, **_kwargs):
            return by_scenario[scenario]

        with (
            mock.patch(
                "impacts.thermoelectric.siting.cells.load_isimip_variable",
                side_effect=_load_isimip_variable,
            ),
            mock.patch(
                "impacts.thermoelectric.siting.cells.build_region_country_rasters",
                return_value=(region, country),
            ),
        ):
            wet = build_valid_cells(
                paths=None,
                gcm="GFDL-ESM4",
                scenario=["ssp126", "ssp370"],
                region_paths=None,
                require_hydrology=True,
            )

        self.assertEqual(
            set(zip(wet["grid_lat"], wet["grid_lon"], strict=True)),
            {(0.25, 0.25), (0.75, 0.75)},
        )


if __name__ == "__main__":
    unittest.main()
