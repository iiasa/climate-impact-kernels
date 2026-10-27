from __future__ import annotations

import unittest

import pandas as pd
import xarray as xr

from impacts.engine import evaluate_mapped, prepare_climate
from impacts.thermoelectric.dry import (
    DRY_COOLING_KERNEL,
    DryCoolingAssumptions,
    build_dry_cooling_mapping,
    dry_cooling,
)
from impacts.thermoelectric.regions import aggregate_thermoelectric_regions


class DryCoolingContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.unit_cell_mapping = pd.DataFrame(
            {
                "UNITID": [101, 202, 303, 404],
                "LOCATIONID": [10, 20, 30, 40],
                "grid_lat": [23.25, 37.25, 41.25, 48.25],
                "grid_lon": [121.25, 127.25, -87.25, 2.25],
                "mw": [100.0, 200.0, 80.0, 900.0],
            }
        )
        self.cooling_mapping = pd.DataFrame(
            {
                "UNITID": [101, 202, 303, 404, 505],
                "LOCATIONID": [10, 20, 30, 40, 50],
                "country": ["Taiwan", "Korea", "USA", "France", "Canada"],
                "fuel": ["COAL", "LNG", "DIESEL", "NUCLEAR", "COAL"],
                "utype": ["ST", "GT/C", "GT", "ST", "ST"],
                "technology_family": [
                    "steam",
                    "combined_cycle",
                    "gas_turbine",
                    "steam",
                    "steam",
                ],
                "cooling_category": ["air", "air", "air", "air", "cl_fresh"],
                "cooling_physics": ["dry", "dry", "dry", "dry", "recirculating"],
            }
        )
        self.climate = xr.Dataset(
            data_vars={
                "tas": (
                    ("time", "lat", "lon"),
                    [
                        [
                            [10.0, 10.5, 11.0, 11.5],
                            [12.0, 12.5, 13.0, 13.5],
                            [14.0, 14.5, 15.0, 15.5],
                            [16.0, 16.5, 17.0, 17.5],
                        ],
                        [
                            [20.0, 20.5, 21.0, 21.5],
                            [22.0, 22.5, 23.0, 23.5],
                            [24.0, 24.5, 25.0, 25.5],
                            [26.0, 26.5, 27.0, 27.5],
                        ],
                        [
                            [30.0, 30.5, 31.0, 31.5],
                            [32.0, 32.5, 33.0, 33.5],
                            [34.0, 34.5, 35.0, 35.5],
                            [36.0, 36.5, 37.0, 37.5],
                        ],
                    ],
                )
            },
            coords={
                "time": pd.to_datetime(["2001-01-01", "2001-02-01", "2001-06-01"]),
                "lat": [23.25, 37.25, 41.25, 48.25],
                "lon": [-87.25, 2.25, 121.25, 127.25],
            },
        )
        self.climate["tas"].attrs["units"] = "C"

    def test_normalize_dry_cooling_inputs_renames_and_converts_units(self) -> None:
        normalized = prepare_climate(DRY_COOLING_KERNEL, self.climate)

        self.assertEqual(set(normalized.data_vars), {"air_temperature_k"})
        self.assertAlmostEqual(normalized["air_temperature_k"].isel(time=0, lat=0, lon=0).item(), 283.15)

    def test_build_dry_cooling_mapping_classifies_qin_families_and_exclusions(self) -> None:
        mapping = build_dry_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)

        self.assertEqual(mapping["UNITID"].tolist(), [404, 202, 101, 303])
        self.assertEqual(
            mapping["dry_cooling_engine_type"].tolist(),
            ["nuclear_steam", "combined_cycle", "combustible_steam", "no_steam_cycle"],
        )
        self.assertEqual(mapping["dry_cooling_supported"].tolist(), [True, True, True, False])
        self.assertEqual(
            mapping["dry_cooling_support_reason"].tolist(),
            [
                "supported_qin_family",
                "supported_qin_family",
                "supported_qin_family",
                "excluded_no_steam_cycle",
            ],
        )

    def test_build_dry_cooling_mapping_accepts_artifact_schema_without_country(self) -> None:
        mapping = build_dry_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)

        self.assertEqual(mapping["country"].tolist(), ["France", "Korea", "Taiwan", "USA"])

    def test_caller_set_assumption_is_sourced_to_the_caller(self) -> None:
        result = dry_cooling(
            self.unit_cell_mapping,
            self.cooling_mapping,
            self.climate,
            "native",
            assumptions=DryCoolingAssumptions(nuclear_steam_parasitic_load_fraction=0.02),
        )

        self.assertEqual(result.attrs["nuclear_steam_parasitic_load_fraction"], 0.02)
        self.assertEqual(
            result.attrs["nuclear_steam_parasitic_load_fraction_source"],
            "set by the caller; default 0.0304 from EPA 2001, Table 3-19 (dry-tower fan penalty)",
        )
        self.assertEqual(
            result.attrs["unsupported_capacity_factor_source"],
            "assumption: units without a Qin steam-cycle family keep full capacity",
        )

    def test_every_kernel_output_aggregates_to_regions(self) -> None:
        # The regional reduction tables must list every variable the kernel writes.
        for duration in ("native", "seasonal"):
            result = dry_cooling(self.unit_cell_mapping, self.cooling_mapping, self.climate, duration)
            reduced = aggregate_thermoelectric_regions(result, region_set="country")
            self.assertEqual(set(reduced.data_vars) - {"valid_mw"}, set(result.data_vars))

    def test_one_call_matches_the_explicit_path(self) -> None:
        mapping = build_dry_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)
        explicit = evaluate_mapped(DRY_COOLING_KERNEL, mapping, self.climate, duration="seasonal", site_id_column="UNITID")
        one_call = dry_cooling(self.unit_cell_mapping, self.cooling_mapping, self.climate, "seasonal")
        xr.testing.assert_identical(one_call, explicit)

    def test_compute_monthly_dry_cooling_impacts_returns_capacity_outputs(self) -> None:
        mapping = build_dry_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)
        monthly = evaluate_mapped(DRY_COOLING_KERNEL, mapping, self.climate, duration="native", site_id_column="UNITID")

        self.assertEqual(monthly.sizes["entity"], 4)
        self.assertEqual(monthly["capacity_factor"].dims, ("time", "entity"))

        combustible_steam = monthly["capacity_factor"].sel(entity=101)
        combined_cycle = monthly["capacity_factor"].sel(entity=202)
        unsupported = monthly["capacity_factor"].sel(entity=303)
        nuclear = monthly["capacity_factor"].sel(entity=404)

        self.assertGreater(combustible_steam.isel(time=0).item(), combustible_steam.isel(time=2).item())
        self.assertGreater(combined_cycle.isel(time=0).item(), combined_cycle.isel(time=2).item())
        self.assertGreater(nuclear.isel(time=0).item(), nuclear.isel(time=2).item())
        self.assertAlmostEqual(unsupported.isel(time=0).item(), 1.0)
        self.assertAlmostEqual(
            monthly["usable_capacity_mw"].sel(time="2001-01-01", entity=303).item(),
            80.0,
        )
        self.assertAlmostEqual(
            monthly["parasitic_load"].sel(entity=404).item(),
            0.0304,
        )
        # Qin et al. 2023, Methods, eq. 5: p = 1.031 exp(0.019 T_F), T_F in Fahrenheit.
        # The unit's cell is at 16.5 C = 61.7 F: p = 1.031 * exp(1.1723) = 3.3295 inHg.
        self.assertAlmostEqual(
            monthly["backpressure_inhg"].sel(time="2001-01-01", entity=404).item(),
            3.3295,
            places=3,
        )
        self.assertIn("Fahrenheit", monthly.attrs["backpressure_source_note"])
        self.assertIn("3.04%", monthly.attrs["parasitic_load_source_note"])

    def test_aggregate_seasonal_dry_cooling_impacts_uses_climateresponse(self) -> None:
        mapping = build_dry_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)
        # Dec 2000, Jan and Feb 2001 are the three months of DJF 2001.
        climate = self.climate.assign_coords(
            time=pd.to_datetime(["2000-12-01", "2001-01-01", "2001-02-01"])
        )
        seasonal = evaluate_mapped(
            DRY_COOLING_KERNEL, mapping, climate, duration="seasonal",
            site_id_column="UNITID",
        ).compute()

        self.assertEqual(seasonal["capacity_factor"].dims, ("season", "year", "entity"))
        self.assertEqual(seasonal.coords["season"].values.tolist(), ["DJF", "MAM", "JJA", "SON"])
        monthly = evaluate_mapped(DRY_COOLING_KERNEL, mapping, climate, duration="native", site_id_column="UNITID")
        self.assertAlmostEqual(
            seasonal["capacity_factor"].sel(season="DJF", year=2001, entity=101).item(),
            monthly["capacity_factor"].sel(entity=101).mean().item(),
        )
        self.assertAlmostEqual(
            seasonal["capacity_factor"].sel(season="DJF", year=2001, entity=303).item(),
            1.0,
        )

    def test_seasons_missing_a_month_are_not_reported(self) -> None:
        mapping = build_dry_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)
        # Jan, Feb and Jun 2001: DJF lacks December and JJA lacks July and August.
        seasonal = evaluate_mapped(
            DRY_COOLING_KERNEL, mapping, self.climate, duration="seasonal",
            site_id_column="UNITID",
        ).compute()

        for season in ("DJF", "JJA"):
            self.assertTrue(
                seasonal["capacity_factor"].sel(season=season, year=2001).isnull().all().item(),
                season,
            )


# Per family, (air temperature C, p inHg, capacity factor), each worked by hand
# from Qin et al. 2023. Example, combustible steam at 31 C = 87.8 F:
# p = 1.031 exp(0.019 * 87.8) = 5.467 inHg; loss = 0.0063 * 5.467^2 - 0.004 * 5.467
# - 0.0062 = 0.1602; capacity factor = 1 - 0.1602 - 0.0243 = 0.8155.
QIN_EXPECTED = {
    "combustible_steam": ((11.0, 2.7586, 0.9450), (31.0, 5.4670, 0.8155), (36.0, 6.4865, 0.7428)),
    "nuclear_steam": ((11.0, 2.7586, 0.9531), (31.0, 5.4670, 0.8744), (36.0, 6.4865, 0.8406)),
    "non_combustible_steam": ((11.0, 2.7586, 0.9622), (31.0, 5.4670, 0.8483), (36.0, 6.4865, 0.8138)),
    "combined_cycle": ((11.0, 2.7586, 0.9787), (31.0, 5.4670, 0.8964), (36.0, 6.4865, 0.8565)),
}
QIN_UNITS = {
    "combustible_steam": ("COAL", "steam"),
    "nuclear_steam": ("NUCLEAR", "steam"),
    "non_combustible_steam": ("GEOTHERMAL", "steam"),
    "combined_cycle": ("LNG", "combined_cycle"),
}


class QinEquationsTest(unittest.TestCase):
    """Capacity factor against Qin et al. (2023) worked by hand.

    Efficiency loss is a polynomial in the exhaust backpressure p, inches Hg:
    combustible steam eq. 1, nuclear steam eq. 2, non-nuclear non-combustible
    steam eq. 3, combined cycle eq. 4 (Methods). For dry cooling,
    p = 1.031 exp(0.019 T_F), T_F the air temperature in Fahrenheit (eq. 5). Capacity factor is
    1 - loss - fan load, the fan load being the Dry row of the Methods table
    (combustible 2.43%, non-combustible 0.56%, combined cycle 0.81%). The table has
    no nuclear column; the library's nuclear fan load is 3.04%. The library writes the
    equations as sums of exponentials of air temperature with coefficients rounded
    to three figures, which moves the capacity factor by up to 1.1e-3, so the tolerance is 1.5e-3.
    """

    def test_capacity_factor_and_backpressure_follow_the_published_equations(self) -> None:
        families = list(QIN_EXPECTED)
        cells = pd.DataFrame(
            {
                "UNITID": range(1, 5),
                "LOCATIONID": range(1, 5),
                "grid_lat": [0.25] * 4,
                "grid_lon": [0.25] * 4,
                "mw": [100.0] * 4,
            }
        )
        cooling = pd.DataFrame(
            {
                "UNITID": range(1, 5),
                "LOCATIONID": range(1, 5),
                "country": ["X"] * 4,
                "fuel": [QIN_UNITS[family][0] for family in families],
                "utype": ["ST"] * 4,
                "technology_family": [QIN_UNITS[family][1] for family in families],
                "cooling_category": ["air"] * 4,
                "cooling_physics": ["dry"] * 4,
            }
        )
        temperatures = [11.0, 31.0, 36.0]
        climate = xr.Dataset(
            {"tas": (("time", "lat", "lon"), [[[t]] for t in temperatures])},
            coords={
                "time": pd.date_range("2001-01-01", periods=3, freq="MS"),
                "lat": [0.25],
                "lon": [0.25],
            },
        )
        climate["tas"].attrs["units"] = "C"

        result = dry_cooling(cells, cooling, climate, "native")

        for unit_id, family in enumerate(families, start=1):
            for step, (_, pressure, capacity_factor) in enumerate(QIN_EXPECTED[family]):
                with self.subTest(family=family, temperature_c=temperatures[step]):
                    at = result.isel(time=step).sel(entity=unit_id)
                    self.assertAlmostEqual(at["backpressure_inhg"].item(), pressure, places=3)
                    self.assertAlmostEqual(at["capacity_factor"].item(), capacity_factor, delta=1.5e-3)


class DryCoolingDaskTest(DryCoolingContractTest):
    """Re-run all dry-cooling tests with dask-backed climate arrays.

    Kernels must produce correct results from chunked inputs.
    Assertions call .compute() where needed since .item() is not
    available on dask arrays.
    """

    def setUp(self) -> None:
        super().setUp()
        self.climate = self.climate.chunk({"time": 1})

    def _compute(self, result):
        """Materialize dask arrays before scalar assertions."""
        return result.compute() if hasattr(result, "compute") else result

    def test_normalize_dry_cooling_inputs_renames_and_converts_units(self) -> None:
        normalized = prepare_climate(DRY_COOLING_KERNEL, self.climate).compute()
        self.assertEqual(set(normalized.data_vars), {"air_temperature_k"})
        self.assertAlmostEqual(normalized["air_temperature_k"].isel(time=0, lat=0, lon=0).item(), 283.15)

    def test_compute_monthly_dry_cooling_impacts_returns_capacity_outputs(self) -> None:
        mapping = build_dry_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)
        monthly = evaluate_mapped(DRY_COOLING_KERNEL, mapping, self.climate, duration="native", site_id_column="UNITID").compute()

        self.assertEqual(monthly.sizes["entity"], 4)
        combustible_steam = monthly["capacity_factor"].sel(entity=101)
        combined_cycle = monthly["capacity_factor"].sel(entity=202)
        unsupported = monthly["capacity_factor"].sel(entity=303)
        nuclear = monthly["capacity_factor"].sel(entity=404)

        self.assertGreater(combustible_steam.isel(time=0).item(), combustible_steam.isel(time=2).item())
        self.assertGreater(combined_cycle.isel(time=0).item(), combined_cycle.isel(time=2).item())
        self.assertGreater(nuclear.isel(time=0).item(), nuclear.isel(time=2).item())
        self.assertAlmostEqual(unsupported.isel(time=0).item(), 1.0)

    def test_aggregate_seasonal_dry_cooling_impacts_uses_climateresponse(self) -> None:
        mapping = build_dry_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)
        seasonal = evaluate_mapped(DRY_COOLING_KERNEL, mapping, self.climate, duration="seasonal", site_id_column="UNITID").compute()
        self.assertEqual(seasonal["capacity_factor"].dims, ("season", "year", "entity"))


if __name__ == "__main__":
    unittest.main()
