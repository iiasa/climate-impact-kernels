from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from impacts.engine import evaluate_mapped, prepare_climate
from impacts.thermoelectric.regions import aggregate_thermoelectric_regions
from impacts.thermoelectric.wet import (
    FUEL_EFFICIENCY_CURVE_PARAMETERS,
    WET_COOLING_KERNEL,
    WetCoolingAssumptions,
    _moist_air_enthalpy_kj_per_kg,
    assign_saline_intakes,
    build_wet_cooling_mapping,
    fuel_efficiency,
    wet_cooling,
)


class WetCoolingContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.unit_cell_mapping = pd.DataFrame(
            {
                "UNITID": [101, 102, 202, 303, 404, 505],
                "LOCATIONID": [10, 11, 20, 30, 40, 50],
                "country": ["Taiwan", "Taiwan", "Korea", "USA", "Japan", "Japan"],
                "grid_lat": [23.25, 23.25, 37.25, 37.25, 37.25, 23.25],
                "grid_lon": [121.25, 121.25, 127.25, 127.25, 127.25, 121.25],
                "mw": [100.0, 200.0, 150.0, 80.0, 120.0, 70.0],
            }
        )
        self.cooling_mapping = pd.DataFrame(
            {
                "UNITID": [101, 102, 202, 303, 404, 505],
                "LOCATIONID": [10, 11, 20, 30, 40, 50],
                "country": ["Taiwan", "Taiwan", "Korea", "USA", "Japan", "Japan"],
                "fuel": ["COAL", "UR", "LNG", "COAL", "WSTH", "KERO"],
                "technology_family": [
                    "steam",
                    "steam",
                    "combined_cycle",
                    "steam",
                    "steam",
                    "combined_cycle",
                ],
                "cooling_category": ["ot_fresh", "ot_saline", "cl_fresh", "air", "cl_fresh", "ot_saline"],
                "cooling_physics": [
                    "once_through",
                    "once_through",
                    "recirculating",
                    "dry",
                    "recirculating",
                    "once_through",
                ],
                "requires_water": [True, True, True, False, True, True],
            }
        )
        self.climate = xr.Dataset(
            data_vars={
                "tw": (
                    ("time", "lat", "lon"),
                    [
                        [[288.15, 293.15], [293.15, 294.15]],
                        [[292.15, 295.15], [295.15, 296.15]],
                        [[297.15, 297.15], [297.15, 298.15]],
                    ],
                ),
                "dis": (
                    ("time", "lat", "lon"),
                    [
                        [[1.0, 0.30], [0.30, 0.50]],
                        [[0.6, 0.20], [0.20, 0.40]],
                        [[0.3, 0.10], [0.10, 0.20]],
                    ],
                ),
                "tas": (
                    ("time", "lat", "lon"),
                    [
                        [[298.15, 300.15], [300.15, 301.15]],
                        [[300.15, 302.15], [302.15, 303.15]],
                        [[303.15, 304.15], [304.15, 305.15]],
                    ],
                ),
                "hurs": (
                    ("time", "lat", "lon"),
                    [
                        [[60.0, 65.0], [65.0, 70.0]],
                        [[70.0, 75.0], [75.0, 80.0]],
                        [[80.0, 82.0], [82.0, 85.0]],
                    ],
                ),
                "ps": (
                    ("time", "lat", "lon"),
                    [
                        [[101325.0, 100500.0], [100500.0, 100000.0]],
                        [[101325.0, 100000.0], [100000.0, 99500.0]],
                        [[101325.0, 99500.0], [99500.0, 99000.0]],
                    ],
                ),
            },
            coords={
                "time": pd.to_datetime(["2001-01-01", "2001-02-01", "2001-06-01"]),
                "lat": [23.25, 37.25],
                "lon": [121.25, 127.25],
            },
        )
        # Sea-surface temperature is finite only in the cell of the saline units.
        self.climate["tos"] = (
            ("time", "lat", "lon"),
            [
                [[20.0, np.nan], [np.nan, np.nan]],
                [[21.0, np.nan], [np.nan, np.nan]],
                [[25.0, np.nan], [np.nan, np.nan]],
            ],
        )
        self.climate["tw"].attrs["units"] = "K"
        self.climate["tas"].attrs["units"] = "K"
        self.climate["ps"].attrs["units"] = "Pa"
        self.climate["tos"].attrs["units"] = "degC"

    def _mapping(self) -> pd.DataFrame:
        mapping = build_wet_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)
        return assign_saline_intakes(mapping, self.climate["tos"])

    def test_caller_set_assumption_is_sourced_to_the_caller(self) -> None:
        result = wet_cooling(
            self.unit_cell_mapping,
            self.cooling_mapping,
            self.climate,
            "native",
            assumptions=WetCoolingAssumptions(concentration_cycles=3.0),
        )
        fields = WetCoolingAssumptions.__dataclass_fields__

        self.assertEqual(result.attrs["concentration_cycles"], 3.0)
        self.assertEqual(
            result.attrs["concentration_cycles_source"],
            f"set by the caller; default 5.0 from {fields['concentration_cycles'].metadata['source']}",
        )
        self.assertEqual(result.attrs["concentration_cycles_unit"], "1")
        # Fields left at their default keep the default's source.
        self.assertEqual(result.attrs["tower_approach_c_source"], fields["tower_approach_c"].metadata["source"])

    def test_null_requires_water_is_refused(self) -> None:
        cooling = self.cooling_mapping.astype({"requires_water": object})
        cooling.loc[[1, 3], "requires_water"] = None

        with self.assertRaisesRegex(
            ValueError,
            r"cooling_mapping has null requires_water for 2 units \(first UNITIDs: \[102, 303\]\); "
            "set it to True or False for every unit",
        ):
            build_wet_cooling_mapping(self.unit_cell_mapping, cooling)

    def test_water_cooling_with_unsupported_physics_is_refused(self) -> None:
        cooling = self.cooling_mapping.assign(
            cooling_physics=["once_through", "once_through", "recirculating", "dry", "hybrid", "once_through"]
        )

        with self.assertRaisesRegex(ValueError, r"unsupported wet cooling physics: \['hybrid'\]"):
            build_wet_cooling_mapping(self.unit_cell_mapping, cooling)

    def test_every_kernel_output_aggregates_to_regions(self) -> None:
        # The regional reduction tables must list every variable the kernel writes.
        monthly = wet_cooling(self.unit_cell_mapping, self.cooling_mapping, self.climate, "native")
        seasonal = wet_cooling(self.unit_cell_mapping, self.cooling_mapping, self.climate, "seasonal")

        for result in (monthly, seasonal):
            reduced = aggregate_thermoelectric_regions(result, region_set="country")
            self.assertEqual(set(reduced.data_vars) - {"valid_mw"}, set(result.data_vars))

    def test_one_call_matches_the_explicit_path(self) -> None:
        explicit = evaluate_mapped(WET_COOLING_KERNEL, self._mapping(), self.climate, duration="native", site_id_column="UNITID")
        one_call = wet_cooling(self.unit_cell_mapping, self.cooling_mapping, self.climate, "native")
        xr.testing.assert_identical(one_call, explicit)

    def test_normalize_wet_cooling_inputs_renames_and_converts_units(self) -> None:
        normalized = prepare_climate(WET_COOLING_KERNEL, self.climate)

        self.assertEqual(
            set(normalized.data_vars),
            {
                "water_temperature",
                "discharge",
                "air_temperature",
                "relative_humidity",
                "surface_pressure",
                "sea_surface_temperature",
            },
        )
        self.assertAlmostEqual(normalized["water_temperature"].isel(time=0, lat=0, lon=0).item(), 15.0)
        self.assertAlmostEqual(normalized["surface_pressure"].isel(time=0, lat=1, lon=0).item(), 100.5)

    def test_build_wet_cooling_mapping_adds_fuel_coefficients_and_cell_shares(self) -> None:
        mapping = build_wet_cooling_mapping(self.unit_cell_mapping, self.cooling_mapping)

        self.assertEqual(mapping["UNITID"].tolist(), [404, 505, 202, 101, 102])
        self.assertEqual(mapping["wet_fuel_type"].tolist(), ["steam_bottoming", "oil", "gas", "coal", "nuclear"])
        # Unit 101 is the only river-drawing unit in its cell; the two saline units there take no share.
        self.assertAlmostEqual(mapping.loc[mapping["UNITID"] == 101, "unit_share_ratio"].item(), 1.0)
        self.assertAlmostEqual(mapping.loc[mapping["UNITID"] == 101, "cell_total_mw"].item(), 100.0)
        self.assertAlmostEqual(mapping.loc[mapping["UNITID"] == 102, "unit_share_ratio"].item(), 0.0)
        self.assertAlmostEqual(mapping.loc[mapping["UNITID"] == 505, "unit_share_ratio"].item(), 0.0)
        self.assertAlmostEqual(mapping.loc[mapping["UNITID"] == 102, "thermal_efficiency"].item(), 0.35)
        # Steam units of 120 and 70 MW sit in the range of fossil plants, 25-45% (the curve itself
        # is tested in FuelEfficiencyCurveTest).
        for unit_id in (404, 505):
            self.assertTrue(0.25 < mapping.loc[mapping["UNITID"] == unit_id, "thermal_efficiency"].item() < 0.45)
        self.assertAlmostEqual(mapping.loc[mapping["UNITID"] == 101, "heat_rejection_coefficient"].item(), 0.12)
        self.assertAlmostEqual(mapping.loc[mapping["UNITID"] == 202, "evaporation_coefficient"].item(), 0.0088)
        self.assertAlmostEqual(mapping.loc[mapping["UNITID"] == 404, "evaporation_coefficient"].item(), 0.0042)
        self.assertAlmostEqual(mapping.loc[mapping["UNITID"] == 505, "evaporation_coefficient"].item(), 0.0088)

    def test_build_wet_cooling_mapping_accepts_artifact_schema_without_country(self) -> None:
        mapping = build_wet_cooling_mapping(
            self.unit_cell_mapping.drop(columns=["country"]),
            self.cooling_mapping,
        )

        self.assertEqual(mapping["country"].tolist(), ["Japan", "Japan", "Korea", "Taiwan", "Taiwan"])

    def test_compute_monthly_wet_cooling_impacts_matches_li_equations(self) -> None:
        mapping = self._mapping()
        monthly = evaluate_mapped(WET_COOLING_KERNEL, mapping, self.climate, duration="native", site_id_column="UNITID")

        self.assertEqual(monthly.sizes["entity"], 5)
        self.assertEqual(monthly.coords["UNITID"].values.tolist(), [404, 505, 202, 101, 102])
        self.assertEqual(monthly["capacity_factor"].dims, ("time", "entity"))

        # Coal unit 101, 100 MW, once-through, river at 15 C with 1 m3/s in January.
        # Its efficiency from the library's size curve is 0.2924 and its heat-loss
        # coefficient 0.12 (the paper cites earlier studies for both and prints
        # neither). Eq. 1: WI_ot = 3.6e6 * (1 - 0.2924 - 0.12) / 0.2924
        # / (1000 * 4.184 * min(32 - 15, 10)) = 172.92 m3/MWh. Eq. 4, discharge
        # per unit: (1 - 0.8) * 1 * share 1 = 0.2 m3/s. Eq. 3:
        # UC = min(3600 * 0.2 / 172.92, 100) = 4.164 MW.
        january = monthly.sel(time="2001-01-01")
        self.assertAlmostEqual(january["withdrawal_intensity_m3_per_mwh"].sel(entity=101).item(), 172.92, places=1)
        self.assertAlmostEqual(january["unit_available_discharge_m3_per_s"].sel(entity=101).item(), 0.2, places=8)
        self.assertAlmostEqual(january["usable_capacity_mw"].sel(entity=101).item(), 4.164, places=2)

        # Saline nuclear unit 102 draws seawater, so the river does not limit it. In
        # June the sea is at 25 C: the allowed temperature rise min(32 - 25, 10) = 7 K
        # is positive, so it runs at nameplate (200 MW).
        self.assertFalse(monthly["discharge_constraint_applies"].sel(entity=102).item())
        self.assertAlmostEqual(monthly["usable_capacity_mw"].sel(time="2001-06-01", entity=102).item(), 200.0, places=8)
        self.assertAlmostEqual(monthly["capacity_factor"].sel(time="2001-06-01", entity=102).item(), 1.0, places=8)
        self.assertLess(monthly["capacity_factor"].sel(time="2001-06-01", entity=101).item(), 1.0)

        # LNG unit 202, 150 MW, recirculating, in a cell shared with the 120 MW unit
        # 404: river 21 C, air 28 C, relative humidity 70%, pressure 100 kPa, 0.5 m3/s.
        # Efficiency 0.4696. Eq. 4: 0.5 * (1 - 0.8) * 150 / 270 = 0.05556 m3/s.
        # Eq. 2 (psychrometric terms as in the library) gives WI_rc = 1.590 m3/MWh,
        # so eq. 3 gives UC = min(3600 * 0.05556 / 1.5896, 150) = 125.8 MW.
        self.assertAlmostEqual(january["unit_available_discharge_m3_per_s"].sel(entity=202).item(), 0.5 * 0.2 * 150 / 270, places=8)
        self.assertAlmostEqual(january["withdrawal_intensity_m3_per_mwh"].sel(entity=202).item(), 1.5896, places=3)
        self.assertAlmostEqual(january["usable_capacity_mw"].sel(entity=202).item(), 125.82, places=1)

        # Eq. 5: usable capacity ratio = UC / NC.
        self.assertAlmostEqual(monthly["capacity_factor"].sel(time="2001-06-01", entity=102).item(), 1.0, places=8)
        self.assertAlmostEqual(january["capacity_factor"].sel(entity=202).item(), 125.82 / 150.0, places=3)
        self.assertIn("ot_saline once-through units draw seawater", monthly.attrs["ot_saline_policy"])

    def test_aggregate_seasonal_wet_cooling_impacts_uses_climateresponse(self) -> None:
        mapping = self._mapping()
        # Dec 2000, Jan and Feb 2001 are the three months of DJF 2001.
        climate = self.climate.assign_coords(
            time=pd.to_datetime(["2000-12-01", "2001-01-01", "2001-02-01"])
        )
        seasonal = evaluate_mapped(
            WET_COOLING_KERNEL, mapping, climate, duration="seasonal",
            site_id_column="UNITID",
        ).compute()

        self.assertEqual(seasonal["capacity_factor"].dims, ("season", "year", "entity"))
        self.assertEqual(seasonal.coords["season"].values.tolist(), ["DJF", "MAM", "JJA", "SON"])
        monthly = evaluate_mapped(WET_COOLING_KERNEL, mapping, climate, duration="native", site_id_column="UNITID")
        expected_djf = monthly["capacity_factor"].sel(entity=101).mean().item()
        self.assertAlmostEqual(
            seasonal["capacity_factor"].sel(season="DJF", year=2001, entity=101).item(),
            expected_djf,
        )

    def test_seasons_missing_a_month_are_not_reported(self) -> None:
        mapping = self._mapping()
        # Jan, Feb and Jun 2001: DJF lacks December and JJA lacks July and August.
        seasonal = evaluate_mapped(
            WET_COOLING_KERNEL, mapping, self.climate, duration="seasonal",
            site_id_column="UNITID",
        ).compute()

        for season in ("DJF", "JJA"):
            self.assertTrue(
                seasonal["capacity_factor"].sel(season=season, year=2001).isnull().all().item(),
                season,
            )


class WetCoolingDryRiverTest(unittest.TestCase):
    """Zero discharge means zero capacity for units that draw on the river."""

    def test_dry_river_gives_zero_capacity_whatever_the_river_temperature(self) -> None:
        # Month 0: dry river, temperature undefined. Month 1: dry river, warm.
        # Month 2: wet river, temperature known (control).
        climate = _one_cell_climate(
            river_temperature_c=[math.nan, 20.0, 20.0],
            discharge=[0.0, 0.0, 5.0],
        )
        monthly = _one_cell_run(climate)

        for unit_id in (1, 2, 3):
            for month in (0, 1):
                usable = monthly["usable_capacity_mw"].isel(time=month).sel(entity=unit_id).item()
                self.assertEqual(usable, 0.0, (unit_id, month))
                factor = monthly["capacity_factor"].isel(time=month).sel(entity=unit_id).item()
                self.assertEqual(factor, 0.0, (unit_id, month))
        # With water in the river the same units run.
        for unit_id in (1, 2, 3):
            self.assertGreater(
                monthly["usable_capacity_mw"].isel(time=2).sel(entity=unit_id).item(), 0.0
            )
        # Saline once-through units never drew on the river.
        self.assertEqual(monthly["usable_capacity_mw"].isel(time=1).sel(entity=4).item(), 200.0)


class WetCoolingMissingRiverTemperatureTest(unittest.TestCase):
    """An unknown river temperature stays unknown; it is not booked as zero or infinite."""

    def test_missing_river_temperature_propagates_nan(self) -> None:
        climate = _one_cell_climate(
            river_temperature_c=[math.nan, 20.0],
            discharge=[5.0, 5.0],
        )
        monthly = _one_cell_run(climate)

        unknown = monthly.isel(time=0)
        known = monthly.isel(time=1)
        # Every river unit is unknown rather than 0.
        for unit_id in (1, 2, 3):
            self.assertTrue(math.isnan(unknown["usable_capacity_mw"].sel(entity=unit_id).item()), unit_id)
            self.assertTrue(math.isnan(unknown["capacity_factor"].sel(entity=unit_id).item()), unit_id)
        # Once-through intensity is NaN, not infinite.
        for unit_id in (1, 3):
            self.assertTrue(
                math.isnan(unknown["withdrawal_intensity_m3_per_mwh"].sel(entity=unit_id).item()), unit_id
            )
        # The saline unit reads the sea, so a missing river temperature does not touch it.
        self.assertEqual(unknown["usable_capacity_mw"].sel(entity=4).item(), 200.0)
        # With a known temperature the same units are finite.
        for unit_id in (1, 2, 3, 4):
            self.assertTrue(math.isfinite(known["usable_capacity_mw"].sel(entity=unit_id).item()), unit_id)

    def test_dry_river_still_gives_zero_for_river_units_when_temperature_is_missing(self) -> None:
        climate = _one_cell_climate(river_temperature_c=[math.nan], discharge=[0.0])
        monthly = _one_cell_run(climate)

        for unit_id in (1, 2, 3):
            self.assertEqual(monthly["usable_capacity_mw"].isel(time=0).sel(entity=unit_id).item(), 0.0)
        # The saline unit does not use the river; it runs on the sea.
        self.assertEqual(monthly["usable_capacity_mw"].isel(time=0).sel(entity=4).item(), 200.0)


class SalineIntakeTest(unittest.TestCase):
    """Saline once-through units take their temperature limits from the sea, not the river."""

    def test_saline_headroom_follows_sea_temperature_not_river(self) -> None:
        # Month 0: river at the 32 C cap, sea cool. Month 1: river cool, sea at 33 C.
        climate = _one_cell_climate(
            river_temperature_c=[32.0, 15.0],
            discharge=[5.0, 5.0],
            sea_temperature_c=[20.0, 33.0],
        )
        monthly = _one_cell_run(climate)

        self.assertEqual(monthly["usable_capacity_mw"].isel(time=0).sel(entity=4).item(), 200.0)
        self.assertEqual(monthly["usable_capacity_mw"].isel(time=1).sel(entity=4).item(), 0.0)
        # Freshwater once-through units still read the river: no allowed temperature rise at 32 C.
        self.assertEqual(monthly["usable_capacity_mw"].isel(time=0).sel(entity=1).item(), 0.0)
        self.assertEqual(monthly["intake_temperature"].isel(time=1).sel(entity=4).item(), 33.0)
        self.assertEqual(monthly["intake_temperature"].isel(time=1).sel(entity=1).item(), 15.0)
        # The allowed temperature rise is the 32 C cap less the intake, at most the
        # 10 K rise.
        self.assertAlmostEqual(monthly["temperature_headroom_c"].isel(time=0).sel(entity=4).item(), 10.0)
        self.assertEqual(monthly["temperature_headroom_c"].isel(time=0).sel(entity=1).item(), 0.0)

    def test_saline_units_need_sea_surface_temperature(self) -> None:
        climate = _one_cell_climate(river_temperature_c=[20.0], discharge=[5.0]).drop_vars("tos")
        with self.assertRaisesRegex(ValueError, "sea-surface temperature"):
            evaluate_mapped(WET_COOLING_KERNEL, _one_cell_fleet(), climate, duration="native", site_id_column="UNITID")

    def test_river_only_fleet_needs_no_sea_surface_temperature(self) -> None:
        climate = _one_cell_climate(river_temperature_c=[20.0], discharge=[5.0]).drop_vars("tos")
        fleet = _one_cell_fleet()
        fleet = fleet.loc[fleet["cooling_category"] != "ot_saline"]
        monthly = evaluate_mapped(WET_COOLING_KERNEL, fleet, climate, duration="native", site_id_column="UNITID")
        self.assertEqual(monthly.sizes["entity"], 3)

    def test_intake_is_nearest_ocean_cell_within_reach(self) -> None:
        # Two cells on the equator 0.5 degrees (about 56 km) apart; only the east one is sea.
        sea = xr.DataArray(
            [[[np.nan, 21.0]], [[np.nan, 22.0]]],
            dims=("time", "lat", "lon"),
            coords={"time": pd.date_range("2001-01-01", periods=2, freq="MS"), "lat": [0.25], "lon": [10.25, 10.75]},
        )
        fleet = _one_cell_fleet().assign(grid_lat=0.25, grid_lon=10.25)

        near = assign_saline_intakes(fleet, sea)
        saline = near["cooling_category"] == "ot_saline"
        self.assertEqual(near.loc[saline, ["intake_lat", "intake_lon"]].values.tolist(), [[0.25, 10.75]])
        self.assertAlmostEqual(near.loc[saline, "intake_distance_km"].item(), 55.6, delta=0.2)
        self.assertTrue(near.loc[~saline, ["intake_lat", "intake_lon", "intake_distance_km"]].isna().all().all())

        far = assign_saline_intakes(fleet, sea, max_distance_km=50.0)
        self.assertTrue(far.loc[saline, ["intake_lat", "intake_lon"]].isna().all().all())
        self.assertAlmostEqual(far.loc[saline, "intake_distance_km"].item(), 55.6, delta=0.2)

    def test_unit_without_intake_has_unknown_capacity(self) -> None:
        climate = _one_cell_climate(river_temperature_c=[20.0], discharge=[5.0])
        fleet = assign_saline_intakes(_one_cell_fleet(), climate["tos"], max_distance_km=-1.0)
        monthly = evaluate_mapped(WET_COOLING_KERNEL, fleet, climate, duration="native", site_id_column="UNITID").compute()
        self.assertTrue(math.isnan(monthly["usable_capacity_mw"].isel(time=0).sel(entity=4).item()))
        self.assertGreater(monthly["usable_capacity_mw"].isel(time=0).sel(entity=1).item(), 0.0)


def _one_cell_fleet() -> pd.DataFrame:
    """Four coal units in one cell: fresh once-through, recirculating, another fresh, saline."""

    unit_cell_mapping = pd.DataFrame(
        {
            "UNITID": [1, 2, 3, 4],
            "LOCATIONID": [10, 10, 10, 10],
            "country": ["Japan"] * 4,
            "grid_lat": [35.25] * 4,
            "grid_lon": [139.25] * 4,
            "mw": [100.0, 150.0, 300.0, 200.0],
        }
    )
    cooling_mapping = pd.DataFrame(
        {
            "UNITID": [1, 2, 3, 4],
            "LOCATIONID": [10, 10, 10, 10],
            "country": ["Japan"] * 4,
            "fuel": ["COAL"] * 4,
            "technology_family": ["steam"] * 4,
            "cooling_category": ["ot_fresh", "cl_fresh", "ot_fresh", "ot_saline"],
            "cooling_physics": ["once_through", "recirculating", "once_through", "once_through"],
            "requires_water": [True] * 4,
        }
    )
    return build_wet_cooling_mapping(unit_cell_mapping, cooling_mapping)


def _one_cell_climate(
    *,
    river_temperature_c: list[float],
    discharge: list[float],
    relative_humidity: float = 60.0,
    sea_temperature_c: list[float] | None = None,
) -> xr.Dataset:
    """Monthly inputs for the cell of ``_one_cell_fleet``; water temperatures in degrees Celsius.

    The cell is coastal: its sea-surface temperature defaults to 20 C every month.
    """

    steps = len(discharge)
    shape = (steps, 1, 1)

    def field(values: list[float]) -> tuple[tuple[str, str, str], np.ndarray]:
        return (("time", "lat", "lon"), np.asarray(values, dtype=float).reshape(shape))

    climate = xr.Dataset(
        data_vars={
            "tw": field([value + 273.15 for value in river_temperature_c]),
            "tos": field(sea_temperature_c if sea_temperature_c is not None else [20.0] * steps),
            "dis": field(discharge),
            "tas": field([298.15] * steps),
            "hurs": field([relative_humidity] * steps),
            "ps": field([101325.0] * steps),
        },
        coords={
            "time": pd.date_range("2001-01-01", periods=steps, freq="MS"),
            "lat": [35.25],
            "lon": [139.25],
        },
    )
    climate["tw"].attrs["units"] = "K"
    climate["tas"].attrs["units"] = "K"
    climate["ps"].attrs["units"] = "Pa"
    climate["tos"].attrs["units"] = "degC"
    return climate


def _one_cell_run(climate: xr.Dataset) -> xr.Dataset:
    mapping = assign_saline_intakes(_one_cell_fleet(), climate["tos"])
    return evaluate_mapped(WET_COOLING_KERNEL, mapping, climate, duration="native", site_id_column="UNITID").compute()


class WetCoolingSalineShareTest(unittest.TestCase):
    """Saline once-through units do not take a share of the river."""

    def test_freshwater_units_share_the_river_without_saline_neighbours(self) -> None:
        mapping = _one_cell_fleet()  # fresh 100 and 300 MW, recirculating 150 MW, saline 200 MW

        shares = mapping.set_index("UNITID")["unit_share_ratio"]
        self.assertAlmostEqual(shares[1], 100.0 / 550.0)
        self.assertAlmostEqual(shares[2], 150.0 / 550.0)
        self.assertAlmostEqual(shares[3], 300.0 / 550.0)
        self.assertEqual(shares[4], 0.0)
        self.assertAlmostEqual(shares[[1, 2, 3]].sum(), 1.0)
        self.assertTrue((mapping["cell_total_mw"] == 550.0).all())

    def test_freshwater_unit_gets_the_whole_discharge_next_to_a_saline_unit(self) -> None:
        unit_cell_mapping = pd.DataFrame(
            {
                "UNITID": [1, 2],
                "LOCATIONID": [10, 10],
                "grid_lat": [35.25, 35.25],
                "grid_lon": [139.25, 139.25],
                "mw": [100.0, 400.0],
            }
        )
        cooling_mapping = pd.DataFrame(
            {
                "UNITID": [1, 2],
                "LOCATIONID": [10, 10],
                "country": ["Japan", "Japan"],
                "fuel": ["COAL", "COAL"],
                "technology_family": ["steam", "steam"],
                "cooling_category": ["ot_fresh", "ot_saline"],
                "cooling_physics": ["once_through", "once_through"],
                "requires_water": [True, True],
            }
        )
        climate = _one_cell_climate(river_temperature_c=[20.0], discharge=[10.0])
        mapping = assign_saline_intakes(
            build_wet_cooling_mapping(unit_cell_mapping, cooling_mapping), climate["tos"]
        )
        monthly = evaluate_mapped(
            WET_COOLING_KERNEL, mapping, climate, duration="native",
            site_id_column="UNITID",
        ).compute()

        # The saline unit is 80% of the cell's MW, but it does not draw on the river.
        self.assertAlmostEqual(
            monthly["unit_available_discharge_m3_per_s"].isel(time=0).sel(entity=1).item(), 2.0
        )
        self.assertEqual(
            monthly["unit_available_discharge_m3_per_s"].isel(time=0).sel(entity=2).item(), 0.0
        )

    def test_saline_only_cell_has_finite_zero_shares(self) -> None:
        unit_cell_mapping = pd.DataFrame(
            {
                "UNITID": [1],
                "LOCATIONID": [10],
                "grid_lat": [35.25],
                "grid_lon": [139.25],
                "mw": [400.0],
            }
        )
        cooling_mapping = pd.DataFrame(
            {
                "UNITID": [1],
                "LOCATIONID": [10],
                "country": ["Japan"],
                "fuel": ["COAL"],
                "technology_family": ["steam"],
                "cooling_category": ["ot_saline"],
                "cooling_physics": ["once_through"],
                "requires_water": [True],
            }
        )

        mapping = build_wet_cooling_mapping(unit_cell_mapping, cooling_mapping)

        self.assertEqual(mapping["unit_share_ratio"].tolist(), [0.0])
        self.assertEqual(mapping["cell_total_mw"].tolist(), [0.0])


class WetCoolingSaturatedAirTest(unittest.TestCase):
    """Relative humidity of exactly 100% leaves the recirculating intensity defined."""

    def _recirculating_intensity(self, relative_humidity: float) -> float:
        climate = _one_cell_climate(
            river_temperature_c=[20.0],
            discharge=[5.0],
            relative_humidity=relative_humidity,
        )
        monthly = _one_cell_run(climate)
        return monthly["withdrawal_intensity_m3_per_mwh"].isel(time=0).sel(entity=2).item()

    def test_saturated_air_has_a_finite_intensity_continuous_with_its_neighbours(self) -> None:
        saturated = self._recirculating_intensity(100.0)
        below = self._recirculating_intensity(99.9)
        above = self._recirculating_intensity(100.1)

        self.assertTrue(math.isfinite(saturated))
        self.assertGreater(saturated, 0.0)
        # The intensity varies smoothly through 100%, so it sits between its neighbours.
        self.assertAlmostEqual(saturated, below, delta=0.01 * below)
        self.assertAlmostEqual(saturated, above, delta=0.01 * above)

    def test_saturated_air_gives_finite_capacity(self) -> None:
        climate = _one_cell_climate(
            river_temperature_c=[20.0], discharge=[5.0], relative_humidity=100.0
        )
        monthly = _one_cell_run(climate)

        usable = monthly["usable_capacity_mw"].isel(time=0).sel(entity=2).item()
        self.assertTrue(math.isfinite(usable))
        self.assertGreater(usable, 0.0)


class FuelEfficiencyCurveTest(unittest.TestCase):
    """The size curve eta(MW) = a2 + (a1 - a2) / (1 + (MW / x0)^power), in percent.

    The paper takes efficiency from earlier studies and prints neither the curve nor
    its parameters, so the tests check what the form implies: a rise with unit size
    from a1 (small units) to a2 (large units), the midpoint at MW = x0, and a range
    that fits real plants.
    """

    def test_efficiency_rises_with_unit_size_between_the_two_limits(self) -> None:
        for fuel_type, (a1, a2, _x0, _power) in FUEL_EFFICIENCY_CURVE_PARAMETERS.items():
            with self.subTest(fuel_type=fuel_type):
                sizes = [1.0, 10.0, 50.0, 200.0, 1000.0, 5000.0]
                values = [fuel_efficiency(fuel_type, size) for size in sizes]
                self.assertEqual(values, sorted(values))
                self.assertAlmostEqual(fuel_efficiency(fuel_type, 1e-6), a1 / 100.0, places=4)
                self.assertAlmostEqual(fuel_efficiency(fuel_type, 1e9), a2 / 100.0, places=4)

    def test_efficiency_is_the_mean_of_the_limits_at_the_midpoint_size(self) -> None:
        for fuel_type, (a1, a2, x0, _power) in FUEL_EFFICIENCY_CURVE_PARAMETERS.items():
            with self.subTest(fuel_type=fuel_type):
                self.assertAlmostEqual(fuel_efficiency(fuel_type, x0), (a1 + a2) / 200.0, places=10)

    def test_efficiencies_of_plants_of_500_mw_fit_real_plants(self) -> None:
        # Fossil steam and gas plants of this size convert 35-60% of fuel heat (combined cycle
        # at the top), and nuclear plants about a third.
        for fuel_type in FUEL_EFFICIENCY_CURVE_PARAMETERS:
            with self.subTest(fuel_type=fuel_type):
                self.assertTrue(0.25 < fuel_efficiency(fuel_type, 500.0) < 0.65)
        self.assertEqual(fuel_efficiency("nuclear", 1000.0), 0.35)


def _nuclear_fleet() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One 100 MW nuclear unit per cell, once-through and recirculating.

    Nuclear plants have efficiency 0.35 and no flue-gas heat loss in the library, so
    equations 1 and 2 need no tabulated fuel curve.
    """

    cells = pd.DataFrame(
        {
            "UNITID": [1, 2],
            "LOCATIONID": [1, 2],
            "grid_lat": [0.25, 10.25],
            "grid_lon": [0.25, 0.25],
            "mw": [100.0, 100.0],
        }
    )
    cooling = pd.DataFrame(
        {
            "UNITID": [1, 2],
            "LOCATIONID": [1, 2],
            "country": ["X", "X"],
            "fuel": ["UR", "UR"],
            "technology_family": ["steam", "steam"],
            "cooling_category": ["ot_fresh", "cl_fresh"],
            "cooling_physics": ["once_through", "recirculating"],
            "requires_water": [True, True],
        }
    )
    return cells, cooling


def _nuclear_run(river_c: list[float], *, air_c: float = 25.0, relative_humidity: float = 60.0) -> xr.Dataset:
    """Evaluate ``_nuclear_fleet`` at one river temperature per month; 5 m3/s everywhere."""

    cells, cooling = _nuclear_fleet()

    def field(values: list[list[float]]) -> tuple[tuple[str, str, str], np.ndarray]:
        return (("time", "lat", "lon"), np.asarray(values, dtype=float).reshape(len(values), 2, 1))

    steps = len(river_c)
    climate = xr.Dataset(
        {
            "tw": field([[value + 273.15] * 2 for value in river_c]),
            "dis": field([[5.0, 5.0]] * steps),
            "tas": field([[air_c + 273.15] * 2] * steps),
            "hurs": field([[relative_humidity] * 2] * steps),
            "ps": field([[101325.0] * 2] * steps),
        },
        coords={
            "time": pd.date_range("2001-01-01", periods=steps, freq="MS"),
            "lat": [0.25, 10.25],
            "lon": [0.25],
        },
    )
    for name in ("tw", "tas"):
        climate[name].attrs["units"] = "K"
    climate["ps"].attrs["units"] = "Pa"
    return wet_cooling(cells, cooling, climate, "native").compute()


class LiEquationsTest(unittest.TestCase):
    """Withdrawal intensity and usable capacity of a nuclear unit, worked by hand from Li et al. (2025).

    Constants: rho_w = 1000 kg m-3, C_p = 4.184 J g-1 K-1 (so rho_w C_p = 4184 kJ m-3 K-1),
    T_lmax = 32 C, dT_lmax = 10 K, n_cc = 5, T_app = 6 K; efficiency 0.35 and no heat loss
    in the flue gas for nuclear units, so (1 - eta - k_os) / eta = 0.65 / 0.35. One MWh is 3.6e6 kJ.
    """

    def test_once_through_intensity_follows_the_temperature_headroom(self) -> None:
        # Eq. 1 at river temperatures of 15, 27 and 32 C (allowed temperature rise
        # min(32 - Tw, 10) = 10, 5, 0 K):
        #   WI_ot = 3.6e6 * 0.65 / 0.35 / (4184 * 10) = 159.79 m3/MWh at 15 C,
        #   twice that, 319.58, at 5 K of allowed rise, and no water can be withdrawn at 32 C.
        monthly = _nuclear_run([15.0, 27.0, 32.0])
        intensity = monthly["withdrawal_intensity_m3_per_mwh"].sel(entity=1)

        self.assertAlmostEqual(intensity.isel(time=0).item(), 159.79, places=2)
        self.assertAlmostEqual(intensity.isel(time=1).item(), 319.58, places=2)
        self.assertEqual(intensity.isel(time=2).item(), math.inf)

    def test_usable_capacity_is_the_water_limited_quotient(self) -> None:
        # Eq. 4: (5 - 0.8 * 5) m3/s = 1 m3/s for the only unit in the cell.
        # Eq. 3: UC = min(3600 * 1 / WI, 100): 3600 / 159.79 = 22.53 MW at 15 C,
        # 3600 / 319.58 = 11.26 MW at 27 C and 0 MW at 32 C. Eq. 5: usable capacity ratio = UC / 100.
        monthly = _nuclear_run([15.0, 27.0, 32.0])
        usable = monthly["usable_capacity_mw"].sel(entity=1)

        self.assertAlmostEqual(usable.isel(time=0).item(), 22.53, places=2)
        self.assertAlmostEqual(usable.isel(time=1).item(), 11.26, places=2)
        self.assertEqual(usable.isel(time=2).item(), 0.0)
        self.assertAlmostEqual(monthly["capacity_factor"].sel(entity=1).isel(time=0).item(), 0.2253, places=4)

    def test_recirculating_intensity_matches_the_equation_on_table_values(self) -> None:
        # Air at 25 C, 60% relative humidity, 101.325 kPa; river 20 C. By hand, from tables:
        #   saturation vapour pressure at 25 C, 3.169 kPa; w_out = 0.62198 * 3.169 / (101.325 - 3.169) = 20.08 g/kg;
        #   vapour pressure 0.6 * 3.169 = 1.901 kPa; w_in = 0.62198 * 1.901 / (101.325 - 1.901) = 11.89 g/kg;
        #   enthalpy h = 1.006 t + w (2501 + 1.86 t) (ASHRAE): h_out = 76.31, h_in = 55.45 kJ/kg;
        #   wet-bulb temperature at 25 C and 60%, 19.5 C (psychrometric chart).
        # Eq. 2: WI_rc = 3.6e6 * 0.65 / 0.35 * dw / (1000 * (dh * (1 - 1/5)
        #   + ((19.5 + 6) / 5 - 20) * 4.184 * dw)), dw = 0.008186 kg/kg, dh = 20.85 kJ/kg
        #   = 3.384 m3/MWh. The library's psychrometric constants are rounded, which
        #   moves the result by about 0.01%, so the tolerance is 0.2%.
        monthly = _nuclear_run([20.0])

        intensity = monthly["withdrawal_intensity_m3_per_mwh"].sel(entity=2).item()
        self.assertAlmostEqual(intensity, 3.384, delta=0.007)

    def test_intensity_of_once_through_water_is_inversely_proportional_to_headroom(self) -> None:
        # Eq. 1 puts the allowed temperature rise alone in the denominator, so 7 K
        # and 10 K of allowed rise (river at 25 and 15 C) give intensities in the ratio 10 : 7.
        monthly = _nuclear_run([15.0, 25.0])
        intensity = monthly["withdrawal_intensity_m3_per_mwh"].sel(entity=1)

        self.assertAlmostEqual(intensity.isel(time=1).item() / intensity.isel(time=0).item(), 10.0 / 7.0, places=10)


class PsychrometricsTest(unittest.TestCase):
    """The air-state terms of eq. 2 against tabulated values."""

    def test_wet_bulb_matches_stulls_check_point(self) -> None:
        # Stull (2011) gives 13.7 C for an air temperature of 20 C and relative humidity of 50%.
        monthly = _nuclear_run([20.0], air_c=20.0, relative_humidity=50.0)

        self.assertAlmostEqual(monthly["wet_bulb_temperature"].sel(entity=2).item(), 13.7, delta=0.05)

    def test_saturation_vapour_pressure_matches_the_tables(self) -> None:
        # Saturation pressure of water: 2.339 kPa at 20 C, 3.169 kPa at 25 C, 4.247 kPa at 30 C.
        # The Magnus form with Tetens's coefficients stays within 0.5% over this range.
        for air_c, table_pa in ((20.0, 2339.0), (25.0, 3169.0), (30.0, 4247.0)):
            with self.subTest(air_c=air_c):
                monthly = _nuclear_run([20.0], air_c=air_c)
                value = monthly["saturation_vapor_pressure_pa"].sel(entity=2).item()
                self.assertAlmostEqual(value, table_pa, delta=0.005 * table_pa)

    def test_humidity_ratio_matches_hand_arithmetic(self) -> None:
        # w = 0.62198 e / (p - e), e the vapour pressure. At 25 C and 101.325 kPa with e at
        # saturation, 3.169 kPa: 20.08 g/kg; at 60% relative humidity, e = 1.901 kPa: 11.89 g/kg.
        monthly = _nuclear_run([20.0], air_c=25.0, relative_humidity=60.0)

        self.assertAlmostEqual(monthly["humidity_ratio_out_g_per_kg"].sel(entity=2).item(), 20.08, delta=0.1)
        self.assertAlmostEqual(monthly["humidity_ratio_in_g_per_kg"].sel(entity=2).item(), 11.89, delta=0.06)


class MoistAirEnthalpyTest(unittest.TestCase):
    def test_matches_psychrometric_reference(self) -> None:
        # ASHRAE Fundamentals, h = 1.006 t + W (2501 + 1.86 t), t in C, W in kg/kg,
        # worked by hand. The library's rounded constants differ by under 0.2%.
        for air_temperature_c, humidity_ratio_g_per_kg, reference in [
            (5.0, 4.0, 15.071),
            (25.0, 10.0, 50.625),
            (35.0, 20.0, 86.532),
        ]:
            enthalpy = _moist_air_enthalpy_kj_per_kg(
                xr.DataArray(air_temperature_c),
                xr.DataArray(humidity_ratio_g_per_kg),
            ).item()
            self.assertAlmostEqual(enthalpy, reference, delta=0.003 * reference)

class WetCoolingDaskTest(WetCoolingContractTest):
    """Re-run all wet-cooling tests with dask-backed climate arrays.

    Kernels must produce correct results from chunked inputs.
    Tests that extract scalars override the parent to .compute() first.
    """

    def setUp(self) -> None:
        super().setUp()
        self.climate = self.climate.chunk({"time": 1})

    def test_normalize_wet_cooling_inputs_renames_and_converts_units(self) -> None:
        normalized = prepare_climate(WET_COOLING_KERNEL, self.climate).compute()
        self.assertEqual(
            set(normalized.data_vars),
            {
                "water_temperature",
                "discharge",
                "air_temperature",
                "relative_humidity",
                "surface_pressure",
                "sea_surface_temperature",
            },
        )
        self.assertAlmostEqual(normalized["water_temperature"].isel(time=0, lat=0, lon=0).item(), 15.0)
        self.assertAlmostEqual(normalized["surface_pressure"].isel(time=0, lat=1, lon=0).item(), 100.5)

    def test_compute_monthly_wet_cooling_impacts_matches_li_equations(self) -> None:
        mapping = self._mapping()
        monthly = evaluate_mapped(WET_COOLING_KERNEL, mapping, self.climate, duration="native", site_id_column="UNITID").compute()

        self.assertEqual(monthly.sizes["entity"], 5)
        self.assertEqual(monthly["capacity_factor"].dims, ("time", "entity"))
        # The values worked out in the parent test.
        january = monthly.sel(time="2001-01-01")
        self.assertAlmostEqual(january["withdrawal_intensity_m3_per_mwh"].sel(entity=101).item(), 172.92, places=1)
        self.assertAlmostEqual(january["usable_capacity_mw"].sel(entity=101).item(), 4.164, places=2)
        self.assertFalse(monthly["discharge_constraint_applies"].sel(entity=102).item())

    def test_aggregate_seasonal_wet_cooling_impacts_uses_climateresponse(self) -> None:
        mapping = self._mapping()
        seasonal = evaluate_mapped(WET_COOLING_KERNEL, mapping, self.climate, duration="seasonal", site_id_column="UNITID").compute()
        self.assertEqual(seasonal["capacity_factor"].dims, ("season", "year", "entity"))




def test_water_limited_capacity_edge_rules_are_the_callers_choice() -> None:
    from impacts.thermoelectric.wet import water_limited_capacity_mw

    discharge = np.array([1.0, 1.0, 1.0, 1.0, 0.0, np.nan, 0.0])
    intensity = np.array([7200.0, 0.0, -3600.0, np.inf, 7200.0, 7200.0, np.nan])
    nan, inf = np.nan, np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        cases = {
            # The kernel's per-unit capacity: the plain quotient, capped at nameplate.
            "kernel": (dict(nameplate_mw=0.4), [0.4, 0.4, -1.0, 0.0, 0.0, nan, nan]),
            # The water limit of a site in the site evaluation: uncapped, unknown at zero or negative
            # intensity, zero in a dry month whatever the intensity.
            "ceiling": (
                dict(zero_intensity=nan, negative_intensity=nan, zero_discharge=0.0),
                [0.5, nan, nan, 0.0, 0.0, nan, 0.0],
            ),
            # The reference capacity per unit flow in the wet capacity loss split: finite and positive only.
            "reference": (
                dict(zero_intensity=nan, negative_intensity=nan, finite_positive_only=True),
                [0.5, nan, nan, nan, nan, nan, nan],
            ),
            # The capacity per unit flow in the wet capacity loss split: infinite at zero, none if negative.
            "conductance": (
                dict(zero_intensity=inf, negative_intensity=0.0),
                [0.5, inf, 0.0, 0.0, 0.0, nan, nan],
            ),
        }
    for name, (rules, expected) in cases.items():
        actual = water_limited_capacity_mw(discharge, intensity, **rules)
        np.testing.assert_array_equal(actual, np.array(expected), err_msg=name)
