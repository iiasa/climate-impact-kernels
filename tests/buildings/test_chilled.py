from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from impacts.buildings.archetypes import (
    build_chilled_archetype_mapping,
    build_chilled_parameter_grids,
)
from impacts.buildings.chilled import (
    CHILLED_CHUNKS,
    CHILLED_KERNEL,
    CHILLED_REGION_SETPOINT_KERNEL,
    CHILLED_SITE_ID_COLUMN,
    ChilledEnergyAssumptions,
    chilled_energy,
)
from impacts.engine import evaluate_gridded, prepare_climate
from impacts.engine import evaluate_mapped as _evaluate_mapped

from . import fixtures


def evaluate_mapped(kernel, mapping, *args, **kwargs):
    """Evaluate a CHILLED mapping, numbering its rows as sites."""

    return _evaluate_mapped(
        kernel,
        mapping.assign(**{CHILLED_SITE_ID_COLUMN: np.arange(len(mapping))}),
        *args,
        site_id_column=CHILLED_SITE_ID_COLUMN,
        **kwargs,
    )


class BuildingsChilledContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.climate = fixtures.climate()
        self.solar = fixtures.solar()
        self.spatial_units = fixtures.spatial_units()
        self.archetypes = fixtures.archetypes()

    def test_normalize_chilled_climate_inputs_converts_kelvin_to_celsius(self) -> None:
        normalized = prepare_climate(CHILLED_KERNEL, self.climate)

        self.assertEqual(set(normalized.data_vars), {"air_temperature_c"})
        self.assertAlmostEqual(normalized["air_temperature_c"].isel(time=0, lat=0, lon=0).item(), 0.0)

    def test_assumptions_load_from_parametric_row(self) -> None:
        assumptions = ChilledEnergyAssumptions.from_parametric_row(
            {
                "t_sp_c": 23.0,
                "t_sp_c_max": 26.0,
                "t_sp_h": 20.0,
                "f_c": 0.45,
                "f_f": 0.25,
                "f_h": 0.9,
                "cop": 3.2,
                "eff": 0.9,
            }
        )

        self.assertEqual(assumptions.cooling_setpoint_c, 23.0)
        self.assertEqual(assumptions.fan_power_w, 55.0)

    def test_chilled_kernel_evaluation_runs_daily_and_monthly(self) -> None:
        assumptions = ChilledEnergyAssumptions(
            cooling_setpoint_c=26.0,
            cooling_max_c=28.0,
            heating_setpoint_c=20.0,
            cooling_hours_fraction=0.333,
            fan_hours_fraction=0.2,
            heating_hours_fraction=1.0,
            ac_cop=3.2,
            heating_efficiency=0.9,
        )

        mapping, solar = build_chilled_archetype_mapping(self.spatial_units, self.archetypes, self.solar)
        daily = evaluate_mapped(
            CHILLED_KERNEL, mapping, self.climate, duration="native", gridded_coords=solar, assumptions=assumptions
        )

        self.assertEqual(daily["cooling_energy_total"].dims, ("time", "entity"))
        self.assertEqual(daily.sizes["time"], 4)
        self.assertEqual(daily.coords["region"].values.tolist(), ["AFR", "CPA"])
        self.assertEqual(daily.coords["spatial_unit_id"].values.tolist(), [0, 1])
        self.assertIn("climate_contract", daily.attrs)

        monthly = evaluate_mapped(
            CHILLED_KERNEL, mapping, self.climate, duration="monthly", gridded_coords=solar, assumptions=assumptions
        )
        self.assertIn("time", monthly.dims)

    def test_a_hot_day_and_a_cold_day_match_hand_arithmetic(self) -> None:
        # Archetype AFR_sfh_s1 in the cell at 10 N, 100 E, default assumptions. Daily
        # variable degree-day method of Mastrucci et al. (2019), worked by hand.
        # Heat-loss coefficients, W/K: transmission 3.475 * 3.7 = 12.8575; ventilation
        # 1200 * 3.0 * 0.5 / 3600 = 0.5 closed and 1200 * 3.0 * 1.5 / 3600 = 1.5 open.
        #
        # Hot day, August, 30 C. Solar 108 W/m2 vertical, 208 horizontal. Solar gain
        # 108 * 0.125 * 0.85 * 0.8 + 208 * 0.04 * 3.3 * 1.0 * 0.6 = 25.6536 W; with
        # internal gains 2.14, gains are 27.7936 W. Balance temperature
        # 26 - 27.7936 / (12.8575 + 0.5) = 23.919 C; maximum 28 - 27.7936 /
        # (12.8575 + 1.5) = 26.064 C. The day is 30 - 26.064 = 3.936 K above the
        # maximum, so the AC runs. Cooling need 13.3575 * 0.333 * (3.936 + (26.064 -
        # 23.919)) * 0.0864 MJ per W day = 2.3369, over COP 3.2 = 0.7303 MJ/m2/day;
        # the fan, 0.2 * 55 W / 25 m2 * 0.0864 = 0.0380. Total 0.7683.
        #
        # Cold day, January, 0 C. Solar 101 and 201 W/m2: gains 2.14 + 101 * 0.085 +
        # 201 * 0.04 * 3.3 * 0.6 = 26.6442 W. Heating balance 20 - 26.6442 / 13.3575 =
        # 18.005 C, so 18.005 degree-days. Heating need 13.3575 * 1.0 * 18.005 * 0.0864 =
        # 20.780, over efficiency 0.9 = 23.089 MJ/m2/day. No cooling.
        mapping, solar = build_chilled_archetype_mapping(self.spatial_units, self.archetypes, self.solar)
        result = evaluate_mapped(
            CHILLED_KERNEL, mapping, self.climate, duration="native", gridded_coords=solar
        ).compute()
        afr = result.isel(entity=result.coords["region"].values.tolist().index("AFR"))

        hot = afr.isel(time=3)
        self.assertAlmostEqual(hot["cooling_energy_ac"].item(), 0.7303, places=3)
        self.assertAlmostEqual(hot["cooling_energy_fan"].item(), 0.0380, places=4)
        self.assertAlmostEqual(hot["cooling_energy_total"].item(), 0.7683, places=3)
        cold = afr.isel(time=0)
        self.assertAlmostEqual(cold["heating_energy"].item(), 23.089, places=2)
        self.assertEqual(cold["cooling_energy_total"].item(), 0.0)

    def test_region_setpoints_override_scalar_per_entity_with_fallback(self) -> None:
        # AFR gets a high region setpoint; CPA is left out of the table and must
        # fall back to the global scalar. A higher cooling setpoint raises the
        # balance temperature, so AFR cooling energy must drop while CPA is
        # identical to the global-scalar run.
        assumptions = ChilledEnergyAssumptions(cooling_setpoint_c=26.0, cooling_max_c=28.0)
        setpoints = pd.DataFrame(
            {
                "region": ["AFR"],
                "cooling_setpoint_c": [40.0],
                "cooling_max_c": [42.0],
            }
        )

        global_mapping, solar = build_chilled_archetype_mapping(self.spatial_units, self.archetypes, self.solar)
        region_mapping, _ = build_chilled_archetype_mapping(
            self.spatial_units, self.archetypes, self.solar, setpoints=setpoints
        )
        self.assertIn("cooling_setpoint_c", region_mapping.columns)
        self.assertTrue(pd.isna(region_mapping.loc[region_mapping["region"] == "CPA", "cooling_max_c"]).all())

        base = evaluate_mapped(
            CHILLED_KERNEL, global_mapping, self.climate, duration="native", gridded_coords=solar,
            assumptions=assumptions,
        )
        conditioned = evaluate_mapped(
            CHILLED_REGION_SETPOINT_KERNEL, region_mapping, self.climate, duration="native", gridded_coords=solar,
            assumptions=assumptions,
        )

        regions = base.coords["region"].values.tolist()
        afr = regions.index("AFR")
        cpa = regions.index("CPA")

        base_cooling = base["cooling_energy_total"].sum("time").values
        cond_cooling = conditioned["cooling_energy_total"].sum("time").values

        self.assertLess(cond_cooling[afr], base_cooling[afr])
        np.testing.assert_allclose(cond_cooling[cpa], base_cooling[cpa])

    def test_one_call_matches_the_explicit_path(self) -> None:
        setpoints = pd.DataFrame({"region": ["AFR"], "cooling_setpoint_c": [40.0], "cooling_max_c": [42.0]})
        mapping, solar = build_chilled_archetype_mapping(
            self.spatial_units, self.archetypes, self.solar, setpoints=setpoints
        )
        explicit = evaluate_mapped(
            CHILLED_REGION_SETPOINT_KERNEL, mapping, self.climate, duration="monthly", gridded_coords=solar
        )
        one_call = chilled_energy(
            self.spatial_units, self.archetypes, self.solar, self.climate, "monthly", setpoints=setpoints
        )
        xr.testing.assert_identical(one_call.compute(), explicit.compute())

    def test_region_setpoints_flag_off_preserves_global_scalar(self) -> None:
        # CHILLED_KERNEL does not carry the setpoint columns, so a mapping holding
        # them gives the pure global run.
        assumptions = ChilledEnergyAssumptions(cooling_setpoint_c=26.0, cooling_max_c=28.0)
        setpoints = pd.DataFrame(
            {"region": ["AFR"], "cooling_setpoint_c": [40.0], "cooling_max_c": [42.0]}
        )

        global_mapping, solar = build_chilled_archetype_mapping(self.spatial_units, self.archetypes, self.solar)
        region_mapping, _ = build_chilled_archetype_mapping(
            self.spatial_units, self.archetypes, self.solar, setpoints=setpoints
        )

        base = evaluate_mapped(
            CHILLED_KERNEL, global_mapping, self.climate, duration="native", gridded_coords=solar,
            assumptions=assumptions,
        )
        ignored = evaluate_mapped(
            CHILLED_KERNEL, region_mapping, self.climate, duration="native", gridded_coords=solar,
            assumptions=assumptions,
        )

        np.testing.assert_allclose(
            ignored["cooling_energy_total"].values, base["cooling_energy_total"].values
        )

    def _parameter_grids(self, setpoints: pd.DataFrame | None = None) -> xr.Dataset:
        region_raster = xr.DataArray(
            [["AFR", "CPA"], ["AFR", "CPA"]],
            dims=("lat", "lon"),
            coords={"lat": [10.0, 20.0], "lon": [100.0, 110.0]},
        )
        return build_chilled_parameter_grids(
            region_raster, self.archetypes, self.solar, building_type="sfh", urt="rur", vintage="s1",
            setpoints=setpoints,
        )

    def test_gridded_evaluation_defaults_to_chilled_chunks(self) -> None:
        # The loader's chunks (one day each) are replaced by the kernel's own
        # performance default unless the caller passes chunks.
        climate = self.climate.chunk({"time": 1})
        expected = {dim: (min(width, 4),) for dim, width in CHILLED_CHUNKS.items()}

        default = evaluate_gridded(CHILLED_KERNEL, climate, self._parameter_grids(), "native")
        explicit = evaluate_gridded(
            CHILLED_KERNEL, climate, self._parameter_grids(), "native", chunks={"time": 1}
        )

        self.assertEqual(dict(default["cooling_energy_total"].chunksizes), expected)
        self.assertEqual(explicit["cooling_energy_total"].chunksizes["time"], (1, 1, 1, 1))

    def test_per_site_setpoints_are_named_in_the_output_attrs(self) -> None:
        setpoints = fixtures.setpoint_table(self)
        mapping, solar = build_chilled_archetype_mapping(
            self.spatial_units, self.archetypes, self.solar, setpoints=setpoints
        )
        outputs = {
            "mapped": evaluate_mapped(
                CHILLED_REGION_SETPOINT_KERNEL, mapping, self.climate, "native", gridded_coords=solar
            ),
            "gridded": evaluate_gridded(
                CHILLED_KERNEL, self.climate, self._parameter_grids(setpoints), "native"
            ),
            "one_call": chilled_energy(
                self.spatial_units, self.archetypes, self.solar, self.climate, "native", setpoints=setpoints
            ),
        }
        for path, result in outputs.items():
            with self.subTest(path=path):
                for name, default in (("cooling_setpoint_c", 26.0), ("cooling_max_c", 28.0)):
                    self.assertEqual(
                        result.attrs[name],
                        "per-region setpoints from the setpoint table setpoints.csv; "
                        f"{default!r} where a site has none",
                    )
                    self.assertIn("setpoints.csv, a region cooling setpoint table", result.attrs[f"{name}_source"])
                    self.assertIn("Fallback value: message-ix-buildings", result.attrs[f"{name}_source"])
                # Assumptions the sites do not override keep their scalar, and the
                # recorded source stays off the site coordinate.
                self.assertEqual(result.attrs["heating_setpoint_c"], 20.0)
                self.assertNotIn("source", result["cooling_setpoint_c"].attrs)

        # Without the setpoint columns the scalars are what the physics used.
        plain = evaluate_mapped(CHILLED_KERNEL, mapping, self.climate, "native", gridded_coords=solar)
        self.assertEqual(plain.attrs["cooling_setpoint_c"], 26.0)

        # A table that records nothing about itself still says the values were per site.
        bare = pd.DataFrame({"region": ["AFR"], "cooling_setpoint_c": [40.0], "cooling_max_c": [42.0]})
        result = chilled_energy(
            self.spatial_units, self.archetypes, self.solar, self.climate, "native", setpoints=bare
        )
        self.assertEqual(
            result.attrs["cooling_setpoint_c"], "per-site values of 'cooling_setpoint_c'; 26.0 where a site has none"
        )
        self.assertTrue(result.attrs["cooling_setpoint_c_source"].startswith("not recorded with the site values"))

    def test_unedited_setpoints_keep_provenance_when_sorted_subset_or_flipped(self) -> None:
        described = "per-region setpoints from the setpoint table setpoints.csv"
        setpoints = fixtures.setpoint_table(self)
        sorted_table = setpoints.sort_values("cooling_setpoint_c", ascending=False)
        mapping, solar = build_chilled_archetype_mapping(
            self.spatial_units, self.archetypes, self.solar, setpoints=setpoints
        )
        # One region of the mapping.
        afr_only = mapping.loc[mapping["region"] == "AFR"].reset_index(drop=True)
        flipped_climate = self.climate.isel(lat=slice(None, None, -1))
        outputs = {
            "sorted_table_one_call": chilled_energy(
                self.spatial_units, self.archetypes, self.solar, self.climate, "native", setpoints=sorted_table
            ),
            "sorted_table_gridded": evaluate_gridded(
                CHILLED_KERNEL, self.climate, self._parameter_grids(sorted_table), "native"
            ),
            "mapping_row_subset": evaluate_mapped(
                CHILLED_REGION_SETPOINT_KERNEL, afr_only, self.climate, "native", gridded_coords=solar
            ),
            "mapping_reordered": evaluate_mapped(
                CHILLED_REGION_SETPOINT_KERNEL, mapping.iloc[::-1].reset_index(drop=True), self.climate,
                "native", gridded_coords=solar,
            ),
            "spatial_unit_subset_one_call": chilled_energy(
                self.spatial_units.iloc[:1], self.archetypes, self.solar, self.climate, "native",
                setpoints=setpoints,
            ),
            "north_to_south_climate_gridded": evaluate_gridded(
                CHILLED_KERNEL, flipped_climate, self._parameter_grids(setpoints), "native"
            ),
        }
        for path, result in outputs.items():
            with self.subTest(path=path):
                for name in ("cooling_setpoint_c", "cooling_max_c"):
                    self.assertTrue(result.attrs[name].startswith(described), result.attrs[name])
                    self.assertIn("setpoints.csv", result.attrs[f"{name}_source"])

    def test_edited_setpoints_lose_the_recorded_provenance(self) -> None:
        unrecorded = "not recorded with the site values"
        setpoints = fixtures.setpoint_table(self)
        edited_table = setpoints.assign(cooling_setpoint_c=setpoints["cooling_setpoint_c"] + 5.0)
        # pandas carries the table's attrs through the edit; the digest does not match.
        self.assertEqual(edited_table.attrs["description"], setpoints.attrs["description"])

        mapping, solar = build_chilled_archetype_mapping(
            self.spatial_units, self.archetypes, self.solar, setpoints=setpoints
        )
        grids = self._parameter_grids(setpoints)
        outputs = {
            "edited_table_one_call": chilled_energy(
                self.spatial_units, self.archetypes, self.solar, self.climate, "native", setpoints=edited_table
            ),
            "edited_table_gridded": evaluate_gridded(
                CHILLED_KERNEL, self.climate, self._parameter_grids(edited_table), "native"
            ),
            "edited_mapping_column": evaluate_mapped(
                CHILLED_REGION_SETPOINT_KERNEL, mapping.assign(cooling_setpoint_c=30.0), self.climate,
                "native", gridded_coords=solar,
            ),
            "edited_grid": evaluate_gridded(
                CHILLED_KERNEL, self.climate,
                grids.assign(cooling_setpoint_c=(grids["cooling_setpoint_c"] + 1.0).assign_attrs(
                    grids["cooling_setpoint_c"].attrs
                )),
                "native",
            ),
        }
        for path, result in outputs.items():
            with self.subTest(path=path):
                self.assertTrue(result.attrs["cooling_setpoint_c"].startswith("per-site values of"))
                self.assertTrue(result.attrs["cooling_setpoint_c_source"].startswith(unrecorded))
        # A column left alone keeps its recorded source when another is edited.
        self.assertTrue(
            outputs["edited_mapping_column"].attrs["cooling_max_c"].startswith(
                "per-region setpoints from the setpoint table setpoints.csv"
            )
        )

    def test_caller_set_assumptions_are_sourced_to_the_caller(self) -> None:
        # A parametric-run row sets some fields away from their defaults.
        assumptions = ChilledEnergyAssumptions.from_parametric_row(
            {"t_sp_c": 24.0, "t_sp_c_max": 28.0, "t_sp_h": 20.0, "f_c": 0.45, "f_f": 0.2,
             "f_h": 1.0, "cop": 3.2, "eff": 0.9}
        )
        result = chilled_energy(
            self.spatial_units, self.archetypes, self.solar, self.climate, "native", assumptions=assumptions
        )
        fields = ChilledEnergyAssumptions.__dataclass_fields__

        self.assertEqual(result.attrs["cooling_setpoint_c"], 24.0)
        self.assertEqual(
            result.attrs["cooling_setpoint_c_source"],
            f"set by the caller; default 26.0 from {fields['cooling_setpoint_c'].metadata['source']}",
        )
        self.assertTrue(result.attrs["cooling_hours_fraction_source"].startswith("set by the caller; default 0.333"))
        self.assertEqual(result.attrs["cooling_max_c_source"], fields["cooling_max_c"].metadata["source"])

        # With per-site setpoints the fallback scalar's source says the caller set it.
        regional = chilled_energy(
            self.spatial_units, self.archetypes, self.solar, self.climate, "native",
            setpoints=fixtures.setpoint_table(self), assumptions=assumptions,
        )
        self.assertEqual(
            regional.attrs["cooling_setpoint_c"],
            "per-region setpoints from the setpoint table setpoints.csv; 24.0 where a site has none",
        )
        self.assertIn("Fallback value: set by the caller; default 26.0", regional.attrs["cooling_setpoint_c_source"])


if __name__ == "__main__":
    unittest.main()
