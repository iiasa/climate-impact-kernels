from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from impacts.domain.buildings.chilled import (
    ChilledEnergyAssumptions,
    create_chilled_kernel,
    normalize_chilled_climate_inputs,
)
from impacts.domain.buildings.chilled_config import DEFAULT_ARCHETYPE_INPUT_PATH
from impacts.domain.buildings.chilled_io import (
    build_chilled_archetype_mapping,
    load_message_ix_buildings_archetypes,
    load_message_ix_buildings_parametric_runs,
)
from impacts.kernel import evaluate_kernel


class BuildingsChilledContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.climate = xr.Dataset(
            data_vars={
                "tas": (
                    ("time", "lat", "lon"),
                    [
                        [[273.15, 274.15], [275.15, 276.15]],
                        [[283.15, 284.15], [285.15, 286.15]],
                        [[293.15, 294.15], [295.15, 296.15]],
                        [[303.15, 304.15], [305.15, 306.15]],
                    ],
                )
            },
            coords={
                "time": pd.to_datetime(["2001-01-01", "2001-02-01", "2001-07-01", "2001-08-01"]),
                "lat": [10.0, 20.0],
                "lon": [100.0, 110.0],
            },
        )
        self.climate["tas"].attrs["units"] = "K"

        month = np.arange(1, 13)
        self.solar = xr.Dataset(
            data_vars={
                "solar_vertical": (
                    ("month", "lat", "lon"),
                    np.stack(
                        [
                            [[100.0 + month_id, 110.0 + month_id], [120.0 + month_id, 130.0 + month_id]]
                            for month_id in month
                        ]
                    ),
                ),
                "solar_horizontal": (
                    ("month", "lat", "lon"),
                    np.stack(
                        [
                            [[200.0 + month_id, 210.0 + month_id], [220.0 + month_id, 230.0 + month_id]]
                            for month_id in month
                        ]
                    ),
                ),
            },
            coords={"month": month, "lat": [10.0, 20.0], "lon": [100.0, 110.0]},
        )

        self.spatial_units = pd.DataFrame(
            {
                "grid_lat": [10.0, 20.0],
                "grid_lon": [100.0, 110.0],
                "region": ["AFR", "CPA"],
                "grid_label": ["cell_a", "cell_b"],
            }
        )
        self.archetypes = pd.DataFrame(
            {
                "name": ["AFR_sfh_s1", "CPA_sfh_s1"],
                "building_type": ["sfh", "sfh"],
                "urt": ["rur", "urb"],
                "arch": ["s1", "s1"],
                "region": ["AFR", "CPA"],
                "permanent": [1, 1],
                "area_env": [3.7, 3.6],
                "gl_perc": [0.125, 0.125],
                "vol": [3.0, 3.0],
                "fl_cnd": [1.0, 1.0],
                "u_val": [3.475, 3.150],
                "ach_cl": [0.5, 0.5],
                "ach_op": [1.5, 1.5],
                "gn_int": [2.14, 2.14],
                "gl_g": [0.85, 0.85],
                "gl_sh": [0.8, 0.8],
                "roof_area": [1.0, 1.0],
                "roof_abs": [0.6, 0.6],
                "u_roof": [3.3, 2.0],
            }
        )

    def test_normalize_chilled_climate_inputs_converts_kelvin_to_celsius(self) -> None:
        normalized = normalize_chilled_climate_inputs(self.climate)

        self.assertEqual(set(normalized.data_vars), {"air_temperature_c"})
        self.assertAlmostEqual(normalized["air_temperature_c"].isel(time=0, lat=0, lon=0).item(), 0.0)

    def test_build_chilled_archetype_mapping_attaches_monthly_solar_columns(self) -> None:
        mapping = build_chilled_archetype_mapping(self.spatial_units, self.archetypes, self.solar)

        self.assertEqual(mapping["region"].tolist(), ["AFR", "CPA"])
        self.assertEqual(mapping["spatial_unit_id"].tolist(), [0, 1])
        self.assertAlmostEqual(mapping.loc[0, "solar_vertical_m01"], 101.0)
        self.assertAlmostEqual(mapping.loc[1, "solar_vertical_m12"], 142.0)
        self.assertAlmostEqual(mapping.loc[0, "solar_horizontal_m01"], 201.0)
        self.assertAlmostEqual(mapping.loc[1, "solar_horizontal_m12"], 242.0)

    def test_validate_archetype_table_rejects_pre_alps2025_proxy_labels(self) -> None:
        legacy = self.archetypes.assign(building_type="residential", urt="tot")

        with self.assertRaisesRegex(ValueError, "unsupported building_type values"):
            build_chilled_archetype_mapping(self.spatial_units, legacy, self.solar)

    @unittest.skipUnless(
        DEFAULT_ARCHETYPE_INPUT_PATH.exists(),
        "requires local message-ix-buildings archetype CSV",
    )
    def test_load_archetypes_accepts_real_message_ix_buildings_schema(self) -> None:
        archetype_path = DEFAULT_ARCHETYPE_INPUT_PATH

        archetypes = load_message_ix_buildings_archetypes(archetype_path)

        self.assertIn("region", archetypes.columns)
        self.assertEqual(archetypes.loc[0, "region"], "AFR")
        self.assertEqual(sorted(archetypes["building_type"].unique().tolist()), ["mfh", "sfh"])
        self.assertEqual(sorted(archetypes["urt"].unique().tolist()), ["rur", "urb"])
        self.assertEqual(sorted(archetypes["arch"].unique().tolist()), ["s1", "s2", "s3", "s51", "s52"])

    def test_parametric_runs_require_explicit_upstream_path(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "ALPS2025-only CHILLED support does not infer"):
            load_message_ix_buildings_parametric_runs(None)

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

    def test_default_assumptions_match_physical_chilled_baseline(self) -> None:
        assumptions = ChilledEnergyAssumptions()

        self.assertEqual(assumptions.cooling_setpoint_c, 26.0)
        self.assertEqual(assumptions.cooling_max_c, 28.0)
        self.assertEqual(assumptions.heating_setpoint_c, 20.0)
        self.assertAlmostEqual(assumptions.cooling_hours_fraction, 0.333)
        self.assertAlmostEqual(assumptions.fan_hours_fraction, 0.2)
        self.assertEqual(assumptions.heating_hours_fraction, 1.0)
        self.assertEqual(assumptions.ac_cop, 3.2)
        self.assertEqual(assumptions.heating_efficiency, 0.9)

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

        mapping = build_chilled_archetype_mapping(self.spatial_units, self.archetypes, self.solar)
        kernel = create_chilled_kernel(assumptions)
        daily = evaluate_kernel(kernel, mapping, self.climate, duration="daily")

        self.assertEqual(daily["cooling_energy_total"].dims, ("time", "entity"))
        self.assertEqual(daily.sizes["time"], 4)
        self.assertEqual(daily.coords["region"].values.tolist(), ["AFR", "CPA"])
        self.assertEqual(daily.coords["spatial_unit_id"].values.tolist(), [0, 1])
        self.assertIn("climate_contract", daily.attrs)

        monthly = evaluate_kernel(kernel, mapping, self.climate, duration="monthly")
        self.assertIn("time", monthly.dims)


if __name__ == "__main__":
    unittest.main()
