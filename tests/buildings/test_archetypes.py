from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from impacts.buildings.archetypes import (
    build_chilled_archetype_mapping,
    load_ewembi_solar_climatology,
    load_message_ix_buildings_archetypes,
    load_message_ix_buildings_parametric_runs,
    load_r12_region_raster,
    load_region_cooling_setpoints,
)
from impacts.buildings.regions import load_country_raster

from . import fixtures

REAL_ARCHETYPE_PATH = os.environ.get("CHILLED_ARCHETYPE_PATH")


class BuildingsArchetypeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.solar = fixtures.solar()
        self.spatial_units = fixtures.spatial_units()
        self.archetypes = fixtures.archetypes()

    def test_build_chilled_archetype_mapping_returns_mapping_and_solar(self) -> None:
        mapping, solar = build_chilled_archetype_mapping(self.spatial_units, self.archetypes, self.solar)

        self.assertEqual(mapping["region"].tolist(), ["AFR", "CPA"])
        self.assertEqual(mapping["spatial_unit_id"].tolist(), [0, 1])
        self.assertNotIn("solar_vertical_m01", mapping.columns)
        self.assertEqual(set(solar.data_vars), {"solar_vertical", "solar_horizontal"})
        self.assertEqual(solar["solar_vertical"].dims, ("month", "lat", "lon"))
        self.assertAlmostEqual(solar["solar_vertical"].sel(month=1, lat=10.0, lon=100.0).item(), 101.0)
        self.assertAlmostEqual(solar["solar_horizontal"].sel(month=12, lat=20.0, lon=110.0).item(), 242.0)

    def test_validate_archetype_table_rejects_unsupported_proxy_labels(self) -> None:
        proxy = self.archetypes.assign(building_type="residential", urt="tot")

        with self.assertRaisesRegex(ValueError, "unsupported building_type values"):
            build_chilled_archetype_mapping(self.spatial_units, proxy, self.solar)

    @unittest.skipUnless(
        REAL_ARCHETYPE_PATH and Path(REAL_ARCHETYPE_PATH).exists(),
        "set CHILLED_ARCHETYPE_PATH to an archetype CSV to check its schema",
    )
    def test_load_archetypes_reads_an_external_archetype_csv(self) -> None:
        archetypes = load_message_ix_buildings_archetypes()

        self.assertIn("region", archetypes.columns)
        self.assertEqual(archetypes.loc[0, "region"], "AFR")
        self.assertEqual(sorted(archetypes["building_type"].unique().tolist()), ["mfh", "sfh"])
        self.assertEqual(sorted(archetypes["urt"].unique().tolist()), ["rur", "urb"])
        self.assertEqual(sorted(archetypes["arch"].unique().tolist()), ["s1", "s2", "s3", "s51", "s52"])

    def test_parametric_runs_require_explicit_upstream_path(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "ALPS2025-only CHILLED support does not infer"):
            load_message_ix_buildings_parametric_runs(None)

    def test_load_region_cooling_setpoints_reads_the_named_table(self) -> None:
        table = fixtures.setpoint_table(self)

        self.assertEqual(set(table.columns), {"region", "cooling_setpoint_c", "cooling_max_c"})
        self.assertEqual(table["region"].tolist(), ["AFR", "CPA"])
        self.assertEqual(table["cooling_setpoint_c"].tolist(), [25.0, 23.0])

    def test_load_region_cooling_setpoints_has_no_default_table(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("CHILLED_SETPOINT_PATH", None)
            with self.assertRaisesRegex(FileNotFoundError, "CHILLED_SETPOINT_PATH"):
                load_region_cooling_setpoints()

    def test_load_region_cooling_setpoints_reads_the_environment_variable(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "from_environment.csv"
        pd.DataFrame(
            {"region": ["AFR"], "cooling_setpoint_c": [24.0], "cooling_max_c": [26.0]}
        ).to_csv(path, index=False)

        with mock.patch.dict(os.environ, {"CHILLED_SETPOINT_PATH": str(path)}):
            table = load_region_cooling_setpoints()

        self.assertEqual(table["region"].tolist(), ["AFR"])
        self.assertIn("from_environment.csv", table.attrs["description"])

    def test_input_locations_come_from_the_environment_or_an_argument(self) -> None:
        with mock.patch.dict(os.environ):
            for variable in (
                "ALPS_DATA_PATH",
                "CHILLED_ARCHETYPE_PATH",
                "MESSAGE_R12_YAML_PATH",
            ):
                os.environ.pop(variable, None)
            for name, load in (
                ("ALPS_DATA_PATH", load_country_raster),
                ("ALPS_DATA_PATH", load_ewembi_solar_climatology),
                ("ALPS_DATA_PATH", load_r12_region_raster),
                ("CHILLED_ARCHETYPE_PATH", load_message_ix_buildings_archetypes),
            ):
                with (
                    self.subTest(variable=name, loader=load.__name__),
                    self.assertRaisesRegex(FileNotFoundError, name),
                ):
                    load()


if __name__ == "__main__":
    unittest.main()
