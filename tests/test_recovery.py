from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from impacts.thermoelectric.mapping import (
    build_imputed_wet_unit_cell_mapping,
    build_recovered_unit_cell_mapping,
    write_imputed_wet_unit_cell_mapping,
    write_recovered_unit_cell_mapping,
)


class RecoveryContractTest(unittest.TestCase):
    def test_recovered_mapping_inherits_only_single_point_locations(self) -> None:
        frame = pd.DataFrame(
            {
                "UNITID": [1, 2, 3, 4, 5],
                "LOCATIONID": [10, 10, 20, 20, 30],
                "country": ["Taiwan", "Taiwan", "Korea", "Korea", "Chile"],
                "lat": [23.3979, None, 37.4764, 37.5454, None],
                "lon": [121.3603, None, 126.6169, 126.6760, None],
                "mw": [100.0, 150.0, 75.0, 75.0, 50.0],
                "has_coords": [True, False, True, True, False],
            }
        )

        recovered = build_recovered_unit_cell_mapping(frame)

        self.assertEqual(set(recovered["UNITID"].tolist()), {1, 2, 3, 4})
        inherited = recovered.loc[recovered["coordinate_source"] == "inherited_location"]
        self.assertEqual(inherited["UNITID"].tolist(), [2])
        self.assertEqual(inherited["grid_lat"].tolist(), [23.25])
        self.assertEqual(inherited["grid_lon"].tolist(), [121.25])

    def test_recovered_writer_materializes_provenance_column(self) -> None:
        frame = pd.DataFrame(
            {
                "UNITID": [1, 2],
                "LOCATIONID": [10, 10],
                "country": ["Taiwan", "Taiwan"],
                "lat": [23.3979, None],
                "lon": [121.3603, None],
                "mw": [100.0, 150.0],
                "has_coords": [True, False],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            source = f"{tmpdir}/plants.parquet"
            target = f"{tmpdir}/recovered.parquet"
            frame.to_parquet(source, index=False)

            summary = write_recovered_unit_cell_mapping(source, target)
            written = pd.read_parquet(target)

            self.assertEqual(summary.coordinate_units, 2)
            self.assertIn("coordinate_source", written.columns)
            self.assertEqual(
                written["coordinate_source"].tolist(),
                ["source", "inherited_location"],
            )

    def test_imputed_wet_mapping_keeps_recovery_and_adds_missing_wet_units(self) -> None:
        plants = pd.DataFrame(
            {
                "UNITID": [1, 2, 3, 4, 5, 6],
                "LOCATIONID": [10, 10, 20, 30, 40, 50],
                "country": ["A", "A", "A", "A", "B", "A"],
                "lat": [10.2, None, 11.1, 12.2, 30.1, 13.4],
                "lon": [20.2, None, 21.1, 22.2, 40.1, 23.4],
                "mw": [120.0, 90.0, 150.0, 80.0, 140.0, 110.0],
                "has_coords": [True, False, True, False, False, True],
            }
        )
        cooling = pd.DataFrame(
            {
                "UNITID": [1, 2, 3, 4, 5, 6],
                "LOCATIONID": [10, 10, 20, 30, 40, 50],
                "country": ["A", "A", "A", "A", "B", "A"],
                "mw": [120.0, 90.0, 150.0, 80.0, 140.0, 110.0],
                "requires_water": [True, True, True, True, True, False],
                "cooling_physics": [
                    "once_through",
                    "once_through",
                    "once_through",
                    "once_through",
                    "recirculating",
                    "dry",
                ],
            }
        )

        mapped = build_imputed_wet_unit_cell_mapping(plants, cooling, random_seed=0)

        self.assertEqual(set(mapped["UNITID"].tolist()), {1, 2, 3, 4, 5})
        self.assertEqual(
            mapped["coordinate_source"].tolist(),
            ["source", "inherited_location", "source", "imputed_unit_donor", "imputed_unit_donor"],
        )
        inherited = mapped.loc[mapped["UNITID"].eq(2)].iloc[0]
        self.assertEqual(inherited["grid_lat"], 10.25)
        self.assertEqual(inherited["grid_lon"], 20.25)

        imputed = mapped.loc[mapped["coordinate_source"].eq("imputed_unit_donor")].copy()
        self.assertTrue(imputed["imputation_rule"].notna().all())
        self.assertTrue(imputed["donor_UNITID"].notna().all())
        self.assertEqual(
            imputed.loc[imputed["UNITID"].eq(4), "imputation_rule"].iloc[0],
            "country_cooling_size",
        )
        self.assertEqual(
            imputed.loc[imputed["UNITID"].eq(5), "imputation_rule"].iloc[0],
            "size",
        )

    def test_imputed_wet_writer_materializes_single_artifact_with_provenance(self) -> None:
        plants = pd.DataFrame(
            {
                "UNITID": [1, 2, 3],
                "LOCATIONID": [10, 10, 20],
                "country": ["A", "A", "A"],
                "lat": [10.2, None, 11.1],
                "lon": [20.2, None, 21.1],
                "mw": [120.0, 90.0, 150.0],
                "has_coords": [True, False, False],
            }
        )
        cooling = pd.DataFrame(
            {
                "UNITID": [1, 2, 3],
                "LOCATIONID": [10, 10, 20],
                "country": ["A", "A", "A"],
                "mw": [120.0, 90.0, 150.0],
                "requires_water": [True, True, True],
                "cooling_physics": ["once_through", "once_through", "once_through"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            source = f"{tmpdir}/plants.parquet"
            cooling_path = f"{tmpdir}/cooling.parquet"
            target = f"{tmpdir}/wet_imputed.parquet"
            plants.to_parquet(source, index=False)
            cooling.to_parquet(cooling_path, index=False)

            summary = write_imputed_wet_unit_cell_mapping(source, cooling_path, target, random_seed=0)
            written = pd.read_parquet(target)

            self.assertEqual(summary.coordinate_units, 3)
            self.assertIn("coordinate_source", written.columns)
            self.assertIn("imputation_rule", written.columns)
            self.assertEqual(
                written["coordinate_source"].tolist(),
                ["source", "inherited_location", "imputed_unit_donor"],
            )


if __name__ == "__main__":
    unittest.main()
