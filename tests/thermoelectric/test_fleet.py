from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from impacts.thermoelectric.fleet import (
    build_country_totals,
    build_imputed_dry_unit_cell_mapping,
    build_imputed_wet_unit_cell_mapping,
    build_location_cell_mapping,
    build_recovered_unit_cell_mapping,
    build_unit_cell_mapping,
    build_unit_cooling_mapping,
    classify_technology_family,
    snap_to_half_degree_cell,
    summarize_mapping,
    write_country_totals,
    write_imputed_dry_unit_cell_mapping,
    write_imputed_wet_unit_cell_mapping,
    write_location_cell_mapping,
    write_recovered_unit_cell_mapping,
    write_unit_cell_mapping,
    write_unit_cooling_mapping,
)


class SnapRuleTest(unittest.TestCase):
    def test_snap_to_quarter_centered_half_degree_grid(self) -> None:
        values = pd.Series([23.3979, 23.501, -6.001, -6.5001])

        result = snap_to_half_degree_cell(values)

        self.assertEqual(result.tolist(), [23.25, 23.75, -6.25, -6.75])


class MappingContractTest(unittest.TestCase):
    def test_build_mapping_filters_to_units_with_coordinates(self) -> None:
        frame = pd.DataFrame(
            {
                "UNITID": [1, 2, 3],
                "LOCATIONID": [10, 10, 20],
                "country": ["Taiwan", "Taiwan", "Chile"],
                "lat": [23.3979, 23.3979, None],
                "lon": [121.3603, 121.3603, None],
                "mw": [100.0, 150.0, 50.0],
                "has_coords": [True, True, False],
            }
        )

        mapped = build_unit_cell_mapping(frame)

        self.assertEqual(mapped["UNITID"].tolist(), [1, 2])
        self.assertEqual(mapped["country"].tolist(), ["Taiwan", "Taiwan"])
        self.assertEqual(mapped["grid_lat"].tolist(), [23.25, 23.25])
        self.assertEqual(mapped["grid_lon"].tolist(), [121.25, 121.25])
        self.assertEqual(mapped["cell_id"].tolist(), ["23.25,121.25", "23.25,121.25"])
        self.assertEqual(mapped["location_cell_count"].tolist(), [1, 1])
        self.assertEqual(mapped["location_is_single_cell"].tolist(), [True, True])

    def test_build_mapping_treats_null_has_coords_as_false(self) -> None:
        frame = pd.DataFrame(
            {
                "UNITID": [1, 2, 3],
                "LOCATIONID": [10, 10, 20],
                "country": ["Taiwan", "Taiwan", "Chile"],
                "lat": [23.3979, 23.3979, 37.4764],
                "lon": [121.3603, 121.3603, 126.6169],
                "mw": [100.0, 150.0, 75.0],
                "has_coords": [True, None, False],
            }
        )

        mapped = build_unit_cell_mapping(frame)

        self.assertEqual(mapped["UNITID"].tolist(), [1])

    def test_summary_flags_location_ids_spanning_multiple_cells(self) -> None:
        frame = pd.DataFrame(
            {
                "UNITID": [1, 2, 3, 4],
                "LOCATIONID": [10, 10, 20, 20],
                "country": ["Taiwan", "Taiwan", "Korea", "Korea"],
                "lat": [23.3979, 23.3979, 37.4764, 37.5454],
                "lon": [121.3603, 121.3603, 126.6169, 126.6760],
                "mw": [100.0, 150.0, 75.0, 75.0],
                "has_coords": [True, True, True, True],
            }
        )

        mapped = build_unit_cell_mapping(frame)
        summary = summarize_mapping(mapped)

        self.assertEqual(summary.coordinate_units, 4)
        self.assertEqual(summary.occupied_cells, 3)
        self.assertEqual(summary.ambiguity.ambiguous_locations, 1)
        self.assertEqual(summary.ambiguity.ambiguous_units, 2)
        self.assertAlmostEqual(summary.ambiguity.ambiguous_mw, 150.0)
        self.assertAlmostEqual(summary.ambiguity.share_of_coordinate_units, 0.5)
        self.assertAlmostEqual(summary.ambiguity.share_of_coordinate_mw, 0.375)
        self.assertEqual(mapped["location_is_single_cell"].tolist(), [True, True, False, False])
        self.assertEqual(mapped["location_cell_count"].tolist(), [1, 1, 2, 2])

    def test_writer_materializes_default_lookup_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "UNITID": [1, 2],
                "LOCATIONID": [10, 20],
                "country": ["Taiwan", "Korea"],
                "lat": [23.3979, 37.4764],
                "lon": [121.3603, 126.6169],
                "mw": [100.0, 75.0],
                "has_coords": [True, True],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            source = f"{tmpdir}/plants.parquet"
            target = f"{tmpdir}/mapping.parquet"
            frame.to_parquet(source, index=False)

            summary = write_unit_cell_mapping(source, target)
            written = pd.read_parquet(target)

            self.assertEqual(summary.coordinate_units, 2)
            self.assertEqual(
                written.columns.tolist(),
                [
                    "UNITID",
                    "LOCATIONID",
                    "country",
                    "lat",
                    "lon",
                    "grid_lat",
                    "grid_lon",
                    "cell_id",
                    "mw",
                    "location_cell_count",
                    "location_is_single_cell",
                ],
            )
            self.assertEqual(written["location_is_single_cell"].tolist(), [True, True])

    def test_location_mapping_collapses_only_single_cell_locations(self) -> None:
        frame = pd.DataFrame(
            {
                "UNITID": [1, 2, 3, 4],
                "LOCATIONID": [10, 10, 20, 20],
                "country": ["Taiwan", "Taiwan", "Korea", "Korea"],
                "lat": [23.3979, 23.3979, 37.4764, 37.5454],
                "lon": [121.3603, 121.3603, 126.6169, 126.6760],
                "mw": [100.0, 150.0, 75.0, 75.0],
                "has_coords": [True, True, True, True],
            }
        )

        mapped = build_unit_cell_mapping(frame)
        location_mapping = build_location_cell_mapping(mapped)

        self.assertEqual(len(location_mapping), 3)
        self.assertEqual(location_mapping["LOCATIONID"].tolist(), [20, 20, 10])
        self.assertEqual(location_mapping["country"].tolist(), ["Korea", "Korea", "Taiwan"])
        self.assertEqual(location_mapping["units"].tolist(), [1, 1, 2])
        self.assertEqual(location_mapping["mw"].tolist(), [75.0, 75.0, 250.0])
        self.assertEqual(
            location_mapping["source_resolution"].tolist(),
            ["location_cell_split", "location_cell_split", "location"],
        )

    def test_location_writer_materializes_location_lookup_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "UNITID": [1, 2, 3],
                "LOCATIONID": [10, 10, 20],
                "country": ["Taiwan", "Taiwan", "Korea"],
                "lat": [23.3979, 23.3979, 37.4764],
                "lon": [121.3603, 121.3603, 126.6169],
                "mw": [100.0, 150.0, 75.0],
                "has_coords": [True, True, True],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            source = f"{tmpdir}/plants.parquet"
            target = f"{tmpdir}/location_mapping.parquet"
            frame.to_parquet(source, index=False)

            summary = write_location_cell_mapping(source, target)
            written = pd.read_parquet(target)

            self.assertEqual(summary.coordinate_units, 3)
            self.assertEqual(
                written.columns.tolist(),
                [
                    "LOCATIONID",
                    "country",
                    "grid_lat",
                    "grid_lon",
                    "cell_id",
                    "units",
                    "mw",
                    "location_is_single_cell",
                    "source_resolution",
                ],
            )
            self.assertEqual(written["source_resolution"].tolist(), ["location", "location"])

    def test_country_rollup_aggregates_country_capacity(self) -> None:
        frame = pd.DataFrame(
            {
                "UNITID": [1, 2, 3, 4],
                "LOCATIONID": [10, 10, 20, 20],
                "country": ["Taiwan", "Taiwan", "Korea", "Korea"],
                "lat": [23.3979, 23.3979, 37.4764, 37.5454],
                "lon": [121.3603, 121.3603, 126.6169, 126.6760],
                "mw": [100.0, 150.0, 75.0, 75.0],
                "has_coords": [True, True, True, True],
            }
        )

        mapped = build_unit_cell_mapping(frame)
        rollup = build_country_totals(mapped, source_resolution="unit")

        self.assertEqual(rollup["country"].tolist(), ["Korea", "Taiwan"])
        self.assertEqual(rollup["locations"].tolist(), [1, 1])
        self.assertEqual(rollup["mw"].tolist(), [150.0, 250.0])
        self.assertEqual(rollup["source_resolution"].tolist(), ["unit", "unit"])

    def test_country_rollup_counts_a_split_location_once(self) -> None:
        # Location 10 has rows in two cells, so it appears twice in a location-cell mapping.
        location_cells = pd.DataFrame(
            {
                "LOCATIONID": [10, 10, 20],
                "country": ["Taiwan", "Taiwan", "Taiwan"],
                "grid_lat": [23.25, 23.75, 23.25],
                "grid_lon": [121.25, 121.25, 121.25],
                "mw": [100.0, 150.0, 75.0],
            }
        )

        rollup = build_country_totals(location_cells, source_resolution="location")

        self.assertEqual(rollup["locations"].tolist(), [2])
        self.assertEqual(rollup["mw"].tolist(), [325.0])

    def test_country_writer_materializes_country_rollup(self) -> None:
        frame = pd.DataFrame(
            {
                "UNITID": [1, 2, 3],
                "LOCATIONID": [10, 10, 20],
                "country": ["Taiwan", "Taiwan", "Korea"],
                "lat": [23.3979, 23.3979, 37.4764],
                "lon": [121.3603, 121.3603, 126.6169],
                "mw": [100.0, 150.0, 75.0],
                "has_coords": [True, True, True],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            source = f"{tmpdir}/plants.parquet"
            target = f"{tmpdir}/country_rollup.parquet"
            frame.to_parquet(source, index=False)

            summary = write_country_totals(source, target, from_resolution="location")
            written = pd.read_parquet(target)

            self.assertEqual(summary.coordinate_units, 3)
            self.assertEqual(
                written.columns.tolist(),
                ["country", "locations", "mw", "source_resolution"],
            )
            self.assertEqual(written["country"].tolist(), ["Korea", "Taiwan"])
            self.assertEqual(written["source_resolution"].tolist(), ["location", "location"])

    def test_country_writer_refuses_an_unknown_source_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = f"{tmpdir}/country_rollup.parquet"

            with self.assertRaisesRegex(
                ValueError,
                r"unknown from_resolution 'units'; pass one of \['location', 'unit'\]",
            ):
                write_country_totals(f"{tmpdir}/missing.parquet", target, from_resolution="units")

            self.assertFalse(Path(target).exists())


class CoolingContractTest(unittest.TestCase):
    def test_classify_technology_family(self) -> None:
        self.assertEqual(classify_technology_family("GT/C"), "combined_cycle")
        self.assertEqual(classify_technology_family("CCSS"), "combined_cycle")
        self.assertEqual(classify_technology_family("ST"), "steam")
        self.assertEqual(classify_technology_family("GT"), "gas_turbine")
        self.assertEqual(classify_technology_family("IC"), "internal_combustion")
        self.assertEqual(classify_technology_family("ECE"), "internal_combustion")

    def test_build_unit_cooling_mapping_normalizes_contract(self) -> None:
        frame = pd.DataFrame(
            {
                "UNITID": [1, 2, 3],
                "LOCATIONID": [10, 20, 30],
                "country": ["Taiwan", "Korea", "USA"],
                "fuel": ["LNG", "COAL", "OIL"],
                "utype": ["GT/C", "ST", "IC"],
                "mw": [100.0, 250.0, 30.0],
                "cool_imputed": ["ot_saline", "cl_fresh", "air"],
                "cool_source": ["known", "country_fuel", "engine_type"],
            }
        )

        mapping = build_unit_cooling_mapping(frame)

        self.assertEqual(
            mapping.columns.tolist(),
            [
                "UNITID",
                "LOCATIONID",
                "country",
                "fuel",
                "utype",
                "technology_family",
                "cooling_category",
                "cooling_source",
                "requires_water",
                "cooling_physics",
                "mw",
            ],
        )
        self.assertEqual(
            mapping["technology_family"].tolist(),
            ["combined_cycle", "steam", "internal_combustion"],
        )
        self.assertEqual(mapping["cooling_physics"].tolist(), ["once_through", "recirculating", "dry"])
        self.assertEqual(mapping["requires_water"].tolist(), [True, True, False])

    def test_write_unit_cooling_mapping_materializes_artifact(self) -> None:
        frame = pd.DataFrame(
            {
                "UNITID": [1],
                "LOCATIONID": [10],
                "country": ["Taiwan"],
                "fuel": ["LNG"],
                "utype": ["GT/C"],
                "mw": [100.0],
                "cool_imputed": ["ot_saline"],
                "cool_source": ["known"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            source = f"{tmpdir}/plants.parquet"
            target = f"{tmpdir}/cooling.parquet"
            frame.to_parquet(source, index=False)

            write_unit_cooling_mapping(source, target)
            written = pd.read_parquet(target)

            self.assertEqual(written["cooling_category"].tolist(), ["ot_saline"])
            self.assertEqual(written["cooling_physics"].tolist(), ["once_through"])


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

    def test_imputed_dry_mapping_targets_only_qin_supported_units(self) -> None:
        plants = pd.DataFrame(
            {
                "UNITID": [1, 2, 3, 4, 5],
                "LOCATIONID": [10, 10, 20, 30, 40],
                "country": ["A", "A", "A", "A", "B"],
                "lat": [10.2, None, 11.1, None, None],
                "lon": [20.2, None, 21.1, None, None],
                "mw": [120.0, 90.0, 150.0, 80.0, 140.0],
                "has_coords": [True, False, True, False, False],
            }
        )
        cooling = pd.DataFrame(
            {
                "UNITID": [1, 2, 3, 4, 5],
                "LOCATIONID": [10, 10, 20, 30, 40],
                "country": ["A", "A", "A", "A", "B"],
                "mw": [120.0, 90.0, 150.0, 80.0, 140.0],
                "fuel": ["COAL", "COAL", "GAS", "DIESEL", "SOLAR"],
                "technology_family": [
                    "steam",
                    "steam",
                    "combined_cycle",
                    "gas_turbine",
                    "steam",
                ],
                "cooling_physics": ["dry", "dry", "dry", "dry", "dry"],
            }
        )

        mapped = build_imputed_dry_unit_cell_mapping(plants, cooling, random_seed=0)

        self.assertEqual(set(mapped["UNITID"].tolist()), {1, 2, 3, 5})
        self.assertEqual(
            mapped["coordinate_source"].tolist(),
            ["source", "inherited_location", "source", "imputed_unit_donor"],
        )
        self.assertEqual(mapped.loc[mapped["UNITID"].eq(5), "imputation_rule"].iloc[0], "cooling_size")

    def test_imputed_dry_writer_materializes_supported_subset_with_provenance(self) -> None:
        plants = pd.DataFrame(
            {
                "UNITID": [1, 2, 3],
                "LOCATIONID": [10, 10, 20],
                "country": ["A", "A", "A"],
                "lat": [10.2, None, None],
                "lon": [20.2, None, None],
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
                "fuel": ["COAL", "COAL", "GAS"],
                "technology_family": ["steam", "steam", "combined_cycle"],
                "cooling_physics": ["dry", "dry", "dry"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            source = f"{tmpdir}/plants.parquet"
            cooling_path = f"{tmpdir}/cooling.parquet"
            target = f"{tmpdir}/dry_imputed.parquet"
            plants.to_parquet(source, index=False)
            cooling.to_parquet(cooling_path, index=False)

            summary = write_imputed_dry_unit_cell_mapping(source, cooling_path, target, random_seed=0)
            written = pd.read_parquet(target)

            self.assertEqual(summary.coordinate_units, 3)
            self.assertIn("coordinate_source", written.columns)
            self.assertIn("imputation_rule", written.columns)
            self.assertEqual(
                written["coordinate_source"].tolist(),
                ["source", "inherited_location", "imputed_unit_donor"],
            )
