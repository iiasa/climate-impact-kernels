from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from thermo_electric_power.mapping import (
    build_country_rollup,
    build_location_cell_mapping,
    build_unit_cell_mapping,
    snap_to_half_degree_cell,
    summarize_mapping,
    write_country_rollup,
    write_location_cell_mapping,
    write_unit_cell_mapping,
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
        rollup = build_country_rollup(mapped, source_resolution="unit")

        self.assertEqual(rollup["country"].tolist(), ["Korea", "Taiwan"])
        self.assertEqual(rollup["entities"].tolist(), [2, 2])
        self.assertEqual(rollup["mw"].tolist(), [150.0, 250.0])
        self.assertEqual(rollup["source_resolution"].tolist(), ["unit", "unit"])

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

            summary = write_country_rollup(source, target, from_resolution="location")
            written = pd.read_parquet(target)

            self.assertEqual(summary.coordinate_units, 3)
            self.assertEqual(
                written.columns.tolist(),
                ["country", "entities", "mw", "source_resolution"],
            )
            self.assertEqual(written["country"].tolist(), ["Korea", "Taiwan"])
            self.assertEqual(written["source_resolution"].tolist(), ["location", "location"])


if __name__ == "__main__":
    unittest.main()
