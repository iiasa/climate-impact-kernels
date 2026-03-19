from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from impacts.thermoelectric.mapping import (
    build_recovered_unit_cell_mapping,
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


if __name__ == "__main__":
    unittest.main()
