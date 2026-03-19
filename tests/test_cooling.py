from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from impacts.thermoelectric.cooling import (
    build_unit_cooling_mapping,
    classify_technology_family,
    write_unit_cooling_mapping,
)


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


if __name__ == "__main__":
    unittest.main()
