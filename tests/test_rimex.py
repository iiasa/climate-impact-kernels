from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from impacts.rimex import (
    attach_warming_levels,
    build_quantile_map,
    build_warming_level_assignments,
)


class RimexWorkflowTest(unittest.TestCase):
    def test_builds_generic_warming_level_assignments_and_quantile_map(self) -> None:
        frame = pd.DataFrame(
            {
                "impact_type": ["dry", "dry", "wet", "wet"],
                "gcm": ["GFDL-ESM4"] * 4,
                "scenario": ["historical", "historical", "ssp126", "ssp126"],
                "time": ["1850-01-01", "1851-01-01", "2015-01-01", "2015-02-01"],
                "region": ["A", "A", "A", "A"],
                "value": [0.4, 0.6, 0.7, 0.9],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pd.DataFrame(
                {
                    "date": ["1850-01-15", "1851-01-15"],
                    "mean_anomaly": [0.1, 0.6],
                }
            ).to_csv(
                root / "gfdl-esm4_historical_monthly_temperature_1850_2014.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "date": ["2015-01-15", "2015-02-15"],
                    "mean_anomaly": [1.0, 1.0],
                }
            ).to_csv(
                root / "gfdl-esm4_ssp126_monthly_temperature_2015_2100.csv",
                index=False,
            )

            assignments = build_warming_level_assignments(frame, gmt_root=root, bin_width=0.5)
            enriched = attach_warming_levels(frame, assignments)
            quantile_map = build_quantile_map(
                enriched,
                value_column="value",
                group_columns=("impact_type", "region"),
                quantiles=(0.5,),
            )

        self.assertEqual(assignments.target_levels, (0.0, 0.5, 1.0))
        self.assertEqual(enriched["warming_level"].tolist(), [0.0, 0.5, 1.0, 1.0])
        self.assertEqual(
            quantile_map["sample_size"].sel(impact_type="wet", region="A", warming_level=1.0).item(),
            2,
        )
        self.assertAlmostEqual(
            quantile_map["value"].sel(impact_type="wet", region="A", warming_level=1.0, quantile=0.5).item(),
            0.8,
        )


if __name__ == "__main__":
    unittest.main()
