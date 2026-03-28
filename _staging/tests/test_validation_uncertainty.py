from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from _staging.thermoelectric.validation_uncertainty import (
    add_size_class,
    assess_benchmark_interval,
    sample_missing_unit_outcomes,
    summarise_metrics,
    synthetic_missingness_mask,
)


class ValidationUncertaintyContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mapped = add_size_class(
            pd.DataFrame(
                {
                    "UNITID": [1, 2, 3, 4],
                    "LOCATIONID": [10, 11, 20, 21],
                    "country": ["A", "A", "B", "B"],
                    "cooling_physics": ["once_through", "once_through", "dry", "dry"],
                    "mw": [120.0, 80.0, 150.0, 90.0],
                    "historical_ucr": [0.8, 0.6, 0.9, 0.7],
                    "future_decline": [False, True, False, True],
                }
            )
        )

    def test_add_size_class_uses_100_mw_threshold(self) -> None:
        classified = add_size_class(pd.DataFrame({"mw": [99.9, 100.0, 100.1]}))

        self.assertEqual(classified["size_class"].tolist(), ["small", "large", "large"])

    def test_summarise_metrics_returns_capacity_weighted_headlines(self) -> None:
        metrics = summarise_metrics(self.mapped)

        self.assertAlmostEqual(metrics["fleet_ucr"], 0.7772727272727272)
        self.assertAlmostEqual(metrics["large_ucr"], 0.8555555555555555)
        self.assertAlmostEqual(metrics["small_ucr"], 0.6529411764705882)
        self.assertAlmostEqual(metrics["high_risk_share"], 0.0)
        self.assertAlmostEqual(metrics["future_decline_share"], 170.0 / 440.0)

    def test_sample_missing_unit_outcomes_uses_hierarchical_backoff(self) -> None:
        donors = self.mapped.loc[:, ["UNITID", "country", "cooling_physics", "mw", "size_class", "historical_ucr", "future_decline"]]
        missing = pd.DataFrame(
            {
                "UNITID": [10, 11],
                "LOCATIONID": [110, 111],
                "country": ["A", "C"],
                "cooling_physics": ["once_through", "dry"],
                "mw": [130.0, 95.0],
                "size_class": ["large", "small"],
            }
        )

        sampled = sample_missing_unit_outcomes(missing, donors, np.random.default_rng(0))

        self.assertEqual(sampled["backoff_rule"].tolist(), ["country_cooling_size", "cooling_size"])
        self.assertEqual(sampled["donor_unitid"].tolist(), [1, 4])
        self.assertEqual(sampled["historical_ucr"].tolist(), [0.8, 0.7])
        self.assertEqual(sampled["future_decline"].tolist(), [False, True])

    def test_synthetic_missingness_mask_preserves_at_least_one_donor_per_group(self) -> None:
        expanded = pd.concat([self.mapped] * 3, ignore_index=True)
        expanded["UNITID"] = np.arange(1, len(expanded) + 1)
        mask = synthetic_missingness_mask(expanded, np.random.default_rng(0), mask_fraction=0.5, min_group_size=3)

        self.assertEqual(mask.dtype, np.dtype(bool))
        grouped = expanded.groupby(["country", "cooling_physics", "size_class"]).indices
        for indices in grouped.values():
            masked_count = int(mask[np.asarray(indices)].sum())
            self.assertGreaterEqual(masked_count, 1)
            self.assertLess(masked_count, len(indices))

    def test_assess_benchmark_interval_flags_material_unspanned_gap(self) -> None:
        assessment = assess_benchmark_interval(
            {"mean": 0.80, "p05": 0.79, "p95": 0.81},
            benchmark=0.72,
            synthetic_mae_pp=0.2,
        )

        self.assertFalse(assessment.benchmark_spanned)
        self.assertAlmostEqual(assessment.residual_gap_pp, 8.0)
        self.assertAlmostEqual(assessment.interval_width_pp, 2.0)
        self.assertAlmostEqual(assessment.tolerance_pp, 1.2)
        self.assertTrue(assessment.warrants_latent_location)
        self.assertEqual(assessment.direction_vs_benchmark, "above")


if __name__ == "__main__":
    unittest.main()
