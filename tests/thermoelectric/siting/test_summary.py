from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from impacts.thermoelectric.siting.summary import (
    _census_metrics,
    compare_potential_to_existing_sites,
    summarize_site_evaluation,
)

from .fixtures import COAL_ONCE_THROUGH


class BootstrapStandardErrorTest(unittest.TestCase):
    @staticmethod
    def _census(years: list[int]) -> pd.DataFrame:
        rows = []
        for year in years:
            for index, cell in enumerate(("a", "b", "c")):
                rows.append(
                    {
                        "population": "candidate",
                        "region_id": "AFR",
                        "archetype": COAL_ONCE_THROUGH,
                        "gcm": "GFDL-ESM4",
                        "season": "JJA",
                        "warming_level": 2.0,
                        "response": "capacity_factor",
                        "unit": "1",
                        "scenario": "ssp585",
                        "year": year,
                        "cell_id": cell,
                        "cell_area_m2": 1.0e9,
                        "value": 0.5 + 0.1 * index + 0.01 * (year % 7),
                    }
                )
        return pd.DataFrame(rows)

    def test_a_single_sampled_year_has_no_standard_error(self) -> None:
        summary = summarize_site_evaluation(self._census([2030]), bootstrap_replicates=20)

        mean = summary[summary["metric"] == "area_weighted_mean"]
        self.assertEqual(len(mean), 1)
        self.assertTrue(np.isnan(mean["sampling_standard_error"].item()))

    def test_several_sampled_years_have_a_standard_error(self) -> None:
        summary = summarize_site_evaluation(self._census([2030, 2031, 2032, 2033]), bootstrap_replicates=50)

        mean = summary[summary["metric"] == "area_weighted_mean"]
        self.assertTrue(np.isfinite(mean["sampling_standard_error"].item()))

    def test_grid_to_fleet_comparison_needs_both_populations(self) -> None:
        census = self._census([2030, 2031])
        candidate_only = summarize_site_evaluation(census, bootstrap_replicates=0)
        with self.assertRaisesRegex(
            ValueError,
            r"the summary has no incumbent population \(found \['candidate'\]\); "
            "comparing potential sites with existing sites needs both",
        ):
            compare_potential_to_existing_sites(candidate_only)

        incumbent_only = summarize_site_evaluation(
            census.assign(population="incumbent"), bootstrap_replicates=0
        )
        with self.assertRaisesRegex(ValueError, "the summary has no candidate population"):
            compare_potential_to_existing_sites(incumbent_only)

        both = summarize_site_evaluation(
            pd.concat([census, census.assign(population="incumbent", value=census["value"] + 0.1)]),
            bootstrap_replicates=0,
        )
        verdict = compare_potential_to_existing_sites(both)
        mean = verdict[verdict["metric"] == "area_weighted_mean"]
        self.assertAlmostEqual(mean["fleet_minus_grid"].item(), 0.1)


class CensusAreaTest(unittest.TestCase):
    def test_area_total_counts_distinct_cells(self) -> None:
        # Two cells, each appearing in two sampled years -> four cell-year rows.
        data = pd.DataFrame(
            {
                "cell_id": ["a", "a", "b", "b"],
                "value": [0.9, 0.7, 0.6, 0.5],
                "cell_area_m2": [100.0, 100.0, 200.0, 200.0],
            }
        )
        metrics = _census_metrics(data, "capacity_factor", {"capacity_factor": 0.8})
        self.assertEqual(metrics["area_total_m2"], 300.0)
        self.assertEqual(metrics["n_cells"], 2.0)
        # The share of cell area below the threshold is a ratio in which the year multiplicity cancels.
        self.assertAlmostEqual(metrics["bad_tail_area_fraction"], 5.0 / 6.0)


if __name__ == "__main__":
    unittest.main()
