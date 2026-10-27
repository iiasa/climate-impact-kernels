import unittest

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from impacts.climatecore import (
    assign_years_to_levels,
    bin_by_warming_level,
    pool_by_warming_level,
    weighted_group_mean,
)


class WarmingLevelTest(unittest.TestCase):
    def test_assign_years_to_levels(self) -> None:
        assignments = assign_years_to_levels(
            gmt=xr.DataArray(
                [0.9, 1.2, 1.8],
                dims=["year"],
                coords={"year": [2000, 2001, 2002]},
            ),
            target_levels=[1.0, 2.0],
        )

        self.assertEqual(assignments["warming_level"].values.tolist(), [1.0, 1.0, 2.0])

    def test_bin_by_warming_level_groups_values(self) -> None:
        bins = bin_by_warming_level(
            values=xr.DataArray(
                [10.0, 11.0, 12.0],
                dims=["year"],
                coords={"year": [2000, 2001, 2002]},
            ),
            gmt=xr.DataArray(
                [0.9, 1.2, 1.8],
                dims=["year"],
                coords={"year": [2000, 2001, 2002]},
            ),
            target_levels=[1.0, 2.0],
        )

        self.assertTrue(np.allclose(bins.values, [10.5, 12.0]))


def test_empty_target_level_has_nan_statistics_and_zero_count():
    gcm, year = ["a", "b"], [2000, 2001, 2002]
    values = xr.DataArray(
        np.arange(6.0).reshape(2, 1, 3),
        dims=("gcm", "scenario", "year"),
        coords={"gcm": gcm, "scenario": ["s"], "year": year},
    )
    assignments = xr.Dataset(
        {"warming_level": (("gcm", "scenario", "year"), np.array([[[1.0, 1.0, 2.0]], [[1.0, 2.0, np.nan]]]))},
        coords={"gcm": gcm, "scenario": ["s"], "year": year},
    )
    summary = pool_by_warming_level(values, assignments, [1.0, 1.5, 2.0], name="v")
    assert summary["n_obs"].dtype == np.int64
    assert summary["n_obs"].values.tolist() == [3, 0, 2]
    assert np.isnan(summary["v"].sel(gwl=1.5))
    assert summary["v"].sel(gwl=1.0).item() == pytest.approx((0 + 1 + 3) / 3)
    assert list(summary.data_vars) == ["v", "v_std", "v_p10", "v_p50", "v_p90", "n_obs"]


def test_weighted_group_mean_skips_missing_values_and_weightless_entities():
    frame = pd.DataFrame(
        {
            "country": ["a", "b", "c", "d", "a"],
            "region": ["r", "r", "r", "r", "s"],
            "v": [1.0, np.nan, 5.0, 100.0, np.nan],
        }
    )
    weights = pd.Series({"a": 1.0, "b": 10.0, "c": 3.0})
    out = weighted_group_mean(
        frame, group_keys=["region"], value_columns=["v"], weights=weights
    )
    assert out["region"].tolist() == ["r", "s"]
    assert out["v"].iloc[0] == pytest.approx((1.0 * 1 + 3.0 * 5.0) / 4.0)
    assert np.isnan(out["v"].iloc[1])
