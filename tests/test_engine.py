import unittest
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from impacts.engine import (
    ClimateInput,
    Kernel,
    evaluate_gridded,
    evaluate_mapped,
    prepare_climate,
)


def _capacity_physics(climate: xr.Dataset, sites: xr.Dataset, assumptions: None) -> xr.Dataset:
    return xr.Dataset({"capacity_factor": (1.0 - (climate["air_temperature"] - 300.0)).clip(0.0, 1.0)})


def _sea_physics(climate: xr.Dataset, sites: xr.Dataset, assumptions: None) -> xr.Dataset:
    return xr.Dataset({name: climate[name] for name in climate.data_vars})


def _heat_physics(climate: xr.Dataset, sites: xr.Dataset, assumptions: None) -> xr.Dataset:
    return xr.Dataset(
        {"heat": (climate["air_temperature"] - 300.0) * sites["scale"] - sites["elevation"] / 100.0}
    )


TAS = {"air_temperature": ClimateInput("tas")}
SEA = {
    "land": ClimateInput("land"),
    "sea": ClimateInput("sea", point=("intake_lat", "intake_lon"), optional=True),
}


@dataclass(frozen=True)
class _ToyAssumptions:
    threshold: float = field(default=float("nan"), metadata={"source": "toy threshold"})
    weights: np.ndarray = field(default_factory=lambda: np.array([1.0, 2.0]), metadata={"source": "toy weights"})
    scale: float = field(default=1.0, metadata={"source": "toy scale"})


class EvaluationTest(unittest.TestCase):
    def _toy_sources(self, assumptions: _ToyAssumptions) -> dict[str, object]:
        kernel = Kernel(
            name="toy", inputs=TAS, physics=_capacity_physics, assumptions=_ToyAssumptions()
        )
        evaluated = evaluate_mapped(
            kernel,
            pd.DataFrame({"UNITID": [101], "grid_lat": [0.5], "grid_lon": [1.5]}),
            xr.Dataset(
                data_vars={"tas": (("time", "lat", "lon"), [[[300.15, 301.15]]])},
                coords={"time": [0], "lat": [0.5], "lon": [0.5, 1.5]},
            ),
            "native",
            assumptions=assumptions,
            site_id_column="UNITID",
        )
        return {name: evaluated.attrs[f"{name}_source"] for name in ("threshold", "weights", "scale")}

    def test_defaults_compare_robustly_for_nan_and_arrays(self) -> None:
        # NaN matches a NaN default and arrays compare by content, so defaults keep
        # their own source.
        self.assertEqual(
            self._toy_sources(_ToyAssumptions(weights=np.array([1.0, 2.0]))),
            {"threshold": "toy threshold", "weights": "toy weights", "scale": "toy scale"},
        )
        changed = self._toy_sources(
            _ToyAssumptions(threshold=0.5, weights=np.array([1.0, 3.0]), scale=2.0)
        )
        self.assertEqual(changed["threshold"], "set by the caller; default nan from toy threshold")
        self.assertTrue(changed["weights"].startswith("set by the caller; default array([1., 2.])"))
        self.assertEqual(changed["scale"], "set by the caller; default 1.0 from toy scale")

    def test_mapped_evaluation_runs_full_pipeline(self) -> None:
        kernel = Kernel(name="toy", inputs=TAS, physics=_capacity_physics, site_columns=("UNITID", "mw"))
        evaluated = evaluate_mapped(
            kernel,
            pd.DataFrame({"UNITID": [101], "grid_lat": [0.5], "grid_lon": [1.5], "mw": [100.0]}),
            xr.Dataset(
                data_vars={"tas": (("time", "lat", "lon"), [[[300.15, 301.15]]])},
                coords={"time": [0], "lat": [0.5], "lon": [0.5, 1.5]},
            ),
            "native",
            site_id_column="UNITID",
        )

        self.assertEqual(evaluated["capacity_factor"].dims, ("time", "entity"))
        self.assertEqual(evaluated.coords["UNITID"].values.tolist(), [101])
        self.assertAlmostEqual(evaluated["capacity_factor"].sel(entity=101).item(), 0.0)

    def test_mapped_evaluation_reads_named_variable_at_a_second_point(self) -> None:
        mapping = pd.DataFrame(
            {
                "UNITID": [1, 2, 3],
                "grid_lat": [0.5, 0.5, 0.5],
                "grid_lon": [0.5, 0.5, 0.5],
                "intake_lat": [0.5, float("nan"), 0.5],
                "intake_lon": [1.5, float("nan"), 2.5],
            }
        )
        climate = xr.Dataset(
            data_vars={
                "land": (("time", "lat", "lon"), [[[10.0, 11.0, 12.0]]]),
                "sea": (("time", "lat", "lon"), [[[20.0, 21.0, 22.0]]]),
            },
            coords={"time": [0], "lat": [0.5], "lon": [0.5, 1.5, 2.5]},
        )
        kernel = Kernel(name="toy", inputs=SEA, physics=_sea_physics, site_columns=("UNITID",))
        evaluated = evaluate_mapped(kernel, mapping, climate, "native", site_id_column="UNITID")

        # The land variable stays at the site's own cell; the sea variable moves.
        self.assertEqual(evaluated["land"].isel(time=0).values.tolist(), [10.0, 10.0, 10.0])
        sea = evaluated["sea"].isel(time=0).values.tolist()
        self.assertEqual(sea[0], 21.0)
        self.assertNotEqual(sea[1], sea[1])  # no second point: NaN
        self.assertEqual(sea[2], 22.0)
        self.assertEqual(evaluated["grid_lon"].values.tolist(), [0.5, 0.5, 0.5])

    def test_mapped_evaluation_drops_variable_without_point_columns(self) -> None:
        mapping = pd.DataFrame({"UNITID": [1], "grid_lat": [0.5], "grid_lon": [0.5]})
        climate = xr.Dataset(
            data_vars={
                "land": (("time", "lat", "lon"), [[[10.0]]]),
                "sea": (("time", "lat", "lon"), [[[20.0]]]),
            },
            coords={"time": [0], "lat": [0.5], "lon": [0.5]},
        )
        kernel = Kernel(name="toy", inputs=SEA, physics=_sea_physics)
        self.assertNotIn("sea", evaluate_mapped(kernel, mapping, climate, "native", site_id_column="UNITID").data_vars)

    def test_optional_input_is_left_out_when_absent(self) -> None:
        climate = xr.Dataset(
            data_vars={"land": (("time", "lat", "lon"), [[[10.0]]])},
            coords={"time": [0], "lat": [0.5], "lon": [0.5]},
        )
        kernel = Kernel(name="toy", inputs=SEA, physics=_sea_physics)
        self.assertEqual(list(prepare_climate(kernel, climate).data_vars), ["land"])

    def test_gridded_evaluation_stacks_and_attaches_parameters(self) -> None:
        kernel = Kernel(name="toy", inputs=TAS, physics=_heat_physics)
        evaluated = evaluate_gridded(
            kernel,
            xr.Dataset(
                data_vars={"tas": (("time", "lat", "lon"), [[[300.0, 302.0]]])},
                coords={"time": [0], "lat": [0.5], "lon": [0.5, 1.5]},
            ),
            xr.Dataset(
                data_vars={
                    "elevation": (("lat", "lon"), [[0.0, 100.0]]),
                    "scale": ((), 2.0),
                },
                coords={"lat": [0.5], "lon": [0.5, 1.5]},
            ),
            "native",
        )

        self.assertEqual(evaluated["heat"].dims, ("time", "entity"))
        self.assertEqual(evaluated.sizes["entity"], 2)
        # cell (lat=0.5, lon=0.5): (300-300)*2 - 0/100 = 0
        # cell (lat=0.5, lon=1.5): (302-300)*2 - 100/100 = 3
        self.assertEqual(evaluated["heat"].isel(time=0).values.tolist(), [0.0, 3.0])


if __name__ == "__main__":
    unittest.main()
