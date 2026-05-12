import unittest

import pandas as pd
import xarray as xr

from impacts.climatecore import evaluate_mapped_dataset


class EvaluationTest(unittest.TestCase):
    def test_evaluate_mapped_dataset_runs_full_pipeline(self) -> None:
        evaluated = evaluate_mapped_dataset(
            mapped_entities=pd.DataFrame(
                {
                    "UNITID": [101],
                    "grid_lat": [0.5],
                    "grid_lon": [1.5],
                    "mw": [100.0],
                }
            ),
            climate_inputs=xr.Dataset(
                data_vars={
                    "tas": (
                        ("time", "lat", "lon"),
                        [[[300.15, 301.15]]],
                    )
                },
                coords={"time": [0], "lat": [0.5], "lon": [0.5, 1.5]},
            ),
            normalize_inputs=lambda dataset: dataset.rename({"tas": "air_temperature"}),
            physics_kernel=lambda attached: xr.Dataset(
                data_vars={
                    "capacity_factor": ((1.0 - (attached["air_temperature"] - 300.0))).rename("capacity_factor"),
                }
            ),
            metadata_columns=["UNITID", "mw"],
            clips={"capacity_factor": (0.0, 1.0)},
        )

        self.assertEqual(evaluated["capacity_factor"].dims, ("time", "entity"))
        self.assertEqual(evaluated.coords["UNITID"].values.tolist(), [101])
        self.assertAlmostEqual(evaluated["capacity_factor"].sel(entity=101).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
