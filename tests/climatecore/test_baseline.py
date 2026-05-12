import unittest

import numpy as np
import xarray as xr

from impacts.climatecore import compute_anomaly


class BaselineTest(unittest.TestCase):
    def test_compute_anomaly_uses_baseline_mean(self) -> None:
        anomalies = compute_anomaly(
            data=xr.DataArray([1.0, 3.0, 5.0], dims=["year"], coords={"year": [2000, 2001, 2002]}),
            baseline_period=(2000, 2001),
        )

        self.assertTrue(np.allclose(anomalies.values, [-1.0, 1.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
