import unittest

import xarray as xr

from impacts.climatecore import normalize_pressure, normalize_temperature, resolve_unit


class UnitsTest(unittest.TestCase):
    def test_resolve_and_normalize_temperature(self) -> None:
        data = xr.DataArray([300.15], dims=["time"], name="tas", attrs={"units": "K"})

        unit = resolve_unit(data)
        normalized = normalize_temperature(data, target_unit="C", unit=unit)

        self.assertEqual(unit, "K")
        self.assertAlmostEqual(normalized.item(), 27.0)

    def test_normalize_pressure(self) -> None:
        normalized = normalize_pressure(
            xr.DataArray([101325.0], dims=["time"], name="ps", attrs={"units": "Pa"}),
            target_unit="kPa",
            unit="Pa",
        )

        self.assertAlmostEqual(normalized.item(), 101.325)


if __name__ == "__main__":
    unittest.main()
