from __future__ import annotations

import unittest

from impacts.thermoelectric.dry import classify_dry_cooling_engine_type
from impacts.thermoelectric.fuels import classify_fuel
from impacts.thermoelectric.wet import (
    _classify_wet_fuel_type,
)

# The eight codes of the WEPP-derived fleet, with the wet class and dry steam family each must get.
FLEET_CODES = {
    "COAL": ("coal", "combustible_steam"),
    "OIL": ("oil", "combustible_steam"),
    "UR": ("nuclear", "nuclear_steam"),
    "LNG": ("gas", "combustible_steam"),
    "LGAS": ("gas", "combustible_steam"),
    "COKE": ("coal", "combustible_steam"),
    "KERO": ("oil", "combustible_steam"),
    "BGAS": ("gas", "combustible_steam"),
}

# Strings that contain a known code as a substring but are not that fuel.
LOOKALIKES = ["BLAST FURNACE GAS", "SUR", "BFG", "COG", "MSW", "REF", "BL", "HYDRO", "mystery", ""]


class FuelClassTest(unittest.TestCase):
    def test_fleet_codes_get_the_same_class_in_wet_and_dry(self) -> None:
        for code, (wet_class, dry_family) in FLEET_CODES.items():
            with self.subTest(code=code):
                self.assertEqual(_classify_wet_fuel_type(code, "steam"), wet_class)
                self.assertEqual(
                    classify_dry_cooling_engine_type(technology_family="steam", fuel=code)[0],
                    dry_family,
                )

    def test_codes_are_case_insensitive(self) -> None:
        self.assertEqual(classify_fuel("coal"), classify_fuel("COAL"))

    def test_lookalike_strings_are_not_matched_by_substring(self) -> None:
        for fuel in LOOKALIKES:
            with self.subTest(fuel=fuel):
                with self.assertRaisesRegex(ValueError, "unknown fuel"):
                    _classify_wet_fuel_type(fuel, "steam")
                with self.assertRaisesRegex(ValueError, "unknown fuel"):
                    classify_dry_cooling_engine_type(technology_family="steam", fuel=fuel)

    def test_waste_heat_steam_follows_the_explicit_technology_rule(self) -> None:
        self.assertEqual(_classify_wet_fuel_type("WSTH", "steam"), "steam_bottoming")
        self.assertEqual(_classify_wet_fuel_type("WSTH", "combined_cycle"), "gas")
        self.assertEqual(
            classify_dry_cooling_engine_type(technology_family="steam", fuel="WSTH")[0],
            "non_combustible_steam",
        )

    def test_kerosene_is_combustible_in_dry_steam(self) -> None:
        self.assertEqual(
            classify_dry_cooling_engine_type(technology_family="steam", fuel="KERO")[0],
            "combustible_steam",
        )


if __name__ == "__main__":
    unittest.main()
