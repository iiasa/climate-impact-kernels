import os
import unittest

from impacts.climatecore.regions import (
    common_definitions_root,
    list_country_geometry_sources,
    message_r12_country_to_region,
    message_r12_region_labels,
    resolve_region_labels,
)


class RegionRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        if os.environ.get("COMMON_DEFINITIONS_ROOT") is None:
            self.skipTest("COMMON_DEFINITIONS_ROOT is required for region registry tests")

    def test_message_r12_resolution_uses_upstream_mapping_and_aliases(self) -> None:
        resolved = resolve_region_labels(
            ["USA", "VIETNAM", "BOSNIA-HERZEGOVINA", "CHANNEL ISLANDS"],
            region_set="message_r12",
        )

        self.assertEqual(resolved, ("R12_NAM", "R12_RCPA", "R12_EEU", "R12_WEU"))
        self.assertEqual(message_r12_region_labels()["R12_NAM"], "North America")
        self.assertEqual(message_r12_country_to_region()["United States"], "R12_NAM")
        self.assertEqual(common_definitions_root().name, "common-definitions")

    def test_country_geometry_source_registry_contains_natural_earth(self) -> None:
        source = list_country_geometry_sources()[0]

        self.assertEqual(source.source, "Natural Earth")
        self.assertEqual(source.dataset_slug, "ne_110m_admin_0_countries")


if __name__ == "__main__":
    unittest.main()
