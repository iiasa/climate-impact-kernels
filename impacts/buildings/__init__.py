"""Building cooling-energy demand: the CHILLED kernel, its building archetype inputs, country aggregation and warming-level summaries."""

from impacts.buildings.archetypes import (
    build_chilled_archetype_mapping,
    build_chilled_parameter_grids,
    load_ewembi_solar_climatology,
    load_message_ix_buildings_archetypes,
    load_message_ix_buildings_parametric_runs,
    load_r12_region_raster,
    load_region_cooling_setpoints,
    normalize_chilled_solar_climatology,
    validate_archetype_table,
)
from impacts.buildings.chilled import (
    CHILLED_KERNEL,
    CHILLED_REGION_SETPOINT_KERNEL,
    ChilledEnergyAssumptions,
    chilled_energy,
)
from impacts.buildings.regions import (
    aggregate_chilled_to_country,
    aggregate_site_to_country,
    expand_chilled_metadata,
    load_country_raster,
    parse_chilled_matrix_filename,
)
from impacts.buildings.warming_levels import (
    buildings_warming_level_summaries,
    country_land_area,
    country_population,
    load_iso3_to_r12,
    write_buildings_warming_level_summaries,
)

__all__ = [
    "CHILLED_KERNEL",
    "CHILLED_REGION_SETPOINT_KERNEL",
    "ChilledEnergyAssumptions",
    "aggregate_chilled_to_country",
    "aggregate_site_to_country",
    "build_chilled_archetype_mapping",
    "build_chilled_parameter_grids",
    "buildings_warming_level_summaries",
    "chilled_energy",
    "country_land_area",
    "country_population",
    "expand_chilled_metadata",
    "load_country_raster",
    "load_ewembi_solar_climatology",
    "load_iso3_to_r12",
    "load_message_ix_buildings_archetypes",
    "load_message_ix_buildings_parametric_runs",
    "load_r12_region_raster",
    "load_region_cooling_setpoints",
    "normalize_chilled_solar_climatology",
    "parse_chilled_matrix_filename",
    "validate_archetype_table",
    "write_buildings_warming_level_summaries",
]
