"""The table of library functions the end-to-end tests call, and where each is found.

A test calls ``get("<stable name>")`` and receives a callable. The table below
maps each stable name to the module and attribute that provide it. When a
function moves or is renamed, only its row here changes (or a small wrapper in
``CALLS``, when its call shape differs); the tests do not.

Entries in ``CALLS`` are composite entry points whose call shape belongs to the
test code: a kernel evaluated end to end through a kernel factory taking
assumptions and chunk widths and a per-kernel climate normalizer, using
``impacts.engine.Kernel``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

# stable name -> (module, attribute)
TABLE: dict[str, tuple[str, str]] = {
    # climatecore: units, temporal, spatial, stats
    "units.resolve_unit": ("impacts.climatecore", "resolve_unit"),
    "units.normalize_temperature": ("impacts.climatecore", "normalize_temperature"),
    "units.normalize_pressure": ("impacts.climatecore", "normalize_pressure"),
    "units.convert_units": ("impacts.climatecore", "convert_units"),
    "temporal.aggregate_monthly": ("impacts.climatecore", "aggregate_monthly"),
    "temporal.aggregate_seasonal": ("impacts.climatecore", "aggregate_seasonal"),
    "spatial.grid_cell_area": ("impacts.climatecore", "grid_cell_area"),
    "spatial.attach_gridded_dataset": ("impacts.climatecore", "attach_gridded_dataset"),
    "spatial.attach_gridded_dataarray": ("impacts.climatecore", "attach_gridded_dataarray"),
    "spatial.refine_to_grid": ("impacts.climatecore", "refine_to_grid"),
    "spatial.nearest_valid_cells": ("impacts.climatecore", "nearest_valid_cells"),
    "stats.weighted_group_mean": ("impacts.climatecore", "weighted_group_mean"),
    # climatecore: warming levels
    "warming.assign_years_to_levels": ("impacts.climatecore", "assign_years_to_levels"),
    "warming.bin_by_warming_level": ("impacts.climatecore", "bin_by_warming_level"),
    "warming.build_warming_level_assignments": (
        "impacts.climatecore",
        "build_warming_level_assignments",
    ),
    "warming.pool_by_warming_level": ("impacts.climatecore", "pool_by_warming_level"),
    # climatecore: regions
    "regions.message_r12_region_labels": (
        "impacts.climatecore.regions",
        "message_r12_region_labels",
    ),
    "regions.message_r12_country_to_region": (
        "impacts.climatecore.regions",
        "message_r12_country_to_region",
    ),
    "regions.resolve_region_labels": ("impacts.climatecore.regions", "resolve_region_labels"),
    "regions.aggregate_entity_dataarray": (
        "impacts.climatecore.regions",
        "aggregate_site_dataarray",
    ),
    "regions.aggregate_entity_dataset": (
        "impacts.climatecore.regions",
        "aggregate_site_dataset",
    ),
    # evaluation engine and ``Kernel``
    "kernel.ClimateInput": ("impacts.engine", "ClimateInput"),
    "kernel.prepare_climate": ("impacts.engine", "prepare_climate"),
    "kernel.evaluate_mapped": ("impacts.engine", "evaluate_mapped"),
    "kernel.evaluate_gridded": ("impacts.engine", "evaluate_gridded"),
    "kernel.ImpactKernel": ("impacts.engine", "Kernel"),
    # ISIMIP
    "isimip.ISIMIPVariable": ("impacts.isimip", "ISIMIPVariable"),
    "isimip.ISIMIPPaths": ("impacts.isimip", "ISIMIPPaths"),
    "isimip.isimip3b_paths": ("impacts.isimip", "isimip3b_paths"),
    "isimip.isimip3a_counterclim_paths": ("impacts.isimip", "isimip3a_counterclim_paths"),
    "isimip.discover_isimip_files": ("impacts.isimip", "discover_isimip_files"),
    "isimip.load_isimip_variable": ("impacts.isimip", "load_isimip_variable"),
    "isimip.load_isimip_dataset": ("impacts.isimip", "load_isimip_dataset"),
    "isimip.load_isimip_gmt": ("impacts.isimip", "load_isimip_gmt"),
    "isimip.isimip_warming_level_assignments": (
        "impacts.isimip",
        "isimip_warming_level_assignments",
    ),
    # warming-level summaries (buildings energy intensity, thermoelectric capacity)
    "surfaces.load_iso3_to_r12": (
        "impacts.buildings.warming_levels",
        "load_iso3_to_r12",
    ),
    "surfaces.country_population": (
        "impacts.buildings.warming_levels",
        "country_population",
    ),
    "surfaces.country_land_area": (
        "impacts.buildings.warming_levels",
        "country_land_area",
    ),
    "surfaces.buildings": (
        "impacts.buildings.warming_levels",
        "buildings_warming_level_summaries",
    ),
    "surfaces.write_buildings": (
        "impacts.buildings.warming_levels",
        "write_buildings_warming_level_summaries",
    ),
    "surfaces.build_country_name_to_r12": (
        "impacts.thermoelectric.warming_levels",
        "build_country_name_to_r12",
    ),
    "surfaces.thermoelectric": (
        "impacts.thermoelectric.warming_levels",
        "thermoelectric_warming_level_summary",
    ),
    "surfaces.write_thermoelectric": (
        "impacts.thermoelectric.warming_levels",
        "write_thermoelectric_warming_level_summary",
    ),
    # thermoelectric: fleet mapping and cooling classification
    "fleet.snap_to_half_degree_cell": (
        "impacts.thermoelectric.fleet",
        "snap_to_half_degree_cell",
    ),
    "fleet.build_unit_cell_mapping": (
        "impacts.thermoelectric.fleet",
        "build_unit_cell_mapping",
    ),
    "fleet.build_recovered_unit_cell_mapping": (
        "impacts.thermoelectric.fleet",
        "build_recovered_unit_cell_mapping",
    ),
    "fleet.build_imputed_wet_unit_cell_mapping": (
        "impacts.thermoelectric.fleet",
        "build_imputed_wet_unit_cell_mapping",
    ),
    "fleet.build_imputed_dry_unit_cell_mapping": (
        "impacts.thermoelectric.fleet",
        "build_imputed_dry_unit_cell_mapping",
    ),
    "fleet.summarize_location_ambiguity": (
        "impacts.thermoelectric.fleet",
        "summarize_location_ambiguity",
    ),
    "fleet.summarize_mapping": ("impacts.thermoelectric.fleet", "summarize_mapping"),
    "fleet.build_location_cell_mapping": (
        "impacts.thermoelectric.fleet",
        "build_location_cell_mapping",
    ),
    "fleet.build_country_rollup": (
        "impacts.thermoelectric.fleet",
        "build_country_totals",
    ),
    "fleet.load_and_map_thermal_plants": (
        "impacts.thermoelectric.fleet",
        "load_and_map_thermal_plants",
    ),
    "fleet.write_unit_cell_mapping": (
        "impacts.thermoelectric.fleet",
        "write_unit_cell_mapping",
    ),
    "fleet.write_recovered_unit_cell_mapping": (
        "impacts.thermoelectric.fleet",
        "write_recovered_unit_cell_mapping",
    ),
    "fleet.write_imputed_wet_unit_cell_mapping": (
        "impacts.thermoelectric.fleet",
        "write_imputed_wet_unit_cell_mapping",
    ),
    "fleet.write_imputed_dry_unit_cell_mapping": (
        "impacts.thermoelectric.fleet",
        "write_imputed_dry_unit_cell_mapping",
    ),
    "fleet.write_location_cell_mapping": (
        "impacts.thermoelectric.fleet",
        "write_location_cell_mapping",
    ),
    "fleet.write_country_rollup": (
        "impacts.thermoelectric.fleet",
        "write_country_totals",
    ),
    "cooling.classify_technology_family": (
        "impacts.thermoelectric.fleet",
        "classify_technology_family",
    ),
    "cooling.build_unit_cooling_mapping": (
        "impacts.thermoelectric.fleet",
        "build_unit_cooling_mapping",
    ),
    "cooling.write_unit_cooling_mapping": (
        "impacts.thermoelectric.fleet",
        "write_unit_cooling_mapping",
    ),
    # thermoelectric: wet and dry kernels
    "wet.kernel": ("impacts.thermoelectric.wet", "WET_COOLING_KERNEL"),
    "wet.evaluate": ("impacts.thermoelectric.wet", "wet_cooling"),
    "wet.build_mapping": ("impacts.thermoelectric.wet", "build_wet_cooling_mapping"),
    "wet.assign_saline_intakes": ("impacts.thermoelectric.wet", "assign_saline_intakes"),
    "wet.Assumptions": ("impacts.thermoelectric.wet", "WetCoolingAssumptions"),
    "wet.fuel_efficiency": ("impacts.thermoelectric.wet", "fuel_efficiency"),
    "dry.kernel": ("impacts.thermoelectric.dry", "DRY_COOLING_KERNEL"),
    "dry.evaluate": ("impacts.thermoelectric.dry", "dry_cooling"),
    "dry.build_mapping": ("impacts.thermoelectric.dry", "build_dry_cooling_mapping"),
    "dry.Assumptions": ("impacts.thermoelectric.dry", "DryCoolingAssumptions"),
    "dry.classify_engine_type": (
        "impacts.thermoelectric.dry",
        "classify_dry_cooling_engine_type",
    ),
    # thermoelectric: regional aggregation
    "aggregation.aggregate_regions": (
        "impacts.thermoelectric.regions",
        "aggregate_thermoelectric_regions",
    ),
    "aggregation.aggregate_file": (
        "impacts.thermoelectric.regions",
        "aggregate_thermoelectric_file",
    ),
    "aggregation.write_aggregated_file": (
        "impacts.thermoelectric.regions",
        "write_aggregated_thermoelectric_file",
    ),
    # thermoelectric: evaluation of synthetic plants
    "archetypes.default_wet": (
        "impacts.thermoelectric.siting.cells",
        "default_wet_synthetic_plants",
    ),
    "archetypes.default_dry": (
        "impacts.thermoelectric.siting.cells",
        "default_dry_synthetic_plants",
    ),
    "archetypes.build_synthetic_entities": (
        "impacts.thermoelectric.siting.cells",
        "build_synthetic_sites",
    ),
    "archetypes.evaluate_wet": (
        "impacts.thermoelectric.siting.cells",
        "evaluate_wet_potential_sites",
    ),
    "archetypes.evaluate_wet_seasonal_and_monthly": (
        "impacts.thermoelectric.siting.cells",
        "evaluate_wet_potential_sites_seasonal_and_monthly",
    ),
    "archetypes.evaluate_dry": (
        "impacts.thermoelectric.siting.cells",
        "evaluate_dry_potential_sites",
    ),
    # cells considered and audit
    "universe.RegionRasterPaths": ("impacts.thermoelectric.siting.cells", "RegionRasterPaths"),
    "universe.build_region_country_rasters": (
        "impacts.thermoelectric.siting.cells",
        "build_region_country_rasters",
    ),
    "universe.build_valid_cell_universe": (
        "impacts.thermoelectric.siting.cells",
        "build_valid_cells",
    ),
    "universe.extract_fleet_cells": ("impacts.thermoelectric.siting.cells", "extract_existing_site_cells"),
    "universe.tag_cells": ("impacts.thermoelectric.siting.cells", "tag_cells"),
    "audit.run_census": ("impacts.thermoelectric.siting.site_evaluation", "run_site_evaluation"),
    "audit.CensusSummary": ("impacts.thermoelectric.siting.site_evaluation", "SiteEvaluationSummary"),
    "audit.reference_climate_paths": (
        "impacts.thermoelectric.siting.site_evaluation",
        "reference_climate_paths",
    ),
    "audit.GWLSampling": ("impacts.thermoelectric.siting.site_evaluation", "GWLSampling"),
    "audit.ReferenceSampling": (
        "impacts.thermoelectric.siting.site_evaluation",
        "ReferenceSampling",
    ),
    "audit.sample_years_within_gwl": (
        "impacts.thermoelectric.siting.site_evaluation",
        "sample_years_within_gwl",
    ),
    "audit.sample_years_within_reference": (
        "impacts.thermoelectric.siting.site_evaluation",
        "sample_years_within_reference",
    ),
    "audit.summarize_census": (
        "impacts.thermoelectric.siting.summary",
        "summarize_site_evaluation",
    ),
    "audit.compare_grid_to_fleet": (
        "impacts.thermoelectric.siting.summary",
        "compare_potential_to_existing_sites",
    ),
    # splitting the wet capacity loss, and output files
    "decomposition.build_low_gwl_reference": (
        "impacts.thermoelectric.siting.loss_split",
        "build_low_warming_reference",
    ),
    "decomposition.decompose_censuses": (
        "impacts.thermoelectric.siting.loss_split",
        "split_site_evaluations",
    ),
    "export.load_cell_means": (
        "impacts.thermoelectric.siting.curves",
        "_load_cell_means",
    ),
    "export.load_decomposition_samples": (
        "impacts.thermoelectric.siting.curves",
        "_load_decomposition_samples",
    ),
    "export.load_reach_water_ceilings": (
        "impacts.thermoelectric.siting.curves",
        "_load_reach_water_ceilings",
    ),
    "export.load_topology": (
        "impacts.thermoelectric.siting.curves",
        "_load_topology",
    ),
    "export.build_capacity_curves": (
        "impacts.thermoelectric.siting.curves",
        "build_capacity_curves",
    ),
    "export.build_pooled_median_curves": (
        "impacts.thermoelectric.siting.curves",
        "build_pooled_median_curves",
    ),
    "export.reference_fleet_additivity_oracle": (
        "impacts.thermoelectric.siting.curves",
        "reference_fleet_additivity_check",
    ),
    "export.validate_reference_fleet_oracle": (
        "impacts.thermoelectric.siting.curves",
        "validate_reference_fleet_additivity_check",
    ),
    "export.build_fixed_cell_ordering": (
        "impacts.thermoelectric.siting.curves",
        "build_fixed_cell_ordering",
    ),
    "export.build_residual_water_cell_ordering": (
        "impacts.thermoelectric.siting.curves",
        "build_residual_water_cell_ordering",
    ),
    "export.residual_water_reference_samples": (
        "impacts.thermoelectric.siting.curves",
        "residual_water_reference_samples",
    ),
    "export.attach_incumbent_water_demand": (
        "impacts.thermoelectric.siting.curves",
        "attach_existing_site_water_demand",
    ),
    "export.build_reach_aware_candidate_curves": (
        "impacts.thermoelectric.siting.curves",
        "build_potential_site_curves_with_river_network",
    ),
    "export.build_reach_aware_candidate_sample_curves": (
        "impacts.thermoelectric.siting.curves",
        "build_potential_site_sample_curves_with_river_network",
    ),
    "export.average_reach_aware_candidate_sample_curves": (
        "impacts.thermoelectric.siting.curves",
        "average_potential_site_sample_curves_with_river_network",
    ),
    "export.export_wet_ucr_curves": (
        "impacts.thermoelectric.siting.curves",
        "export_wet_usable_capacity_curves",
    ),
    "rime.build_incumbent_sample_endpoints": (
        "impacts.thermoelectric.siting.rime",
        "build_existing_site_totals",
    ),
    "rime.build_wet_rime_samples": (
        "impacts.thermoelectric.siting.rime",
        "build_wet_rime_samples",
    ),
    "rime.build_wet_rime_samples_from_curves": (
        "impacts.thermoelectric.siting.rime",
        "build_wet_rime_samples_from_curves",
    ),
    "rime.build_gwl_window_percentiles": (
        "impacts.thermoelectric.siting.rime",
        "build_warming_level_bin_percentiles",
    ),
    "rime.percentiles_to_dataset": (
        "impacts.thermoelectric.siting.rime",
        "percentiles_to_dataset",
    ),
    "rime.export_wet_rime_samples": (
        "impacts.thermoelectric.siting.rime",
        "export_wet_rime_samples",
    ),
    "rime.export_wet_rime_dataset": (
        "impacts.thermoelectric.siting.rime",
        "export_wet_rime_dataset",
    ),
    # composition of the wet-cooling outputs
    "products.build_wet_candidate_products": (
        "impacts.thermoelectric.siting.outputs",
        "build_wet_siting_outputs",
    ),
    "products.write_wet_candidate_products": (
        "impacts.thermoelectric.siting.outputs",
        "write_wet_siting_outputs",
    ),
    "products.incumbent_candidate_overlap": (
        "impacts.thermoelectric.siting.outputs",
        "existing_potential_site_overlap",
    ),
    "products.build_residual_water_ordering": (
        "impacts.thermoelectric.siting.outputs",
        "build_residual_water_ordering",
    ),
    "products.write_residual_water_ordering": (
        "impacts.thermoelectric.siting.outputs",
        "write_residual_water_ordering",
    ),
    "products.pool_wet_candidate_products": (
        "impacts.thermoelectric.siting.outputs",
        "pool_wet_siting_outputs",
    ),
    "products.write_pooled_wet_products": (
        "impacts.thermoelectric.siting.outputs",
        "write_pooled_wet_outputs",
    ),
    "products.finalize_wet_candidate_products": (
        "impacts.thermoelectric.siting.outputs",
        "finalize_wet_siting_outputs",
    ),
    "products.read_gcm_outputs": (
        "impacts.thermoelectric.siting.outputs",
        "read_gcm_outputs",
    ),
    "products.pool": ("impacts.thermoelectric.siting.outputs", "_pool"),
    "products.moves": ("impacts.thermoelectric.siting.outputs", "_moves"),
    # river network, splitting the wet capacity loss and site evaluation
    "topology.read_candidate_cells": (
        "impacts.thermoelectric.siting.river_network",
        "read_potential_site_cells",
    ),
    "topology.read_flow_directions": (
        "impacts.thermoelectric.siting.river_network",
        "read_flow_directions",
    ),
    "topology.build_reach_topologies": (
        "impacts.thermoelectric.siting.river_network",
        "build_river_networks",
    ),
    "topology.write_reach_topologies": (
        "impacts.thermoelectric.siting.river_network",
        "write_river_networks",
    ),
    "topology.assign_downstream": (
        "impacts.thermoelectric.siting.river_network",
        "assign_downstream",
    ),
    "decomposition.decompose_wet_channels": (
        "impacts.thermoelectric.siting.loss_split",
        "split_wet_losses",
    ),
    "decomposition.DEFAULT_BASELINE_REFERENCE": (
        "impacts.thermoelectric.siting.loss_split",
        "DEFAULT_REFERENCE",
    ),
    "decomposition.REF_PD_BASELINE_REFERENCE": (
        "impacts.thermoelectric.siting.loss_split",
        "PRESENT_DAY_REFERENCE",
    ),
    "audit.REFERENCE_SAMPLE_STARTS": (
        "impacts.thermoelectric.siting.site_evaluation",
        "REFERENCE_SAMPLE_STARTS",
    ),
    "audit.build_census_cells": (
        "impacts.thermoelectric.siting.site_evaluation",
        "build_site_cells",
    ),
    "audit.CensusCells": ("impacts.thermoelectric.siting.site_evaluation", "SiteCells"),
    "audit.evaluate_candidate_census": (
        "impacts.thermoelectric.siting.site_evaluation",
        "evaluate_sites",
    ),
    "audit.write_candidate_census": (
        "impacts.thermoelectric.siting.site_evaluation",
        "write_site_evaluation",
    ),
    # water shared along the river
    "reach.topological_order": ("impacts.thermoelectric.siting.river_network", "topological_order"),
    "reach.reach_capped_allocation": (
        "impacts.thermoelectric.siting.river_network",
        "river_network_capped_allocation",
    ),
    "reach.cumulative_upstream_usage": (
        "impacts.thermoelectric.siting.river_network",
        "cumulative_upstream_usage",
    ),
    "reach.residual_ceilings": ("impacts.thermoelectric.siting.river_network", "remaining_water_limits"),
    # RIME-X input loading of the thermoelectric and buildings warming-level summaries
    "rimex_thermo.load_thermoelectric_ucr": (
        "impacts.thermoelectric.warming_levels",
        "load_thermoelectric_usable_capacity_ratio",
    ),
    "rimex_thermo.aggregate_thermoelectric_seasonal": (
        "impacts.thermoelectric.warming_levels",
        "aggregate_thermoelectric_seasonal",
    ),
    "rimex_buildings.parse_chilled_matrix_filename": (
        "impacts.buildings.regions",
        "parse_chilled_matrix_filename",
    ),
    "rimex_buildings.aggregate_chilled_to_country": (
        "impacts.buildings.regions",
        "aggregate_chilled_to_country",
    ),
    "rimex_buildings.expand_chilled_metadata": (
        "impacts.buildings.regions",
        "expand_chilled_metadata",
    ),
    # buildings: CHILLED
    "country.load_country_raster": ("impacts.buildings.regions", "load_country_raster"),
    "country.aggregate_entity_to_country": (
        "impacts.buildings.regions",
        "aggregate_site_to_country",
    ),
    "chilled.Assumptions": ("impacts.buildings.chilled", "ChilledEnergyAssumptions"),
    "chilled.kernel": ("impacts.buildings.chilled", "CHILLED_KERNEL"),
    "chilled.evaluate": ("impacts.buildings.chilled", "chilled_energy"),
    "chilled.region_setpoint_kernel": ("impacts.buildings.chilled", "CHILLED_REGION_SETPOINT_KERNEL"),
    "chilled.build_archetype_mapping": (
        "impacts.buildings.archetypes",
        "build_chilled_archetype_mapping",
    ),
    "chilled.build_parameter_grids": (
        "impacts.buildings.archetypes",
        "build_chilled_parameter_grids",
    ),
    "chilled.validate_archetype_table": (
        "impacts.buildings.archetypes",
        "validate_archetype_table",
    ),
    "chilled.normalize_solar_climatology": (
        "impacts.buildings.archetypes",
        "normalize_chilled_solar_climatology",
    ),
    "chilled.load_archetypes": (
        "impacts.buildings.archetypes",
        "load_message_ix_buildings_archetypes",
    ),
    "chilled.load_parametric_runs": (
        "impacts.buildings.archetypes",
        "load_message_ix_buildings_parametric_runs",
    ),
    "chilled.load_region_cooling_setpoints": (
        "impacts.buildings.archetypes",
        "load_region_cooling_setpoints",
    ),
    "chilled.load_ewembi_solar_climatology": (
        "impacts.buildings.archetypes",
        "load_ewembi_solar_climatology",
    ),
    "chilled.load_r12_region_raster": (
        "impacts.buildings.archetypes",
        "load_r12_region_raster",
    ),
    # public constants and result types re-exported by the package __init__ modules
    "types.CellMappingSummary": ("impacts.thermoelectric", "CellMappingSummary"),
    "types.LocationAmbiguity": ("impacts.thermoelectric", "LocationAmbiguity"),
    "constants.CLIMATOLOGICAL_SEASONS": ("impacts.climatecore", "CLIMATOLOGICAL_SEASONS"),
    "constants.DEFAULT_COOLING_COLUMNS": ("impacts.thermoelectric", "DEFAULT_COOLING_COLUMNS"),
    "constants.DEFAULT_COUNTRY_COLUMNS": ("impacts.thermoelectric", "DEFAULT_COUNTRY_COLUMNS"),
    "constants.DEFAULT_IMPUTED_DRY_MAPPING_COLUMNS": (
        "impacts.thermoelectric",
        "DEFAULT_IMPUTED_DRY_MAPPING_COLUMNS",
    ),
    "constants.DEFAULT_IMPUTED_WET_MAPPING_COLUMNS": (
        "impacts.thermoelectric",
        "DEFAULT_IMPUTED_WET_MAPPING_COLUMNS",
    ),
    "constants.DEFAULT_LOCATION_COLUMNS": ("impacts.thermoelectric", "DEFAULT_LOCATION_COLUMNS"),
    "constants.DEFAULT_MAPPING_COLUMNS": ("impacts.thermoelectric", "DEFAULT_MAPPING_COLUMNS"),
    "constants.DEFAULT_RECOVERED_MAPPING_COLUMNS": (
        "impacts.thermoelectric",
        "DEFAULT_RECOVERED_MAPPING_COLUMNS",
    ),
    "constants.DRY_COOLING_VARIABLES": ("impacts.thermoelectric", "DRY_COOLING_VARIABLES"),
    "constants.DOWNSTREAM_COLUMN": ("impacts.thermoelectric.siting", "DOWNSTREAM_COLUMN"),
    "constants.WET_COOLING_VARIABLES": ("impacts.thermoelectric", "WET_COOLING_VARIABLES"),
    "constants.SALINE_INTAKE_VARIABLES": ("impacts.thermoelectric", "SALINE_INTAKE_VARIABLES"),
}


# --- caches ---------------------------------------------------------------------


def reset_region_caches() -> None:
    """Forget cached MESSAGE R12 definitions so a new definitions root takes effect."""

    from impacts.climatecore import regions

    for name in (
        "message_r12_region_labels",
        "message_r12_country_to_region",
        "_normalized_message_r12_lookup",
    ):
        getattr(regions, name).cache_clear()


# --- composite entry points --------------------------------------------------------


@dataclass(frozen=True)
class ConfiguredKernel:
    """A kernel with the assumptions and chunk widths an evaluation passes it."""

    kernel: Any
    assumptions: Any = None
    chunks: Any = None


def _resolve(name: str) -> Any:
    module_name, attribute = TABLE[name]
    return getattr(importlib.import_module(module_name), attribute)


def _kernel_factory(kernel_name: str) -> Callable[..., ConfiguredKernel]:
    def create(assumptions: Any = None) -> ConfiguredKernel:
        return ConfiguredKernel(_resolve(kernel_name), assumptions)

    return create


def _chilled_kernel(
    assumptions: Any = None,
    *,
    region_setpoints: bool = False,
    entity_chunk: int = 10_000,
    time_chunk: int = 365,
) -> ConfiguredKernel:
    kernel = _resolve("chilled.region_setpoint_kernel" if region_setpoints else "chilled.kernel")
    return ConfiguredKernel(kernel, assumptions, {"time": time_chunk, "entity": entity_chunk})


def _normalizer(kernel_name: str) -> Callable[..., Any]:
    def normalize(dataset: Any, *, variable_names: Any = None, variable_units: Any = None) -> Any:
        return _resolve("kernel.prepare_climate")(
            _resolve(kernel_name), dataset, variable_names=variable_names, variable_units=variable_units
        )

    return normalize


def _evaluate_mapped(configured: ConfiguredKernel, mapping: Any, climate: Any, duration: Any, **kwargs: Any) -> Any:
    # Thermoelectric sites carry a UNITID; a CHILLED mapping has no identifier, so
    # its rows are numbered, as the one-call function does.
    if "site_id_column" not in kwargs:
        if "UNITID" in mapping.columns:
            kwargs["site_id_column"] = "UNITID"
        else:
            mapping = mapping.reset_index(drop=True).assign(entity_id=range(len(mapping)))
            kwargs["site_id_column"] = "entity_id"
    return _resolve("kernel.evaluate_mapped")(
        configured.kernel,
        mapping,
        climate,
        duration,
        assumptions=configured.assumptions,
        chunks=configured.chunks,
        **kwargs,
    )


def _evaluate_gridded(configured: ConfiguredKernel, climate: Any, parameters: Any, duration: Any, **kwargs: Any) -> Any:
    return _resolve("kernel.evaluate_gridded")(
        configured.kernel,
        climate,
        parameters,
        duration,
        assumptions=configured.assumptions,
        chunks=configured.chunks,
        **kwargs,
    )


# stable name -> test-helper callable
def _census_from_settings(settings: dict[str, Any]) -> Path:
    """Write the site evaluation that a settings mapping describes; return its path.

    ``settings`` is a mapping in the shape of the settings the stored outputs were produced from (``gcm``,
    ``watxene_root``, ``gmt_root``, ``scenarios``, ``gwl_sampling`` or
    ``reference_sampling``, ``region``, ``universe``, ``fleet``, ``responses``,
    ``output`` and ``summary``), and this composes the library calls from it: the
    sampled years, the cells of the site evaluation and ``run_site_evaluation``.
    """

    gcm = settings["gcm"]
    root = settings.get("watxene_root")
    reference = settings.get("reference_sampling")
    if reference is None:
        scenarios = list(settings["scenarios"])
        paths = _resolve("isimip.isimip3b_paths")(Path(root) if root else None)
        years = _resolve("audit.sample_years_within_gwl")(
            settings["gmt_root"],
            gcm,
            scenarios,
            _resolve("audit.GWLSampling")(**settings.get("gwl_sampling", {})),
        )
    else:
        sampling = _resolve("audit.ReferenceSampling")(**reference)
        paths = _resolve("audit.reference_climate_paths")(sampling, gcm, root)
        years = _resolve("audit.sample_years_within_reference")(sampling)
        scenarios = sorted({year.scenario for year in years})
    universe = settings.get("universe", {})
    fleet = settings["fleet"]
    # As the runner: without fleet.scenario, the first of the settings' own
    # scenarios (a reference site evaluation without either has no fleet scenario).
    fleet_scenario = fleet.get("scenario")
    if fleet_scenario is None:
        fleet_scenario = settings["scenarios"][0]
    fleet_names = {"gcm": fleet.get("gcm", gcm), "scenario": fleet_scenario}
    caches = {
        key: universe[key].format(gcm=gcm)
        for key in ("dry_cache", "wet_cache")
        if universe.get(key) is not None
    }
    cells = _resolve("audit.build_census_cells")(
        paths,
        gcm,
        scenarios,
        _resolve("universe.RegionRasterPaths")(
            **{key: Path(value) for key, value in settings["region"].items()}
        ),
        fleet_wet=fleet["wet"].format(**fleet_names),
        fleet_dry=fleet["dry"].format(**fleet_names),
        cells_considered_scenarios=universe.get("scenarios"),
        **caches,
    )
    summary = settings.get("summary", {})
    options = {"responses": settings["responses"]} if "responses" in settings else {}
    written = _resolve("audit.run_census")(
        cells,
        years,
        paths=paths,
        gcm=gcm,
        output=settings["output"],
        summary=None
        if summary.get("skip", False)
        else _resolve("audit.CensusSummary")(
            low_value_thresholds=summary.get("bad_tail_thresholds", {}),
            bootstrap_replicates=int(summary.get("bootstrap_replicates", 200)),
            seed=int(summary.get("bootstrap_seed", 730)),
        ),
        overwrite=True,
        **options,
    )
    return written[0]


def _load_or_build_universe(*args: Any, cache_path: Any = None, **kwargs: Any) -> Any:
    """The cache of the cells considered, with the ``{gcm}`` template the stored outputs were produced with.

    The library takes the cache path itself; the template is filled here.
    """

    if isinstance(cache_path, str):
        cache_path = cache_path.format(gcm=args[1])
    load = importlib.import_module("impacts.thermoelectric.siting.site_evaluation")._load_or_build_universe
    return load(*args, cache_path=cache_path, **kwargs)


CALLS: dict[str, Callable[..., Any]] = {
    "audit.run": _census_from_settings,
    "audit.load_or_build_universe": _load_or_build_universe,
    "kernel.evaluate_mapped_kernel": _evaluate_mapped,
    "kernel.evaluate_gridded_kernel": _evaluate_gridded,
    "wet.create_kernel": _kernel_factory("wet.kernel"),
    "wet.normalize_climate_inputs": _normalizer("wet.kernel"),
    "dry.create_kernel": _kernel_factory("dry.kernel"),
    "dry.normalize_climate_inputs": _normalizer("dry.kernel"),
    "chilled.create_kernel": _chilled_kernel,
    "chilled.normalize_climate_inputs": _normalizer("chilled.kernel"),
}


def _default_sources(kernel_name: str) -> dict[str, str]:
    """The default source variable of each required input of a kernel."""

    inputs = _resolve(kernel_name).inputs
    return {name: spec.source for name, spec in inputs.items() if not spec.optional}


# stable name -> zero-argument function returning a value the library derives
DERIVED: dict[str, Callable[[], Any]] = {
    "constants.DEFAULT_DRY_COOLING_VARIABLE_NAMES": lambda: _default_sources("dry.kernel"),
    "constants.DEFAULT_ISIMIP_VARIABLE_NAMES": lambda: _default_sources("wet.kernel"),
}


# --- resolution -------------------------------------------------------------------


def get(name: str) -> Any:
    """Return the library callable (or class) registered under a stable name."""

    if name in CALLS:
        return CALLS[name]
    if name in DERIVED:
        return DERIVED[name]()
    return _resolve(name)
