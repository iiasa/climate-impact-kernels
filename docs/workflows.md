# Workflows

How the library's functions are called together. The sketches are illustrative,
not runnable: lower-case names are the caller's paths, tables and settings
(`gcm` is the name of a global climate model). A kernel is one impact
calculation; the [README](../README.md#impact-kernels) describes it. Every
path is passed explicitly or read from the environment variables listed in the
[README](../README.md#data-paths).

## ISIMIP inputs

ISIMIP is the Inter-Sectoral Impact Model Intercomparison Project;
`impacts.isimip` loads its bias-adjusted climate data.

```python
from impacts.isimip import isimip3b_paths, load_isimip_dataset
from impacts.thermoelectric import WET_COOLING_VARIABLES

paths = isimip3b_paths(isimip_root)  # or $WATXENE_DATA_PATH
climate = load_isimip_dataset(WET_COOLING_VARIABLES, gcm, scenario, paths,
                              year_range=(2041, 2060))
```

- `load_isimip_variable` returns `target_frequency="monthly"` (the default) or
  `"native"`. Use `"native"` for kernels that need daily values, such as CHILLED
  and the dry-cooling site evaluation of [siting.md](siting.md).
- `load_isimip_dataset` merges variables after frequency handling and requires
  identical time, lat and lon coordinates; a mismatch raises. A variable on a
  coarser grid nested in the finest one (the 1-degree ocean grid under the
  0.5-degree land grid) is copied onto it first.
- With `year_range`, discovery keeps files whose names end in
  `_<start>_<end>.nc` and overlap the range; a file named otherwise raises.
- Discovery looks for one named WaterGAP water-use forcing and never falls back
  to another; see [assumptions.md](assumptions.md#assumptions-register).
- `load_isimip_gmt` stitches historical and scenario annual global mean
  temperature (GMT). A file with
  several `mean_anomaly*` columns raises unless they are copies of one column.

## Thermoelectric kernels

```python
from impacts.thermoelectric import wet_cooling, aggregate_thermoelectric_regions

units = wet_cooling(unit_cell_mapping, cooling_mapping, climate, duration="monthly")
regions = aggregate_thermoelectric_regions(units, region_set="message_r12")
```

1. Build the unit-to-cell and cooling mappings with `fleet.py`.
2. `wet_cooling` or `dry_cooling` builds the kernel mapping and evaluates it. A
   wet fleet with `ot_saline` units also needs `SALINE_INTAKE_VARIABLES` in
   `climate`.
3. Aggregate to countries or R12 regions (MESSAGE's twelve world regions) with
   `regions.py`.

For a warming-level summary, write the country results as a monthly parquet and
call `thermoelectric_warming_level_summary(parquet, reference, assignments)`,
with `assignments` from `isimip_warming_level_assignments`.

## Siting of new plants

The methods are in [siting.md](siting.md). One GCM's site evaluation:

```python
from impacts.isimip import isimip3b_paths
from impacts.thermoelectric.siting import (
    GWLSampling, RegionRasterPaths, build_site_cells, run_site_evaluation,
    sample_years_within_gwl,
)

paths = isimip3b_paths(isimip_root)
cells = build_site_cells(
    paths, gcm, scenarios, RegionRasterPaths(gaul_raster, gaul_csv, r12_yaml),
    fleet_wet=wet_fleet_nc, fleet_dry=dry_fleet_nc,
    cells_considered_scenarios=river_scenarios,
)
years = sample_years_within_gwl(gmt_root, gcm, scenarios, GWLSampling())
run_site_evaluation(
    cells, years, paths=paths, gcm=gcm, output=site_evaluation_parquet,
    responses={"wet": {...}, "wet_monthly": {...}, "dry": {...}},
    summary=None,
)
```

A site evaluation for a reference period (`REF-PD` or `REF-PI`) takes its years
from `sample_years_within_reference(ReferenceSampling(mode))`, its paths from
`reference_climate_paths`, and its reference scenario as `scenarios`.

From site evaluations to outputs:

1. `split_wet_losses(low_warming_evaluation_paths, census_paths, output, reference=...)`
   splits the wet capacity loss into thermal loss and water-availability loss.
   Run it for each of the two calculations: the warming-level calculation and the
   present-day reference calculation.
2. `build_river_networks(read_potential_site_cells(decomposition), read_flow_directions(ddm30))`
   builds the river network, written with `write_river_networks`.
3. A cell ordering: `build_fixed_cell_ordering` on the split loss of the `REF-PD` site evaluation, or
   `build_residual_water_ordering` (written with `write_residual_water_ordering`).
   Use the same table for every GCM and both calculations.
4. `build_wet_siting_outputs(decompositions, census_ceilings, topologies, ordering)`
   per GCM and calculation, written with `write_wet_siting_outputs` to one directory
   per GCM.
5. Pool both calculations across GCMs and write the output files with
   `finalize_wet_siting_outputs(gwl_root, reference_root, output_directory, gcm_count=...)`.

`export_wet_usable_capacity_curves` writes the capacity curves of existing sites from the
parquet files that hold the split loss.

## Buildings

```python
from impacts.buildings import (
    chilled_energy, load_ewembi_solar_climatology,
    load_message_ix_buildings_archetypes, load_region_cooling_setpoints,
)

archetypes = load_message_ix_buildings_archetypes(archetype_csv)
solar = load_ewembi_solar_climatology(vertical_path=..., horizontal_path=...)
setpoints = load_region_cooling_setpoints(setpoint_csv)  # optional
result = chilled_energy(spatial_units, archetypes, solar, daily_climate,
                        duration="seasonal", setpoints=setpoints)
```

For a grid, build the parameters with `build_chilled_parameter_grids` (one
building type, urban/rural class and vintage at a time) and call
`evaluate_gridded(CHILLED_KERNEL, climate, parameters, duration)`. Without
`setpoints`, the kernel uses the scalar setpoints of `ChilledEnergyAssumptions`.

Seasonal outputs stay dask-backed until written or computed.

To build warming-level summaries from gridded seasonal outputs:

1. Build a country raster with `load_country_raster`.
2. `aggregate_chilled_to_country(dataset, country_raster)` reduces every output
   to countries as a cell-area-weighted mean; `parse_chilled_matrix_filename`
   and `expand_chilled_metadata` attach GCM, scenario and archetype labels
   from the file name.
3. `buildings_warming_level_summaries(...)` pools the country table by warming
   level, with country weights from `country_population` or
   `country_land_area`.
