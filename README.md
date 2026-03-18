`thermo_electric_power` holds the upstream contract artifacts needed before
plant-level cooling physics can run against gridded climate and hydrology data.
It is a library package with materialized data artifacts, not a pipeline CLI.

## Separation Of Concerns

`impacts` owns plant-level contracts, plant-to-grid joins, and cooling physics.
It does not open NetCDF files, discover datasets, or provide general xarray
preprocessing helpers. Those array-native transforms belong in
`climateresponse`, which is an explicit local dependency of this package.

The intended boundary is:

- `climateresponse`: reusable xarray transforms such as annual or seasonal
  aggregation once arrays are already loaded.
- `impacts`: fleet mappings, cooling assignments, and plant-level impact
  equations operating on already-opened arrays or tables.

If a reusable xarray manipulation step is missing for this workflow, add it to
`climateresponse` first and then consume it here rather than reimplementing the
helper inside `impacts`.

Current artifacts:

- `data/unit_cell_mapping.parquet`
  Exact coordinate-only unit-to-cell mapping.
- `data/recovered_unit_cell_mapping.parquet`
  Conservative extension of the unit mapping that adds only safe inherited
  coordinates from shared `LOCATIONID`s where all coordinate-bearing siblings
  share one exact point.
- `data/location_cell_mapping.parquet`
  Location-normalized mapping that collapses only single-cell `LOCATIONID`s and
  preserves ambiguous sites as explicit location-cell splits.
- `data/country_rollup.parquet`
  Country rollup derived from the location mapping. MESSAGE regional semantics
  are intentionally out of scope here.
- `data/unit_cooling_mapping.parquet`
  Per-unit technology-to-cooling contract for downstream wet/dry physics.

## Spatial Gap

The source fleet in `/home/raghunathan/Li_et_al_2025/thermal_plants.parquet`
contains `43,031` thermo-electric units. Exact source coordinates exist for
`25,652` units (`2,596.9 GW`).

The conservative recovery path implemented here adds only units whose missing
coordinates can be inherited safely from the same `LOCATIONID`. This recovers
about `605` additional units (`76.7 GW`). The remainder of the missing fleet is
left unresolved on purpose.

What is not done here:

- fuzzy record linkage against external registries
- name-based geocoding
- heuristic country-centroid or subnational imputation
- MESSAGE region assignment

The unresolved spatial gap is therefore a source-data limitation, not a hidden
inference step. Downstream workflows can choose between exact-only mappings and
the conservative recovered mapping depending on how much provenance risk they
want to accept.
