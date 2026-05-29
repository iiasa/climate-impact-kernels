# Thermoelectric

`impacts.domain.thermoelectric` owns the thermo-electric power workflows:

- unit-to-grid mapping
- cooling classification
- wet-cooling kernel construction
- dry-cooling kernel construction
- aggregation from plant/entity outputs to countries or MESSAGE R12 regions

## Module Map

- `mapping.py`
  Builds unit-cell mappings, recovery mappings, location-normalized mappings,
  and country rollups.
- `cooling.py`
  Classifies technology families and assigns cooling contracts.
- `wet.py`
  Defines wet-cooling assumptions, climate normalization, and wet kernel
  construction.
- `dry.py`
  Defines dry-cooling assumptions, engine classification, climate
  normalization, and dry kernel construction.
- `aggregation.py`
  Aggregates entity-level outputs to countries or MESSAGE R12 regions and
  writes reduced NetCDF products.

## Core Artifacts

These derived artifacts are **not distributed with the repository** —
they derive from license-restricted WEPP/PLATTS source data and are
git-ignored. Regenerate them with the `build_*`/`write_*` helpers below
from the licensed source fleet table, or supply your own under `data/`:

- `unit_cell_mapping.parquet`
  Exact coordinate-only unit-to-cell mapping.
- `recovered_unit_cell_mapping.parquet`
  Conservative extension that inherits coordinates only when a shared
  `LOCATIONID` unambiguously resolves to one exact point.
- `location_cell_mapping.parquet`
  Location-normalized mapping that preserves ambiguous sites as explicit
  location-cell splits.
- `country_rollup.parquet`
  Country rollup derived from the location mapping.
- `unit_cooling_mapping.parquet`
  Per-unit technology-to-cooling contract.

## Aggregation Contract

`aggregation.py` aggregates entity-level outputs using variable-specific
reductions:

- capacity-like quantities are summed
- most diagnostic variables are MW-weighted means

The current implementation defaults unknown entity-level variables to
MW-weighted means. That is permissive rather than final; if the production
variable set stabilizes further, tighten that contract explicitly rather than
depending on the fallback.

## Scope Boundaries

In scope:

- plant/entity mappings
- cooling assignments
- wet/dry climate normalization for the kernel inputs
- wet/dry physics kernels
- aggregation to countries and MESSAGE R12 regions

Out of scope:

- heuristic geocoding of unresolved plants
- fuzzy registry linkage
- project-specific file discovery
- generic xarray transforms

## Provenance

The wet-cooling kernel follows the withdrawal-intensity and usable-capacity
framework reported in Li et al., `Nature Sustainability`
(<https://doi.org/10.1038/s41893-025-01692-9>).

The dry-cooling kernel follows the dry-cooling efficiency-loss framework
reported in Qin et al., `Nature Water`
(<https://doi.org/10.1038/s44221-023-00120-6>).
