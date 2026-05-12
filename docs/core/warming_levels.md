# Warming Levels

`climateresponse.warming_levels` provides the reusable warming-level mechanics
used to move between annual GMT paths and warming-level-indexed response data.

## Main Entry Points

- `match_warming_levels`
  Returns all years that fall inside each warming-level bucket and carries
  diagnostics including `actual_warming`, `absolute_error`,
  `warming_rate`, and `accumulated_warming`.
- `assign_years_to_levels`
  Assigns each year to the nearest target warming level.
- `bin_by_warming_level`
  Reduces a year-indexed value array by the assigned warming levels.
- `build_warming_level_assignments`
  Builds an xarray assignment dataset indexed by explicit path dims plus year.
- `attach_warming_levels`
  Attaches those path/year assignments back onto a time- or year-indexed
  xarray object as coordinates.
- `build_warming_level_quantile_surface`
  Builds quantile surfaces from a GWL-labelled indicator `DataArray`.

## Current Interpretation

The current design follows the production `rimeX` findings:

- warming-level matching is bucket-based rather than nearest-neighbor matching
- diagnostic quantities such as `actual_warming`, `accumulated_warming`, and
  `warming_rate` are part of the contract
- the shared surface supports two-dimensional GMT lookup patterns across
  multiple paths without converting the hot path to tables

## Boundary

This module owns the warming-level mechanics themselves.

It does not own:

- file discovery for GMT series
- project-specific scenario objects
- sector adapters that turn parquet tables or domain-specific xarray layouts
  into indicator arrays
- the non-stationary GEV workflow used in production `rimeX`
