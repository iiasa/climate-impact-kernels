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
  Builds a table of year-to-level assignments across multiple GMT paths.
- `attach_warming_levels`
  Joins that assignment table back onto a long-format data frame.
- `build_warming_level_quantile_map`
  Builds quantile maps over warming levels within explicit groups.

## Current Interpretation

The current design follows the production `rimeX` findings:

- warming-level matching is bucket-based rather than nearest-neighbor matching
- diagnostic quantities such as `actual_warming`, `accumulated_warming`, and
  `warming_rate` are part of the contract
- the shared surface supports two-dimensional GMT lookup patterns across
  multiple paths

## Boundary

This module owns the warming-level mechanics themselves.

It does not own:

- file discovery for GMT series
- project-specific scenario objects
- the non-stationary GEV workflow used in production `rimeX`
