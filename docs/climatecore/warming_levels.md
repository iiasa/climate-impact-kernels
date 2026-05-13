# Warming levels

`impacts.climatecore.warming` provides the reusable
warming-level mechanics that move between annual GMT paths and
warming-level-indexed response data.

## Entry points

- `match_warming_levels` — returns the years that fall inside each
  warming-level bucket. Carries diagnostics: `actual_warming`,
  `absolute_error`, `warming_rate`, `accumulated_warming`.
- `assign_years_to_levels` — assigns each year to its nearest target
  warming level.
- `bin_by_warming_level` — reduces a year-indexed value array by the
  assigned levels.
- `build_warming_level_assignments` — builds an xarray assignment
  dataset indexed by explicit path dims plus year.
- `attach_warming_levels` — attaches those assignments back onto a time
  or year-indexed xarray object as coordinates.
- `build_warming_level_quantile_surface` — builds quantile surfaces from
  a GWL-labelled indicator `DataArray`.

## Matching semantics

Matching is bucket-based, not nearest-neighbor: a centered running mean
over annual GMT determines which bucket each year falls in, and every
year inside the bucket is recorded. Diagnostic quantities
(`actual_warming`, `accumulated_warming`, `warming_rate`) are part of
the contract, not optional.

## Out of scope

- GMT-file discovery and project-specific scenario objects.
- Sector adapters that turn parquet tables or domain-specific xarray
  layouts into indicator arrays.
- The non-stationary GEV extreme-value workflow used by production
  `rimeX`.
