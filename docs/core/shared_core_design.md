# Shared Core Design

## Boundary

`climateresponse` is a computational shared core for future climate-response workflows.
It owns reusable transforms and contracts that recur across `hydro_preprocess`,
`rimeX`, and the direct cooling-impact pipeline.

The package includes:

- baseline normalization and anomaly transforms
- GMT running means and warming-level matching
- warming-level binning for yearly outputs
- annual and seasonal aggregation
- gridded point attachment for mapped entities
- unit normalization for common climate variables
- mapped-entity evaluation orchestration for reusable array-native kernels
- masked spatial weighting
- response-surface contracts and warming-level interpolation
- lightweight deterministic statistics helpers

The package does not own:

- config loading
- catalog or dataset discovery
- file I/O conventions
- project-boundary schema validation
- domain physics such as hydrology, thermal cooling, or crop-response logic

Consumers are expected to validate and load their own datasets, then hand explicit
arrays or tables into this library.

This package is not a reproduction of production `rimeX`.
`rimeX` keeps substantial workflow logic above any reusable kernel:

- config-driven season definitions
- xarray-native map assembly and quantile calculation
- project-specific file discovery and dataset opening
- GMT-conditioned non-stationary GEV fitting for extremes

The shared core here only claims the parts that remain coherent once those adapters
and workflow decisions are removed.

The intended execution model is xarray-first and compatible with dask-backed arrays.
Hot-path transforms operate directly on `xarray.DataArray` or `xarray.Dataset` objects
so consumers do not have to extract Python dictionaries or reconstruct array objects
around the library.

## Canonical Contracts

The core contracts are intentionally small and explicit.

- `GMTSeries` carries `(model, scenario, baseline_start_year, baseline_end_year, values)`.
- `WarmingLevelMatch` records the target level, matched year, actual warming, absolute error,
  and an optional local warming-rate estimate.
- `YearLevelBin` records the year-to-level assignment used when grouping yearly outputs by GWL.
- `RegionResponseSurface` and `GridQuantileResponseSurface` remain distinct contracts.
  Region-aggregated surfaces and gridded quantile maps have different downstream consumers and
  should not be forced into a single polymorphic schema.

The response-surface contracts wrap xarray-native values.
`RegionResponseSurface` requires `region` and `warming_level` dimensions.
`GridQuantileResponseSurface` requires `warming_level`, `quantile`, `lat`, and `lon`.
The contract describes required axes while still using array-native storage in hot paths.

## Observed Production rimeX Behavior

Direct inspection of `/home/raghunathan/rime/rimeX` matters here because the earlier
design discussion compressed several distinct behaviors into one generic "shared core".

- Warming-level matching is bucket-based, not nearest-neighbor. `rimeX` computes a
  centered running mean over annual GMT and records every year that falls inside each
  warming-level bucket.
- The warming-level table carries diagnostic fields beyond `(level, year)`: production
  records include `actual_warming`, `accumulated_warming`, and `warming_rate`.
- Seasonal aggregation uses fixed month sets from configuration. `rimeX` does not
  implement a dynamic wet/dry regime-detection algorithm in production.
- Quantile-map generation is xarray-native and grid-scale. The production workflow
  collects slices per warming level and computes weighted quantiles across samples.
- Extremes processing is split. Generic percentile-triplet fitting exists in
  `stats.fit_dist`, but production extreme-value preprocessing uses a separate
  non-stationary GEV pipeline conditioned on GMT.

## Salvage And Rejection

The reusable mechanics worth preserving from existing codebases are:

- anomaly and baseline transforms from both pipelines
- GMT smoothing and warming-level lookup logic, but not necessarily the surrounding
  file and config workflow
- seasonal and annual aggregation mechanics where the month groupings are passed in
  explicitly
- spatial weighting once masks and file management are stripped away
- interpolation-ready response-surface handling and contract definitions
- lightweight statistical helpers

In this repository those reusable mechanics are implemented as xarray-native kernels:

- baseline anomalies operate directly on year-indexed arrays
- running means use xarray rolling windows
- warming-level tables and bins are derived from xarray GMT series
- seasonal and annual aggregation operate on time coordinates
- standard climatological seasonal aggregation uses quarterly resampling and
  consolidated `season`/`year` chunks so dask-backed outputs do not fragment
  into one-chunk-per-season-year layouts
- spatial weighting works on masked arrays over explicit dimensions
- warming-level interpolation stays on named warming-level axes

The following stay outside the core:

- `hydro_preprocess` repo-specific file discovery and shell wrappers
- `rimeX` global config object and project-specific parameter access patterns
- `rimeX` quantile-map assembly as an xarray orchestration layer
- non-stationary GEV fitting and return-period production for extremes
- output layouts tied to a single downstream consumer
- Julia GMT implementation details unless later benchmarks show Python is insufficient

## Implementation Principles

- Keep functions explicit and argument-driven rather than hiding behavior behind config state.
- Favor simple data contracts and pure transforms over framework-style abstractions.
- Keep the computational core array-native.
  `xarray` is the primary interface and dask compatibility comes from preserving xarray
  operations in the hot paths rather than converting through Python containers.
- Treat file discovery, dataset opening, and workflow orchestration as adapter layers
  above the core even when those adapters also use xarray.
- Keep response interpolation one-dimensional along the warming-level axis.
- Do not claim a generic "extremes" core unless the reusable algorithm is actually shared.
  `rimeX`'s production extremes path is a GMT-conditioned GEV workflow and remains outside
  this first shared-core cut.
- Treat testing as part of the contract: numerical helpers need deterministic examples and invariant checks.

## Review Resolution

The main review concern was valid: the earlier dict-based implementation contradicted the
stated xarray+dask-first scope. That concern is addressed in the current implementation by
moving the hot-path functions onto xarray objects directly instead of routing data through
Python mappings.

- The shared core remains computation-focused and adapter-agnostic.
- Grid-scale xarray workflows now stay in-core for the reusable numerical transforms.
- File opening, catalog discovery, and project-specific orchestration remain above the core.
- The production `rimeX` non-stationary GEV path remains outside scope.
- Distinct region and gridded response contracts stay valid and now use xarray-native values.

## Initial Testing Contract

This repository starts with deterministic unit tests for the core transforms.
Future work should add property-style checks for conservation and monotonicity, plus
reference fixtures derived from validated `hydro_preprocess` and `rimeX` outputs.
