# climatecore scope

`impacts.climatecore` is the reusable xarray-native core of the `impacts`
package. It owns the transforms and contracts that recur across the
domain modules and downstream RIME-X workflows.

## What it owns

- Baseline anomalies on year-indexed arrays.
- GMT running means; warming-level matching, binning, attachment, and
  quantile-surface construction.
- Temporal aggregation (monthly, annual, seasonal — quarterly tilings).
- Spatial mechanics: gridded-point attachment, masked weighted averages,
  grid-cell areas, region-wise reduction.
- Region loading and entity-to-region aggregation against the upstream
  `common-definitions` source of truth.
- Unit normalization for temperature and pressure.
- Generic kernel orchestration for mapped and gridded evaluation.
- Lightweight deterministic statistics helpers.

## What it does not own

- Catalog, dataset discovery, file I/O conventions.
- Domain physics — cooling equations, building-energy kernels,
  crop-response logic.
- Project-specific workflow orchestration.
- The non-stationary GEV extreme-value pipeline used by production
  `rimeX`.

Consumers load and validate their own data, then hand explicit
`xarray.DataArray` or `xarray.Dataset` objects into the public API.

## Subpackage layout

| Path | Contents |
|---|---|
| `climatecore/baseline.py` | Anomaly computation against a baseline window. |
| `climatecore/evaluation.py` | Mapped and gridded kernel-evaluation orchestration. |
| `climatecore/warming.py` | GMT running means, warming-level matching/binning, response-surface interpolation, canonical dataclasses. |
| `climatecore/transforms/` | Temporal, spatial, unit, and statistical transforms. |
| `climatecore/regions.py` | Region loading and entity-to-region aggregation. |

## Canonical contracts

The contracts in `climatecore.warming` are intentionally small and
explicit:

- `GMTSeries` carries `(model, scenario, baseline_start_year,
  baseline_end_year, values)`.
- `WarmingLevelMatch` records `level`, matched `year`, `actual_warming`,
  `absolute_error`, and an optional `warming_rate`.
- `YearLevelBin` records the year-to-level assignment.
- `RegionResponseSurface` requires `region` and `warming_level`
  dimensions.
- `GridQuantileResponseSurface` requires `warming_level`, `quantile`,
  `lat`, `lon`.

Region-aggregated surfaces and gridded quantile maps are separate
contracts. Their downstream consumers differ; collapsing them into one
polymorphic schema is not supported.

## Execution model

The core is xarray-first. Hot paths take `xarray.DataArray` or
`xarray.Dataset` and return the same — consumers never have to extract
Python containers or rebuild xarray objects around the library.

Dask backing is structural: custom kernels use
`xr.apply_ufunc(..., dask="parallelized")` or composed xarray operations.
The seasonal-aggregation path consolidates `season`/`year` chunks so
dask-backed outputs do not fragment into one chunk per `(season, year)`
pair.

## Implementation principles

- Functions are argument-driven, not config-state-driven.
- Pure transforms over framework-style abstractions.
- Response interpolation is one-dimensional along the warming-level axis.
- File discovery, dataset opening, and project orchestration belong
  above this layer.
