# climatecore

`impacts.climatecore` is the xarray-native core of the `impacts`
package. It owns the transforms and contracts that recur across the
domain modules and downstream RIME-X workflows. The layered structure
and the `climatecore` ↓ `domain` boundary are described in
[architecture.md](architecture.md).

## Layout

| Path | Contents |
|---|---|
| `climatecore/baseline.py` | Baseline anomaly computation. |
| `climatecore/evaluation.py` | Mapped and gridded kernel-evaluation orchestration. |
| `climatecore/warming.py` | GMT running means, warming-level matching and binning, response-surface interpolation, canonical dataclasses. |
| `climatecore/regions.py` | Region loading and entity-to-region aggregation against `common-definitions`. |
| `climatecore/transforms/` | Temporal, spatial, unit, and statistical transforms. |

The hot path is xarray-first: functions take `xarray.DataArray` or
`xarray.Dataset` and return the same. Custom kernels are written with
`xr.apply_ufunc(..., dask="parallelized")` so dask backing remains
viable end-to-end.

## Baseline

`compute_anomaly(data, baseline_period, dim="year", offset=0.0)`
subtracts the baseline-window mean along `dim` and adds an optional
offset.

## Evaluation

The evaluation engine lives in `impacts.climatecore.evaluation` and is reached
through `impacts.kernel.evaluate_mapped_kernel` / `evaluate_gridded_kernel`,
which adapt an `ImpactKernel` onto it. The two functions below are that internal
seam — not part of the `impacts.climatecore` public API.

- `evaluate_mapped_dataset` — workflows where a tabular mapping
  supplies entity coordinates. Normalizes climate inputs, attaches
  gridded values to mapped entities, optionally samples extra
  `gridded_coords` fields per entity, runs the physics, applies clips,
  and packages the result.
- `evaluate_gridded_dataset` — workflows where gridded climate inputs
  and gridded parameter fields share spatial dimensions. Stacks the
  spatial dimensions into `entity`, attaches parameters, runs the
  physics, applies clips, and packages the result.
- `EvaluationResult` — structured return type (public, re-exported from
  `impacts.climatecore`) for physics functions that supply data
  variables and dataset attributes without constructing the full output
  dataset themselves.

A kernel's physics operates on already-normalized, already-attached data.
Domain physics — cooling equations, building energy kernels,
crop-response logic — lives above this layer in the domain modules.

## Warming levels

Entry points:

- `compute_running_mean` — centered running mean over annual GMT.
- `match_warming_levels(gmt, target_levels, bucket_width=...)` —
  returns every year that falls inside each warming-level bucket, with
  diagnostics (`actual_warming`, `absolute_error`, `warming_rate`,
  `accumulated_warming`).
- `assign_years_to_levels(gmt, target_levels)` — assigns each year to
  its nearest target warming level.
- `bin_by_warming_level(values, gmt, target_levels)` — reduces a
  year-indexed value array by the assigned levels.
- `build_warming_level_assignments(gmt_by_path, path_columns=...)` —
  builds an xarray assignment dataset indexed by explicit path
  dimensions plus year.
- `attach_warming_levels(data, assignments)` — attaches assignments
  back onto a time- or year-indexed xarray object as coordinates.
- `interpolate_along_warming_level` — one-dimensional interpolation
  along the warming-level axis.
- `build_warming_level_quantile_surface` — quantile surfaces from a
  GWL-labelled indicator `DataArray`.

Matching is bucket-based, not nearest-neighbor: a centered running
mean over annual GMT determines which bucket each year falls in, and
every year inside the bucket is recorded.

Canonical dataclasses (all frozen):

- `GMTPoint`, `GMTSeries` — annual GMT path with baseline metadata.
- `WarmingLevelMatch` — `(level, year, actual_warming,
  absolute_error, warming_rate)`.
- `YearLevelBin` — year-to-level assignment.
- `RegionLevelValue`, `RegionResponseSurface` — region-aggregated
  response surfaces; require `region` and `warming_level` dimensions.
- `GridQuantileResponseSurface` — gridded quantile surfaces; require
  `warming_level`, `quantile`, `lat`, `lon`.

Region-aggregated surfaces and gridded quantile maps are separate
contracts with different downstream consumers; they do not collapse
into one polymorphic schema.

Minimal example:

```python
from impacts.climatecore import (
    attach_warming_levels,
    build_warming_level_assignments,
)

assignments = build_warming_level_assignments(
    {("CanESM5", "ssp370"): gmt_anomaly},
    path_columns=("model", "scenario"),
)
labelled = attach_warming_levels(indicator, assignments)
```

## Regions

Country and MESSAGE-region metadata are read from the local
`common-definitions` checkout. Set `COMMON_DEFINITIONS_ROOT` before
using these helpers.

Entry points:

- `common_definitions_root` — resolves the configured
  `common-definitions` checkout.
- `message_r12_region_labels` — MESSAGEix-GLOBIOM 2.1 R12
  code-to-label mappings.
- `message_r12_country_to_region` — canonical country-to-R12
  membership.
- `resolve_region_labels(region_set)` — member labels for the
  supported region sets (`country`, `message_r12`).
- `aggregate_entity_dataarray` and `aggregate_entity_dataset` —
  aggregate entity-indexed outputs by region. The dataset form takes
  variable-specific reductions (`sum` or `weighted_mean`).

Geometry files, downstream mapping artifacts, and domain-specific
choices about which variables to sum or average belong in the domain
modules.

## Transforms

The `transforms/` subpackage contains the smaller utility surface:

- `temporal.py` — monthly, annual, and seasonal (quarterly tilings)
  aggregation. The seasonal path consolidates `season`/`year` chunks
  so dask-backed outputs do not fragment into one chunk per
  `(season, year)` pair.
- `spatial.py` — gridded-point attachment, masked weighted averages,
  grid-cell areas, region-wise reduction.
- `units.py` — temperature and pressure unit normalization.
- `stats.py` — deterministic resampling and weighted quantiles.

## Out of scope

- Catalog, dataset discovery, file I/O conventions.
- Domain physics — cooling equations, building energy kernels,
  crop-response logic.
- Project-specific workflow orchestration.
- The non-stationary GEV extreme-value pipeline used by production
  `rimeX`.

Consumers load and validate their own data, then hand explicit
`xarray.DataArray` or `xarray.Dataset` objects into the public API.
