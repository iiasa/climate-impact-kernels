# climatecore

`impacts.climatecore` is the xarray-native core of the `impacts`
package. It owns the operations and definitions that recur across the
domain modules and downstream RIME-X workflows (RIME-X is IIASA's emulator of
regional climate impacts by warming level). The layered structure and the rule
that `climatecore` imports nothing from the domain modules are described in
[architecture.md](architecture.md).

## Layout

| Path | Contents |
|---|---|
| `climatecore/warming.py` | Warming-level assignment of GMT years, pooling of samples by warming level, and weighted group means. |
| `climatecore/regions.py` | Region loading and site-to-region aggregation against `common-definitions`. |
| `climatecore/time.py` | Temporal transforms. |
| `climatecore/space.py` | Spatial transforms. |
| `climatecore/units.py` | Unit transforms. |

The main computation is xarray-first: functions take `xarray.DataArray` or
`xarray.Dataset` and return the same. The kernels are plain xarray
operations, so dask backing remains viable end-to-end.

## Evaluation

The evaluation engine lives in `impacts.engine`, above `impacts.climatecore`.
It uses the climatecore operations (`attach_gridded_dataset`, `convert_units`,
`aggregate_monthly`, `aggregate_seasonal`) but is not one of the public
functions of `impacts.climatecore`.

A `Kernel` (one impact calculation) declares its climate inputs (`ClimateInput`: default source
variable, the unit the physics receives, an optional second sampling point,
whether it is optional), the site columns it carries from a mapping table, its
default assumptions, and its physics, `physics(climate, sites, assumptions)`.
`evaluate_mapped` and `evaluate_gridded` do everything around the physics:

- `prepare_climate` selects, renames and unit-converts the inputs; an input
  without a unit passes through unconverted and unchecked.
- The mapped path labels each site (a row of the mapping table) with the value
  of the mapping column the caller names as `site_id_column` (it must exist and hold unique values; the
  thermoelectric kernels use `UNITID`), samples climate at each mapping row's
  `grid_lat`/`grid_lon`
  (and, for an input with a `point`, at the second point named by a pair of
  mapping columns: NaN where a site has none, dropped when the mapping lacks
  the columns), and samples any `gridded_coords` fields at the same cells.
- The gridded path stacks the spatial grid to `entity` (one site per grid cell)
  and takes every parameter variable as a site value.
- The physics receives climate as a `(time, entity)` Dataset and site values as
  a Dataset indexed by `entity`, both of plain data variables, and returns one
  Dataset. The engine attaches the site columns as output coordinates, writes the
  assumptions and the kernel's `attrs`, drops coordinates no output uses, and
  resamples to the requested `duration`. `chunks=` rechunks the sampled data
  before the physics runs; left `None`, it takes the kernel's `chunks`, a
  performance default (CHILLED sets one), and `{}` keeps the loader's chunks.
- Each assumption is recorded as its value, `<name>_unit` and `<name>_source`
  from the field metadata. That source describes the field's default, so for a
  value the caller changed `<name>_source` reads "set by the caller; default
  <default> from <source>".
- An assumption field whose metadata names a `site` value is overridden per site
  when the sites carry that value. The output's `<name>` attr then describes the
  per-site values and `<name>_source` gives their source, read from the site
  variable's `description`/`source` attrs (a mapping table holds them in
  `mapping.attrs["site_value_attrs"]`), followed by the scalar used as fallback.
  The recorded source counts only while it is bound to the values the engine sees,
  either by a key-to-value map (`values_by_key`, checked site by site) or by a
  `values_digest` of the values sorted by named coordinates (`values_order`);
  both ignore the order and subset of sites. Otherwise the source reads "not
  recorded with the site values".

Domain physics (cooling equations, building energy kernels,
crop-response logic) lives above this layer in the domain modules.

## Warming levels

A warming level is a level of global warming, such as 1.5 or 2 K above
pre-industrial; years are grouped by the level they reach. GMT is global mean
near-surface air temperature change from pre-industrial.

Entry points:

- `assign_years_to_levels(gmt, target_levels)` — assigns each year to
  its nearest target warming level.
- `bin_by_warming_level(values, gmt, target_levels)` — reduces a
  year-indexed value array by the assigned levels.
- `build_warming_level_assignments(gmt_by_path, path_columns=...)` —
  builds an xarray assignment dataset indexed by explicit path
  dimensions plus year.
- `pool_by_warming_level(values, assignments, target_levels, name=...)` —
  pools (path, year) samples into warming-level bins and summarises each
  bin (mean, standard deviation, quantiles, sample count).

Assignment is nearest-neighbour: each year takes the target level closest
to its warming, and years with no finite warming get no level. Every year
with a finite warming gets a level, including years below the lowest level and,
when a maximum is given, years above the highest. In
`build_warming_level_assignments` the levels run from `min_warming_level`
(default 0) in steps of `bin_width` up to `max_warming_level`, which defaults to
the highest warming in any path.

Minimal example:

```python
from impacts.climatecore import (
    build_warming_level_assignments,
    pool_by_warming_level,
)

assignments = build_warming_level_assignments(
    {("CanESM5", "ssp370"): gmt_anomaly},
    path_columns=("model", "scenario"),
)
summary = pool_by_warming_level(
    indicator, assignments, target_levels=[1.5, 2.0], name="indicator",
    sample_dims=("model", "scenario", "year"),
)
```

## Regions

Country and MESSAGE-region metadata are read from a checkout of
[common-definitions](https://github.com/IAMconsortium/common-definitions),
named by `COMMON_DEFINITIONS_ROOT` (see the [README](../README.md#data-paths)).

Entry points:

- `common_definitions_root` — resolves the configured
  `common-definitions` checkout.
- `message_r12_region_labels` — MESSAGEix-GLOBIOM 2.1 R12
  code-to-label mappings.
- `message_r12_country_to_region` — canonical country-to-R12
  membership.
- `resolve_region_labels(region_set)` — member labels for the
  supported region sets (`country`, `message_r12`).
- `aggregate_site_dataarray` and `aggregate_site_dataset` —
  aggregate outputs indexed by `entity` (one entry per site) by region. The dataset form takes
  variable-specific reductions (`sum` or `weighted_mean`) and raises for a
  variable indexed by `entity` that it has no reduction for. A weighted mean gives a site
  with no value no weight and is NaN for a region with no valid weight; a sum
  over a region with no value is NaN, not zero.

Geometry files, downstream mapping files, and domain-specific
choices about which variables to sum or average belong in the domain
modules.

## Transforms

`time.py`, `space.py` and `units.py` hold the smaller helper functions:

- `time.py` — monthly and seasonal (four three-month seasons: DJF, MAM, JJA,
  SON) aggregation. A season-year (a season and the year it belongs to) needs all three of its months in the record
  (otherwise it is NaN, as at the edges of a record that does not start or end
  on a season boundary); missing values within a complete season are skipped.
  The seasonal path consolidates `season`/`year` chunks
  so dask-backed outputs do not fragment into one chunk per
  `(season, year)` pair.
- `space.py` — gridded-point attachment (`attach_gridded_dataset` takes the
  column that labels the sites as a required `site_id_column`), grid-cell areas (a coordinate with a single cell has no width of its own, so
  `grid_cell_area` takes `resolution_deg` for it and raises without), copying a coarse
  grid onto a finer one nested in it (`refine_to_grid`), and the nearest cell
  where a mask holds (`nearest_valid_cells`).
- `units.py` — temperature and pressure unit normalization.
- `warming.py` also holds `weighted_group_mean`, weighted group means over site weights.

## Out of scope

- Catalog, dataset discovery, file I/O conventions.
- Domain physics — cooling equations, building energy kernels,
  crop-response logic.
- Running the steps of a project-specific workflow.
- The non-stationary GEV extreme-value pipeline of `rimeX`.

Consumers load and validate their own data, then hand explicit
`xarray.DataArray` or `xarray.Dataset` objects to the public functions.
