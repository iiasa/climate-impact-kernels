# Buildings

`impacts.domain.buildings` currently centers on the CHILLED pathway. The
implementation is split across three files so the kernel can be reasoned
about, tested, and re-used independently of the operational I/O scaffolding:

| Module | Owns |
| --- | --- |
| `impacts.domain.buildings.chilled` | Kernel contract: physics, climate-input normalizer, `ChilledEnergyAssumptions`, `create_chilled_kernel`. |
| `impacts.domain.buildings.chilled_io` | File I/O: archetype/parametric/solar/region loaders, validators, gridded-parameter and archetype-mapping builders. |
| `impacts.domain.buildings.chilled_config` | Operational path defaults. The only file in `impacts.domain.buildings` that reads from `os.environ`. |
| `impacts.domain.buildings.country` | Country-label raster + generic area-weighted entity→country reduction. Reusable by any gridded buildings analysis that needs country aggregation. |
| `impacts.domain.buildings.rimex` | CHILLED → RIME-X shape adapter: matrix-filename parsing, CHILLED-specific reduction map, metadata expansion. Composes the `country` primitives. |

The kernel itself never touches the filesystem. Callers compose it with
`impacts.kernel.evaluate_mapped_kernel` (mapped DataFrame inputs) or
`impacts.kernel.evaluate_gridded_kernel` (gridded parameter Dataset
inputs); both reach the same physics.

## Current Contract

The current implementation targets the ALPS2025 archetype contract only.

Important assumptions:

- `arch_input_reg.csv` is consumed with `building_type`, `urt`, `arch`, and
  `region` as first-class dimensions
- the explicit cross-combination rows in the source table are treated as
  upstream modeling assumptions, not repaired back into older proxy rules
- climate input is daily near-surface air temperature
- solar enters as a fixed EWEMBI monthly `(month, lat, lon)` climatology; the
  kernel reads it as a `(month, entity)` coordinate — the gridded path stacks
  it, the mapped path samples it via `gridded_coords`
- output duration is part of kernel evaluation: `native`, `daily`, `monthly`,
  or `seasonal`
- `seasonal` outputs can remain dask-backed until the caller materializes or
  writes them

## Main Entry Points

From `impacts.domain.buildings.chilled`:

- `normalize_chilled_climate_inputs` — selects, renames, and unit-normalizes
  the required climate inputs.
- `ChilledEnergyAssumptions` — equipment and setpoint parameters; build from
  a parametric-run row with `ChilledEnergyAssumptions.from_parametric_row`.
- `create_chilled_kernel(assumptions, *, entity_chunk=10_000, time_chunk=365)`
  — builds the `ImpactKernel` instance for CHILLED evaluation. The chunk
  sizes control the dask rechunk at kernel entry and are surfaced as
  explicit args so a calling script can tune them per-run (e.g. via env
  vars) without mutating module state.

From `impacts.domain.buildings.chilled_io`:

- `load_message_ix_buildings_archetypes` — loads and validates the ALPS2025
  archetype table.
- `load_message_ix_buildings_parametric_runs` — loads parametric-run
  assumptions when an upstream workflow provides `CHILLED_PARAMETRIC_PATH`.
- `load_ewembi_solar_climatology`, `load_r12_region_raster` — supporting
  inputs for the gridded path.
- `build_chilled_parameter_grids` — builds the `(lat, lon)` parameter
  Dataset (with solar carried as `(month, lat, lon)`) consumed by
  `evaluate_gridded_kernel`.
- `build_chilled_archetype_mapping` — explodes spatial units against the
  archetype table; returns `(mapping, solar)` — the mapping plus the
  `(month, lat, lon)` solar climatology to pass as `gridded_coords`.
- `validate_archetype_table` — exposed so callers can validate ad-hoc
  archetype tables outside the loader path.

From `impacts.kernel`:

- `evaluate_mapped_kernel(kernel, mapping, climate, duration, *, gridded_coords=None)`
  — mapped-entity evaluation. `gridded_coords` carries gridded fields (e.g. the
  CHILLED solar climatology) to sample per entity. Used by the CHILLED test
  suite and any analysis where the analytical units are intrinsically a list
  (e.g. country × archetype).
- `evaluate_gridded_kernel(kernel, climate, parameters, duration)` —
  gridded evaluation. Used by the production matrix runner. Output is
  entity-stacked: `(time, entity)` for `native`/`daily` (and
  `(season, year, entity)` after seasonal aggregation), with `lat`/`lon`
  carried as coordinates on `entity`. The production runner calls
  `reset_index("entity")` to write a `(time, lat, lon)` product.

Both primitives present the same `(time, entity)` shape to the physics
kernel internally; the kernel body is shape-agnostic.

## Environment And Input Paths

Path defaults live in `impacts.domain.buildings.chilled_config` and are
derived from:

- `WATXENE_PATH`
- `CHILLED_ARCHETYPE_PATH`
- `CHILLED_PARAMETRIC_PATH`
- `MESSAGE_R12_YAML_PATH`

These defaults are operational conveniences. Upstream workflows remain
responsible for choosing scenario/run definitions and providing any
non-default inputs; every loader accepts an explicit `path` argument.

The kernel-entry dask chunk sizes are also tunable via env vars by the
calling script (e.g. `run_chilled_diag.py` reads
`CHILLED_KERNEL_ENTITY_CHUNK` and `CHILLED_KERNEL_TIME_CHUNK` and forwards
them as `create_chilled_kernel(..., entity_chunk=..., time_chunk=...)`).
The kernel module itself does not read environment variables.

## Scope Boundaries

In scope for `impacts.domain.buildings`:

- archetype-table validation
- mapping archetypes onto spatial units
- climate normalization for CHILLED
- kernel definition and the physics it expresses
- kernel-entry chunking strategy

Out of scope:

- scenario assembly
- end-to-end pipeline orchestration
- raster generation beyond the validated upstream bundle
- implicit inference of parametric-run tables
- production-runner orchestration (lives in `scripts/`)

If a workflow depends on `par_var.csv`, it must pass
`CHILLED_PARAMETRIC_PATH` explicitly.
