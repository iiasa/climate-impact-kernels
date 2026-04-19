# Buildings

`impacts.buildings` currently centers on the CHILLED pathway implemented in
`impacts.buildings.chilled`.

The buildings kernel uses the shared `ImpactKernel` contract from
`impacts.kernel`, but the parameters, contracts, and validation rules are
domain-specific.

## Current Contract

The current implementation targets the ALPS2025 archetype contract only.

Important assumptions:

- `arch_input_reg.csv` is consumed with `building_type`, `urt`, `arch`, and
  `region` as first-class dimensions
- the explicit cross-combination rows in the source table are treated as
  upstream modeling assumptions, not repaired back into older proxy rules
- climate input is daily near-surface air temperature
- solar enters as a fixed EWEMBI monthly climatology attached to the mapping
  table
- output duration is part of kernel evaluation: `native`, `daily`, `monthly`,
  or `seasonal`
- `seasonal` outputs can remain dask-backed until the caller materializes or
  writes them

## Main Entry Points

- `normalize_chilled_climate_inputs`
  Selects, renames, and unit-normalizes the required climate inputs.
- `load_message_ix_buildings_archetypes`
  Loads and validates the ALPS2025 archetype table.
- `load_message_ix_buildings_parametric_runs`
  Loads parametric-run assumptions when an upstream workflow provides
  `CHILLED_PARAMETRIC_PATH`.
- `create_chilled_kernel`
  Builds the `ImpactKernel` instance for CHILLED evaluation.
- `evaluate_kernel` / `evaluate_gridded_kernel`
  Execute the kernel at the requested duration.

## Environment And Input Paths

The module currently derives default paths from:

- `WATXENE_PATH`
- `CHILLED_ARCHETYPE_PATH`
- `CHILLED_PARAMETRIC_PATH`
- `MESSAGE_R12_YAML_PATH`

These defaults are operational conveniences, not a claim that path resolution
belongs inside the pure kernel contract. Upstream workflows remain responsible
for choosing scenario/run definitions and providing any non-default inputs.

## Scope Boundaries

In scope:

- archetype-table validation
- mapping archetypes onto spatial units
- climate normalization for CHILLED
- kernel execution and duration-aware aggregation

Out of scope:

- scenario assembly
- end-to-end pipeline orchestration
- raster generation beyond the validated upstream bundle
- implicit inference of parametric-run tables

If a workflow depends on `par_var.csv`, it must pass
`CHILLED_PARAMETRIC_PATH` explicitly.
