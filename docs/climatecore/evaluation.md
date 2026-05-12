# Evaluation

`impacts.climatecore.evaluation` orchestrates the application of a
physics kernel to climate inputs after normalization and spatial
attachment.

## Entry points

- `evaluate_mapped_dataset` — for workflows where a tabular mapping
  provides entity coordinates and metadata. Normalizes climate inputs,
  attaches the gridded values to mapped entities, runs the kernel,
  applies clips, and packages the result.
- `evaluate_gridded_dataset` — for workflows where gridded climate
  inputs and gridded parameter fields share spatial dimensions. Stacks
  the spatial dimensions into `entity`, attaches parameters, runs the
  kernel, applies clips, and packages the result.
- `EvaluationResult` — structured return type for kernels that want to
  supply data variables and dataset attributes without constructing the
  full output dataset themselves.

## Constraints

- Inputs and outputs stay xarray-native.
- Dask-backed execution remains viable.
- Downstream temporal aggregation preserves named xarray axes; the
  standard seasonal path consolidates `season`/`year` chunks.
- Kernel callables operate on already-normalized, already-attached data.

## Out of scope

- Plant-specific cooling equations, building-specific assumptions, and
  any other domain physics.
- Project-specific file discovery or scenario assembly.

Those belong above this layer, in the domain modules.
