# Evaluation

`climateresponse.evaluation` provides reusable orchestration for applying a
physics kernel to climate inputs after normalization and spatial attachment.

It is the shared contract behind downstream repos that want to evaluate
array-native kernels without rebuilding the same xarray plumbing.

## Main Entry Points

- `evaluate_mapped_dataset`
  For workflows where a tabular mapping provides entity coordinates and
  metadata. The function normalizes climate inputs, attaches the gridded values
  to mapped entities, runs the kernel, applies clips, and packages the result.
- `evaluate_gridded_dataset`
  For workflows where gridded climate inputs and gridded parameter fields share
  spatial dimensions. The function stacks spatial dimensions into `entity`,
  attaches parameters, runs the kernel, applies clips, and packages the result.
- `EvaluationResult`
  A structured return type for kernels that want to supply data variables and
  dataset attributes without constructing the full output dataset themselves.

## Design Constraints

- inputs and outputs stay xarray-native
- dask-backed execution remains viable
- downstream temporal aggregation keeps named xarray axes, and the standard
  seasonal path consolidates `season`/`year` chunks for dask-backed outputs
- kernel functions operate on already-normalized, already-attached data
- evaluation is generic orchestration, not domain physics

## Boundary

If a workflow needs:

- plant-specific cooling equations
- buildings-specific assumptions
- project-specific file discovery
- scenario assembly

that belongs above this layer, not inside `evaluation.py`.
