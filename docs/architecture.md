# Architecture

`impacts` has two layers:

- **`impacts.core`** — reusable xarray-native mechanics: temporal/spatial
  aggregation, unit normalization, warming-level matching and binning,
  baseline anomalies, response-surface contracts, evaluation orchestration,
  and region loading.
- **Domain modules** — `impacts.thermoelectric`, `impacts.buildings`,
  `impacts.isimip`, `impacts.kernel`. These own domain contracts, joins,
  physics kernels, RIME-X adapters, and ISIMIP ingress.

Dependencies point downward only: domain → core. Core never imports from
any domain module.

## The boundary contract

`impacts.core` must not import from `impacts.thermoelectric`,
`impacts.buildings`, `impacts.isimip`, or `impacts.kernel`.

This is enforced by `import-linter` (configured in `pyproject.toml` under
`[tool.importlinter]`). Run `uv run lint-imports` locally and in CI.

Contract violations are not negotiable. If a generic helper appears to need
domain knowledge to function, the design is wrong — either generalize the
helper to take the domain concept as an argument, or keep the
domain-coupled version inside the domain module.

## What belongs in core

- Operations expressible in terms of `xarray.DataArray` / `xarray.Dataset`
  inputs and outputs.
- Mechanics that recur or could plausibly recur across multiple domains.
- Helpers whose contract is stable under the rule "if a non-`impacts`
  consumer ever appears, this would still apply to them".
- Anything dask-parallelizable in the hot path.

## What does not belong in core

- ISIMIP file discovery, scenario metadata, project-specific path
  conventions.
- Domain physics — cooling equations, building energy kernels, crop yields.
- Adapter orchestration for a single workflow.
- Output schemas tied to one downstream consumer.

When a step looks generic but turns out to need domain-specific knobs,
default to keeping it in the domain module. Promotion to core happens only
when a second use site emerges. Premature promotion is the failure mode
this rule prevents.

## History

`impacts.core` was originally a standalone Python package
(`climateresponse`) consumed by `impacts` via a `path =` dependency. It was
folded in when it became clear that `impacts` was the only real consumer
and the two repositories were co-evolving on essentially every feature.

The boundary survived the fold-in as a code-enforced rule rather than a
repo-enforced one. The standalone repo at `~/climateresponse` is archived
at the pre-fold point.

## Extraction recovery

If a second non-`impacts` consumer ever materializes and the core is large
enough to justify separate hosting again, extraction is one
`git filter-repo --subdirectory-filter impacts/core` plus the inverse
operations on `tests/core/` and `docs/core/`. The boundary linter
guarantees that no domain imports have leaked across, so the extracted
package will build standalone without rewriting imports.

Do not let this safety valve become a planning assumption. The cost of
extraction grows with how much code lives on the wrong side of the
boundary; the linter keeps that cost low, but only as long as it actually
runs in CI.
