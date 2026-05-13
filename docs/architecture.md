# Architecture

`impacts` is two layers and dependencies point downward only. See the
mental-model diagram in [README.md](../README.md) for the visual
summary.

- **`impacts.climatecore`** — reusable xarray-native mechanics:
  `baseline.py`, `evaluation.py`, `warming.py` (warming-level matching,
  response surfaces, canonical dataclasses), `regions.py` (region
  loading and entity-to-region aggregation), and `transforms/`
  (temporal, spatial, units, stats).
- **`impacts.domain`** — domain-specific contracts, mappings, physics
  kernels, and RIME-X adapters under `thermoelectric/` and
  `buildings/`.
- **`impacts.kernel`** — the shared evaluation contract that every
  domain kernel implements.
- **`impacts.isimip`** — concrete ISIMIP3b file discovery and variable
  loading.

## Boundary contract

`impacts.climatecore` must not import from `impacts.domain`,
`impacts.kernel`, or `impacts.isimip`. The contract is enforced by
`import-linter` (`[tool.importlinter]` in `pyproject.toml`); run
`uv run lint-imports` locally and in CI.

## What belongs in climatecore

- Operations expressible in terms of `xarray.DataArray` /
  `xarray.Dataset` inputs and outputs.
- Mechanics that recur or could plausibly recur across multiple
  domains.
- Helpers that remain coherent under the hypothetical "any future
  non-`impacts` consumer".
- Anything that needs to stay dask-parallelizable in the hot path.

## What does not belong in climatecore

- ISIMIP file discovery, scenario metadata, project-specific path
  conventions.
- Domain physics — cooling equations, building-energy kernels, crop
  yields.
- Adapter orchestration for a single workflow.
- Output schemas tied to one downstream consumer.

When a step looks generic but turns out to need domain-specific knobs,
default to keeping it in the domain module. Promotion to
`climatecore` happens only when a second use site emerges.

## Extraction recovery

If a non-`impacts` consumer ever materializes, extraction is a
`git filter-repo --subdirectory-filter impacts/climatecore` plus the
matching operations on `tests/climatecore/` and `docs/climatecore.md`.
The boundary linter guarantees no domain imports have leaked across,
so the extracted package builds standalone without rewriting imports.
