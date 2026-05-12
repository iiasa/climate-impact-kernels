`impacts` is a library of domain-specific climate-impact contracts, mappings,
and physics kernels, built on a shared xarray-native core.

It is organized as `impacts.<domain>`, not as a pipeline CLI.

Current layout:

- `impacts.core` — shared xarray-native mechanics (warming levels, temporal
  and spatial aggregation, unit normalization, region loading, response
  surfaces, evaluation orchestration)
- `impacts.thermoelectric` — thermoelectric domain
- `impacts.buildings` — buildings domain
- `impacts.isimip` — ISIMIP3b ingress
- `impacts.kernel` — kernel contract used by domain modules

## What belongs here

`impacts` owns:

- domain contracts and validation
- joins from climate inputs to domain entities
- physics kernels and assumptions
- domain-specific aggregation targets
- checked-in artifacts that encode validated mappings
- reusable xarray-native mechanics that recur across domains (inside
  `impacts.core`)

`impacts` does not own:

- dataset discovery or catalog logic outside of `impacts.isimip`
- project-specific workflow orchestration above the kernel contract
- non-stationary GEV extremes (lives in `rimeX`)
- output layouts tied to a single downstream consumer

## Read this first

- [docs/architecture.md](docs/architecture.md) — layered structure and
  boundary contract between `impacts.core` and the domain modules
- [docs/thermoelectric.md](docs/thermoelectric.md)
- [docs/buildings.md](docs/buildings.md)
- [docs/workflows.md](docs/workflows.md)
- [docs/core/shared_core_design.md](docs/core/shared_core_design.md) and the
  rest of [docs/core/](docs/core/) for the shared-core scope

Then open the concrete modules you need:

- `impacts/core/__init__.py`
- `impacts/thermoelectric/__init__.py`
- `impacts/buildings/chilled.py`
- `impacts/thermoelectric/rimex.py`
- `impacts/buildings/rimex.py`
- `impacts/kernel.py`
- `impacts/isimip.py`

## Boundary discipline

`impacts.core` must not import from `impacts.thermoelectric`,
`impacts.buildings`, `impacts.isimip`, or `impacts.kernel`. This is
enforced by `import-linter` (`uv run lint-imports`) and is expected to run
in CI. See [docs/architecture.md](docs/architecture.md) for the rationale
and the extraction recovery path.

If a generic-looking helper appears to need domain knowledge, generalize the
signature; do not punch a hole in the contract.

## Package shape

### `impacts.core`

Reusable xarray-native mechanics. Hot paths operate directly on
`xarray.DataArray` / `xarray.Dataset` objects and remain dask-compatible.
See [docs/core/shared_core_design.md](docs/core/shared_core_design.md) for
the full scope and what is intentionally out of scope.

### `impacts.thermoelectric`

Thermo-electric plant mapping, cooling classification, wet/dry kernel
construction, and regional aggregation.

Important modules:

- `mapping.py`
- `cooling.py`
- `wet.py`
- `dry.py`
- `aggregation.py`
- `rimex.py`

Important checked-in artifacts under `data/`:

- `unit_cell_mapping.parquet`
- `recovered_unit_cell_mapping.parquet`
- `location_cell_mapping.parquet`
- `country_rollup.parquet`
- `unit_cooling_mapping.parquet`

### `impacts.buildings`

The `buildings` domain currently centers on
`impacts.buildings.chilled`, which evaluates the CHILLED pathway
through the shared `ImpactKernel` contract.

Important modules:

- `buildings/chilled.py`
- `buildings/rimex.py`
- `kernel.py`
- `isimip.py`

## Design constraints

- The package stays domain-first. Reusable numerical transforms belong in
  `impacts.core`, not duplicated inside each domain.
- `kernel.py` is the shared evaluation contract for domain kernels.
  Duration is part of evaluation, not a separate downstream convention.
- Thermoelectric outputs may be aggregated to countries or MESSAGE R12
  regions, but MESSAGE regional semantics are not embedded in the mapping
  artifacts.
- The buildings CHILLED kernel targets the ALPS2025 archetype contract. Do
  not silently coerce it back to older proxy schemas.

## RIME-X adapters

RIME-X preprocessing follows the layered structure directly:

- `impacts.core` owns the generic xarray-first warming-level attachment and
  quantile-surface builders
- `impacts.thermoelectric.rimex` reshapes the thermoelectric country
  parquet into an xarray UCR field and derives seasonal UCR
- `impacts.buildings.rimex` aggregates CHILLED entity outputs to countries
  and attaches filename-derived archetype metadata

The durable artifact scripts under `scripts/` remain the I/O boundary.

## Provenance notes

The thermoelectric artifacts are built from a preconstructed global fleet
table with unit attributes, cooling assignments, and partial coordinate
coverage. The package materializes contract artifacts from that source
table; it does not reconstruct the fleet from raw registries in this
repository.

The conservative recovery path intentionally leaves much of the missing
fleet unresolved rather than inferring locations heuristically.

## Development

Use `uv` for local execution.

Install: `uv sync` (installs the project editable and the dev group with
`pytest` and `import-linter`).

Tests: `uv run pytest` or pick a focused subset:

- `tests/core/test_core.py`
- `tests/test_mapping.py`
- `tests/test_cooling.py`
- `tests/test_wet.py`
- `tests/test_dry.py`
- `tests/test_aggregation.py`
- `tests/test_isimip.py`
- `tests/test_buildings.py`
- `tests/test_recovery.py`
- `tests/test_rimex.py`

Boundary check: `uv run lint-imports` (runs the `import-linter` contracts
declared in `pyproject.toml`).
