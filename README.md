# impacts

A library of domain-specific climate-impact contracts, mappings, and
physics kernels, built on a shared xarray-native core.

Organized as `impacts.<area>`, not as a pipeline CLI.

## Install

```sh
pip install git+https://github.com/iiasa/climate-impact-kernels.git
# or, for development:
git clone https://github.com/iiasa/climate-impact-kernels.git
cd climate-impact-kernels
uv sync
```

Requires Python 3.13+. Cluster path defaults (ISIMIP forcing, CHILLED
inputs) are controlled via environment variables — see
[docs/buildings.md](docs/buildings.md) and
[docs/thermoelectric.md](docs/thermoelectric.md).

## Mental model

```
        ┌────────────────────────────────────────────────────────┐
        │ impacts.domain         (thermoelectric, buildings)      │
        │                                                         │
        │   physics kernels • RIME-X adapters • mapping artifacts │
        └─────────────┬──────────────────────────┬────────────────┘
                      │ depends on               │ depends on
                      ▼                          ▼
        ┌──────────────────────┐    ┌──────────────────────────┐
        │ impacts.kernel       │    │ impacts.isimip           │
        │ evaluation contract  │    │ ISIMIP3b ingress         │
        └──────────┬───────────┘    └──────────────────────────┘
                   │ depends on
                   ▼
        ┌────────────────────────────────────────────────────────┐
        │ impacts.climatecore    (xarray-native, dask-friendly)   │
        │                                                         │
        │   baseline.py • evaluation.py • warming.py              │
        │   regions.py • transforms/                              │
        └────────────────────────────────────────────────────────┘
```

The arrow direction is the rule: everything above may depend on
everything below it. The reverse is forbidden and enforced by
`import-linter`. See [docs/architecture.md](docs/architecture.md) for
the boundary contract and the extraction recovery path.

## Read this first

- [docs/architecture.md](docs/architecture.md) — layered structure and
  the `climatecore` ↓ `domain` boundary contract.
- [docs/climatecore.md](docs/climatecore.md) — the shared core's
  layout, entry points, and contracts.
- [docs/thermoelectric.md](docs/thermoelectric.md)
- [docs/buildings.md](docs/buildings.md)
- [docs/workflows.md](docs/workflows.md)

Then open the concrete modules you need:

- `impacts/climatecore/__init__.py`
- `impacts/domain/thermoelectric/__init__.py`
- `impacts/domain/buildings/chilled.py`
- `impacts/domain/thermoelectric/rimex.py`
- `impacts/domain/buildings/rimex.py`
- `impacts/kernel.py`
- `impacts/isimip.py`

## Package shape

### `impacts.climatecore`

Reusable xarray-native mechanics. Hot paths take `xarray.DataArray` or
`xarray.Dataset` and return the same. Layout:

- `baseline.py`, `evaluation.py` at the top.
- `warming.py` — GMT running means, warming-level matching and
  binning, response-surface interpolation, canonical dataclasses.
- `regions.py` — region loading and entity-to-region aggregation
  against `common-definitions`.
- `transforms/` — temporal, spatial, unit, and statistical transforms.

See [docs/climatecore.md](docs/climatecore.md) for module-level docs.

### `impacts.domain.thermoelectric`

Thermo-electric plant mapping, cooling classification, wet/dry kernel
construction, regional aggregation, and the RIME-X adapter.

Modules: `mapping.py`, `cooling.py`, `wet.py`, `dry.py`,
`aggregation.py`, `rimex.py`.

Checked-in artifacts under `data/`:

- `unit_cell_mapping.parquet`
- `recovered_unit_cell_mapping.parquet`
- `location_cell_mapping.parquet`
- `country_rollup.parquet`
- `unit_cooling_mapping.parquet`

### `impacts.domain.buildings`

The buildings domain centers on `impacts.domain.buildings.chilled`,
which evaluates the CHILLED pathway through the shared `ImpactKernel`
contract. The RIME-X adapter lives at
`impacts.domain.buildings.rimex`.

### `impacts.kernel`

Shared evaluation contract for domain kernels. Duration is part of
evaluation, not a separate downstream convention.

### `impacts.isimip`

Concrete ISIMIP3b file discovery and variable loading. Input root is
configurable via the `WATXENE_PATH` environment variable; the default
points at the IIASA cluster mount (`/mnt/p/watxene`).

## Design constraints

- The package stays domain-first. Reusable numerical transforms belong
  in `impacts.climatecore`.
- Thermoelectric outputs may be aggregated to countries or MESSAGE R12
  regions, but MESSAGE regional semantics are not embedded in the
  mapping artifacts.
- The CHILLED kernel targets the ALPS2025 archetype contract. Do not
  silently coerce it back to older proxy schemas.

## Development

```sh
uv sync                  # installs the project editable + dev group
uv run pytest            # 101 tests
uv run lint-imports      # boundary contract
```

Focused test subsets:

- `tests/climatecore/{test_baseline,test_evaluation,test_warming,test_transforms,test_regions}.py`
- `tests/test_{mapping,cooling,wet,dry,aggregation,isimip,buildings,recovery,rimex}.py`

## RIME-X adapters

[RIME-X](https://github.com/iiasa/rimeX) is the IIASA probabilistic
emulator for regional climate indicators. Domain modules expose adapters
that translate impact-kernel outputs into the xarray fields RIME-X
consumes.

- `impacts.climatecore` owns the generic xarray-first warming-level
  attachment and quantile-surface builders.
- `impacts.domain.thermoelectric.rimex` reshapes the thermoelectric
  country parquet into an xarray UCR field.
- `impacts.domain.buildings.rimex` aggregates CHILLED entity outputs to
  countries and attaches filename-derived archetype metadata.

## Provenance notes

The thermoelectric artifacts are built from a preconstructed global
fleet table with unit attributes, cooling assignments, and partial
coordinate coverage. The conservative recovery path intentionally
leaves much of the missing fleet unresolved rather than inferring
locations heuristically.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Citation

This package integrates physics kernels and methods from upstream
scientific work. Results derived from a specific kernel should cite the
underlying method paper or upstream repository in addition to (or in
place of) this software:

- **CHILLED** (buildings cooling demand) — see
  [iiasa/message-ix-buildings](https://github.com/iiasa/message-ix-buildings)
  and the references in [docs/buildings.md](docs/buildings.md).
- **Thermoelectric production and cooling** — see method notes in
  [docs/thermoelectric.md](docs/thermoelectric.md).
- **RIME-X** (regional climate-indicator emulation) — see
  [iiasa/rimeX](https://github.com/iiasa/rimeX).

For software-level citation, point at
<https://github.com/iiasa/climate-impact-kernels>. A formal citation
entry will be added when an `impacts`-specific paper or DOI is available.
