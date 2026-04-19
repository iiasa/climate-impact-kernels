`impacts` is a library of domain-specific climate-impact contracts, mappings,
and physics kernels.

It is organized as `impacts.<domain>`, not as a pipeline CLI.

Current domains:

- `impacts.thermoelectric`
- `impacts.buildings`

## What belongs here

`impacts` owns:

- domain contracts and validation
- joins from climate inputs to domain entities
- physics kernels and assumptions
- domain-specific aggregation targets
- checked-in artifacts that encode validated mappings

`impacts` does not own:

- generic xarray preprocessing helpers
- dataset discovery or catalog logic
- project-specific workflow orchestration
- generic warming-level mechanics

Those reusable array-native transforms belong in `climateresponse`, which is an
explicit dependency of this package.

If a missing step is generic across domains, add it to `climateresponse` first
and consume it here rather than rebuilding it inside `impacts`.

## Read This First

- [docs/thermoelectric.md](docs/thermoelectric.md)
- [docs/buildings.md](docs/buildings.md)
- [docs/workflows.md](docs/workflows.md)

Then open the concrete modules you need:

- `impacts/thermoelectric/__init__.py`
- `impacts/buildings/chilled.py`
- `impacts/kernel.py`
- `impacts/isimip.py`

## Package Shape

### `impacts.thermoelectric`

Thermo-electric plant mapping, cooling classification, wet/dry kernel
construction, and regional aggregation.

Important modules:

- `mapping.py`
- `cooling.py`
- `wet.py`
- `dry.py`
- `aggregation.py`

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
- `kernel.py`
- `isimip.py`

## Design Constraints

- The package stays domain-first. Reusable numerical transforms belong in
  `climateresponse`.
- `kernel.py` is the shared evaluation contract for domain kernels. Duration is
  part of evaluation, not a separate downstream convention.
- Thermoelectric outputs may be aggregated to countries or MESSAGE R12 regions,
  but MESSAGE regional semantics are not embedded in the mapping artifacts.
- The buildings CHILLED kernel targets the ALPS2025 archetype contract. Do not
  silently coerce it back to older proxy schemas.

## Provenance Notes

The thermoelectric artifacts are built from a preconstructed global fleet table
with unit attributes, cooling assignments, and partial coordinate coverage. The
package materializes contract artifacts from that source table; it does not
reconstruct the fleet from raw registries in this repository.

The conservative recovery path intentionally leaves much of the missing fleet
unresolved rather than inferring locations heuristically.

## Development

Use `uv` for local execution.

Start with the relevant tests for the surface you changed:

- `tests/test_mapping.py`
- `tests/test_cooling.py`
- `tests/test_wet.py`
- `tests/test_dry.py`
- `tests/test_aggregation.py`
- `tests/test_isimip.py`
- `tests/test_buildings.py`
- `tests/test_recovery.py`

Run a focused subset unless your change crosses domain boundaries.
