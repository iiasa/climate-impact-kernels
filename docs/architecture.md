# Architecture

`impacts` is two layers, and dependencies point downward only: the domain
modules build on a shared core, never the reverse. The diagram is in the
[README](../README.md#architecture). A kernel is one impact calculation: the
climate inputs it reads, its constants and its equations.

## Module map

| Path | Contents |
| --- | --- |
| `impacts/engine.py` | The kernel definition (`Kernel`, `ClimateInput`) and the evaluation entry points `evaluate_mapped` and `evaluate_gridded`. |
| `impacts/isimip.py` | File discovery and loading for ISIMIP3b and ISIMIP3a counterclim, the Inter-Sectoral Impact Model Intercomparison Project's bias-adjusted climate data (counterclim is observed weather with the warming trend removed) (`isimip3b_paths`, `load_isimip_variable`, `load_isimip_dataset`, `load_isimip_gmt`). |
| `impacts/climatecore/warming.py` | Warming-level assignment, pooling by warming level, weighted group means. |
| `impacts/climatecore/regions.py` | Country and R12 region labels (R12 is MESSAGE's twelve world regions) from `common-definitions`; site-to-region aggregation. |
| `impacts/climatecore/time.py` | Monthly and seasonal aggregation. |
| `impacts/climatecore/space.py` | Sampling grids at points, cell areas, grid refinement, nearest valid cells. |
| `impacts/climatecore/units.py` | Temperature and pressure unit conversion. |
| `impacts/thermoelectric/wet.py` | Wet-cooling kernel (Li et al. 2025), saline intakes, `wet_cooling`. |
| `impacts/thermoelectric/dry.py` | Dry-cooling kernel (Qin et al. 2023), `dry_cooling`. |
| `impacts/thermoelectric/fuels.py` | The fuel-code table both kernels read. |
| `impacts/thermoelectric/fleet.py` | Unit-to-cell mappings, cooling classification, country totals. |
| `impacts/thermoelectric/regions.py` | Aggregation of unit outputs to countries or R12 regions. |
| `impacts/thermoelectric/warming_levels.py` | Usable-capacity summaries by warming level. |
| `impacts/thermoelectric/siting/cells.py` | Potential-site and existing-site cells, synthetic plants, evaluation at potential sites. |
| `impacts/thermoelectric/siting/site_evaluation.py` | Year sampling (by warming level and from a fixed reference period) and the site evaluation (`run_site_evaluation`). |
| `impacts/thermoelectric/siting/summary.py` | Per-region summary of the site evaluation and the comparison of potential sites with existing sites. |
| `impacts/thermoelectric/siting/loss_split.py` | Splitting the wet capacity loss into thermal loss and water-availability loss. |
| `impacts/thermoelectric/siting/river_network.py` | River network and the allocation of water along it. |
| `impacts/thermoelectric/siting/curves.py` | Cell orderings, capacity curves of existing sites and of potential sites with water shared along the river network, the additivity check. |
| `impacts/thermoelectric/siting/rime.py` | Samples by new-build ratio and warming-level percentiles. |
| `impacts/thermoelectric/siting/outputs.py` | Output files for potential sites, the residual-water ordering, and pooling across GCMs. |
| `impacts/thermoelectric/siting/_io.py` | Write helpers that refuse existing files and write outputs together or not at all. |
| `impacts/buildings/chilled.py` | CHILLED kernel (building cooling and heating energy, after Mastrucci et al. 2019) and `chilled_energy`. |
| `impacts/buildings/archetypes.py` | CHILLED input loaders, validators and builders. |
| `impacts/buildings/_config.py` | CHILLED input paths from environment variables. |
| `impacts/buildings/regions.py` | Country raster and site-to-country aggregation of CHILLED outputs. |
| `impacts/buildings/warming_levels.py` | Energy-intensity summaries by warming level. |

The pages that describe them: [thermoelectric.md](thermoelectric.md),
[siting.md](siting.md), [buildings.md](buildings.md),
[climatecore.md](climatecore.md), [assumptions.md](assumptions.md) and
[workflows.md](workflows.md).

## Evaluation

`evaluate_mapped` and `evaluate_gridded` select and unit-convert the climate
inputs a kernel declares, sample them at the sites (rows of a mapping table, such
as power-plant units or grid cells), run the physics, attach the
site coordinates and assumption attributes, and resample to the requested
duration. A kernel's physics is a function of climate, site values and
assumptions only; it reads no files. [climatecore.md](climatecore.md#evaluation)
describes what a kernel declares and what the engine does around it.

## Import rule

`impacts.climatecore` must not import from `impacts.thermoelectric`,
`impacts.buildings`, `impacts.engine` or `impacts.isimip`. `import-linter`
enforces it (`[tool.importlinter]` in `pyproject.toml`); run
`uv run lint-imports`.

What belongs in `climatecore`:

- operations on `xarray.DataArray` or `xarray.Dataset` inputs and outputs;
- operations that recur, or could recur, across domains;
- anything that must stay dask-parallelizable in the main computation.

What does not:

- ISIMIP file discovery, scenario metadata and path conventions;
- domain physics;
- running the steps of a single workflow in order;
- output layouts tied to one downstream consumer.

A step that looks generic but needs domain-specific settings stays in the
domain module until a second use appears.

Because no domain import crosses the boundary, `impacts/climatecore`,
`tests/climatecore/` and `docs/climatecore.md` can be extracted into a
standalone package without rewriting imports; the one test fixture they share with
`tests/regression/` (a miniature region definitions checkout) goes with them.
