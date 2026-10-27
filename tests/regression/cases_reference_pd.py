"""End-to-end test of the present-day reference calculation in the steps that produce the wet-cooling outputs.

The residual-water order and the present-day reference part of the outputs come
from the present-day reference period: the 1995-2014 ``historical`` climate,
written on the warming-level axis under the string ``REF-PD``, with its wet
capacity loss split against its own site evaluation. The test runs each step as
the library functions do, on a made-up ``historical`` run.

Each step is fed the site evaluation written from the ``REF-PD`` settings and the
split capacity loss prepared in ``world.py``. Each step is therefore checked on
its own input, not end to end.
"""

from __future__ import annotations

import shutil

import pandas as pd

from . import entrypoints as ep
from . import fixtures
from .registry import case

GRID = 7  # capacity steps: keeps the curve frames small


def _fresh(w, name: str):
    directory = w.work / name
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    return directory


def _read_tree(directory, pattern="*.parquet"):
    return {path.name: pd.read_parquet(path) for path in sorted(directory.glob(pattern))}


def _write(frame, path):
    frame.to_parquet(path, index=False)
    return path


@case("library.reference_pd")
def library_reference_pd(w):
    scenarios = [fixtures.HISTORICAL]
    region = ep.get("universe.RegionRasterPaths")(**w.region_paths)
    wet_file, dry_file = w.fleet_files
    directory = _fresh(w, "library_reference_pd")

    # The site evaluation: the cells and the years of the ``REF-PD`` settings, then the evaluation.
    with w.short_reference_window():
        years = ep.get("audit.sample_years_within_reference")(ep.get("audit.ReferenceSampling")("REF-PD"))
    cells = ep.get("audit.build_census_cells")(
        w.isimip_paths, fixtures.GCM, scenarios, region,
        fleet_wet=wet_file, fleet_dry=dry_file,
        cells_considered_scenarios=[*fixtures.SCENARIOS, fixtures.HISTORICAL],
    )  # fmt: skip
    windows = ep.get("audit.evaluate_candidate_census")(
        paths=w.isimip_paths, gcm=fixtures.GCM, sampled_years=years,
        wet_cells=cells.wet_cells, dry_cells=cells.dry_cells, responses=w._REF_PD_MONTHLY,
    )  # fmt: skip
    census = pd.read_parquet(ep.get("audit.write_candidate_census")(windows, directory / "census.parquet"))

    # The split of the wet capacity loss: the site evaluation is its own reference and carries the ``REF-PD`` label.
    runner_census = w.audit_reference_pd_run
    decomposed = ep.get("decomposition.decompose_wet_channels")(
        [runner_census], [runner_census], directory / "decomposition.parquet",
        reference=ep.get("decomposition.REF_PD_BASELINE_REFERENCE"),
    )  # fmt: skip
    decomposition = w.reference_pd_decomposition_path

    # The river network of the historical run.
    cells_of_decomposition = ep.get("topology.read_candidate_cells")(decomposition)
    flowdir = ep.get("topology.read_flow_directions")(w.flow_direction_files["acyclic"])
    topologies = ep.get("topology.build_reach_topologies")(cells_of_decomposition, flowdir)
    topology_dir = directory / "topology"
    river_network_paths = ep.get("topology.write_reach_topologies")(topologies, fixtures.GCM, topology_dir)

    # The residual-water order, then the outputs on it and on the shared fixed order.
    result = ep.get("products.build_residual_water_ordering")(
        [decomposition], [runner_census], river_network_paths
    )
    ordering_path = ep.get("products.write_residual_water_ordering")(
        result, directory / "residual_ordering.parquet"
    )
    with_baseline = ep.get("products.build_residual_water_ordering")(
        [decomposition], [runner_census], river_network_paths,
        baseline_ordering=pd.read_parquet(w.fixed_ordering_path),
    )  # fmt: skip
    build = ep.get("products.build_wet_candidate_products")
    write = ep.get("products.write_wet_candidate_products")

    def export(name, fixed_ordering):
        products = build(
            [decomposition], [runner_census], river_network_paths, fixed_ordering, capacity_grid_points=GRID
        )
        write(products, directory / name)
        return _read_tree(directory / name)

    shared = export("shared", pd.read_parquet(w.fixed_ordering_path))
    residual = export("residual", pd.read_parquet(ordering_path))
    return {
        "census": census,
        "sampled_years": years,
        "decomposition_reference": decomposed.low_gwl_reference,
        "decomposition": pd.read_parquet(decomposed.output),
        "topology_file_names": sorted(path.name for path in river_network_paths),
        "topology_tables": _read_tree(topology_dir),
        "ordering": result.ordering,
        "ordering_with_baseline": with_baseline.ordering,
        "ordering_moves_against_baseline": with_baseline.moves,
        "shared_ordering_per_gcm": shared["wet_ucr_curves_per_gcm.parquet"],
        "shared_ordering_pooled_median": shared["wet_ucr_curves_pooled_median.parquet"],
        "shared_ordering_additivity_oracle": shared["wet_ucr_curve_additivity_oracle.parquet"],
        "shared_ordering_rime_samples": shared["wet_ucr_rime_samples.parquet"],
        "residual_ordering_per_gcm": residual["wet_ucr_curves_per_gcm.parquet"],
        "residual_ordering_rime_samples": residual["wet_ucr_rime_samples.parquet"],
    }
