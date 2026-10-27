"""End-to-end tests of the wet-cooling output files, the RIME-X samples and how the library composes them."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import xarray as xr

from . import entrypoints as ep
from . import fixtures
from .canon import Raised
from .registry import case, outcome, outcome_type

GRID = 7  # capacity steps: keeps the curve frames small


def read_back(path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _write(frame: pd.DataFrame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _directory(w, name: str):
    directory = w.work / name
    directory.mkdir(exist_ok=True)
    return directory


def _topology_path(directory, name: str, frame: pd.DataFrame):
    return _write(frame, directory / name)


# --- loaders ---------------------------------------------------------------------------------


@case("export.loaders")
def export_loaders(w):
    directory = _directory(w, "export_loaders")
    decomposition = [w.decomposition_path]
    samples = ep.get("export.load_decomposition_samples")
    ceilings = ep.get("export.load_reach_water_ceilings")
    topology = ep.get("export.load_topology")
    census = [w.audit_monthly_run]
    frame = pd.read_parquet(w.topology_files[0])
    conflicting = frame.assign(
        downstream_cell_id=frame["downstream_cell_id"].where(
            frame["cell_id"] != frame["cell_id"].iloc[0], frame["cell_id"].iloc[-1]
        )
    )
    conflicting = pd.concat([frame, conflicting.iloc[:1]], ignore_index=True)
    unnamed = _write(frame, directory / "topology_without_scenario.parquet")
    conflict_path = _write(conflicting, directory / "conflict_ssp126_topology.parquet")
    incumbent_topology = _write(
        frame.assign(population="incumbent"), directory / "incumbent_ssp126_topology.parquet"
    )
    stripped = _write(
        pd.read_parquet(w.decomposition_path).drop(columns=["capacity_factor"]),
        directory / "decomposition_missing_column.parquet",
    )
    return {
        "cell_means_candidate": ep.get("export.load_cell_means")(decomposition, population="candidate"),
        "cell_means_incumbent": ep.get("export.load_cell_means")(decomposition, population="incumbent"),
        "cell_means_unknown_population": outcome(
            ep.get("export.load_cell_means"), decomposition, population="neither"
        ),
        "samples_candidate": samples(decomposition, population="candidate"),
        "samples_incumbent": samples(decomposition, population="incumbent"),
        "samples_missing_column": outcome(samples, [stripped], population="candidate"),
        "ceilings_candidate": ceilings(census, population="candidate"),
        "ceilings_year_preserved": ceilings(census, population="candidate", preserve_year=True),
        "ceilings_incumbent_year_preserved": ceilings(
            census, population="incumbent", preserve_year=True
        ),
        "ceilings_unknown_population": outcome(ceilings, census, population="neither"),
        "topology_both_scenarios": topology(w.topology_files, population="candidate"),
        "topology_single": topology(w.topology_files[:1], population="candidate"),
        "topology_no_scenario_in_name": outcome(topology, [unnamed], population="candidate"),
        "topology_conflict": outcome(topology, [conflict_path], population="candidate"),
        "topology_wrong_population": outcome(topology, [incumbent_topology], population="candidate"),
    }


# --- ordering -------------------------------------------------------------------------------------


@case("export.orderings")
def export_orderings(w):
    candidate = w.samples["candidate"]
    incumbent = w.samples["incumbent"]
    ordering = ep.get("export.build_fixed_cell_ordering")
    residual_samples = ep.get("export.residual_water_reference_samples")
    residual_ordering = ep.get("export.build_residual_water_cell_ordering")
    ceilings, topology = w.water_limits, w.topology
    nonfinite = candidate.assign(usable_capacity_mw=candidate["usable_capacity_mw"].astype(float))
    nonfinite.loc[nonfinite.index[0], "usable_capacity_mw"] = np.nan
    return {
        "fixed": w.fixed_ordering,
        "fixed_from_incumbent": ordering(incumbent),
        "fixed_empty": outcome(ordering, candidate.iloc[0:0]),
        "fixed_missing_columns": outcome(ordering, candidate.drop(columns=["region_id"])),
        "fixed_nonfinite": outcome(ordering, nonfinite),
        "residual_samples": residual_samples(candidate, ceilings, topology, existing_site_samples=incumbent),
        "residual_samples_no_incumbent": residual_samples(
            candidate, ceilings, topology, existing_site_samples=None
        ),
        "residual_ordering": residual_ordering(candidate, ceilings, topology, existing_site_samples=incumbent),
        "residual_ordering_no_incumbent": residual_ordering(
            candidate, ceilings, topology, existing_site_samples=None
        ),
        "residual_empty": outcome(
            residual_samples, candidate.iloc[0:0], ceilings, topology, existing_site_samples=incumbent
        ),
        "residual_missing_ceilings": outcome(
            residual_samples, candidate, ceilings.iloc[:3], topology, existing_site_samples=incumbent
        ),
        "residual_missing_topology": outcome(
            residual_samples, candidate, ceilings, topology.iloc[:2], existing_site_samples=incumbent
        ),
    }  # fmt: skip


# --- curves ------------------------------------------------------------------------------------------


@case("export.capacity_curves")
def export_capacity_curves(w):
    cells = ep.get("export.load_cell_means")([w.decomposition_path], population="incumbent")
    candidate_cells = ep.get("export.load_cell_means")([w.decomposition_path], population="candidate")
    build = ep.get("export.build_capacity_curves")
    pooled = ep.get("export.build_pooled_median_curves")
    oracle = ep.get("export.reference_fleet_additivity_oracle")
    validate = ep.get("export.validate_reference_fleet_oracle")
    curves = build(cells)
    ordering = w.fixed_ordering
    reversed_ordering = ordering.assign(fixed_order=ordering.groupby(["region_id", "archetype"]).fixed_order.transform("max") - ordering["fixed_order"])
    two_gcm = pd.concat([curves, curves.assign(gcm="IPSL-CM6A-LR", usable_capacity_mw=curves["usable_capacity_mw"] * 0.9)], ignore_index=True)
    candidate_oracle = oracle(candidate_cells, build(candidate_cells, population="candidate", placement_rule="candidate_fixed_order", fixed_ordering=ordering), population="candidate", placement_rule="candidate_fixed_order")
    broken = oracle(cells, curves.assign(usable_capacity_mw=curves["usable_capacity_mw"] + 1.0))
    return {
        "curves": curves,
        "curves_fixed_ordering": build(
            candidate_cells, population="candidate", placement_rule="candidate_fixed_order", fixed_ordering=ordering
        ),
        "curves_reversed_ordering": build(cells, fixed_ordering=reversed_ordering),
        "curves_empty": outcome(build, cells.iloc[0:0]),
        "curves_unrecognised_archetype": outcome(build, cells.assign(archetype="mystery_plant")),
        "curves_ordering_misses_cells": outcome(build, cells, fixed_ordering=ordering.iloc[:3]),
        "curves_ordering_duplicates": outcome(
            build, cells, fixed_ordering=pd.concat([ordering, ordering.iloc[:1]])
        ),
        "curves_ordering_missing_columns": outcome(
            build, cells, fixed_ordering=ordering.drop(columns=["fixed_order"])
        ),
        "pooled": pooled(curves),
        "pooled_two_gcms": pooled(two_gcm),
        "oracle": oracle(cells, curves),
        "oracle_candidate": candidate_oracle,
        "oracle_mismatch": broken,
        "oracle_other_placement_rule": oracle(cells, curves, placement_rule="not_present"),
        "validate_ok": outcome(validate, oracle(cells, curves)),
        "validate_fails": outcome(validate, broken),
        "validate_loose_tolerance": outcome(validate, broken, atol_mw=10.0),
    }  # fmt: skip


@case("export.reach_curves")
def export_reach_curves(w):
    candidate = w.samples["candidate"]
    incumbent = w.samples["incumbent"]
    ceilings, topology, ordering = w.water_limits, w.topology, w.fixed_ordering
    sample_curves = ep.get("export.build_reach_aware_candidate_sample_curves")
    average = ep.get("export.average_reach_aware_candidate_sample_curves")
    curves = ep.get("export.build_reach_aware_candidate_curves")
    attach = ep.get("export.attach_incumbent_water_demand")
    inputs = dict(capacity_grid_points=GRID, fixed_ordering=ordering)
    joint = sample_curves(candidate, ceilings, topology, existing_site_samples=incumbent, **inputs)
    virgin = sample_curves(candidate, ceilings, topology, **inputs)
    band_mean = ceilings.drop(columns=["year"]).groupby(
        [c for c in ceilings.columns if c not in {"year", "water_ceiling_mw"}], as_index=False
    )["water_ceiling_mw"].mean()
    potential_site_inputs = candidate.assign(
        water_ceiling_mw=1.0e9, downstream_cell_id=None
    )
    return {
        "joint": joint,
        "virgin": virgin,
        # The ordering built from the potential-site samples themselves, passed explicitly.
        "joint_default_ordering": sample_curves(
            candidate, ceilings, topology, existing_site_samples=incumbent, capacity_grid_points=GRID,
            fixed_ordering=ep.get("export.build_fixed_cell_ordering")(candidate),
        ),
        "joint_custom_labels": sample_curves(
            candidate, ceilings, topology, existing_site_samples=incumbent, population="trial",
            placement_rule="trial_rule", **inputs,
        ),
        "averaged_joint": average(joint),
        "averaged_virgin": average(virgin),
        "curves_joint": curves(candidate, ceilings, topology, existing_site_samples=incumbent, **inputs),
        "curves_virgin": curves(candidate, ceilings, topology, **inputs),
        "curves_band_mean_ceilings": outcome(curves, candidate, band_mean, topology, **inputs),
        "curves_one_grid_point": outcome(
            curves, candidate, ceilings, topology, capacity_grid_points=1, fixed_ordering=ordering
        ),
        "sample_curves_one_grid_point": outcome(
            sample_curves, candidate, ceilings, topology, capacity_grid_points=1, fixed_ordering=ordering
        ),
        "missing_ceilings": outcome(curves, candidate, ceilings.iloc[:-5], topology, **inputs),
        "missing_topology": outcome(curves, candidate, ceilings, topology.iloc[:-1], **inputs),
        "ordering_misses_cells": outcome(
            curves, candidate, ceilings, topology, capacity_grid_points=GRID, fixed_ordering=ordering.iloc[:5]
        ),
        "attach_demand": attach(potential_site_inputs, incumbent),
        "attach_demand_year_dropped": attach(potential_site_inputs, incumbent.drop(columns=["year"])),
        "attach_demand_missing_key": outcome(attach, potential_site_inputs, incumbent.drop(columns=["season"])),
        "attach_demand_no_overlap": attach(
            potential_site_inputs, incumbent.assign(cell_id="+000.000_+0000.000")
        ),
    }  # fmt: skip


@case("export.export_curves")
def export_export_curves(w):
    directory = _directory(w, "export_curves")
    export = ep.get("export.export_wet_ucr_curves")
    results = {}

    def products(label, *args, **kwargs):
        paths = outcome(export, *args, directory / label, **kwargs)
        if isinstance(paths, Raised):
            return paths
        return {
            "files": [path.name for path in paths],
            "per_gcm": read_back(paths[0]),
            "pooled": read_back(paths[1]),
            "oracle": read_back(paths[2]),
        }

    decomposition = [w.decomposition_path]
    results["incumbent"] = products("incumbent", decomposition)
    results["incumbent_fixed_ordering"] = products(
        "incumbent_fixed", decomposition, fixed_ordering_path=w.fixed_ordering_path
    )
    results["candidate_non_reach"] = products(
        "candidate_plain", decomposition, population="candidate",
        placement_rule="candidate_fixed_order", fixed_ordering_path=w.fixed_ordering_path,
    )  # fmt: skip
    results["unknown_population"] = products("unknown_population", decomposition, population="neither")
    return results


# --- RIME-X samples -----------------------------------------------------------------------------------------


@case("rime.samples")
def rime_samples(w):
    incumbent = w.samples["incumbent"]
    candidate = w.samples["candidate"]
    ceilings, topology, ordering = w.water_limits, w.topology, w.fixed_ordering
    endpoints = ep.get("rime.build_incumbent_sample_endpoints")
    build = ep.get("rime.build_wet_rime_samples")
    from_curves = ep.get("rime.build_wet_rime_samples_from_curves")
    curves = ep.get("export.build_reach_aware_candidate_sample_curves")(
        candidate, ceilings, topology, capacity_grid_points=GRID,
        fixed_ordering=ordering, existing_site_samples=incumbent,
    )  # fmt: skip
    samples = build(
        incumbent, candidate, ceilings, topology, capacity_grid_points=GRID, fixed_ordering=ordering
    )
    return {
        "endpoints": endpoints(incumbent),
        "endpoints_empty": outcome(endpoints, incumbent.iloc[0:0]),
        "samples": samples,
        "samples_equal_composition": bool(
            samples.equals(from_curves(incumbent, curves))
        ),
        "samples_depth_ratios": build(
            incumbent, candidate, ceilings, topology, new_build_ratios=(0.25, 0.75),
            capacity_grid_points=GRID, fixed_ordering=ordering,
        ),
        # The ordering built from the potential-site samples themselves, passed explicitly.
        "samples_default_ordering": build(
            incumbent, candidate, ceilings, topology, capacity_grid_points=GRID,
            fixed_ordering=ep.get("export.build_fixed_cell_ordering")(candidate),
        ),
        "from_curves": from_curves(incumbent, curves),
        "from_curves_depth_beyond_curve": outcome(from_curves, incumbent, curves, new_build_ratios=(500.0,)),
        "from_curves_no_ratios": outcome(from_curves, incumbent, curves, new_build_ratios=()),
        "from_curves_negative_ratio": outcome(from_curves, incumbent, curves, new_build_ratios=(1.0, -1.0)),
        "from_curves_missing_pairs": outcome(
            from_curves, incumbent, curves[curves["season"].ne("DJF") | curves["region_id"].ne("WEU")]
        ),
        "band_mean_ceilings": outcome(
            build, incumbent, candidate, ceilings.drop(columns=["year"]).groupby(
                [c for c in ceilings.columns if c not in {"year", "water_ceiling_mw"}], as_index=False
            )["water_ceiling_mw"].mean(), topology, capacity_grid_points=GRID, fixed_ordering=ordering,
        ),
    }  # fmt: skip


@case("rime.percentiles")
def rime_percentiles(w):
    samples = read_back(_rime_samples_path(w))
    percentiles = ep.get("rime.build_gwl_window_percentiles")
    to_dataset = ep.get("rime.percentiles_to_dataset")
    default = percentiles(samples)
    pooled = pd.concat(
        [samples, samples.assign(gcm="IPSL-CM6A-LR", candidate_usable_fraction=samples["candidate_usable_fraction"] * 0.9)],
        ignore_index=True,
    )
    return {
        "percentiles": default,
        "percentiles_custom_quantiles": percentiles(samples, quantiles=(0.0, 0.25, 1.0)),
        "percentiles_pooled_gcms": percentiles(pooled),
        "percentiles_bad_quantile": outcome(percentiles, samples, quantiles=(1.5,)),
        "percentiles_no_quantiles": outcome(percentiles, samples, quantiles=()),
        "percentiles_missing_column": outcome(percentiles, samples.drop(columns=["depth_ratio"])),
        "dataset": to_dataset(default),
        "dataset_pooled": to_dataset(percentiles(pooled)),
        "dataset_single_statistic": to_dataset(default[default["statistic"] == "p50"]),
    }


def _rime_samples_path(w):
    path = w.work / "rime_samples.parquet"
    if not path.exists():
        ep.get("rime.export_wet_rime_samples")(
            [w.decomposition_path], [w.audit_monthly_run], w.topology_files, path,
            capacity_grid_points=GRID, fixed_ordering_path=w.fixed_ordering_path,
        )  # fmt: skip
    return path


@case("rime.exports")
def rime_exports(w):
    directory = _directory(w, "rime_exports")
    samples_path = _rime_samples_path(w)
    dataset_path = ep.get("rime.export_wet_rime_dataset")([samples_path, samples_path], directory / "nested" / "surface.nc")
    import xarray as xr

    with xr.open_dataset(dataset_path) as stored:
        dataset = stored.load()
    return {
        "samples_file": read_back(samples_path),
        "dataset_file": dataset,
        "dataset_file_name": dataset_path.name,
        "no_ordering": outcome(
            ep.get("rime.export_wet_rime_samples"),
            [w.decomposition_path], [w.audit_monthly_run], w.topology_files, directory / "none.parquet",
            capacity_grid_points=GRID,
        ),
        "no_sample_files": outcome_type(ep.get("rime.export_wet_rime_dataset"), [], directory / "none.nc"),
    }  # fmt: skip


# --- how the library composes the wet-cooling outputs --------------------------------------------
#
# These tests run the library functions that compose the wet-cooling outputs.


@case("library.wet_candidate_products")
def library_wet_candidate_products(w):
    directory = _fresh(w, "library_products")
    build = ep.get("products.build_wet_candidate_products")
    write = ep.get("products.write_wet_candidate_products")
    ordering = pd.read_parquet(w.fixed_ordering_path)
    inputs = w.first_gcm_ssp585_inputs

    def run(target, decomposition, ceiling, topologies, order=ordering):
        products = build([decomposition], [ceiling], topologies, order, capacity_grid_points=GRID)
        write(products, target)
        return products

    products = run(directory / "all", w.decomposition_path, w.audit_monthly_run, w.topology_files)
    single = run(
        directory / "single", inputs["decomposition"], inputs["ceiling"], [inputs["topology"]]
    )
    elsewhere = pd.read_parquet(w.decomposition_path)
    elsewhere.loc[elsewhere["population"] == "incumbent", "cell_id"] = "+000.000_+0000.000"
    isolated = _write(elsewhere, directory / "no_overlap" / "GFDL-ESM4_wet_channel_decomposition.parquet")
    return {
        "per_gcm": read_back(directory / "all" / "wet_ucr_curves_per_gcm.parquet"),
        "pooled_median": read_back(directory / "all" / "wet_ucr_curves_pooled_median.parquet"),
        "additivity_oracle": read_back(directory / "all" / "wet_ucr_curve_additivity_oracle.parquet"),
        "rime_samples": read_back(directory / "all" / "wet_ucr_rime_samples.parquet"),
        "overlap": products.incumbent_overlap,
        "single_scenario_rime": read_back(directory / "single" / "wet_ucr_rime_samples.parquet"),
        "single_scenario_per_gcm": read_back(directory / "single" / "wet_ucr_curves_per_gcm.parquet"),
        "single_scenario_overlap": single.incumbent_overlap,
        "no_overlap": outcome(run, directory / "none", isolated, w.audit_monthly_run, w.topology_files),
        "missing_topology_scenario": outcome(
            run, directory / "bad", w.decomposition_path, w.audit_monthly_run, w.topology_files[:1]
        ),
    }  # fmt: skip


@case("library.residual_water_ordering")
def library_residual_water_ordering(w):
    directory = _fresh(w, "library_ordering")
    first = w.first_gcm_ssp585_inputs
    second = w.derived_gcm_inputs()
    build = ep.get("products.build_residual_water_ordering")
    write = ep.get("products.write_residual_water_ordering")
    decompositions = [first["decomposition"], second["decomposition"]]
    ceilings = [first["ceiling"], second["ceiling"]]
    topologies = [first["topology"], second["topology"]]

    def run(path, decompositions=decompositions, ceilings=ceilings, topologies=topologies, baseline=None):
        result = build(decompositions, ceilings, topologies, baseline_ordering=baseline)
        write(result, directory / path)
        return result

    pooled = run("pooled.parquet")
    run("single.parquet", decompositions[:1], ceilings[:1], topologies[:1])
    baseline_frame = pooled.ordering.assign(
        fixed_order=lambda frame: frame.groupby(["region_id", "archetype"]).fixed_order.transform("max")
        - frame["fixed_order"]
    )
    with_baseline = run("with_baseline.parquet", baseline=baseline_frame)
    return {
        "pooled": read_back(directory / "pooled.parquet"),
        "pooled_charged_share": pooled.charged_share,
        "single_gcm": read_back(directory / "single.parquet"),
        "with_baseline": read_back(directory / "with_baseline.parquet"),
        "with_baseline_moves": with_baseline.moves,
        "no_op_against_identical_baseline": outcome(run, "noop.parquet", baseline=pooled.ordering),
        "baseline_covers_other_cells": outcome(
            run, "other.parquet", baseline=pd.read_parquet(directory / "pooled.parquet").iloc[:-2]
        ),
        "mismatched_gcm_and_topology": outcome(
            run, "mismatch.parquet", decompositions[:1], ceilings[:1], topologies[1:]
        ),
        "full_domain_decomposition": outcome(
            run, "two_scenarios.parquet", [w.decomposition_path], [w.audit_monthly_run], [first["topology"]]
        ),
    }  # fmt: skip


@case("library.ordering_helpers")
def library_ordering_helpers(w):
    pool = ep.get("products.pool")
    moves = ep.get("products.moves")
    overlap = ep.get("products.incumbent_candidate_overlap")
    keys = ["region_id", "archetype", "cell_id"]

    def part(rows):
        frame = pd.DataFrame(rows, columns=[*keys, "nameplate_mw", "usable_capacity_mw"])
        return frame.astype({key: "category" for key in keys})

    first = part([("AFR", "a", "c2", 100.0, 80.0), ("AFR", "a", "c1", 100.0, 90.0)])
    second = part([("WEU", "b", "c0", 50.0, 25.0), ("AFR", "a", "c3", 100.0, 70.0)])
    ordering = ep.get("export.build_fixed_cell_ordering")(pool([first, second]).astype({k: str for k in keys}))
    baseline = ordering.assign(fixed_order=ordering["fixed_order"][::-1].to_numpy())
    pooled = pool([first, second])
    incumbent, candidate = w.samples["incumbent"], w.samples["candidate"]
    return {
        "pooled": pooled,
        "pooled_category_order": [list(pooled[key].cat.categories) for key in keys],
        "pooled_single": pool([first]),
        "ordering": ordering,
        "moves_all_moved": moves(ordering, baseline),
        "moves_none": moves(ordering, ordering),
        "moves_different_cells": outcome(moves, ordering, baseline.iloc[:-1]),
        "pool_without_parts": outcome_type(pool, []),
        "overlap": overlap(incumbent, candidate),
        "overlap_no_candidates": overlap(incumbent, candidate.iloc[0:0]),
        "overlap_no_incumbents": overlap(incumbent.iloc[0:0], candidate),
        "overlap_partial": overlap(incumbent, candidate[candidate["cell_id"] != incumbent["cell_id"].iloc[0]]),
    }


# --- finalize: pooling the five GCM tasks of both calculations ----------------------------------------
#
# The library runs on a five-task layout and on defective copies of it. ``mutations``
# edits a copy in place; the library must refuse each defective copy.


def _fresh(w, name: str):
    directory = w.work / name
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    return directory


def _rewrite(path, edit):
    frame = pd.read_parquet(path)
    edit(frame).to_parquet(path, index=False)


def _finalize_mutations():
    curves = "wet_ucr_curves_per_gcm.parquet"
    rime = "wet_ucr_rime_samples.parquet"

    def drop_task(roots):
        shutil.rmtree(roots["gwl"] / "4")

    def extra_task(roots):
        shutil.copytree(roots["reference"] / "0", roots["reference"] / "5")

    def drop_rime_file(roots):
        (roots["gwl"] / "3" / rime).unlink()

    def membership_varies_with_season(roots):
        def edit(frame):
            hit = (frame["season"] == "JJA") & (frame["point_index"] == 3)
            frame.loc[hit, "marginal_cell_id"] = "elsewhere"
            return frame

        _rewrite(roots["gwl"] / "0" / curves, edit)

    def reference_membership_varies(roots):
        def edit(frame):
            hit = frame["point_index"] == 4
            frame.loc[hit & (frame.index % 2 == 0), "marginal_cell_id"] = "elsewhere"
            return frame

        _rewrite(roots["reference"] / "1" / curves, edit)

    def reference_misses_points(roots):
        def edit(frame):
            hit = (frame["region_id"] == fixtures.FINALIZE_REGIONS[2]) & (frame["point_index"] == 5)
            return frame[~hit]

        _rewrite(roots["reference"] / "2" / curves, edit)

    def arms_use_other_cells(roots):
        def edit(frame):
            hit = (frame["archetype"] == "gas_once_through") & (frame["point_index"] == 2)
            frame.loc[hit, "marginal_cell_id"] = "swapped"
            return frame

        _rewrite(roots["reference"] / "4" / curves, edit)

    def rime_loses_a_region(roots):
        _rewrite(
            roots["gwl"] / "0" / rime,
            lambda frame: frame[frame["region_id"] != fixtures.FINALIZE_REGIONS[0]],
        )
        for task in range(1, 5):
            _rewrite(
                roots["gwl"] / str(task) / rime,
                lambda frame: frame[frame["region_id"] != fixtures.FINALIZE_REGIONS[0]],
            )

    def rime_has_a_hole(roots):
        def edit(frame):
            hit = (
                (frame["region_id"] == fixtures.FINALIZE_REGIONS[3])
                & (frame["season"] == "SON")
                & (frame["warming_level"] == 3.0)
                & (frame["depth_ratio"] == 2.0)
                & (frame["plant_type"] == "gas")
                & (frame["cooling_type"] == "once_through")
            )
            return frame[~hit]

        for task in range(5):
            _rewrite(roots["gwl"] / str(task) / rime, edit)

    return {
        "four_gwl_tasks": drop_task,
        "six_reference_tasks": extra_task,
        "four_rime_files": drop_rime_file,
        "membership_varies_with_season": membership_varies_with_season,
        "reference_membership_varies": reference_membership_varies,
        "reference_misses_points": reference_misses_points,
        "arms_use_other_cells": arms_use_other_cells,
        "rime_loses_a_region": rime_loses_a_region,
        "rime_has_a_hole": rime_has_a_hole,
    }


def _finalize_variant(w, prefix: str, name: str, mutate):
    """A fresh copy of the task layout with ``mutate`` applied; returns its two root directories."""

    base = _fresh(w, f"{prefix}_{name}")
    layout = w.finalize_task_layout
    roots = {
        "gwl": base / "tasks_gwl",
        "reference": base / "tasks_reference",
    }
    shutil.copytree(layout["gwl"], roots["gwl"])
    shutil.copytree(layout["reference"], roots["reference"])
    mutate(roots)
    return roots


def _read_product_tree(root):
    with xr.open_dataset(root / "gwl" / "r12_thermoelectric_wet_gwl.nc") as dataset:
        rime = dataset.load()
    text = (root / "validation.json").read_text()
    return {
        "gwl_per_gcm": read_back(root / "gwl" / "wet_ucr_curves_per_gcm.parquet"),
        "gwl_pooled_median": read_back(root / "gwl" / "wet_ucr_curves_pooled_median.parquet"),
        "reference_per_gcm": read_back(root / "reference_pd" / "wet_ucr_curves_per_gcm.parquet"),
        "reference_pooled_median": read_back(
            root / "reference_pd" / "wet_ucr_curves_pooled_median.parquet"
        ),
        "rime_dataset": rime,
        "validation": json.loads(text),
        "validation_text": text,
    }




@case("library.finalize_wet_products")
def library_finalize_wet_products(w):
    layout = w.finalize_task_layout
    finalize = ep.get("products.finalize_wet_candidate_products")
    directory = _fresh(w, "library_finalize_ok")
    result = finalize(layout["gwl"], layout["reference"], directory, gcm_count=5)

    def refuse(name, mutate):
        roots = _finalize_variant(w, "library_finalize", name, mutate)
        out = _fresh(w, f"library_finalize_{name}_out")
        refused = outcome(finalize, roots["gwl"], roots["reference"], out, gcm_count=5)
        if isinstance(refused, Raised):
            assert not any(out.iterdir()), "a refused finalize wrote files"
        return refused

    return {
        "validation_returned": result.validation,
        **_read_product_tree(directory),
        **{name: refuse(name, mutate) for name, mutate in _finalize_mutations().items()},
        "other_task_count_and_dimensions": _finalize_other_options(w),
    }


def _finalize_other_options(w):
    """Options beyond the defaults: a different task count and no dimension check."""

    finalize = ep.get("products.finalize_wet_candidate_products")

    def drop_last_task_of_both_arms(roots):
        shutil.rmtree(roots["gwl"] / "4")
        shutil.rmtree(roots["reference"] / "4")

    roots = _finalize_variant(w, "library_finalize", "options", drop_last_task_of_both_arms)
    four_tasks = finalize(
        roots["gwl"], roots["reference"], _fresh(w, "library_finalize_options_out"),
        gcm_count=4, expected_rime_dimensions=None,
    )  # fmt: skip
    return four_tasks.validation


# --- river network -----------------------------------------------------------------------------------


def _topology_inputs(w):
    """Files of the split wet capacity loss used by the river-network tests (their paths appear in errors)."""

    directory = w.work / "topology_inputs"
    directory.mkdir(exist_ok=True)
    frame = pd.read_parquet(w.decomposition_path)
    only_incumbent = _write(
        frame[frame["population"] == "incumbent"], directory / "only_incumbent.parquet"
    )
    return {
        "decomposition": w.decomposition_path,
        "single_scenario": w.first_gcm_ssp585_inputs["decomposition"],
        "only_incumbent": only_incumbent,
    }


def _topology_names(directory):
    return sorted(path.name for path in directory.glob("*.parquet"))


def _topology_tables(directory):
    return {path.name: read_back(path) for path in sorted(directory.glob("*.parquet"))}




@case("library.reach_topology")
def library_reach_topology(w):
    inputs = _topology_inputs(w)
    flowdirs = w.flow_direction_files
    read = ep.get("topology.read_candidate_cells")
    build = ep.get("topology.build_reach_topologies")
    write = ep.get("topology.write_reach_topologies")
    cwd = _fresh(w, "library_topology")

    def run(decomposition, flowdir, gcm, target):
        cells = read(decomposition)
        topologies = build(cells, ep.get("topology.read_flow_directions")(flowdir))
        return write(topologies, gcm, target)

    both = cwd / "nested" / "both"
    written = run(inputs["decomposition"], flowdirs["acyclic"], "GFDL-ESM4", both)
    single = cwd / "single"
    run(inputs["single_scenario"], flowdirs["acyclic"], "GFDL-ESM4", single)
    run(inputs["single_scenario"], flowdirs["acyclic"], "IPSL-CM6A-LR", cwd / "other")
    return {
        "returned_names": [path.name for path in written],
        "file_names": _topology_names(both),
        "tables": _topology_tables(both),
        "single_file_names": _topology_names(single),
        "single_tables": _topology_tables(single),
        "other_gcm_file_names": _topology_names(cwd / "other"),
        "loop_in_flow_directions": outcome(
            run, inputs["decomposition"], flowdirs["cyclic"], "GFDL-ESM4", cwd / "loop"
        ),
        "no_candidate_rows": outcome(
            run, inputs["only_incumbent"], flowdirs["acyclic"], "GFDL-ESM4", cwd / "none"
        ),
        "loop_wrote": _topology_names(cwd / "loop"),
    }  # fmt: skip


def _assign_downstream_inputs():
    """Small rasters and cell tables that reach the corners of ``assign_downstream``."""

    def raster(codes, lats, lons, descending=False):
        array = np.asarray(codes, dtype=float)
        lats = np.asarray(lats, dtype=float)
        if descending:
            array, lats = array[::-1], lats[::-1]
        return xr.DataArray(array, dims=("lat", "lon"), coords={"lat": lats, "lon": lons})

    def cells(rows):
        frame = pd.DataFrame(rows, columns=["cell_id", "grid_lat", "grid_lon"])
        return frame.assign(population="candidate")

    lons = [0.25, 0.75, 1.25]
    lats = [0.25, 0.75, 1.25]
    all_cells = cells([(f"c{i}{j}", lat, lon) for i, lat in enumerate(lats) for j, lon in enumerate(lons)])
    return {
        "east_wraps_to_first_column": (
            all_cells.iloc[[3, 5]], raster([[3, 3, 3], [1, 3, 1], [3, 3, 3]], lats, lons)
        ),
        "west_wraps_to_last_column": (
            all_cells.iloc[[3, 5]], raster([[3, 3, 3], [5, 3, 5], [3, 3, 3]], lats, lons)
        ),
        "all_eight_directions": (
            all_cells.iloc[[4, 0, 1, 2, 3, 5, 6, 7, 8]],
            raster([[0, 0, 0], [1, 2, 3], [4, 5, 6]], lats, lons),
        ),
        "north_and_north_east": (
            all_cells.iloc[[0, 3]], raster([[7, 3, 3], [8, 3, 3], [3, 3, 3]], lats, lons)
        ),
        "leaves_north_edge": (all_cells.iloc[[6]], raster([[3] * 3, [3] * 3, [7] * 3], lats, lons)),
        "latitude_stored_descending": (
            all_cells.iloc[[0, 6]], raster([[7, 3, 3], [3, 3, 3], [3, 3, 3]], lats, lons, descending=True)
        ),
        "off_grid_cells_snap_to_nearest": (
            cells([("a", 0.30, 0.20), ("b", 1.20, 0.80)]),
            raster([[3, 3, 3], [3, 3, 3], [7, 7, 7]], lats, lons),
        ),
        "tie_goes_to_lower_coordinate": (
            cells([("a", 0.5, 0.25), ("b", 0.25, 0.25)]), raster([[3] * 3] * 3, lats, lons)
        ),
        "tie_in_longitude_goes_to_lower_coordinate": (
            cells([("a", 1.25, 0.5), ("b", 1.25, 0.25)]), raster([[3] * 3] * 3, lats, lons)
        ),
        "tie_not_colliding": (
            cells([("a", 0.5, 0.25), ("b", 0.25, 0.75)]), raster([[1, 3, 3], [3, 3, 3], [3, 3, 3]], lats, lons)
        ),
        "outlet_codes_and_missing": (
            all_cells.iloc[[0, 1, 2]], raster([[0, 9, np.nan], [3, 3, 3], [3, 3, 3]], lats, lons)
        ),
        "single_latitude": (
            cells([("a", 0.25, 0.25), ("b", 0.25, 0.75)]), raster([[1, 0, 3]], [0.25], lons)
        ),
        "repeated_cell_rows": (
            pd.concat([all_cells.iloc[[0, 3]], all_cells.iloc[[0, 3]]], ignore_index=True),
            raster([[3, 3, 3], [7, 3, 3], [3, 3, 3]], lats, lons),
        ),
        "loop_between_empty_cells": (
            all_cells.iloc[[0]], raster([[1, 1, 5], [3, 3, 3], [3, 3, 3]], lats, lons)
        ),
        "two_cells_one_raster_cell": (
            cells([("a", 0.25, 0.25), ("b", 0.26, 0.24)]), raster([[3] * 3] * 3, lats, lons)
        ),
        "same_id_two_coordinates": (
            cells([("a", 0.25, 0.25), ("a", 0.75, 0.25)]), raster([[3] * 3] * 3, lats, lons)
        ),
        "missing_coordinate_column": (
            all_cells.iloc[[0]].drop(columns="grid_lon"), raster([[3] * 3] * 3, lats, lons)
        ),
        "unknown_code": (all_cells.iloc[[0]], raster([[10, 3, 3], [3, 3, 3], [3, 3, 3]], lats, lons)),
    }  # fmt: skip




@case("library.assign_downstream")
def library_assign_downstream(w):
    assign = ep.get("topology.assign_downstream")
    return {name: outcome(assign, cells, flow) for name, (cells, flow) in _assign_downstream_inputs().items()}


# --- splitting the wet capacity loss ------------------------------------------------------------------------


def _decomposition_inputs(w):
    directory = w.work / "decomposition_inputs"
    directory.mkdir(exist_ok=True)
    monthly = pd.read_parquet(w.audit_monthly_run)
    without_january = _write(
        pd.read_parquet(w.audit_baseline_run).query("month != 1 or month.isna()", engine="python"),
        directory / "baseline_without_january.parquet",
    )
    return {
        "baseline": w.audit_baseline_run,
        "census": w.audit_monthly_run,
        "summary_census": w.audit_summary_run["long"],
        "without_january": without_january,
        "months": int(monthly["month"].nunique()),
    }




@case("library.wet_channel_decomposition")
def library_wet_channel_decomposition(w):
    inputs = _decomposition_inputs(w)
    directory = _fresh(w, "library_decomposition")
    decompose = ep.get("decomposition.decompose_wet_channels")
    default_label = ep.get("decomposition.DEFAULT_BASELINE_REFERENCE")

    def run(name, baselines, censuses, reference=default_label):
        return decompose(
            baselines, censuses, directory / f"{name}.parquet", reference=reference
        )

    single = run("single", [inputs["baseline"]], [inputs["census"]])
    two = run("two", [inputs["baseline"], inputs["census"]], [inputs["census"], inputs["baseline"]])
    labelled = run("label", [inputs["baseline"]], [inputs["census"]], reference="custom")
    return {
        "single_output": read_back(single.output),
        "single_reference": single.low_gwl_reference,
        "two_inputs_each_output": read_back(two.output),
        "two_inputs_each_reference_rows": len(two.low_gwl_reference),
        "custom_label_output": read_back(labelled.output),
        "summary_census_as_reference": outcome(
            lambda: read_back(run("summary_reference", [inputs["summary_census"]], [inputs["census"]]).output)
        ),
        "reference_without_a_month": outcome(
            lambda: read_back(run("thin", [inputs["without_january"]], [inputs["census"]]).output)
        ),
        "summary_census_to_decompose": outcome(
            lambda: read_back(run("summary_census", [inputs["baseline"]], [inputs["summary_census"]]).output)
        ),
    }


# --- site evaluation ---------------------------------------------------------------------------------------------


def _census_inputs(w):
    """The cells the audit builds from its settings, as the library function builds them."""

    wet_file, dry_file = w.fleet_files
    cells = ep.get("audit.build_census_cells")(
        w.isimip_paths,
        fixtures.GCM,
        list(fixtures.SCENARIOS),
        ep.get("universe.RegionRasterPaths")(**w.region_paths),
        fleet_wet=wet_file,
        fleet_dry=dry_file,
    )
    return {"wet_cells": cells.wet_cells, "dry_cells": cells.dry_cells}


@case("library.candidate_census")
def library_candidate_census(w):
    cells = _census_inputs(w)
    directory = _fresh(w, "library_census")
    evaluate = ep.get("audit.evaluate_candidate_census")
    write = ep.get("audit.write_candidate_census")
    gwl = ep.get("audit.GWLSampling")
    scenarios = list(fixtures.SCENARIOS)

    def census(name, sampling, **options):
        years = ep.get("audit.sample_years_within_gwl")(w.gmt_root, fixtures.GCM, scenarios, sampling)
        windows = evaluate(paths=w.isimip_paths, gcm=fixtures.GCM, sampled_years=years, **cells, **options)
        return read_back(write(windows, directory / f"{name}.parquet"))

    shared = gwl(bin_width=2.0, min_level=1.0, max_level=3.5, years_per_bin=2, seed=730)
    baseline = gwl(target_levels=(1.0,), bin_width=0.5, years_per_bin=2, seed=11)
    command_line = gwl(target_levels=(1.0,), bin_width=0.5, years_per_bin=1, seed=4)
    monthly = w._MONTHLY
    return {
        "monthly": census("monthly", shared, responses=monthly),
        "baseline": census("baseline", baseline, responses=monthly),
        "default_responses": census("default", shared),
        "command_line_equivalent": census("command_line", command_line, responses=monthly),
        "no_years": outcome(
            lambda: evaluate(paths=w.isimip_paths, gcm=fixtures.GCM, sampled_years=[], **cells)
        ),
    }  # fmt: skip
