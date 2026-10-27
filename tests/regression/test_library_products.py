"""Library checks on the cells of the site evaluation, the ``REF-PD`` labels and the finalize guard.

Each test runs library functions on the made-up input data and checks a property
that the stored reference outputs do not state: cache round trips, agreement of
the cell recipe with the cells a site evaluation uses, the ``REF-PD`` label
strings, refusal to overwrite.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

from impacts.thermoelectric.siting.cells import default_wet_synthetic_plants

from . import fixtures


def test_wet_only_census_is_the_wet_rows_of_the_full_census(world):
    from . import cases_exports
    from . import entrypoints as ep

    cells = cases_exports._census_inputs(world)
    years = ep.get("audit.sample_years_within_gwl")(
        world.gmt_root,
        fixtures.GCM,
        list(fixtures.SCENARIOS),
        ep.get("audit.GWLSampling")(bin_width=2.0, min_level=1.0, max_level=3.5, years_per_bin=2),
    )
    run = ep.get("audit.evaluate_candidate_census")
    full = pd.concat(
        run(paths=world.isimip_paths, gcm=fixtures.GCM, sampled_years=years, **cells),
        ignore_index=True,
    )
    wet_only = pd.concat(
        run(
            paths=world.isimip_paths,
            gcm=fixtures.GCM,
            sampled_years=years,
            wet_cells=cells["wet_cells"],
        ),
        ignore_index=True,
    )
    wet_synthetic_plants = {archetype.name for archetype in default_wet_synthetic_plants()}
    expected = full[full["archetype"].isin(wet_synthetic_plants)].reset_index(drop=True)
    assert len(wet_only) == len(expected) > 0
    pd.testing.assert_frame_equal(wet_only, expected)


# --- the cells of the site evaluation ---------------------------------------------------------------------


def _assert_same_cells(actual, expected):
    for kernel in ("wet_cells", "dry_cells"):
        assert list(actual[kernel]) == list(expected[kernel]) == ["candidate", "incumbent"]
        for population in ("candidate", "incumbent"):
            pd.testing.assert_frame_equal(actual[kernel][population], expected[kernel][population])


def test_build_census_cells_reads_and_writes_the_universe_caches(world, tmp_path):
    from . import entrypoints as ep

    region = ep.get("universe.RegionRasterPaths")(**world.region_paths)
    wet_file, dry_file = world.fleet_files
    arguments = (world.isimip_paths, fixtures.GCM, list(fixtures.SCENARIOS), region)
    caches = {
        "dry_cache": tmp_path / "cache" / f"dry_{fixtures.GCM}.parquet",
        "wet_cache": str(tmp_path / "cache" / f"wet_{fixtures.GCM}.parquet"),
    }
    build = ep.get("audit.build_census_cells")
    first = build(*arguments, fleet_wet=wet_file, fleet_dry=dry_file, **caches)
    assert sorted(path.name for path in (tmp_path / "cache").iterdir()) == [
        f"dry_{fixtures.GCM}.parquet",
        f"wet_{fixtures.GCM}.parquet",
    ]
    again = build(*arguments, fleet_wet=wet_file, fleet_dry=dry_file, **caches)
    uncached = build(*arguments, fleet_wet=wet_file, fleet_dry=dry_file)
    for result in (again, uncached):
        _assert_same_cells(
            {"wet_cells": first.wet_cells, "dry_cells": first.dry_cells},
            {"wet_cells": result.wet_cells, "dry_cells": result.dry_cells},
        )


def test_build_census_cells_refuses_no_scenarios(world):
    from . import entrypoints as ep

    region = ep.get("universe.RegionRasterPaths")(**world.region_paths)
    wet_file, dry_file = world.fleet_files
    with pytest.raises(ValueError, match="scenarios"):
        ep.get("audit.build_census_cells")(
            world.isimip_paths, fixtures.GCM, [], region, fleet_wet=wet_file, fleet_dry=dry_file
        )


def _cells_the_census_evaluates(world, monkeypatch, config):
    """Write the site evaluation the settings describe and return the four cell tables it hands the kernels."""

    from . import entrypoints as ep

    module = importlib.import_module(ep.TABLE["audit.run_census"][0])
    original = module.build_synthetic_sites
    seen = []

    def spy(cells, archetypes):
        seen.append(cells.copy())
        return original(cells, archetypes)

    monkeypatch.setattr(module, "build_synthetic_sites", spy)
    with world.short_reference_window():
        ep.get("audit.run")(config)
    wet_candidate, wet_incumbent, dry_candidate, dry_incumbent = seen
    return {
        "wet_cells": {"candidate": wet_candidate, "incumbent": wet_incumbent},
        "dry_cells": {"candidate": dry_candidate, "incumbent": dry_incumbent},
    }


def _cells_from_the_pieces(world, scenarios, cells_considered_scenarios):
    """The recipe spelled out with the lower-level calls, independent of ``build_site_cells``."""

    from . import entrypoints as ep

    region = ep.get("universe.RegionRasterPaths")(**world.region_paths)
    build = ep.get("universe.build_valid_cell_universe")
    extract = ep.get("universe.extract_fleet_cells")
    dry_universe = build(
        world.isimip_paths, fixtures.GCM, scenarios[0], region, require_hydrology=False
    )
    wet_universe = build(
        world.isimip_paths, fixtures.GCM, cells_considered_scenarios, region, require_hydrology=True
    )
    wet_file, dry_file = world.fleet_files
    return {
        "wet_cells": {
            "candidate": wet_universe.assign(population="candidate"),
            "incumbent": extract(wet_file, dry_file, region, restrict_to=wet_universe).assign(
                population="incumbent"
            ),
        },
        "dry_cells": {
            "candidate": dry_universe.assign(population="candidate"),
            "incumbent": extract(wet_file, dry_file, region, restrict_to=dry_universe).assign(
                population="incumbent"
            ),
        },
    }


@pytest.mark.parametrize("arm", ["warming_levels", "reference_pd"])
def test_build_census_cells_are_the_cells_the_census_evaluates(world, monkeypatch, arm):
    from . import entrypoints as ep

    region = ep.get("universe.RegionRasterPaths")(**world.region_paths)
    wet_file, dry_file = world.fleet_files
    if arm == "warming_levels":
        scenarios = list(fixtures.SCENARIOS)
        cells_considered_scenarios = None
        config = world.audit_config(
            "cells_warming_levels",
            summary={"skip": True},
            gwl_sampling={"target_levels": [1.0], "bin_width": 0.5, "years_per_bin": 1, "seed": 4},
        )
    else:
        # The reference run lists one scenario and a wider set of wet cells: the dry cells
        # follow the first of the run's scenarios, the wet cells the run's own list of
        # scenarios for the cells considered.
        scenarios = [fixtures.HISTORICAL]
        cells_considered_scenarios = [*fixtures.SCENARIOS, fixtures.HISTORICAL]
        config = world.reference_pd_config("cells_reference_pd")

    evaluated = _cells_the_census_evaluates(world, monkeypatch, config)
    built = ep.get("audit.build_census_cells")(
        world.isimip_paths, fixtures.GCM, scenarios, region,
        fleet_wet=wet_file, fleet_dry=dry_file, cells_considered_scenarios=cells_considered_scenarios,
    )  # fmt: skip
    assert isinstance(built, ep.get("audit.CensusCells"))
    library = {"wet_cells": built.wet_cells, "dry_cells": built.dry_cells}
    pieces = _cells_from_the_pieces(world, scenarios, cells_considered_scenarios or scenarios)

    _assert_same_cells(library, evaluated)
    _assert_same_cells(library, pieces)
    assert len(library["wet_cells"]["candidate"]) > 0
    assert len(library["dry_cells"]["candidate"]) > len(library["wet_cells"]["candidate"])


# --- the present-day reference calculation ------------------------------------------------------------------------------


def test_reference_pd_labels_are_the_documented_strings():
    from . import entrypoints as ep

    # The strings the output files carry; the constants must keep these values.
    assert (
        ep.get("decomposition.REF_PD_BASELINE_REFERENCE")
        == "max_thermal_conductance_over_ref_pd_1995_2014"
    )
    assert (
        ep.get("decomposition.DEFAULT_BASELINE_REFERENCE")
        == "max_thermal_conductance_over_low_gwl"
    )


def test_decomposing_needs_a_baseline_label(world, tmp_path):
    from . import entrypoints as ep

    census = world.audit_reference_pd_run
    with pytest.raises(TypeError, match="reference"):
        ep.get("decomposition.decompose_wet_channels")([census], [census], tmp_path / "out.parquet")


def test_reference_pd_decomposition_carries_the_label_and_the_string_warming_level(world, tmp_path):
    from . import entrypoints as ep

    census = world.audit_reference_pd_run
    decompose = ep.get("decomposition.decompose_wet_channels")
    labelled = decompose(
        [census], [census], tmp_path / "labelled.parquet",
        reference=ep.get("decomposition.REF_PD_BASELINE_REFERENCE"),
    )  # fmt: skip
    other = decompose(
        [census], [census], tmp_path / "other.parquet",
        reference=ep.get("decomposition.DEFAULT_BASELINE_REFERENCE"),
    )  # fmt: skip
    frame = pd.read_parquet(labelled.output)
    assert set(frame["warming_level"]) == {"REF-PD"}
    assert set(frame["scenario"]) == {fixtures.HISTORICAL}
    assert set(frame["baseline_reference"]) == {ep.get("decomposition.REF_PD_BASELINE_REFERENCE")}
    # The label is only a record: the wrong one changes nothing else, which is why
    # ``split_wet_losses`` makes the caller choose it.
    wrong = pd.read_parquet(other.output)
    assert set(wrong["baseline_reference"]) == {ep.get("decomposition.DEFAULT_BASELINE_REFERENCE")}
    pd.testing.assert_frame_equal(
        frame.drop(columns="baseline_reference"), wrong.drop(columns="baseline_reference")
    )
    # The site evaluation is its own reference, so the reference is built from it alone.
    assert len(labelled.low_gwl_reference) > 0


def test_finalize_refuses_to_replace_an_earlier_product_tree(world, tmp_path):
    from . import entrypoints as ep

    layout = world.finalize_task_layout
    finalize = ep.get("products.finalize_wet_candidate_products")
    first = finalize(layout["gwl"], layout["reference"], tmp_path, gcm_count=5)
    written = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    assert len(written) == 6
    before = {path: path.read_bytes() for path in written}

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        finalize(layout["gwl"], layout["reference"], tmp_path, gcm_count=5)
    assert {path: path.read_bytes() for path in written} == before

    again = finalize(layout["gwl"], layout["reference"], tmp_path, gcm_count=5, overwrite=True)
    assert again.validation == first.validation
