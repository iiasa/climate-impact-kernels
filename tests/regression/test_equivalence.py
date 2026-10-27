"""Run every registered end-to-end test and compare its result with the stored reference output.

Each test runs through ``entrypoints.py``. See ``README.md`` for the comparison
modes and the command that updates the stored outputs.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from . import (  # noqa: F401  (importing registers the tests)
    cases_audit,
    cases_buildings,
    cases_climatecore,
    cases_exports,
    cases_pipeline,
    cases_reference_pd,
    cases_surfaces,
    cases_thermo,
)
from .canon import compare_case, encode_case, load_golden, save_golden
from .registry import CASES

GOLDEN = Path(__file__).parent / "golden"
GRAPH_SIZES = GOLDEN / "dask_graph_sizes.json"
REGENERATE = os.environ.get("IMPACTS_REGRESSION_REGENERATE") == "1"
TIER_OVERRIDE = os.environ.get("IMPACTS_REGRESSION_TIER")
GRAPH_FACTOR = 2.0


def _tier(case) -> str:
    return TIER_OVERRIDE or case.tier


@pytest.mark.parametrize("name", sorted(CASES))
def test_case_matches_golden(name, world):
    case = CASES[name]
    if REGENERATE and case.frozen:
        return
    actual = encode_case(case.fn(world), world.scrub)
    directory = GOLDEN / name
    if REGENERATE:
        again = encode_case(case.fn(world), world.scrub)
        compare_case(again, actual, "exact", name)  # the test must give the same result twice
        save_golden(directory, *actual)
        return
    assert directory.is_dir(), f"no stored reference output for {name}; update the stored outputs (see README.md)"
    compare_case(actual, load_golden(directory), _tier(case), name)


def test_goldens_and_cases_agree():
    if REGENERATE:
        stale = {p.name for p in GOLDEN.iterdir() if p.is_dir()} - set(CASES)
        for name in stale:
            shutil.rmtree(GOLDEN / name)
        return
    stored = {p.name for p in GOLDEN.iterdir() if p.is_dir()}
    assert stored == set(CASES)


def test_chilled_stays_dask_backed_and_graph_is_bounded(world):
    sizes, lazy = cases_buildings.chilled_graph_report(world)
    assert all(lazy.values()), f"CHILLED output is not dask-backed: {lazy}"
    if REGENERATE:
        GRAPH_SIZES.write_text(json.dumps(sizes, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return
    expected = json.loads(GRAPH_SIZES.read_text(encoding="utf-8"))
    assert set(sizes) == set(expected)
    for key, size in sizes.items():
        assert expected[key] / GRAPH_FACTOR <= size <= expected[key] * GRAPH_FACTOR, (
            f"{key}: graph has {size} tasks, stored {expected[key]}"
        )
