"""The wet-cooling output builders share river water between existing and potential sites.

Capacity curves for potential sites are built together with the samples of the
existing sites, so the joint water correction changes the outputs. A
residual-water ordering (cells ranked on the water left after existing plants
upstream take theirs) ranks on every GCM and differs from the full-flow
ordering. Expected values are derived by hand from the river network arithmetic.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from impacts.thermoelectric.siting.outputs import (
    ADDITIVITY_ORACLE_FILE,
    DEFAULT_RIME_DIMENSIONS,
    PER_GCM_CURVES_FILE,
    POOLED_MEDIAN_CURVES_FILE,
    RIME_SAMPLES_FILE,
    build_residual_water_ordering,
    build_wet_siting_outputs,
    existing_potential_site_overlap,
    pool_wet_siting_outputs,
    read_gcm_outputs,
    write_pooled_wet_outputs,
    write_residual_water_ordering,
    write_wet_siting_outputs,
)

ARCHETYPE = "li_coal_500mw_once_through"
CEILING_KEYS = [
    "population", "region_id", "country", "cell_id", "grid_lat", "grid_lon",
    "archetype", "gcm", "scenario", "year", "season", "warming_level",
]  # fmt: skip


# --- outputs for potential sites -------------------------------------------------------------------------


def _sample(population: str, cell_id: str, lon: float, nameplate_mw: float) -> dict:
    return {
        "region_id": "R12_MEA",
        "country": "AAA",
        "grid_lat": 0.25,
        "archetype": ARCHETYPE,
        "gcm": "GFDL-ESM4",
        "scenario": "ssp585",
        "year": 2050,
        "season": "JJA",
        "warming_level": 3.0,
        "thermal_loss_mw": 0.0,
        "withdrawal_loss_mw": 0.0,
        "population": population,
        "cell_id": cell_id,
        "grid_lon": lon,
        "nameplate_mw": nameplate_mw,
        "usable_capacity_mw": nameplate_mw,
    }


def _candidate_inputs(root: Path, incumbent_cell: str):
    candidates = [_sample("candidate", "up", 0.25, 100.0), _sample("candidate", "down", 0.75, 100.0)]
    # One existing site drawing 60 MW worth of water from `incumbent_cell`.
    lon = {"up": 0.25, "down": 0.75}.get(incumbent_cell, 1.25)
    incumbent = _sample("incumbent", incumbent_cell, lon, 60.0)
    decomposition = root / "GFDL-ESM4_wet_channel_decomposition.parquet"
    pd.DataFrame([*candidates, incumbent]).to_parquet(decomposition, index=False)
    census = root / "GFDL-ESM4_monthly_census.parquet"
    pd.DataFrame(candidates)[CEILING_KEYS].assign(
        response="water_limited_capacity_ceiling_mw", value=[100.0, 200.0]
    ).to_parquet(census, index=False)
    river_network = root / "GFDL-ESM4_ssp585_wet_ucr_reach_topology.parquet"
    pd.DataFrame(
        {
            "population": ["candidate", "candidate"],
            "cell_id": ["up", "down"],
            "downstream_cell_id": ["down", None],
        }
    ).to_parquet(river_network, index=False)
    ordering = pd.DataFrame(
        {
            "region_id": ["R12_MEA", "R12_MEA"],
            "archetype": [ARCHETYPE] * 2,
            "cell_id": ["up", "down"],
            "ordering_usable_fraction": [1.0, 1.0],
            "fixed_order": [0, 1],
        }
    )
    return [decomposition], [census], [river_network], ordering


def test_candidate_curves_net_incumbent_withdrawal_out_of_candidate_water(tmp_path):
    products = build_wet_siting_outputs(
        *_candidate_inputs(tmp_path, incumbent_cell="up"), capacity_grid_points=3
    )

    curve = products.per_gcm_curves.sort_values("point_index")
    # Grid 0/100/200 MW, "up" filled first. The existing site's 60 MW leaves 40 of
    # the upstream water limit 100 and 140 of the cumulative downstream water
    # limit 200. Full-flow water would give usable 100 and 200 at the last two points.
    assert curve["capacity_axis_mw"].tolist() == pytest.approx([0.0, 100.0, 200.0])
    assert curve["usable_capacity_mw"].tolist() == pytest.approx([0.0, 40.0, 140.0])

    rime = products.rime_samples.set_index("depth_ratio")
    # The new-build ratio is relative to the 60 MW existing site. At ratio 2.0 (120 MW) the curve
    # interpolates to 40 + 0.2 * 100 = 60 MW usable, a fraction of 0.5; at ratio
    # 0.5 (30 MW) it gives 0.3 * 40 = 12 MW, a fraction of 0.4. Full-flow water
    # would give 1.0 at both ratios.
    assert rime.loc[2.0, "candidate_usable_fraction"] == pytest.approx(0.5)
    assert rime.loc[0.5, "candidate_usable_fraction"] == pytest.approx(0.4)


def test_candidate_products_report_the_overlap_and_respect_depth_ratios(tmp_path):
    products = build_wet_siting_outputs(
        *_candidate_inputs(tmp_path, incumbent_cell="up"),
        capacity_grid_points=3,
        new_build_ratios=[1.0],
    )

    assert products.rime_samples["depth_ratio"].tolist() == [1.0]
    overlap = products.incumbent_overlap
    assert overlap["region_id"].tolist() == ["R12_MEA"]
    assert overlap["nameplate_mw"].tolist() == [60.0]
    assert overlap["matched_share"].tolist() == [1.0]

    # Grid 0/100/200 MW; the 60 MW existing site leaves usable 0/40/140 MW (see the
    # test above). The pooled median over the one GCM is that curve itself, so
    # the last point is 140 MW usable at 200 MW installed.
    pooled = products.pooled_median_curves.sort_values("point_index")
    assert pooled["usable_capacity_mw"].iloc[-1] == pytest.approx(140.0)
    # Additivity check: the direct reference sum over the two potential sites
    # has no losses (200 MW nameplate, 200 MW usable). The final curve point is
    # 200 MW installed but 140 MW usable, the existing site's 60 MW having been
    # netted out of the shared river water, so usable capacity differs by
    # 200 - 140 = 60 MW; the curve's water-availability loss is likewise 60 MW
    # against 0 directly.
    assert products.additivity_oracle["max_abs_error_mw"].tolist() == pytest.approx([60.0])
    assert len(products.additivity_oracle) == 1


def test_candidate_products_are_written_under_their_file_names(tmp_path):
    products = build_wet_siting_outputs(
        *_candidate_inputs(tmp_path, incumbent_cell="up"), capacity_grid_points=3
    )
    paths = write_wet_siting_outputs(products, tmp_path / "out" / "nested")

    assert [path.name for path in paths] == [
        PER_GCM_CURVES_FILE,
        POOLED_MEDIAN_CURVES_FILE,
        ADDITIVITY_ORACLE_FILE,
        RIME_SAMPLES_FILE,
    ]
    pd.testing.assert_frame_equal(pd.read_parquet(paths[3]), products.rime_samples)


def test_candidate_products_refuse_when_no_incumbent_stands_on_a_candidate_cell(tmp_path):
    inputs = _candidate_inputs(tmp_path, incumbent_cell="elsewhere")
    with pytest.raises(ValueError, match="no-op") as refusal:
        build_wet_siting_outputs(*inputs, capacity_grid_points=3)
    # The overlap table that explains the refusal follows the first line.
    first, _, table = str(refusal.value).partition("\n")
    assert first.startswith("no existing site is also a potential site")
    assert "existing-site nameplate subtracted from potential-site water limits, by region:" in table
    assert "R12_MEA" in table
    assert "matched_share" in table


def test_candidate_products_refuse_a_capacity_grid_below_two_points(tmp_path):
    with pytest.raises(ValueError, match="capacity_grid_points must be at least 2"):
        build_wet_siting_outputs(
            *_candidate_inputs(tmp_path, incumbent_cell="up"), capacity_grid_points=1
        )


def test_overlap_of_empty_frames_is_an_empty_table_with_the_columns():
    incumbent = pd.DataFrame(
        {"region_id": [], "cell_id": [], "archetype": [], "nameplate_mw": []}
    ).astype({"nameplate_mw": float})
    candidate = pd.DataFrame({"cell_id": ["x"], "archetype": [ARCHETYPE]})
    overlap = existing_potential_site_overlap(incumbent, candidate)
    assert overlap.empty
    assert overlap.columns.tolist() == ["region_id", "nameplate_mw", "matched_mw", "matched_share"]


def test_overlap_counts_a_plant_once_and_only_where_a_candidate_cell_matches():
    incumbent = pd.DataFrame(
        {
            "region_id": ["A", "A", "B"],
            "cell_id": ["x", "y", "z"],
            "archetype": [ARCHETYPE] * 3,
            "nameplate_mw": [30.0, 10.0, 5.0],
        }
    )
    candidate = pd.DataFrame(
        {"cell_id": ["x", "x"], "archetype": [ARCHETYPE, "other"]}
    )
    overlap = existing_potential_site_overlap(incumbent, candidate).set_index("region_id")
    assert overlap["matched_mw"].to_dict() == {"A": 30.0, "B": 0.0}
    assert overlap["matched_share"].to_dict() == {"A": 0.75, "B": 0.0}


# --- multi-file input ---------------------------------------------------------------------------


def _two_gcm_two_region_inputs(root: Path):
    """One table of the wet capacity loss split and water limits over two GCMs and two regions."""

    decomposition, census, ordering = [], [], []
    for region, offset in (("R12_MEA", 0.0), ("R12_AFR", 10.0)):
        for gcm, loss in (("GFDL-ESM4", 0.0), ("IPSL-CM6A-LR", 20.0)):
            up, down = f"{region}_up", f"{region}_down"
            samples = [
                _sample("candidate", up, offset + 0.25, 100.0),
                _sample("candidate", down, offset + 0.75, 100.0),
                _sample("incumbent", up, offset + 0.25, 60.0),
            ]
            for sample in samples:
                sample.update(region_id=region, gcm=gcm)
                sample["usable_capacity_mw"] -= loss if sample["population"] == "candidate" else 0.0
                sample["thermal_loss_mw"] = loss if sample["population"] == "candidate" else 0.0
            decomposition += samples
            census.append(
                pd.DataFrame(samples[:2])[CEILING_KEYS].assign(
                    response="water_limited_capacity_ceiling_mw", value=[100.0, 200.0]
                )
            )
        ordering.append(
            pd.DataFrame(
                {
                    "region_id": region,
                    "archetype": ARCHETYPE,
                    "cell_id": [f"{region}_up", f"{region}_down"],
                    "ordering_usable_fraction": [1.0, 1.0],
                    "fixed_order": [0, 1],
                }
            )
        )
    river_network = pd.DataFrame(
        {
            "population": "candidate",
            "cell_id": [f"{r}_{c}" for r in ("R12_MEA", "R12_AFR") for c in ("up", "down")],
            "downstream_cell_id": ["R12_MEA_down", None, "R12_AFR_down", None],
        }
    )
    topology_path = root / "ssp585_wet_ucr_reach_topology.parquet"
    river_network.to_parquet(topology_path, index=False)
    return pd.DataFrame(decomposition), pd.concat(census, ignore_index=True), pd.concat(ordering), topology_path


@pytest.mark.parametrize("split_by", ["gcm", "region_id"])
def test_candidate_products_from_split_files_equal_those_from_one_file(tmp_path, split_by):
    decomposition, census, ordering, river_network = _two_gcm_two_region_inputs(tmp_path)

    def write(frame, name):
        path = tmp_path / name
        frame.to_parquet(path, index=False)
        return path

    whole = build_wet_siting_outputs(
        [write(decomposition, "whole_decomposition.parquet")],
        [write(census, "whole_census.parquet")],
        [river_network],
        ordering,
        capacity_grid_points=5,
    )
    values = sorted(decomposition[split_by].unique())
    split = build_wet_siting_outputs(
        [write(decomposition[decomposition[split_by] == v], f"d_{i}.parquet") for i, v in enumerate(values)],
        [write(census[census[split_by] == v], f"c_{i}.parquet") for i, v in enumerate(values)],
        [river_network],
        ordering,
        capacity_grid_points=5,
    )  # fmt: skip

    assert len(whole.per_gcm_curves) > 0
    for name in ("per_gcm_curves", "pooled_median_curves", "additivity_oracle", "rime_samples", "incumbent_overlap"):
        left = getattr(whole, name)
        right = getattr(split, name)
        sort = list(left.columns)
        pd.testing.assert_frame_equal(
            left.sort_values(sort).reset_index(drop=True),
            right.sort_values(sort).reset_index(drop=True),
            obj=name,
        )


# --- residual-water ordering --------------------------------------------------------------------


def _cells(population: str, rows: list[dict], gcm: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "region_id": "R12_MEA",
                "country": "AAA",
                "grid_lat": 0.25,
                "archetype": ARCHETYPE,
                "gcm": gcm,
                "scenario": "historical",
                "year": 2000,
                "season": "JJA",
                "warming_level": "REF-PD",
                "withdrawal_loss_mw": 0.0,
                "population": population,
                "grid_lon": 0.25 + 0.5 * i,
                "nameplate_mw": 100.0,
                "thermal_loss_mw": 100.0 - row.get("usable_capacity_mw", 100.0),
                **row,
            }
            for i, row in enumerate(rows)
        ]
    )


def _write_gcm(root: Path, gcm: str, incumbent_mw: float) -> tuple[Path, Path, Path]:
    candidates = _cells(
        "candidate",
        [
            {"cell_id": "fleet", "usable_capacity_mw": 90.0},
            {"cell_id": "open", "usable_capacity_mw": 70.0},
        ],
        gcm,
    )
    incumbent = _cells(
        "incumbent",
        [{"cell_id": "fleet", "nameplate_mw": incumbent_mw, "thermal_loss_mw": 0.0}],
        gcm,
    )
    decomposition = root / f"{gcm}_REF-PD_wet_channel_decomposition.parquet"
    pd.concat([candidates, incumbent], ignore_index=True).to_parquet(decomposition, index=False)
    census = root / f"{gcm}_REF-PD_monthly_census.parquet"
    candidates[CEILING_KEYS].assign(
        response="water_limited_capacity_ceiling_mw", value=100.0
    ).to_parquet(census, index=False)
    river_network = root / f"{gcm}_historical_wet_ucr_reach_topology.parquet"
    pd.DataFrame(
        {
            "population": "candidate",
            "cell_id": ["fleet", "open"],
            "downstream_cell_id": [None, None],
        }
    ).to_parquet(river_network, index=False)
    return decomposition, census, river_network


def _inputs(root: Path, draws: dict[str, float]):
    triples = [_write_gcm(root, gcm, draw) for gcm, draw in draws.items()]
    return [list(column) for column in zip(*triples, strict=True)]


def _baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region_id": "R12_MEA",
            "archetype": ARCHETYPE,
            "cell_id": ["fleet", "open"],
            "ordering_usable_fraction": [0.9, 0.7],
            "fixed_order": [0, 1],
        }
    )


def _ranked(ordering: pd.DataFrame) -> list[tuple[str, float]]:
    ordered = ordering.sort_values("fixed_order")
    return list(zip(ordered["cell_id"], ordered["ordering_usable_fraction"].round(12), strict=True))


def test_residual_ordering_pools_every_gcm_before_ranking(tmp_path):
    # GFDL's existing site leaves the fleet cell 40 of 100 (0.4); IPSL's 5 MW draw
    # leaves 95, above its 90 usable (0.9). Pooled: fleet 0.65, open 0.7.
    # Ranking GFDL alone would give the same order at 0.4; ranking IPSL alone,
    # or full-flow water, would put the fleet cell first.
    result = build_residual_water_ordering(
        *_inputs(tmp_path, {"GFDL-ESM4": 60.0, "IPSL-CM6A-LR": 5.0}),
        baseline_ordering=_baseline(),
    )
    assert _ranked(result.ordering) == [("open", 0.7), ("fleet", 0.65)]
    assert result.ordering["cell_id"].map(type).eq(str).all()
    # Both GCMs report the share of samples charged; only the fleet cell is.
    assert result.charged_share["gcm"].tolist() == ["GFDL-ESM4", "IPSL-CM6A-LR"]
    assert result.charged_share["charged_share"].tolist() == [0.5, 0.0]
    assert result.moves["moved"].tolist() == [2]


def test_residual_ordering_refuses_an_ordering_identical_to_the_baseline(tmp_path):
    inputs = _inputs(tmp_path, {"GFDL-ESM4": 5.0, "IPSL-CM6A-LR": 5.0})
    with pytest.raises(ValueError, match="no-op") as refusal:
        build_residual_water_ordering(*inputs, baseline_ordering=_baseline())
    # The moves table that explains the refusal follows the first line.
    first, _, table = str(refusal.value).partition("\n")
    assert first.startswith("residual-water ordering equals the baseline")
    assert "cells whose rank differs from the baseline ordering:" in table
    assert "moved" in table


def test_residual_ordering_without_a_baseline_reports_no_moves(tmp_path):
    result = build_residual_water_ordering(*_inputs(tmp_path, {"GFDL-ESM4": 60.0}))
    assert result.moves is None
    assert _ranked(result.ordering) == [("open", 0.7), ("fleet", 0.4)]


def test_residual_ordering_refuses_a_baseline_over_other_cells(tmp_path):
    inputs = _inputs(tmp_path, {"GFDL-ESM4": 60.0})
    with pytest.raises(ValueError, match="different cells"):
        build_residual_water_ordering(*inputs, baseline_ordering=_baseline().iloc[:1])


def test_residual_ordering_refuses_unequal_input_counts(tmp_path):
    decompositions, ceilings, topologies = _inputs(tmp_path, {"GFDL-ESM4": 60.0, "IPSL-CM6A-LR": 5.0})
    with pytest.raises(ValueError, match="one loss-split table, one water-limit file and one river-network file per GCM"):
        build_residual_water_ordering(decompositions, ceilings[:1], topologies)


def test_residual_ordering_refuses_a_topology_of_another_gcm(tmp_path):
    decompositions, ceilings, topologies = _inputs(tmp_path, {"GFDL-ESM4": 60.0, "IPSL-CM6A-LR": 5.0})
    with pytest.raises(ValueError, match="do not pair on one GCM"):
        build_residual_water_ordering(decompositions[:1], ceilings[:1], topologies[1:])


def test_residual_ordering_is_written_as_parquet(tmp_path):
    result = build_residual_water_ordering(*_inputs(tmp_path, {"GFDL-ESM4": 60.0}))
    path = write_residual_water_ordering(result, tmp_path / "out" / "ordering.parquet")
    pd.testing.assert_frame_equal(pd.read_parquet(path), result.ordering)


# --- pooling the warming-level and present-day reference calculations ------------------------------------------------------------------


def _curves(gcm: str, usable_mw: float, cell: str, *, level=2.0) -> pd.DataFrame:
    """One curve of two points: zero capacity, then 100 MW on ``cell``."""

    rows = []
    for point, axis in ((1, 0.0), (2, 100.0)):
        rows.append(
            {
                "population": "candidate", "placement_rule": "candidate_reach_aware",
                "gcm": gcm, "region_id": "R12_MEA", "plant_type": "coal",
                "cooling_type": "once_through", "archetype": ARCHETYPE, "season": "JJA",
                "warming_level": level, "point_index": point, "capacity_axis_mw": axis,
                "marginal_cell_id": cell, "marginal_country": "AAA",
                "marginal_capacity_mw": axis, "marginal_usable_fraction": 1.0,
                "thermal_loss_mw": 0.0, "withdrawal_loss_mw": 0.0,
                "usable_capacity_mw": usable_mw if point == 2 else 0.0,
                "thermal_loss_fraction": 0.0, "withdrawal_loss_fraction": 0.0,
                "usable_fraction": usable_mw / 100.0 if point == 2 else 0.0,
            }
        )  # fmt: skip
    return pd.DataFrame(rows)


def _rime_samples(gcm: str, fraction: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gcm": gcm, "scenario": "ssp585", "year": [2040, 2041], "region_id": "R12_MEA",
            "plant_type": "coal", "cooling_type": "once_through", "season": "JJA",
            "warming_level": 2.0, "depth_ratio": 1.0, "incumbent_capacity_mw": 60.0,
            "incumbent_usable_fraction": 1.0, "candidate_usable_fraction": [fraction, fraction + 0.2],
            "usable_capacity_ratio": [1.0 + fraction, 1.2 + fraction],
        }
    )  # fmt: skip


ONE_CELL_DIMENSIONS = {
    "region_id": 1, "plant_type": 1, "cooling_type": 1, "season": 1, "gwl": 1, "depth_ratio": 1,
}  # fmt: skip


def _three_gcm_arms():
    gwl = pd.concat([_curves(g, u, "c1") for g, u in (("A", 10.0), ("B", 20.0), ("C", 60.0))])
    reference = pd.concat(
        [_curves(g, u, "c1", level="REF-PD") for g, u in (("A", 30.0), ("B", 40.0), ("C", 50.0))]
    )
    rime = pd.concat([_rime_samples(g, f) for g, f in (("A", 0.2), ("B", 0.4), ("C", 0.6))])
    return gwl, reference, rime


def test_pooled_products_take_the_median_over_gcms_and_percentiles_over_samples():
    gwl, reference, rime = _three_gcm_arms()
    products = pool_wet_siting_outputs(
        gwl, reference, rime, expected_rime_dimensions=ONE_CELL_DIMENSIONS
    )

    # Usable capacity at 100 MW: the median of 10, 20 and 60 MW is 20 MW, and of
    # the reference calculation's 30, 40 and 50 MW is 40 MW.
    last = products.gwl_pooled_median_curves.query("point_index == 2")
    assert last["usable_capacity_mw"].tolist() == [20.0]
    assert last["gcm_count"].tolist() == [3]
    last = products.reference_pooled_median_curves.query("point_index == 2")
    assert last["usable_capacity_mw"].tolist() == [40.0]
    # Six samples with potential-site fractions 0.2 0.4 0.6 0.4 0.6 0.8: p50 is their
    # median, (0.4 + 0.6) / 2 = 0.5, and p10 interpolates at rank 0.5 between 0.2
    # and the second smallest, 0.4 (the two 0.4s are equal): 0.2 + 0.5 * 0.2 = 0.3.
    percentiles = products.rime_percentiles.squeeze()
    assert percentiles["candidate_usable_fraction_p50"].item() == pytest.approx(0.5)
    assert percentiles["candidate_usable_fraction_p10"].item() == pytest.approx(0.3)
    assert percentiles["sample_count"].item() == 6
    assert percentiles["gcm_count"].item() == 3
    assert products.validation == {
        "fixed_membership_rows": 3, "gcms": 3, "regions": 1, "archetypes": 1, "seasons": 1,
        "gwl_levels": 1, "curve_points": 2, "rime_dimensions": ONE_CELL_DIMENSIONS,
    }  # fmt: skip


def test_pooled_products_refuse_arms_that_place_different_cells():
    gwl, reference, rime = _three_gcm_arms()
    other = reference.assign(marginal_cell_id=reference["marginal_cell_id"].where(reference["gcm"] != "B", "c2"))
    with pytest.raises(ValueError, match="different physical cells"):
        pool_wet_siting_outputs(gwl, other, rime, expected_rime_dimensions=None)
    # The zero-capacity first point carries no cell and may differ.
    first_point = reference.assign(
        marginal_cell_id=reference["marginal_cell_id"].where(reference["point_index"] > 1, "none")
    )
    pool_wet_siting_outputs(gwl, first_point, rime, expected_rime_dimensions=None)


def test_pooled_products_refuse_dimensions_other_than_the_defaults():
    gwl, reference, rime = _three_gcm_arms()
    with pytest.raises(ValueError, match="unexpected RIME dimensions"):
        pool_wet_siting_outputs(gwl, reference, rime)
    assert DEFAULT_RIME_DIMENSIONS == {
        "region_id": 12, "plant_type": 2, "cooling_type": 2, "season": 4, "gwl": 6, "depth_ratio": 3,
    }  # fmt: skip


def test_gcm_outputs_are_read_in_path_order_and_counted(tmp_path):
    for index in (10, 2, 1):
        directory = tmp_path / str(index)
        directory.mkdir()
        pd.DataFrame({"directory": [index]}).to_parquet(directory / "table.parquet", index=False)
    # Paths sort as text: 1, 10, 2.
    assert read_gcm_outputs(tmp_path, "table.parquet", gcm_count=3)["directory"].tolist() == [1, 10, 2]
    with pytest.raises(ValueError, match=r"expected table\.parquet in 5 per-GCM output directories, found 3"):
        read_gcm_outputs(tmp_path, "table.parquet", gcm_count=5)
    with pytest.raises(ValueError, match=r"expected table\.parquet in 4 per-GCM output directories, found 3"):
        read_gcm_outputs(tmp_path, "table.parquet", gcm_count=4)


def test_pooled_products_are_written_as_the_product_tree(tmp_path):
    gwl, reference, rime = _three_gcm_arms()
    products = pool_wet_siting_outputs(
        gwl, reference, rime, expected_rime_dimensions=ONE_CELL_DIMENSIONS
    )
    paths = write_pooled_wet_outputs(products, tmp_path / "out")

    assert sorted(path.relative_to(tmp_path / "out").as_posix() for path in paths) == [
        "gwl/r12_thermoelectric_wet_gwl.nc",
        "gwl/wet_ucr_curves_per_gcm.parquet",
        "gwl/wet_ucr_curves_pooled_median.parquet",
        "reference_pd/wet_ucr_curves_per_gcm.parquet",
        "reference_pd/wet_ucr_curves_pooled_median.parquet",
        "validation.json",
    ]
    text = (tmp_path / "out" / "validation.json").read_text()
    assert text.endswith("}\n")
    assert json.loads(text) == products.validation


# --- writers refuse to replace earlier output files ---------------------------------------------------


def test_candidate_products_are_not_written_over_an_earlier_product(tmp_path):
    products = build_wet_siting_outputs(
        *_candidate_inputs(tmp_path, incumbent_cell="up"), capacity_grid_points=3
    )
    out = tmp_path / "out"
    out.mkdir()
    (out / RIME_SAMPLES_FILE).write_text("earlier")

    with pytest.raises(FileExistsError, match=RIME_SAMPLES_FILE):
        write_wet_siting_outputs(products, out)
    # Nothing else was written, and the earlier file is as it was.
    assert [path.name for path in out.iterdir()] == [RIME_SAMPLES_FILE]
    assert (out / RIME_SAMPLES_FILE).read_text() == "earlier"

    write_wet_siting_outputs(products, out, overwrite=True)
    pd.testing.assert_frame_equal(pd.read_parquet(out / RIME_SAMPLES_FILE), products.rime_samples)


def test_residual_ordering_is_not_written_over_an_earlier_ordering(tmp_path):
    result = build_residual_water_ordering(*_inputs(tmp_path, {"GFDL-ESM4": 60.0}))
    path = tmp_path / "ordering.parquet"
    path.write_text("earlier")

    with pytest.raises(FileExistsError, match=r"ordering\.parquet"):
        write_residual_water_ordering(result, path)
    assert path.read_text() == "earlier"

    write_residual_water_ordering(result, path, overwrite=True)
    pd.testing.assert_frame_equal(pd.read_parquet(path), result.ordering)


def test_pooled_products_are_not_written_over_an_earlier_tree(tmp_path):
    gwl, reference, rime = _three_gcm_arms()
    products = pool_wet_siting_outputs(
        gwl, reference, rime, expected_rime_dimensions=ONE_CELL_DIMENSIONS
    )
    out = tmp_path / "out"
    (out / "reference_pd").mkdir(parents=True)
    (out / "validation.json").write_text("earlier")

    with pytest.raises(FileExistsError, match=r"validation\.json"):
        write_pooled_wet_outputs(products, out)
    # Neither calculation's directory gained a file.
    assert sorted(path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()) == [
        "validation.json"
    ]

    write_pooled_wet_outputs(products, out, overwrite=True)
    assert json.loads((out / "validation.json").read_text()) == products.validation


def _fail_on_nth_parquet_write(monkeypatch, n):
    real = pd.DataFrame.to_parquet
    calls = []

    def flaky(self, path, *args, **kwargs):
        calls.append(path)
        if len(calls) == n:
            raise OSError("disk full")
        return real(self, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", flaky)


def _snapshot(directory):
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _changed(table):
    return table.assign(extra_marker=1)


def test_candidate_products_that_fail_on_a_later_file_leave_the_earlier_product_whole(
    tmp_path, monkeypatch
):
    products = build_wet_siting_outputs(
        *_candidate_inputs(tmp_path, incumbent_cell="up"), capacity_grid_points=3
    )
    out = tmp_path / "out"
    write_wet_siting_outputs(products, out)
    before = _snapshot(out)

    changed = dataclasses.replace(
        products,
        per_gcm_curves=_changed(products.per_gcm_curves),
        pooled_median_curves=_changed(products.pooled_median_curves),
        additivity_oracle=_changed(products.additivity_oracle),
    )
    _fail_on_nth_parquet_write(monkeypatch, 3)
    with pytest.raises(OSError, match="disk full"):
        write_wet_siting_outputs(changed, out, overwrite=True)

    assert _snapshot(out) == before

    monkeypatch.undo()
    write_wet_siting_outputs(changed, out, overwrite=True)
    assert _snapshot(out) != before


@pytest.mark.parametrize("failing", ["second_parquet", "netcdf", "json"])
def test_pooled_products_that_fail_on_a_later_file_leave_the_earlier_tree_whole(
    tmp_path, monkeypatch, failing
):
    gwl, reference, rime = _three_gcm_arms()
    products = pool_wet_siting_outputs(
        gwl, reference, rime, expected_rime_dimensions=ONE_CELL_DIMENSIONS
    )
    out = tmp_path / "out"
    write_pooled_wet_outputs(products, out)
    before = _snapshot(out)

    changed = dataclasses.replace(
        products,
        gwl_per_gcm_curves=_changed(products.gwl_per_gcm_curves),
        gwl_pooled_median_curves=_changed(products.gwl_pooled_median_curves),
        validation={**products.validation, "marker": 1},
    )
    if failing == "second_parquet":
        _fail_on_nth_parquet_write(monkeypatch, 2)
    elif failing == "netcdf":

        def broken(self, *args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(xr.Dataset, "to_netcdf", broken)
    else:

        def broken_text(self, *args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", broken_text)
    with pytest.raises(OSError, match="disk full"):
        write_pooled_wet_outputs(changed, out, overwrite=True)

    assert _snapshot(out) == before

    fresh = tmp_path / "fresh"
    if failing == "second_parquet":
        monkeypatch.undo()
        _fail_on_nth_parquet_write(monkeypatch, 2)
    with pytest.raises(OSError, match="disk full"):
        write_pooled_wet_outputs(changed, fresh)
    assert _snapshot(fresh) == {}
