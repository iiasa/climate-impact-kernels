"""Joint allocation of river water to existing and potential sites.

These tests catch capacity curves for potential sites that are allocated against
the full-flow water limits while the existing sites withdraw from the same river
network. The same water is then counted twice and the water left at potential
sites looks larger than it is. Expected values are derived by hand from the river
network arithmetic, not from the functions under test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from impacts.thermoelectric.siting.curves import (
    EXISTING_SITE_DEMAND_COLUMN,
    _load_reach_water_ceilings,
    attach_existing_site_water_demand,
    build_capacity_curves,
    build_fixed_cell_ordering,
    build_pooled_median_curves,
    build_potential_site_curves_with_river_network,
    build_potential_site_sample_curves_with_river_network,
    build_residual_water_cell_ordering,
    export_wet_usable_capacity_curves,
    reference_fleet_additivity_check,
    validate_reference_fleet_additivity_check,
)


def _cell_table() -> pd.DataFrame:
    rows = []
    for gcm, multiplier in (("GFDL-ESM4", 1.0), ("IPSL-CM6A-LR", 2.0)):
        rows.extend(
            [
                {
                    "population": "incumbent",
                    "region_id": "R12_MEA",
                    "country": "SAU",
                    "cell_id": "cell-good",
                    "grid_lat": 0.25,
                    "grid_lon": 40.25,
                    "archetype": "li_coal_500mw_once_through",
                    "gcm": gcm,
                    "season": "JJA",
                    "warming_level": 3.0,
                    "nameplate_mw": 500.0,
                    "thermal_loss_mw": 50.0 * multiplier,
                    "withdrawal_loss_mw": 50.0,
                    "usable_capacity_mw": 400.0 - 50.0 * (multiplier - 1.0),
                },
                {
                    "population": "incumbent",
                    "region_id": "R12_MEA",
                    "country": "SAU",
                    "cell_id": "cell-bad",
                    "grid_lat": 0.25,
                    "grid_lon": 40.75,
                    "archetype": "li_coal_500mw_once_through",
                    "gcm": gcm,
                    "season": "JJA",
                    "warming_level": 3.0,
                    "nameplate_mw": 500.0,
                    "thermal_loss_mw": 150.0 * multiplier,
                    "withdrawal_loss_mw": 100.0,
                    "usable_capacity_mw": 250.0 - 150.0 * (multiplier - 1.0),
                },
                {
                    "population": "incumbent",
                    "region_id": "R12_MEA",
                    "country": "SAU",
                    "cell_id": "cell-recirc",
                    "grid_lat": 0.25,
                    "grid_lon": 41.25,
                    "archetype": "li_gas_250mw_recirculating",
                    "gcm": gcm,
                    "season": "JJA",
                    "warming_level": 3.0,
                    "nameplate_mw": 250.0,
                    "thermal_loss_mw": 25.0 * multiplier,
                    "withdrawal_loss_mw": 0.0,
                    "usable_capacity_mw": 225.0 - 25.0 * (multiplier - 1.0),
                },
            ]
        )
    return pd.DataFrame(rows)


def test_capacity_curves_rank_cells_and_accumulate_fractions() -> None:
    curves = build_capacity_curves(_cell_table())

    coal = curves[
        (curves["gcm"] == "GFDL-ESM4")
        & (curves["plant_type"] == "coal")
        & (curves["cooling_type"] == "once_through")
    ].sort_values("point_index")

    assert coal["marginal_cell_id"].tolist() == ["cell-good", "cell-bad"]
    assert coal["capacity_axis_mw"].tolist() == [500.0, 1000.0]
    assert coal["usable_capacity_mw"].tolist() == [400.0, 650.0]
    assert coal["usable_fraction"].tolist() == pytest.approx([0.8, 0.65])
    assert coal["thermal_loss_fraction"].tolist() == pytest.approx([0.1, 0.2])
    assert coal["withdrawal_loss_fraction"].tolist() == pytest.approx([0.1, 0.15])


def test_reference_fleet_oracle_reconstructs_final_curve_sum() -> None:
    cells = _cell_table()
    curves = build_capacity_curves(cells)
    additivity_check = reference_fleet_additivity_check(cells, curves)

    validate_reference_fleet_additivity_check(additivity_check)

    row = additivity_check[
        (additivity_check["gcm"] == "GFDL-ESM4")
        & (additivity_check["region_id"] == "R12_MEA")
        & (additivity_check["season"] == "JJA")
        & (additivity_check["warming_level"] == 3.0)
    ].iloc[0]
    assert row["direct_nameplate_mw"] == pytest.approx(1250.0)
    assert row["curve_nameplate_mw"] == pytest.approx(1250.0)
    assert row["direct_usable_capacity_mw"] == pytest.approx(875.0)
    assert row["curve_usable_capacity_mw"] == pytest.approx(875.0)
    assert row["max_abs_error_mw"] == pytest.approx(0.0)


def test_pooled_median_curves_keep_per_gcm_rank_surface() -> None:
    pooled = build_pooled_median_curves(build_capacity_curves(_cell_table()))

    coal_first = pooled[
        (pooled["plant_type"] == "coal")
        & (pooled["cooling_type"] == "once_through")
        & (pooled["point_index"] == 1)
    ].iloc[0]

    assert coal_first["statistic"] == "pooled_median"
    assert coal_first["gcm_count"] == 2
    assert coal_first["capacity_axis_mw"] == pytest.approx(500.0)
    assert coal_first["thermal_loss_mw"] == pytest.approx(75.0)
    assert coal_first["usable_fraction"] == pytest.approx(0.75)


def test_pooled_median_curves_refuse_a_point_the_gcms_do_not_share() -> None:
    curves = build_capacity_curves(_cell_table())
    first_gcm = curves["gcm"].iloc[0]
    point = (curves["gcm"] == first_gcm) & (curves["point_index"] == 1)

    moved = curves.copy()
    moved.loc[point, "capacity_axis_mw"] += 500.0
    with pytest.raises(ValueError, match="differ in capacity across GCMs"):
        build_pooled_median_curves(moved)

    with pytest.raises(ValueError, match="missing for some"):
        build_pooled_median_curves(curves.loc[~point])


def test_export_writes_curve_tables_and_oracle(tmp_path) -> None:
    path = tmp_path / "decomposition.parquet"
    decomposition = _cell_table().assign(scenario="ssp585", year=2030)
    decomposition.to_parquet(path, index=False)

    per_gcm_path, pooled_path, oracle_path = export_wet_usable_capacity_curves(
        [path], tmp_path / "out"
    )

    per_gcm = pd.read_parquet(per_gcm_path)
    pooled = pd.read_parquet(pooled_path)
    additivity_check = pd.read_parquet(oracle_path)

    assert set(per_gcm["gcm"]) == {"GFDL-ESM4", "IPSL-CM6A-LR"}
    assert set(pooled["statistic"]) == {"pooled_median"}
    assert additivity_check["max_abs_error_mw"].max() == pytest.approx(0.0)


def _candidate_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "population": "candidate",
                "region_id": "R12_MEA",
                "country": "AAA",
                "cell_id": "up",
                "grid_lat": 0.25,
                "grid_lon": 0.25,
                "archetype": "li_coal_500mw_once_through",
                "gcm": "GFDL-ESM4",
                "scenario": "ssp585",
                "year": 2050,
                "season": "JJA",
                "warming_level": 3.0,
                "nameplate_mw": 100.0,
                "thermal_loss_mw": 20.0,
                "withdrawal_loss_mw": 0.0,
                "usable_capacity_mw": 80.0,
            },
            {
                "population": "candidate",
                "region_id": "R12_MEA",
                "country": "AAA",
                "cell_id": "down",
                "grid_lat": 0.25,
                "grid_lon": 0.75,
                "archetype": "li_coal_500mw_once_through",
                "gcm": "GFDL-ESM4",
                "scenario": "ssp585",
                "year": 2050,
                "season": "JJA",
                "warming_level": 3.0,
                "nameplate_mw": 100.0,
                "thermal_loss_mw": 0.0,
                "withdrawal_loss_mw": 0.0,
                "usable_capacity_mw": 100.0,
            },
        ]
    )


def test_candidate_reach_curves_keep_thermal_local_and_share_withdrawal() -> None:
    cells = _candidate_table()
    water = cells[
        [
            "population",
            "region_id",
            "country",
            "cell_id",
            "grid_lat",
            "grid_lon",
            "archetype",
            "gcm",
            "scenario",
            "year",
            "season",
            "warming_level",
        ]
    ].assign(water_ceiling_mw=[100.0, 120.0])
    river_network = pd.DataFrame(
        {
            "population": ["candidate", "candidate"],
            "scenario": ["ssp585", "ssp585"],
            "cell_id": ["up", "down"],
            "downstream_cell_id": ["down", None],
        }
    )

    curves = build_potential_site_curves_with_river_network(
        cells,
        water,
        river_network,
        capacity_grid_points=3,
        fixed_ordering=build_fixed_cell_ordering(cells),
    )
    final = curves.sort_values("point_index").iloc[-1]

    assert final["capacity_axis_mw"] == pytest.approx(200.0)
    assert final["thermal_loss_mw"] == pytest.approx(20.0)
    assert final["usable_capacity_mw"] == pytest.approx(120.0)
    assert final["withdrawal_loss_mw"] == pytest.approx(60.0)


def test_fixed_ordering_preserves_cell_identity_across_climate_slices() -> None:
    base = _candidate_table()
    reference = base.assign(
        usable_capacity_mw=[90.0, 50.0],
        thermal_loss_mw=[10.0, 50.0],
    )
    fixed_ordering = build_fixed_cell_ordering(reference)
    slices = []
    for season, warming_level in (("DJF", 1.5), ("JJA", 3.0)):
        slices.append(
            base.assign(
                season=season,
                warming_level=warming_level,
                usable_capacity_mw=[20.0, 100.0],
                thermal_loss_mw=[80.0, 0.0],
            )
        )
    cells = pd.concat(slices, ignore_index=True)
    water = cells[
        [
            "population",
            "region_id",
            "country",
            "cell_id",
            "grid_lat",
            "grid_lon",
            "archetype",
            "gcm",
            "scenario",
            "year",
            "season",
            "warming_level",
        ]
    ].assign(water_ceiling_mw=1000.0)
    river_network = pd.DataFrame(
        {
            "population": ["candidate", "candidate"],
            "scenario": ["ssp585", "ssp585"],
            "cell_id": ["up", "down"],
            "downstream_cell_id": [None, None],
        }
    )

    candidate = build_potential_site_sample_curves_with_river_network(
        cells,
        water,
        river_network,
        capacity_grid_points=3,
        fixed_ordering=fixed_ordering,
    )
    identities = candidate.loc[candidate["point_index"] > 1].groupby(
        ["season", "warming_level"], sort=True
    )["marginal_cell_id"].apply(list)
    assert identities.tolist() == [["up", "down"], ["up", "down"]]

    incumbent = build_capacity_curves(cells, fixed_ordering=fixed_ordering)
    incumbent_identities = incumbent.groupby(
        ["season", "warming_level"], sort=True
    )["marginal_cell_id"].apply(list)
    assert incumbent_identities.tolist() == [["up", "down"], ["up", "down"]]


def test_candidate_reach_curves_reduce_to_existing_export_without_links() -> None:
    cells = _candidate_table()
    water = cells[
        [
            "population",
            "region_id",
            "country",
            "cell_id",
            "grid_lat",
            "grid_lon",
            "archetype",
            "gcm",
            "scenario",
            "year",
            "season",
            "warming_level",
        ]
    ].assign(water_ceiling_mw=cells["usable_capacity_mw"])
    river_network = pd.DataFrame(
        {
            "population": ["candidate", "candidate"],
            "scenario": ["ssp585", "ssp585"],
            "cell_id": ["up", "down"],
            "downstream_cell_id": [None, None],
        }
    )

    reach = build_potential_site_curves_with_river_network(
        cells,
        water,
        river_network,
        capacity_grid_points=3,
        fixed_ordering=build_fixed_cell_ordering(cells),
    )
    existing = build_capacity_curves(cells, population="candidate")

    assert reach["capacity_axis_mw"].tolist() == pytest.approx([0.0, 100.0, 200.0])
    assert reach["usable_capacity_mw"].tolist() == pytest.approx(
        [
            0.0,
            existing["usable_capacity_mw"].iloc[0],
            existing["usable_capacity_mw"].iloc[-1],
        ]
    )


def test_candidate_reach_curves_accept_present_nan_ceiling_for_thermal_zero() -> None:
    cells = _candidate_table().iloc[[0]].assign(
        thermal_loss_mw=100.0,
        withdrawal_loss_mw=0.0,
        usable_capacity_mw=0.0,
    )
    water = cells[
        [
            "population",
            "region_id",
            "country",
            "cell_id",
            "grid_lat",
            "grid_lon",
            "archetype",
            "gcm",
            "scenario",
            "year",
            "season",
            "warming_level",
        ]
    ].assign(water_ceiling_mw=float("nan"))
    river_network = pd.DataFrame(
        {
            "population": ["candidate"],
            "scenario": ["ssp585"],
            "cell_id": ["up"],
            "downstream_cell_id": [None],
        }
    )

    curves = build_potential_site_curves_with_river_network(
        cells,
        water,
        river_network,
        capacity_grid_points=2,
        fixed_ordering=build_fixed_cell_ordering(cells),
    )

    final = curves.sort_values("point_index").iloc[-1]
    assert final["capacity_axis_mw"] == pytest.approx(100.0)
    assert final["thermal_loss_mw"] == pytest.approx(100.0)
    assert final["withdrawal_loss_mw"] == pytest.approx(0.0)
    assert final["usable_capacity_mw"] == pytest.approx(0.0)


def _candidate_audit_ceiling_rows(
    *, cell_id: str, region_id: str, country: str, archetype: str, gcm: str
) -> list[dict]:
    return [
        {
            "population": "candidate",
            "region_id": region_id,
            "country": country,
            "cell_id": cell_id,
            "grid_lat": 0.25,
            "grid_lon": 0.25,
            "archetype": archetype,
            "gcm": gcm,
            "scenario": "ssp585",
            "year": year,
            "season": "JJA",
            "warming_level": 3.0,
            "response": "water_limited_capacity_ceiling_mw",
            "value": value,
        }
        for year, value in ((2035, 100.0), (2054, 120.0))
    ]


def test_load_reach_water_ceilings_means_over_sampled_years(tmp_path) -> None:
    path = tmp_path / "candidate_audit_ceilings.parquet"
    pd.DataFrame(
        _candidate_audit_ceiling_rows(
            cell_id="cell-x",
            region_id="R12_MEA",
            country="AAA",
            archetype="li_coal_500mw_once_through",
            gcm="GFDL-ESM4",
        )
    ).to_parquet(path, index=False)

    ceilings = _load_reach_water_ceilings([path], population="candidate")

    assert "year" not in ceilings.columns
    assert len(ceilings) == 1
    assert ceilings["water_ceiling_mw"].iloc[0] == pytest.approx(110.0)


def test_archetype_parts_come_from_the_definitions_not_the_name() -> None:
    from impacts.thermoelectric.siting.curves import _with_archetype_parts
    from impacts.thermoelectric.siting.loss_split import _decompose_monthly_to_sums

    cells = _cell_table()
    parts = _with_archetype_parts(cells)
    assert set(zip(parts["archetype"], parts["plant_type"], parts["cooling_type"], strict=True)) == {
        (name, name.split("_")[1], name.split("_", 3)[3]) for name in cells["archetype"].unique()
    }
    # A well-formed name that is not a defined synthetic plant is refused, in the
    # curves and in the nameplate lookup of the wet capacity loss split alike.
    unknown = cells.assign(archetype="li_gas_500mw_once_through")
    with pytest.raises(ValueError, match="unrecognized wet archetype: 'li_gas_500mw_once_through'"):
        _with_archetype_parts(unknown)
    monthly = pd.DataFrame(
        {
            "population": ["candidate"],
            "region_id": ["AFR"],
            "country": ["NGA"],
            "cell_id": ["a"],
            "grid_lat": [0.25],
            "grid_lon": [0.25],
            "cell_area_m2": [1.0],
            "archetype": ["li_coal_250mw_once_through"],
            "gcm": ["GFDL-ESM4"],
            "scenario": ["ssp585"],
            "year": [2050],
            "season": ["JJA"],
            "month": [7],
            "warming_level": [2.0],
            "available_discharge_m3_per_s": [10.0],
            "withdrawal_intensity_m3_per_mwh": [100.0],
        }
    )
    reference = pd.DataFrame(
        {
            "population": ["candidate"],
            "cell_id": ["a"],
            "archetype": ["li_coal_250mw_once_through"],
            "month": [7],
            "baseline_thermal_conductance": [36.0],
        }
    )
    with pytest.raises(ValueError, match="unrecognized wet archetype: 'li_coal_250mw_once_through'"):
        _decompose_monthly_to_sums(monthly, reference)


def test_incumbent_curves_name_their_ordering() -> None:
    cells = _cell_table()
    default = build_capacity_curves(cells)
    named = build_capacity_curves(cells, fixed_ordering="own_mean_usable_fraction")
    explicit = build_capacity_curves(cells, fixed_ordering=build_fixed_cell_ordering(cells))
    pd.testing.assert_frame_equal(default, named)
    pd.testing.assert_frame_equal(default, explicit)
    with pytest.raises(ValueError, match="unknown fixed ordering 'virgin'"):
        build_capacity_curves(cells, fixed_ordering="virgin")
    with pytest.raises(TypeError, match="ordering table"):
        build_capacity_curves(cells, fixed_ordering=None)


def test_incumbent_curve_export_refuses_a_missing_ordering_path(tmp_path) -> None:
    with pytest.raises(TypeError, match="None is not accepted"):
        export_wet_usable_capacity_curves([], tmp_path / "curves", fixed_ordering_path=None)
    assert not (tmp_path / "curves").exists()


SAMPLE = {
    "gcm": "GFDL-ESM4",
    "scenario": "ssp585",
    "year": 2050,
    "season": "JJA",
    "warming_level": 3.0,
    "archetype": "li_coal_500mw_once_through",
}


def _cells(population: str, rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "population": population,
                "country": "AAA",
                "grid_lat": 0.25,
                "grid_lon": 0.25 + 0.5 * i,
                "withdrawal_loss_mw": 0.0,
                **SAMPLE,
                **row,
            }
            for i, row in enumerate(rows)
        ]
    )


def _candidates() -> pd.DataFrame:
    return _cells(
        "candidate",
        [
            {
                "region_id": "R12_MEA",
                "cell_id": "up",
                "nameplate_mw": 100.0,
                "thermal_loss_mw": 0.0,
                "usable_capacity_mw": 100.0,
            },
            {
                "region_id": "R12_MEA",
                "cell_id": "down",
                "nameplate_mw": 100.0,
                "thermal_loss_mw": 0.0,
                "usable_capacity_mw": 100.0,
            },
        ],
    )


def _water(cells: pd.DataFrame, ceilings: list[float]) -> pd.DataFrame:
    keys = [
        "population", "region_id", "country", "cell_id", "grid_lat", "grid_lon",
        "archetype", "gcm", "scenario", "year", "season", "warming_level",
    ]
    return cells[keys].assign(water_ceiling_mw=ceilings)


def _topology() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "population": ["candidate", "candidate"],
            "scenario": ["ssp585", "ssp585"],
            "cell_id": ["up", "down"],
            "downstream_cell_id": ["down", None],
        }
    )


def _final(curves: pd.DataFrame) -> pd.Series:
    return curves.sort_values("point_index").iloc[-1]


def test_incumbent_withdrawal_reduces_candidate_usable_capacity():
    candidates = _candidates()
    water = _water(candidates, [100.0, 200.0])
    # An existing site sits on the upstream cell and takes 60 MW worth of water
    # (nameplate 60, no thermal loss). An upstream water limit of 100 leaves 40
    # for potential sites there; the downstream water limit of 200 is reduced by
    # the existing site's 60 of cumulative upstream use to 140.
    incumbents = _cells(
        "incumbent",
        [
            {
                "region_id": "R12_MEA",
                "cell_id": "up",
                "nameplate_mw": 60.0,
                "thermal_loss_mw": 0.0,
                "usable_capacity_mw": 60.0,
            }
        ],
    )

    virgin = _final(
        build_potential_site_sample_curves_with_river_network(
            candidates, water, _topology(), capacity_grid_points=3,
            fixed_ordering=build_fixed_cell_ordering(candidates),
        )
    )
    joint = _final(
        build_potential_site_sample_curves_with_river_network(
            candidates,
            water,
            _topology(),
            capacity_grid_points=3,
            fixed_ordering=build_fixed_cell_ordering(candidates),
            existing_site_samples=incumbents,
        )
    )

    # Full-flow: 200 MW placed, upstream capped at 100, basin capped at 200 -> 200.
    assert virgin["usable_capacity_mw"] == pytest.approx(200.0)
    # Joint: upstream water left 40, downstream water left 140 -> 40 + 100 = 140.
    assert joint["usable_capacity_mw"] == pytest.approx(140.0)
    assert joint["capacity_axis_mw"] == pytest.approx(virgin["capacity_axis_mw"])
    assert joint["usable_fraction"] < virgin["usable_fraction"]


def test_no_incumbent_overlap_leaves_the_curve_unchanged():
    candidates = _candidates()
    water = _water(candidates, [100.0, 200.0])
    elsewhere = _cells(
        "incumbent",
        [
            {
                "region_id": "R12_MEA",
                "cell_id": "unrelated",
                "nameplate_mw": 500.0,
                "thermal_loss_mw": 0.0,
                "usable_capacity_mw": 500.0,
            }
        ],
    )

    virgin = _final(
        build_potential_site_sample_curves_with_river_network(
            candidates, water, _topology(), capacity_grid_points=3,
            fixed_ordering=build_fixed_cell_ordering(candidates),
        )
    )
    joint = _final(
        build_potential_site_sample_curves_with_river_network(
            candidates,
            water,
            _topology(),
            capacity_grid_points=3,
            fixed_ordering=build_fixed_cell_ordering(candidates),
            existing_site_samples=elsewhere,
        )
    )
    assert joint["usable_capacity_mw"] == pytest.approx(virgin["usable_capacity_mw"])


def test_upstream_incumbent_in_another_region_still_reduces_headroom():
    # Cross-region is the case a per-region subtraction would miss: the two
    # sets of cells (potential and existing sites) share a river across an R12
    # boundary.
    candidates = _candidates()
    candidates.loc[candidates["cell_id"] == "down", "region_id"] = "R12_SAS"
    water = _water(candidates, [100.0, 200.0])
    incumbents = _cells(
        "incumbent",
        [
            {
                "region_id": "R12_MEA",
                "cell_id": "up",
                "nameplate_mw": 80.0,
                "thermal_loss_mw": 0.0,
                "usable_capacity_mw": 80.0,
            }
        ],
    )

    curves = build_potential_site_sample_curves_with_river_network(
        candidates, water, _topology(), capacity_grid_points=3, existing_site_samples=incumbents,
        fixed_ordering=build_fixed_cell_ordering(candidates),
    )
    downstream = curves.loc[curves["region_id"] == "R12_SAS"].sort_values("point_index")
    # Downstream region places 100 MW against a water limit of 200 - 80 = 120, so
    # it is unconstrained, but the water limit it saw must have been reduced.
    assert downstream.iloc[-1]["usable_capacity_mw"] == pytest.approx(100.0)

    heavy = candidates.copy()
    heavy.loc[heavy["cell_id"] == "down", "nameplate_mw"] = 300.0
    heavy.loc[heavy["cell_id"] == "down", "usable_capacity_mw"] = 300.0
    curves_heavy = build_potential_site_sample_curves_with_river_network(
        heavy, _water(heavy, [100.0, 200.0]), _topology(), capacity_grid_points=3,
        existing_site_samples=incumbents, fixed_ordering=build_fixed_cell_ordering(heavy),
    )
    down_heavy = curves_heavy.loc[curves_heavy["region_id"] == "R12_SAS"].sort_values("point_index")
    # 300 MW placed against a remaining water limit of 120 rather than the
    # full-flow 200.
    assert down_heavy.iloc[-1]["usable_capacity_mw"] == pytest.approx(120.0)


def test_demand_uses_post_thermal_capacity_and_sums_within_a_cell():
    candidates = _candidates()
    incumbents = _cells(
        "incumbent",
        [
            {
                "region_id": "R12_MEA",
                "cell_id": "up",
                "nameplate_mw": 100.0,
                "thermal_loss_mw": 30.0,
                "usable_capacity_mw": 70.0,
            },
            {
                "region_id": "R12_MEA",
                "cell_id": "up",
                "nameplate_mw": 50.0,
                "thermal_loss_mw": 10.0,
                "usable_capacity_mw": 40.0,
            },
        ],
    )
    attached = attach_existing_site_water_demand(candidates, incumbents)
    by_cell = attached.set_index("cell_id")[EXISTING_SITE_DEMAND_COLUMN]
    # Only capacity surviving the thermal constraint reaches the water: 70 + 40.
    assert by_cell.loc["up"] == pytest.approx(110.0)
    assert by_cell.loc["down"] == pytest.approx(0.0)


def test_missing_join_keys_raise_rather_than_silently_dropping_demand():
    candidates = _candidates()
    incumbents = _cells(
        "incumbent",
        [
            {
                "region_id": "R12_MEA",
                "cell_id": "up",
                "nameplate_mw": 60.0,
                "thermal_loss_mw": 0.0,
                "usable_capacity_mw": 60.0,
            }
        ],
    ).drop(columns=["season"])
    with pytest.raises(ValueError, match="missing join keys"):
        attach_existing_site_water_demand(candidates, incumbents)


def test_joint_allocation_never_exceeds_the_virgin_result():
    # Monotonicity: subtracting a non-negative demand cannot raise the water left.
    rng = np.random.default_rng(916)
    candidates = _candidates()
    for _ in range(20):
        ceilings = [float(rng.uniform(10, 300)), float(rng.uniform(10, 300))]
        demand = float(rng.uniform(0, 150))
        water = _water(candidates, ceilings)
        incumbents = _cells(
            "incumbent",
            [
                {
                    "region_id": "R12_MEA",
                    "cell_id": "up",
                    "nameplate_mw": demand,
                    "thermal_loss_mw": 0.0,
                    "usable_capacity_mw": demand,
                }
            ],
        )
        virgin = _final(
            build_potential_site_sample_curves_with_river_network(
                candidates, water, _topology(), capacity_grid_points=3,
                fixed_ordering=build_fixed_cell_ordering(candidates),
            )
        )
        joint = _final(
            build_potential_site_sample_curves_with_river_network(
                candidates, water, _topology(), capacity_grid_points=3,
                existing_site_samples=incumbents,
                fixed_ordering=build_fixed_cell_ordering(candidates),
            )
        )
        assert joint["usable_capacity_mw"] <= virgin["usable_capacity_mw"] + 1e-9


ARCHETYPE = "li_coal_500mw_once_through"
ORDERING_SAMPLE = {
    "region_id": "R12_MEA",
    "country": "AAA",
    "grid_lat": 0.25,
    "archetype": ARCHETYPE,
    "gcm": "GFDL-ESM4",
    "scenario": "historical",
    "year": 2000,
    "season": "JJA",
    "warming_level": "REF-PD",
    "withdrawal_loss_mw": 0.0,
}
CEILING_KEYS = [
    "population",
    "region_id",
    "country",
    "cell_id",
    "grid_lat",
    "grid_lon",
    "archetype",
    "gcm",
    "scenario",
    "year",
    "season",
    "warming_level",
]


def _ordering_cells(population: str, rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **ORDERING_SAMPLE,
                "population": population,
                "grid_lon": 0.25 + 0.5 * i,
                "nameplate_mw": 100.0,
                "thermal_loss_mw": 100.0 - row.get("usable_capacity_mw", 100.0),
                **row,
            }
            for i, row in enumerate(rows)
        ]
    )


def _ordering_water(cells: pd.DataFrame, ceilings: list[float]) -> pd.DataFrame:
    return cells[CEILING_KEYS].assign(water_ceiling_mw=ceilings)


def _ordering_topology(downstream: dict[str, str | None]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "population": "candidate",
            "scenario": ORDERING_SAMPLE["scenario"],
            "cell_id": list(downstream),
            "downstream_cell_id": list(downstream.values()),
        }
    )


def _incumbent(cell_id: str, demand_mw: float, **overrides) -> pd.DataFrame:
    # Post-thermal demand equals nameplate when there is no thermal loss.
    return _ordering_cells(
        "incumbent",
        [{"cell_id": cell_id, "nameplate_mw": demand_mw, "thermal_loss_mw": 0.0}],
    ).assign(**overrides)


def _ranked(ordering: pd.DataFrame) -> list[tuple[str, float]]:
    ordered = ordering.sort_values("fixed_order")
    return list(
        zip(
            ordered["cell_id"],
            ordered["ordering_usable_fraction"].round(12),
            strict=True,
        )
    )


def test_fleet_cell_ranks_on_the_water_its_incumbent_leaves():
    candidates = _ordering_cells(
        "candidate",
        [
            {"cell_id": "fleet", "usable_capacity_mw": 90.0},
            {"cell_id": "open", "usable_capacity_mw": 70.0},
        ],
    )
    water = _ordering_water(candidates, [100.0, 100.0])
    river_network = _ordering_topology({"fleet": None, "open": None})
    # The existing site takes 60 of the fleet cell's 100, leaving 40: the fleet
    # cell keeps min(90, 40) = 40 (0.4) and the open cell keeps its 70 (0.7).
    ordering = build_residual_water_cell_ordering(
        candidates, water, river_network, existing_site_samples=_incumbent("fleet", 60.0)
    )
    assert _ranked(ordering) == [("open", 0.7), ("fleet", 0.4)]
    # On full-flow water the fleet cell ranks first.
    assert _ranked(build_fixed_cell_ordering(candidates)) == [
        ("fleet", 0.9),
        ("open", 0.7),
    ]


def test_upstream_incumbent_lowers_the_downstream_cell_it_drains():
    candidates = _ordering_cells(
        "candidate",
        [
            {"cell_id": "up", "usable_capacity_mw": 90.0},
            {"cell_id": "down", "usable_capacity_mw": 100.0},
            {"cell_id": "side", "usable_capacity_mw": 50.0},
        ],
    )
    water = _ordering_water(candidates, [100.0, 120.0, 1000.0])
    river_network = _ordering_topology({"up": "down", "down": None, "side": None})
    # An 80 MW existing site on `up` uses 80 of up's water limit (100 -> 20) and,
    # routed downstream, 80 of down's cumulative water limit (120 -> 40). Scores:
    # up min(90, 20) = 20, down min(100, 40) = 40, side 50. Charging the existing
    # site only at its own cell would leave down at 100 and ranked first.
    ordering = build_residual_water_cell_ordering(
        candidates, water, river_network, existing_site_samples=_incumbent("up", 80.0)
    )
    assert _ranked(ordering) == [("side", 0.5), ("down", 0.4), ("up", 0.2)]


def test_charge_is_matched_to_the_incumbents_own_climate_sample():
    jja = _ordering_cells(
        "candidate",
        [
            {"cell_id": "fleet", "usable_capacity_mw": 90.0},
            {"cell_id": "open", "usable_capacity_mw": 50.0},
        ],
    )
    candidates = pd.concat([jja, jja.assign(season="DJF")], ignore_index=True)
    water = _ordering_water(candidates, [100.0, 100.0, 100.0, 100.0])
    river_network = _ordering_topology({"fleet": None, "open": None})
    # The existing site draws its full 100 MW in JJA and nothing in DJF (all
    # thermal loss). The fleet cell scores 0 in JJA and 90 in DJF, a mean of 0.45, behind
    # the open cell's 0.5. Charging DJF with the JJA draw would give 0.0; capping
    # the season-mean usable (90) by the season-mean water left (50) would give 0.5
    # and, on the cell_id tie-break, rank the fleet cell first.
    incumbents = pd.concat(
        [
            _incumbent("fleet", 100.0),
            _incumbent("fleet", 100.0, season="DJF", thermal_loss_mw=100.0),
        ],
        ignore_index=True,
    )
    ordering = build_residual_water_cell_ordering(
        candidates, water, river_network, existing_site_samples=incumbents
    )
    assert _ranked(ordering) == [("open", 0.5), ("fleet", 0.45)]


def test_without_incumbents_the_virgin_ordering_is_reproduced_exactly():
    # Site evaluation water limits are stored at float32 precision, so a
    # water-limited cell's water limit can sit a rounding step below the kernel's
    # own usable capacity. Capping usable capacity by that limit would break the
    # exact tie between `a` and `b` and move `b` ahead; charging only what
    # existing sites remove must not.
    candidates = _ordering_cells(
        "candidate",
        [
            {"cell_id": "a", "usable_capacity_mw": 100.0},
            {"cell_id": "b", "usable_capacity_mw": 100.0},
            {"cell_id": "c", "usable_capacity_mw": 60.0},
        ],
    )
    water = _ordering_water(candidates, [float(np.float32(100.0 - 5e-6)), 100.0, np.nan])
    river_network = _ordering_topology({"a": None, "b": None, "c": None})
    ordering = build_residual_water_cell_ordering(
        candidates, water, river_network, existing_site_samples=None
    )
    assert _ranked(ordering) == [("a", 1.0), ("b", 1.0), ("c", 0.6)]
    pd.testing.assert_frame_equal(ordering, build_fixed_cell_ordering(candidates))
