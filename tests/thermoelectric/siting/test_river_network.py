"""River network routing functions used by the joint water allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from impacts.thermoelectric.siting.river_network import (
    assign_downstream,
    build_river_networks,
    cumulative_upstream_usage,
    read_flow_directions,
    read_potential_site_cells,
    remaining_water_limits,
    write_river_networks,
)


def test_residual_ceilings_subtract_cumulative_upstream_usage() -> None:
    ceiling = np.array([100.0, 120.0, np.nan])
    usage = np.array([50.0, 0.0, 0.0])
    downstream = np.array([1, 2, -1])
    residual = remaining_water_limits(ceiling, usage, downstream)
    np.testing.assert_allclose(residual[:2], [50.0, 70.0])
    assert np.isnan(residual[2])


def test_cumulative_upstream_usage_accumulates_down_the_reach() -> None:
    usage = np.array([5.0, 7.0, 1.0])
    downstream = np.array([1, 2, -1])
    np.testing.assert_allclose(
        cumulative_upstream_usage(usage, downstream), [5.0, 12.0, 13.0]
    )


def test_residual_ceilings_floor_at_zero() -> None:
    # Use exceeding the water limit must not produce negative water left, which
    # would later read as a credit rather than an exhausted river.
    ceiling = np.array([10.0, 10.0])
    usage = np.array([25.0, 0.0])
    downstream = np.array([1, -1])
    np.testing.assert_allclose(remaining_water_limits(ceiling, usage, downstream), [0.0, 0.0])


AXIS = [0.25, 0.75, 1.25]


def _raster(codes) -> xr.DataArray:
    return xr.DataArray(
        np.asarray(codes, dtype=float), dims=("lat", "lon"), coords={"lat": AXIS, "lon": AXIS}
    )


def _cells(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["cell_id", "grid_lat", "grid_lon"]).assign(population="candidate")


def _pointers(cells, codes) -> dict:
    result = assign_downstream(_cells(cells), _raster(codes))
    return dict(zip(result["cell_id"], result["downstream_cell_id"], strict=True))


def test_a_path_skips_empty_raster_cells_to_the_next_candidate_cell():
    # Row 2 (north) code 3 goes south; row 1 is empty and also south; "low" is at row 0.
    pointers = _pointers(
        [("high", 1.25, 0.25), ("low", 0.25, 0.25)], [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
    )
    assert pointers["high"] == "low"
    assert pd.isna(pointers["low"])  # south of the southernmost row: leaves the raster


def test_outlets_and_missing_codes_end_the_path():
    codes = [[0, 9, np.nan], [3, 3, 3], [3, 3, 3]]
    pointers = _pointers([("a", 0.25, 0.25), ("b", 0.25, 0.75), ("c", 0.25, 1.25)], codes)
    assert all(pd.isna(value) for value in pointers.values())


def test_longitude_wraps_around_the_raster():
    # East from the last column returns to the first; west from the first to the last.
    east = _pointers([("last", 0.75, 1.25), ("first", 0.75, 0.25)], [[3] * 3, [3, 3, 1], [3] * 3])
    assert east["last"] == "first"
    west = _pointers([("last", 0.75, 1.25), ("first", 0.75, 0.25)], [[3] * 3, [5, 3, 3], [3] * 3])
    assert west["first"] == "last"


def test_a_tie_between_two_raster_cells_goes_to_the_lower_coordinate():
    # Latitude 0.5 is 0.25 from both rows 0 and 1: the lower row, which holds "b".
    with pytest.raises(ValueError, match="resolve to the same raster cell"):
        assign_downstream(_cells([("a", 0.5, 0.25), ("b", 0.25, 0.25)]), _raster([[3] * 3] * 3))


def test_a_loop_between_empty_cells_is_refused():
    codes = [[1, 1, 5], [3, 3, 3], [3, 3, 3]]  # (0,0) -> (0,1) <-> (0,2)
    with pytest.raises(ValueError, match="exceeded the maximum path length"):
        assign_downstream(_cells([("a", 0.25, 0.25)]), _raster(codes))


def test_topologies_are_built_per_scenario_and_written_under_their_scenario_names(tmp_path):
    cells = pd.concat(
        [
            _cells([("high", 1.25, 0.25), ("low", 0.25, 0.25)]).assign(scenario="ssp126"),
            _cells([("high", 1.25, 0.25)]).assign(scenario="ssp585"),
        ]
    )
    topologies = build_river_networks(cells, _raster([[3] * 3] * 3))

    assert list(topologies) == ["ssp126", "ssp585"]
    # With "low" absent from ssp585 the path from "high" meets no cell.
    assert topologies["ssp126"].set_index("cell_id")["downstream_cell_id"].loc["high"] == "low"
    assert topologies["ssp585"]["downstream_cell_id"].isna().all()

    paths = write_river_networks(topologies, "GFDL-ESM4", tmp_path / "nested")
    assert [path.name for path in paths] == [
        "GFDL-ESM4_ssp126_wet_ucr_reach_topology.parquet",
        "GFDL-ESM4_ssp585_wet_ucr_reach_topology.parquet",
    ]
    assert pd.read_parquet(paths[0]).columns.tolist() == [
        "population", "cell_id", "grid_lat", "grid_lon", "downstream_cell_id",
    ]  # fmt: skip


def test_candidate_cells_come_from_candidate_rows_only(tmp_path):
    frame = pd.DataFrame(
        {
            "population": ["candidate", "candidate", "incumbent", "candidate"],
            "scenario": ["ssp126", "ssp126", "ssp126", "ssp585"],
            "cell_id": ["a", "a", "b", "a"],
            "grid_lat": [0.25] * 4,
            "grid_lon": [0.25] * 4,
            "year": [2040, 2041, 2040, 2040],
        }
    )
    path = tmp_path / "decomposition.parquet"
    frame.to_parquet(path, index=False)
    cells = read_potential_site_cells(path)
    assert cells[["scenario", "cell_id"]].values.tolist() == [["ssp126", "a"], ["ssp585", "a"]]

    frame[frame["population"] == "incumbent"].to_parquet(path, index=False)
    with pytest.raises(ValueError, match="no rows for potential sites"):
        read_potential_site_cells(path)


def test_topologies_are_not_written_over_an_earlier_topology(tmp_path):
    cells = _cells([("high", 1.25, 0.25), ("low", 0.25, 0.25)]).assign(scenario="ssp126")
    topologies = build_river_networks(cells, _raster([[3] * 3] * 3))
    earlier = tmp_path / "GFDL-ESM4_ssp126_wet_ucr_reach_topology.parquet"
    earlier.write_text("earlier")

    with pytest.raises(FileExistsError, match="GFDL-ESM4_ssp126"):
        write_river_networks(topologies, "GFDL-ESM4", tmp_path)
    assert earlier.read_text() == "earlier"

    # Another model's file does not collide; overwriting replaces the earlier one.
    write_river_networks(topologies, "IPSL-CM6A-LR", tmp_path)
    write_river_networks(topologies, "GFDL-ESM4", tmp_path, overwrite=True)
    assert pd.read_parquet(earlier)["cell_id"].tolist() == ["high", "low"]


def test_flow_directions_are_read_from_the_flowdirection_variable(tmp_path):
    codes = _raster([[3, 3, 3], [1, 1, 0], [9, 5, 5]])
    path = tmp_path / "ddm30.nc"
    xr.Dataset({"flowdirection": codes, "other": codes * 2}).to_netcdf(path)

    flowdir = read_flow_directions(path)

    xr.testing.assert_equal(flowdir, codes.rename("flowdirection"))
    assert flowdir.name == "flowdirection"
    # The file is closed and the values are in memory: the file can be replaced.
    path.unlink()
    assert flowdir.to_numpy().shape == (3, 3)
