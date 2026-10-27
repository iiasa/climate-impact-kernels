"""Sharing of river water between wet-cooled plants along the river network, and the construction of that network.

The capacity curves of potential sites need, for every potential site, the next
potential site downstream. :func:`build_river_networks` walks each cell down a
DDM30-style flow-direction raster (DDM30 is a 0.5-degree river flow-direction
map) until it meets another cell of the same scenario, and
:func:`write_river_networks` names the files the way
``impacts.thermoelectric.siting.curves`` reads them back (the scenario is read
from the file name).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as pads
import xarray as xr

from impacts.thermoelectric.siting._io import (
    refuse_existing_outputs,
    staged_outputs,
)

DOWNSTREAM_COLUMN = "downstream_cell_id"


def topological_order(downstream_index: np.ndarray) -> np.ndarray:
    """Order cells so every cell comes before the cell downstream of it.

    ``downstream_index`` holds, for each cell, the array position of its
    downstream cell, or a negative number for none.

    Raises:
        ValueError: if the downstream links contain a loop.
    """

    down = np.asarray(downstream_index, dtype=int)
    pending = np.zeros(down.size, dtype=int)
    for downstream in down:
        if downstream >= 0:
            pending[downstream] += 1
    stack = [int(i) for i in np.flatnonzero(pending == 0)]
    order: list[int] = []
    while stack:
        node = stack.pop()
        order.append(node)
        downstream = int(down[node])
        if downstream >= 0:
            pending[downstream] -= 1
            if pending[downstream] == 0:
                stack.append(downstream)
    if len(order) != down.size:
        raise ValueError("downstream pointers contain a cycle")
    return np.asarray(order, dtype=int)


def river_network_capped_allocation(
    ceiling_mw: np.ndarray,
    placed_mw: np.ndarray,
    downstream_index: np.ndarray,
) -> np.ndarray:
    """Allocate placed capacity under the cumulative upstream water limits of the river network.

    ``ceiling_mw`` is each cell's water limit: the capacity (MW) that the
    cell's available river water can cool. A cell's limit caps the total
    capacity cooled by water withdrawn at that cell and at every cell upstream
    of it. ``placed_mw`` may be one-dimensional ``(n_cells,)`` or
    two-dimensional ``(n_cells, n_points)``. A NaN water limit marks a cell
    that passes water on with no limit of its own: capacity upstream of it still
    counts, but it imposes no cap.

    Cells are visited from upstream to downstream. A cell's inflow is the
    capacity placed on it plus what the cells directly upstream pass on. The
    cell passes on the smaller of its water limit and its inflow, and its
    scaling factor is that amount divided by the inflow (1 when the inflow is
    0). A cell's allocation is its placed capacity times the product of the
    scaling factors of the cell and of every cell downstream of it, so a
    limiting cell cuts every upstream contribution by the same factor.

    Returns an array with the shape of ``placed_mw``.
    """

    ceiling = np.where(
        np.isnan(ceiling_mw), np.inf, np.asarray(ceiling_mw, dtype=float)
    )
    placed = np.asarray(placed_mw, dtype=float)
    flat = placed.ndim == 1
    placed = np.atleast_2d(placed.T).T
    down = np.asarray(downstream_index, dtype=int)
    order = topological_order(down)

    inflow = placed.copy()
    usable_subtree = np.zeros_like(placed)
    for node in order:
        usable_subtree[node] = np.minimum(ceiling[node], inflow[node])
        downstream = int(down[node])
        if downstream >= 0:
            inflow[downstream] += usable_subtree[node]

    scale = np.ones_like(placed)
    for node in reversed(order):
        ratio = np.divide(
            usable_subtree[node],
            inflow[node],
            out=np.ones_like(inflow[node]),
            where=inflow[node] > 0.0,
        )
        downstream = int(down[node])
        scale[node] = ratio if downstream < 0 else ratio * scale[downstream]
    allocation = placed * scale
    return allocation[:, 0] if flat else allocation


def cumulative_upstream_usage(
    usage_mw: np.ndarray,
    downstream_index: np.ndarray,
) -> np.ndarray:
    """Sum usage over each cell and every cell upstream of it."""

    down = np.asarray(downstream_index, dtype=int)
    cumulative = np.asarray(usage_mw, dtype=float).copy()
    for node in topological_order(down):
        d = int(down[node])
        if d >= 0:
            cumulative[d] += cumulative[node]
    return cumulative


def remaining_water_limits(
    ceiling_mw: np.ndarray,
    usage_mw: np.ndarray,
    downstream_index: np.ndarray,
) -> np.ndarray:
    """Remaining water limits: each limit minus the usage summed over the cell and every cell upstream of it, floored at zero.

    NaN water limits stay NaN (cells that pass water on stay without a limit).
    """

    ceiling = np.asarray(ceiling_mw, dtype=float)
    used = cumulative_upstream_usage(usage_mw, downstream_index)
    return np.where(np.isnan(ceiling), np.nan, np.maximum(ceiling - used, 0.0))


# Flow-direction code -> (step in latitude index, step in longitude index) on a
# raster sorted by ascending latitude and longitude: clockwise from east.
FLOWDIR_STEPS = {
    1: (0, 1),
    2: (-1, 1),
    3: (-1, 0),
    4: (-1, -1),
    5: (0, -1),
    6: (1, -1),
    7: (1, 0),
    8: (1, 1),
}
TERMINAL_CODES = (0, 9)
MAX_WALK_STEPS = 1000
TOPOLOGY_FILE_SUFFIX = "wet_ucr_reach_topology.parquet"
FLOW_DIRECTION_VARIABLE = "flowdirection"


def read_flow_directions(path: str | Path) -> xr.DataArray:
    """Read the flow-direction codes of a DDM30 file into memory.

    Returns the ``flowdirection`` variable (codes on ``lat`` and ``lon``), the input
    :func:`build_river_networks` takes. The file is opened with the ``netcdf4``
    engine and closed before returning.
    """

    with xr.open_dataset(path, engine="netcdf4") as dataset:
        return dataset[FLOW_DIRECTION_VARIABLE].load()


def read_potential_site_cells(decomposition_path: str | Path) -> pd.DataFrame:
    """Read the distinct potential sites of each scenario from a loss-split table.

    The loss-split table is the parquet file that splits each cell's wet
    capacity loss into thermal loss and water-availability loss
    (``split_wet_losses``).

    Columns: ``population`` (always ``candidate``), ``scenario``, ``cell_id``,
    ``grid_lat`` and ``grid_lon`` (cell centre, degrees).

    Raises:
        ValueError: if the table holds no rows for potential sites.
    """

    dataset = pads.dataset(str(decomposition_path), format="parquet")
    table = dataset.to_table(
        columns=["population", "scenario", "cell_id", "grid_lat", "grid_lon"],
        filter=pc.field("population") == "candidate",
    )
    frame = table.to_pandas(self_destruct=True, split_blocks=True)
    if frame.empty:
        raise ValueError(
            f"no rows for potential sites (population 'candidate') in: {decomposition_path}"
        )
    return frame.drop_duplicates(["scenario", "cell_id"]).reset_index(drop=True)


def build_river_networks(
    cells: pd.DataFrame, flowdir: xr.DataArray
) -> dict[str, pd.DataFrame]:
    """Give every potential site the next potential site downstream of it, per scenario.

    Parameters:
        cells: Potential sites with ``population``, ``scenario``, ``cell_id``,
            ``grid_lat`` and ``grid_lon``, as :func:`read_potential_site_cells` returns.
        flowdir: Flow-direction codes on ``lat`` and ``lon`` coordinates (the
            ``flowdirection`` variable of the ISIMIP DDM30 file): 1 to 8 clockwise
            from east, 0 and 9 for outlets.

    Returns:
        Scenario name -> table with ``population``, ``cell_id``, ``grid_lat``,
        ``grid_lon`` and ``downstream_cell_id`` (missing for river outlets and
        for cells whose path leaves the raster before meeting another cell).
    """

    return {
        scenario: assign_downstream(
            group[["population", "cell_id", "grid_lat", "grid_lon"]]
            .drop_duplicates("cell_id")
            .reset_index(drop=True),
            flowdir,
        )
        for scenario, group in cells.groupby("scenario", sort=True)
    }


def write_river_networks(
    topologies: dict[str, pd.DataFrame],
    gcm: str,
    directory: str | Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Write one river-network parquet per scenario and return the paths, in scenario order.

    The files are named ``<gcm>_<scenario>_wet_ucr_reach_topology.parquet``.

    Raises:
        FileExistsError: if any of the files exists and ``overwrite`` is false;
            nothing is written then.
    """

    root = Path(directory)
    paths = [root / f"{gcm}_{scenario}_{TOPOLOGY_FILE_SUFFIX}" for scenario in topologies]
    if not overwrite:
        refuse_existing_outputs(paths)
    with staged_outputs(paths) as staged:
        for path, river_network in zip(paths, topologies.values(), strict=True):
            river_network.to_parquet(staged[path], index=False)
    return paths


def assign_downstream(cells: pd.DataFrame, flowdir: xr.DataArray) -> pd.DataFrame:
    """Add ``downstream_cell_id`` to one scenario's cells.

    Each cell is placed on the nearest raster cell and followed downstream until
    the path reaches another of ``cells``; ties between two raster cells go to the
    lower coordinate, and longitude wraps around the raster.

    Raises:
        ValueError: if ``cells`` lacks a ``cell_id``, ``grid_lat`` or ``grid_lon``
            column, if a ``cell_id`` has two coordinate pairs, if two cells fall on
            one raster cell, or if a path exceeds ``MAX_WALK_STEPS`` steps.
    """

    for column in ("cell_id", "grid_lat", "grid_lon"):
        if column not in cells.columns:
            raise ValueError(f"cells must define {column!r}")

    unique = cells[["cell_id", "grid_lat", "grid_lon"]].drop_duplicates()
    if unique["cell_id"].duplicated().any():
        raise ValueError("'cell_id' maps to more than one coordinate pair")

    array = flowdir.sortby(["lat", "lon"]).transpose("lat", "lon")
    raster = array.to_numpy()
    lat_axis = array["lat"].to_numpy().astype(float)
    lon_axis = array["lon"].to_numpy().astype(float)

    rows = _nearest_index(lat_axis, unique["grid_lat"].to_numpy(dtype=float))
    cols = _nearest_index(lon_axis, unique["grid_lon"].to_numpy(dtype=float))
    ids = unique["cell_id"].to_numpy()

    occupied = {}
    for cell_id, i, j in zip(ids, rows, cols, strict=True):
        key = (int(i), int(j))
        if key in occupied:
            raise ValueError(
                f"cells {occupied[key]!r} and {cell_id!r} resolve to the same raster cell"
            )
        occupied[key] = cell_id

    downstream = [
        _walk_to_table_cell(raster, occupied, int(i), int(j))
        for i, j in zip(rows, cols, strict=True)
    ]
    pointers = pd.DataFrame({"cell_id": ids, "downstream_cell_id": downstream})
    return cells.merge(pointers, on="cell_id", how="left", validate="many_to_one")


def _walk_to_table_cell(
    raster: np.ndarray,
    occupied: dict[tuple[int, int], object],
    row: int,
    col: int,
) -> object:
    n_lat, n_lon = raster.shape
    i, j = row, col
    for _ in range(MAX_WALK_STEPS):
        code = raster[i, j]
        if not np.isfinite(code) or int(code) in TERMINAL_CODES:
            return np.nan
        di, dj = FLOWDIR_STEPS[int(code)]
        i, j = i + di, (j + dj) % n_lon
        if not 0 <= i < n_lat:
            return np.nan
        if (i, j) in occupied:
            return occupied[(i, j)]
    raise ValueError("flow-direction walk exceeded the maximum path length")


def _nearest_index(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    if axis.size == 1:
        return np.zeros(values.shape, dtype=int)
    position = np.clip(np.searchsorted(axis, values), 1, axis.size - 1)
    left = axis[position - 1]
    right = axis[position]
    return np.where(
        np.abs(values - left) <= np.abs(right - values),
        position - 1,
        position,
    )
