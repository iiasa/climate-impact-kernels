"""Diagnostic: measure CHILLED kernel dask graph at production horizon.

Runs the kernel against a synthetic `attached` dataset that mimics the
stacked (time, entity) shape produced by `evaluate_gridded_dataset`, with
tas chunked as the raw ISIMIP loader ships it (time=1, entity=full), and
reports per-variable chunk count, max-chunk memory, and graph task count.

Used to quantify the memory-leak fix on #409 subtask 4: before the entry
rechunk, the graph runs to ~50M tasks at the 85-year horizon. After, it
lands around 2.6M unique tasks with max chunk ~29 MB.
"""

from __future__ import annotations

import argparse
import warnings

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from impacts.buildings.chilled import (
    _KERNEL_ENTITY_CHUNK,
    _KERNEL_TIME_CHUNK,
    ChilledEnergyAssumptions,
    _evaluate_chilled_kernel,
)


def build_attached(*, time: int, entity: int) -> xr.Dataset:
    times = pd.date_range("2015-01-01", periods=time, freq="D")
    tas = xr.DataArray(
        da.zeros((time, entity), chunks=(1, entity), dtype="float64"),
        dims=("time", "entity"),
        coords={"time": times},
    )
    attached = xr.Dataset(data_vars={"air_temperature_c": tas})
    params = {
        "gl_perc": 0.125, "gl_g": 0.85, "gl_sh": 0.8,
        "roof_area": 1.0, "roof_abs": 0.6, "u_roof": 3.3,
        "u_val": 3.5, "area_env": 3.7, "vol": 3.0,
        "ach_cl": 0.5, "ach_op": 1.5, "gn_int": 2.14,
    }
    for name, v in params.items():
        attached = attached.assign_coords({name: ("entity", np.full(entity, v))})
    for month in range(1, 13):
        attached = attached.assign_coords({
            f"solar_vertical_m{month:02d}": ("entity", np.full(entity, 120.0)),
            f"solar_horizontal_m{month:02d}": ("entity", np.full(entity, 220.0)),
        })
    return attached


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=int, default=31411, help="time length (days)")
    parser.add_argument("--entity", type=int, default=259200, help="entity size (lat*lon)")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    print(f"TIME={args.time}, ENTITY={args.entity}")
    print(f"_KERNEL_TIME_CHUNK={_KERNEL_TIME_CHUNK}, _KERNEL_ENTITY_CHUNK={_KERNEL_ENTITY_CHUNK}")

    attached = build_attached(time=args.time, entity=args.entity)
    result = _evaluate_chilled_kernel(attached, assumptions=ChilledEnergyAssumptions())

    all_keys: set[object] = set()
    for name, arr in result.data_vars.items():
        n_chunks = int(np.prod([len(c) for c in arr.chunks])) if arr.chunks else 1
        max_chunk_mb = (
            np.prod([max(c) for c in arr.chunks]) * 8 / 1e6 if arr.chunks else 0.0
        )
        graph = arr.data.__dask_graph__() if arr.chunks else {}
        all_keys.update(graph.keys())
        print(
            f"  {name}: n_chunks={n_chunks}, max_chunk_MB={max_chunk_mb:.2f},"
            f" per_var_tasks={len(graph)}"
        )
    print(f"\nUnion unique graph tasks: {len(all_keys):,}")


if __name__ == "__main__":
    main()
