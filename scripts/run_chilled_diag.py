"""Single-combo CHILLED diagnostic runner.

Runs one (GCM, scenario, building_type, urt, vintage) combo through
``run_chilled_gridded_ei`` with stage-by-stage instrumentation and a
tmp-path + atomic rename write protocol.

Designed to drive a bounded memory sweep on UniCC: dask scheduler,
``num_workers`` and input-time chunk are all controlled by env vars so
the sbatch harness can vary them without editing this file. A single
background sampler thread snapshots ``/proc/self/status`` every second
so peak RSS and its stage get attributed from the timeline without
depending on SLURM ``sacct`` polling.

Exit codes:
  0  tmp write succeeded, at least one sampled cell is finite, renamed
  2  tmp write succeeded but contained no finite cells — not renamed
  3  to_netcdf raised — tmp file removed, no rename
"""

from __future__ import annotations

import gc
import os
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np

_T0 = time.monotonic()
_STAGE: list[tuple[str, float, float]] = []  # (label, t, rss_mb)
_PEAK_LOCK = threading.Lock()
_PEAK = {"rss_mb": 0.0, "t": 0.0, "last_stage": "init",
         "threads_peak": 0, "threads_peak_t": 0.0, "threads_peak_stage": "init"}


def _read_rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


def _count_threads() -> int:
    try:
        return len(list(Path("/proc/self/task").iterdir()))
    except OSError:
        return -1


def _sampler(interval: float, stop: threading.Event) -> None:
    last_threads_logged = 0
    while not stop.is_set():
        rss = _read_rss_mb()
        nthreads = _count_threads()
        now = time.monotonic() - _T0
        cur_stage = _STAGE[-1][0] if _STAGE else "init"
        with _PEAK_LOCK:
            if rss > _PEAK["rss_mb"]:
                _PEAK["rss_mb"] = rss
                _PEAK["t"] = now
                _PEAK["last_stage"] = cur_stage
            if nthreads > _PEAK["threads_peak"]:
                _PEAK["threads_peak"] = nthreads
                _PEAK["threads_peak_t"] = now
                _PEAK["threads_peak_stage"] = cur_stage
        # Log any transition that crosses a log2 boundary, so we can see thread
        # proliferation without spamming.
        if nthreads != last_threads_logged and (
            nthreads >= 2 * max(last_threads_logged, 1)
            or nthreads <= last_threads_logged // 2
        ):
            print(f"[diag {now:7.1f}s] THREADS  live={nthreads}  rss={rss:7.0f} MB  stage={cur_stage!r}", flush=True)
            last_threads_logged = nthreads
        stop.wait(interval)


def stage(label: str, detail: str = "") -> None:
    gc.collect()
    rss = _read_rss_mb()
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    elapsed = time.monotonic() - _T0
    _STAGE.append((label, elapsed, rss))
    suffix = f"  | {detail}" if detail else ""
    print(f"[diag {elapsed:7.1f}s] rss={rss:7.0f} MB  maxrss={maxrss:7.0f} MB  {label}{suffix}", flush=True)


def describe_dataset(ds) -> str:
    import xarray as xr
    if not isinstance(ds, xr.Dataset):
        return repr(type(ds))
    bits = [f"dims={dict(ds.sizes)}", f"data_vars={list(ds.data_vars)}"]
    first = next(iter(ds.data_vars), None)
    if first is not None:
        arr = ds[first]
        if hasattr(arr.data, "chunks"):
            bits.append(f"{first}.chunks={arr.chunks}")
        else:
            bits.append(f"{first}.numpy_shape={arr.shape}")
    return "  ".join(bits)


def describe_array(da) -> str:
    bits = [f"dims={da.dims}", f"shape={tuple(da.shape)}"]
    if hasattr(da.data, "chunks"):
        bits.append(f"chunks={da.chunks}")
    return "  ".join(bits)


def verify_tmp(path: Path) -> dict:
    import netCDF4 as nc
    with nc.Dataset(path, "r") as ds:
        dim_sizes = {k: len(v) for k, v in ds.dimensions.items()}
        var_names = list(ds.variables)
        var = ds.variables["cooling_energy_total"]
        var.set_auto_mask(False)
        sample = var[0, 0, ::5000]
        finite_sample = int(np.isfinite(sample).sum())
        sample_size = int(sample.size)
        # Also scan every (season, mid-year, ::N) stride
        season_finite = {}
        for s in range(var.shape[0]):
            y_mid = var.shape[1] // 2
            row = var[s, y_mid, ::1000]
            season_finite[s] = int(np.isfinite(row).sum())
    return {
        "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
        "dims": dim_sizes,
        "var_count": len(var_names),
        "sample_finite": finite_sample,
        "sample_size": sample_size,
        "season_mid_finite": season_finite,
    }


def main() -> int:
    # Required env
    gcm = os.environ["CHILLED_GCM"]
    scenario = os.environ["CHILLED_SCENARIO"]
    year_start = int(os.environ["CHILLED_YEAR_START"])
    year_end = int(os.environ["CHILLED_YEAR_END"])
    building_type = os.environ["CHILLED_BUILDING_TYPE"]
    urt = os.environ["CHILLED_URT"]
    vintage = os.environ["CHILLED_VINTAGE"]

    # Tunables (all optional)
    dask_scheduler = os.environ.get("DASK_SCHEDULER", "synchronous")
    dask_workers = int(os.environ.get("DASK_NUM_WORKERS", "1"))
    time_chunk = int(os.environ.get("CHILLED_TIME_CHUNK", "90"))
    force = os.environ.get("CHILLED_FORCE", "0") == "1"

    output_root = Path(os.environ.get(
        "CHILLED_OUTPUT_ROOT",
        "/pdrive/projects/watxene/IMPACTS/buildings/seasonal_diag",
    ))
    output_root.mkdir(parents=True, exist_ok=True)

    tag = f"{gcm.replace('-', '_').lower()}_{scenario}_{building_type}_{urt}_{vintage}_{year_start}_{year_end}"
    final_path = output_root / f"seasonal_{tag}.nc"
    tmp_path = final_path.with_suffix(".nc.tmp")

    print(f"[diag] target final: {final_path}", flush=True)
    print(f"[diag] scheduler={dask_scheduler}  num_workers={dask_workers}  time_chunk={time_chunk}", flush=True)
    if final_path.exists() and not force:
        print(f"[diag] final already exists; remove it or set CHILLED_FORCE=1 to overwrite. Exiting 0.", flush=True)
        return 0
    if tmp_path.exists():
        tmp_path.unlink()

    import dask
    dask.config.set(scheduler=dask_scheduler, num_workers=dask_workers)
    try:
        affinity_n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_n = -1
    stage(
        "dask configured",
        f"scheduler={dask_scheduler} num_workers={dask_workers} "
        f"os.cpu_count={os.cpu_count()} sched_affinity={affinity_n} "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')} "
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', 'unset')} "
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', 'unset')} "
        f"threads_at_start={_count_threads()}"
    )

    stop = threading.Event()
    sampler = threading.Thread(target=_sampler, args=(1.0, stop), daemon=True)
    sampler.start()

    # Imports deferred so `diag configured` shows a clean pre-import baseline
    from impacts.buildings import chilled as chilled_mod
    from impacts.buildings.chilled import (
        ChilledEnergyAssumptions,
        build_chilled_parameter_grids,
        load_ewembi_solar_climatology,
        load_message_ix_buildings_archetypes,
        load_message_ix_buildings_parametric_runs,
        load_r12_region_raster,
        run_chilled_gridded_ei,
    )
    from impacts.isimip import ISIMIPVariable, isimip3b_paths, load_isimip_variable

    # Allow override of kernel-entry entity/time chunk without editing chilled.py
    if (ec := os.environ.get("CHILLED_KERNEL_ENTITY_CHUNK")):
        chilled_mod._KERNEL_ENTITY_CHUNK = int(ec)
    if (tc := os.environ.get("CHILLED_KERNEL_TIME_CHUNK")):
        chilled_mod._KERNEL_TIME_CHUNK = int(tc)
    stage(
        "imports done",
        f"_KERNEL_ENTITY_CHUNK={chilled_mod._KERNEL_ENTITY_CHUNK} "
        f"_KERNEL_TIME_CHUNK={chilled_mod._KERNEL_TIME_CHUNK}"
    )

    watxene_root = Path(os.environ["WATXENE_PATH"])
    isimip_paths = isimip3b_paths(watxene_root=watxene_root)

    stage(f"loading climate {gcm}/{scenario} [{year_start}-{year_end}]")
    tas = load_isimip_variable(
        ISIMIPVariable(name="tas", product="climate", frequency="daily"),
        gcm, scenario, isimip_paths,
        year_range=(year_start, year_end),
        chunks={"time": time_chunk, "lat": -1, "lon": -1},
        target_frequency="native",
    ).astype("float32", keep_attrs=True)
    climate = tas.to_dataset(name="tas").sel(time=slice(f"{year_start}", f"{year_end}"))
    stage("climate loaded", describe_dataset(climate))

    archetypes = load_message_ix_buildings_archetypes()
    solar = load_ewembi_solar_climatology()
    region_raster = load_r12_region_raster()
    parametric = load_message_ix_buildings_parametric_runs()
    assumptions = ChilledEnergyAssumptions.from_parametric_row(parametric.iloc[0])
    stage("params+region loaded", f"run={parametric.iloc[0]['name_run']!r}")

    parameters = build_chilled_parameter_grids(
        region_raster, archetypes, solar,
        building_type=building_type, urt=urt, vintage=vintage,
    )
    stage("parameter grids built", describe_dataset(parameters))

    year_batch = int(os.environ.get("CHILLED_YEAR_BATCH", "0"))
    import shutil
    import xarray as xr
    batch_dir = None
    if year_batch > 0:
        # Incremental-write batching: each batch is computed, cast to float32,
        # reset_index'd, and written to its own tmp netcdf, then released from
        # memory. This bounds RSS+cache to one-batch-worth regardless of horizon.
        import gc
        all_years = list(range(year_start, year_end + 1))
        batch_dir = tmp_path.with_suffix(".batches.d")
        # Nuke any stale batch dir from a prior failed run so open_mfdataset at
        # the end doesn't pick up obsolete files.
        if batch_dir.exists():
            shutil.rmtree(batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_files: list[Path] = []
        for i in range(0, len(all_years), year_batch):
            batch_years = all_years[i:i + year_batch]
            b0, b1 = batch_years[0], batch_years[-1]
            # Include Dec of prev year for DJF continuity (except first batch
            # which is always partial on DJF b0 by construction).
            slice_start = f"{b0 - 1}-12-01" if b0 > year_start else f"{b0}-01-01"
            slice_end = f"{b1}-12-31"
            batch_climate = climate.sel(time=slice(slice_start, slice_end))
            stage(f"batch {b0}-{b1} start  time_slice=({slice_start},{slice_end})")
            batch_result = run_chilled_gridded_ei(
                batch_climate, parameters, assumptions=assumptions, duration="seasonal",
            ).load().sel(year=slice(b0, b1))
            batch_result = batch_result.astype({v: "float32" for v in batch_result.data_vars})
            batch_file = batch_dir / f"batch_{b0}_{b1}.nc"
            batch_result.reset_index("entity").to_netcdf(batch_file)
            batch_files.append(batch_file)
            del batch_result
            gc.collect()
            stage(f"batch {b0}-{b1} written  {batch_file.name}  ({batch_file.stat().st_size / 1e6:.0f} MB)")
        # Lazy-combine per-batch files. The final to_netcdf below streams bytes
        # without re-running the kernel.
        reset = xr.open_mfdataset(batch_files, combine="by_coords", chunks={})
        stage("batches combined (lazy)", describe_dataset(reset))
    else:
        result = run_chilled_gridded_ei(
            climate, parameters, assumptions=assumptions, duration="seasonal",
        )
        stage("seasonal graph constructed", describe_dataset(result))

        result32 = result.astype({var: "float32" for var in result.data_vars})
        stage("cast to float32")

        # Coalesce the 1-year × 1-season chunks that aggregate_seasonal emits into
        # a single (season, year) block per entity-stripe. Without this the output
        # has 4 * year * 26 = O(9_000+) chunks per variable and the dask scheduler
        # graph state itself dominates memory at production horizon.
        coalesce = os.environ.get("CHILLED_COALESCE_CHUNKS", "1") == "1"
        if coalesce:
            entity_chunk = int(os.environ.get("CHILLED_ENTITY_CHUNK", "10000"))
            result32 = result32.chunk({"season": -1, "year": -1, "entity": entity_chunk})
            stage("coalesced chunks", describe_dataset(result32))

        reset = result32.reset_index("entity")
        stage("reset_index entity", describe_dataset(reset))

    stage(f"to_netcdf start  tmp={tmp_path}")
    try:
        reset.to_netcdf(tmp_path)
    except BaseException as exc:  # noqa: BLE001 — want to see KeyboardInterrupt/OOM as well
        stage(f"to_netcdf FAILED: {type(exc).__name__}: {exc}")
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    stage("to_netcdf returned")

    info = verify_tmp(tmp_path)
    stage("verify tmp", f"{info}")

    any_finite = info["sample_finite"] > 0 or any(v > 0 for v in info["season_mid_finite"].values())
    if any_finite:
        tmp_path.rename(final_path)
        stage(f"renamed -> {final_path.name}")
        if batch_dir is not None and batch_dir.exists():
            shutil.rmtree(batch_dir, ignore_errors=True)
            stage(f"cleaned up per-batch tmp dir {batch_dir}")
        rc = 0
    else:
        stage("ABORT: tmp file contained zero finite values — not renamed")
        rc = 2

    stop.set()
    sampler.join(timeout=2.0)
    with _PEAK_LOCK:
        peak = dict(_PEAK)
    print(
        f"[diag] PEAK RSS (sampled): {peak['rss_mb']:.0f} MB at t={peak['t']:.1f}s during stage {peak['last_stage']!r}",
        flush=True,
    )
    print(
        f"[diag] PEAK THREADS (live): {peak['threads_peak']} at t={peak['threads_peak_t']:.1f}s during stage {peak['threads_peak_stage']!r}",
        flush=True,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
