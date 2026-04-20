"""Production CHILLED matrix runner.

One SLURM array task runs one (GCM, scenario, building_type, urt, vintage)
combo through ``run_chilled_gridded_ei`` using the #421-verified recipe:
10-year outer year-batching with per-batch to_netcdf to a per-task tmp dir,
``xr.open_mfdataset`` coalesce to ``<final>.tmp``, finite-verify, atomic rename.

Matrix layout (SLURM_ARRAY_TASK_ID in [0, 400)):
  5 GCMs × 4 scenarios × 20 archetype combos
  gcm_idx      = ARRAY_INDEX // 80
  scenario_idx = (ARRAY_INDEX % 80) // 20
  combo_idx    = ARRAY_INDEX % 20
  bt_idx       = combo_idx // 10
  urt_idx      = (combo_idx % 10) // 5
  vintage_idx  = combo_idx % 5

Outputs: ``$CHILLED_OUTPUT_ROOT/seasonal_<tag>.nc`` where
``<tag> = <gcm>_<scenario>_<bt>_<urt>_<vintage>_<year_start>_<year_end>``.
Default root: /pdrive/projects/watxene/IMPACTS/buildings/seasonal.

Exit codes:
  0  success, or skip-if-exists when final already present
  2  verify found no finite cells -- tmp removed, no rename
  3  to_netcdf raised -- tmp removed, no rename
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

GCMS = ("GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL")
SCENARIOS = ("historical", "ssp126", "ssp245", "ssp370")
SCENARIO_YEARS = {
    "historical": (1850, 2014),
    "ssp126": (2015, 2100),
    "ssp245": (2015, 2100),
    "ssp370": (2015, 2100),
}
BUILDING_TYPES = ("sfh", "mfh")
URTS = ("rur", "urb")
VINTAGES = ("s1", "s2", "s3", "s51", "s52")

YEAR_BATCH = 10
TIME_CHUNK = 90

_T0 = time.monotonic()
_PEAK = {"rss_mb": 0.0, "t": 0.0, "stage": "init"}
_PEAK_LOCK = threading.Lock()
_STAGE = "init"


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


def _sampler(interval: float, stop: threading.Event) -> None:
    while not stop.is_set():
        rss = _rss_mb()
        now = time.monotonic() - _T0
        with _PEAK_LOCK:
            if rss > _PEAK["rss_mb"]:
                _PEAK["rss_mb"] = rss
                _PEAK["t"] = now
                _PEAK["stage"] = _STAGE
        stop.wait(interval)


def stage(label: str, detail: str = "") -> None:
    global _STAGE
    gc.collect()
    _STAGE = label
    rss = _rss_mb()
    elapsed = time.monotonic() - _T0
    suffix = f"  | {detail}" if detail else ""
    print(f"[matrix {elapsed:7.1f}s] rss={rss:7.0f} MB  {label}{suffix}", flush=True)


def dispatch(array_index: int) -> dict:
    if not (0 <= array_index < 400):
        raise ValueError(f"SLURM_ARRAY_TASK_ID={array_index} outside [0, 400)")
    gcm = GCMS[array_index // 80]
    scenario = SCENARIOS[(array_index % 80) // 20]
    combo_idx = array_index % 20
    building_type = BUILDING_TYPES[combo_idx // 10]
    urt = URTS[(combo_idx % 10) // 5]
    vintage = VINTAGES[combo_idx % 5]
    year_start, year_end = SCENARIO_YEARS[scenario]
    return {
        "gcm": gcm,
        "scenario": scenario,
        "building_type": building_type,
        "urt": urt,
        "vintage": vintage,
        "year_start": year_start,
        "year_end": year_end,
    }


def verify_tmp(path: Path) -> dict:
    import xarray as xr
    with xr.open_dataset(path) as ds:
        dim_sizes = dict(ds.sizes)
        var = ds["cooling_energy_total"]
        max_val = float(var.max(skipna=True).values)
    any_finite = not np.isnan(max_val)
    return {
        "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
        "dims": dim_sizes,
        "max_cooling_energy": max_val,
        "any_finite": any_finite,
    }


def remove_batch_dir(batch_dir: Path) -> None:
    """Remove a per-task batch directory and raise if it persists."""

    if not batch_dir.exists():
        return

    shutil.rmtree(batch_dir)
    for _ in range(5):
        if not batch_dir.exists():
            return
        time.sleep(0.2)
    raise RuntimeError(f"batch directory persisted after cleanup: {batch_dir}")


def main() -> int:
    array_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    combo = dispatch(array_index)

    gcm = combo["gcm"]
    scenario = combo["scenario"]
    building_type = combo["building_type"]
    urt = combo["urt"]
    vintage = combo["vintage"]
    year_start = combo["year_start"]
    year_end = combo["year_end"]

    force = os.environ.get("CHILLED_FORCE", "0") == "1"
    output_root = Path(os.environ.get(
        "CHILLED_OUTPUT_ROOT",
        "/pdrive/projects/watxene/IMPACTS/buildings/seasonal",
    ))
    output_root.mkdir(parents=True, exist_ok=True)

    tag = (
        f"{gcm.replace('-', '_').lower()}_{scenario}"
        f"_{building_type}_{urt}_{vintage}_{year_start}_{year_end}"
    )
    final_path = output_root / f"seasonal_{tag}.nc"
    tmp_path = final_path.with_suffix(".nc.tmp")
    batch_dir = tmp_path.with_suffix(".batches.d")

    print(f"[matrix] array_index={array_index}  combo={combo}", flush=True)
    print(f"[matrix] final={final_path}", flush=True)

    if final_path.exists() and not force:
        print("[matrix] final already exists; skipping. CHILLED_FORCE=1 to overwrite.", flush=True)
        return 0
    if tmp_path.exists():
        tmp_path.unlink()
    if batch_dir.exists():
        shutil.rmtree(batch_dir)

    import dask
    dask.config.set(scheduler="synchronous", num_workers=1)
    stage("dask configured", "scheduler=synchronous num_workers=1")

    stop = threading.Event()
    sampler = threading.Thread(target=_sampler, args=(1.0, stop), daemon=True)
    sampler.start()

    import xarray as xr

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

    stage("imports done")

    watxene_root = Path(os.environ["WATXENE_PATH"])
    isimip_paths = isimip3b_paths(watxene_root=watxene_root)

    stage(f"loading climate {gcm}/{scenario} [{year_start}-{year_end}]")
    tas = load_isimip_variable(
        ISIMIPVariable(name="tas", product="climate", frequency="daily"),
        gcm, scenario, isimip_paths,
        year_range=(year_start, year_end),
        chunks={"time": TIME_CHUNK, "lat": -1, "lon": -1},
        target_frequency="native",
    ).astype("float32", keep_attrs=True)
    climate = tas.to_dataset(name="tas").sel(time=slice(f"{year_start}", f"{year_end}"))
    stage("climate loaded", f"dims={dict(climate.sizes)}")

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
    stage("parameter grids built")

    # Year-batched incremental-write path (#421 verified recipe): each batch is
    # computed, cast to float32, reset_index'd, and written to its own tmp
    # netcdf, then released from memory. Bounds RSS+cache to one-batch worth
    # regardless of horizon.
    all_years = list(range(year_start, year_end + 1))
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_files: list[Path] = []
    for i in range(0, len(all_years), YEAR_BATCH):
        batch_years = all_years[i:i + YEAR_BATCH]
        b0, b1 = batch_years[0], batch_years[-1]
        # Include Dec of prev year for DJF continuity (except first batch which
        # is always partial on DJF b0 by construction).
        slice_start = f"{b0 - 1}-12-01" if b0 > year_start else f"{b0}-01-01"
        slice_end = f"{b1}-12-31"
        batch_climate = climate.sel(time=slice(slice_start, slice_end))
        stage(f"batch {b0}-{b1} start")
        batch_result = run_chilled_gridded_ei(
            batch_climate, parameters, assumptions=assumptions, duration="seasonal",
        ).load().sel(year=slice(b0, b1))
        batch_result = batch_result.astype({v: "float32" for v in batch_result.data_vars})
        batch_file = batch_dir / f"batch_{b0}_{b1}.nc"
        batch_result.reset_index("entity").to_netcdf(batch_file)
        batch_files.append(batch_file)
        del batch_result
        gc.collect()
        stage(f"batch {b0}-{b1} written", f"{batch_file.stat().st_size / 1e6:.0f} MB")

    combined = xr.open_mfdataset(batch_files, combine="by_coords", chunks={})
    stage("batches combined (lazy)")

    stage(f"to_netcdf start  tmp={tmp_path}")
    try:
        combined.to_netcdf(tmp_path)
    except BaseException as exc:  # noqa: BLE001 — capture OOM / KeyboardInterrupt too
        stage(f"to_netcdf FAILED: {type(exc).__name__}: {exc}")
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    finally:
        combined.close()
        stage("combined dataset closed")
    stage("to_netcdf returned")

    info = verify_tmp(tmp_path)
    stage("verify tmp", f"{info}")

    if info["any_finite"]:
        tmp_path.rename(final_path)
        stage(f"renamed -> {final_path.name}")
        remove_batch_dir(batch_dir)
        stage(f"cleaned up per-batch tmp dir {batch_dir}")
        rc = 0
    else:
        stage("ABORT: tmp file contained zero finite values -- not renamed")
        if tmp_path.exists():
            tmp_path.unlink()
        rc = 2

    stop.set()
    sampler.join(timeout=2.0)
    with _PEAK_LOCK:
        peak = dict(_PEAK)
    print(
        f"[matrix] PEAK RSS (sampled): {peak['rss_mb']:.0f} MB "
        f"at t={peak['t']:.1f}s during stage {peak['stage']!r}",
        flush=True,
    )
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        # Surface unhandled exceptions with rc=3 so SLURM records a clear
        # failure signal for the array task.
        import traceback
        traceback.print_exc()
        sys.exit(3)
