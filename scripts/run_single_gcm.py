"""Run wet and dry cooling kernels on 1 GCM x 1 scenario.

Usage:
    uv run --no-sync python scripts/run_single_gcm.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from impacts.isimip import (
    DRY_COOLING_VARIABLES,
    WET_COOLING_VARIABLES,
    isimip3b_paths,
    load_isimip_dataset,
)
from impacts.runner import run_impact_matrix
from impacts.thermoelectric.dry_cooling import (
    build_dry_cooling_mapping,
    create_dry_cooling_kernel,
)
from impacts.thermoelectric.wet_cooling import (
    build_wet_cooling_mapping,
    create_wet_cooling_kernel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

GCM = "GFDL-ESM4"
SCENARIO = "ssp370"
YEAR_RANGE = (2015, 2020)
OUTPUT_ROOT = Path("output/smoke_test")

# ── Load fleet artifacts ─────────────────────────────────────────────────────

unit_cell_mapping = pd.read_parquet("data/unit_cell_mapping.parquet")
cooling_mapping = pd.read_parquet("data/unit_cooling_mapping.parquet")

paths = isimip3b_paths()

# ── Dry cooling (only needs tas — fast) ──────────────────────────────────────

print(f"\n{'='*70}")
print(f"DRY COOLING: {GCM} / {SCENARIO} / {YEAR_RANGE}")
print(f"{'='*70}")

dry_map = build_dry_cooling_mapping(unit_cell_mapping, cooling_mapping)
print(f"Dry fleet: {len(dry_map)} units, {dry_map['mw'].sum():.0f} MW")
print(f"  Engine types: {dry_map['dry_cooling_engine_type'].value_counts().to_dict()}")

t0 = time.time()
dry_paths = run_impact_matrix(
    kernel=create_dry_cooling_kernel(),
    mapping=dry_map,
    loader=lambda gcm, scenario: load_isimip_dataset(
        DRY_COOLING_VARIABLES, gcm, scenario, paths, year_range=YEAR_RANGE,
    ),
    gcms=[GCM],
    scenarios=[SCENARIO],
    output_dir=OUTPUT_ROOT / "dry",
)
dry_elapsed = time.time() - t0
print(f"Dry cooling completed in {dry_elapsed:.1f}s")
print(f"Output: {dry_paths}")

# ── Wet cooling (needs 5 variables — heavier I/O) ───────────────────────────

print(f"\n{'='*70}")
print(f"WET COOLING: {GCM} / {SCENARIO} / {YEAR_RANGE}")
print(f"{'='*70}")

wet_map = build_wet_cooling_mapping(unit_cell_mapping, cooling_mapping)
print(f"Wet fleet: {len(wet_map)} units, {wet_map['mw'].sum():.0f} MW")
print(f"  Cooling types: {wet_map['cooling_physics'].value_counts().to_dict()}")

t0 = time.time()
wet_paths = run_impact_matrix(
    kernel=create_wet_cooling_kernel(),
    mapping=wet_map,
    loader=lambda gcm, scenario: load_isimip_dataset(
        WET_COOLING_VARIABLES, gcm, scenario, paths, year_range=YEAR_RANGE,
    ),
    gcms=[GCM],
    scenarios=[SCENARIO],
    output_dir=OUTPUT_ROOT / "wet",
)
wet_elapsed = time.time() - t0
print(f"Wet cooling completed in {wet_elapsed:.1f}s")
print(f"Output: {wet_paths}")

# ── Quick sanity check on outputs ────────────────────────────────────────────

import xarray as xr

print(f"\n{'='*70}")
print("OUTPUT SUMMARY")
print(f"{'='*70}")

for label, path_list in [("Dry", dry_paths), ("Wet", wet_paths)]:
    ds = xr.open_dataset(path_list[0])
    cf = ds["capacity_factor"]
    print(f"\n{label} cooling ({path_list[0].name}):")
    print(f"  Dims: {dict(ds.sizes)}")
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  CF range: [{float(cf.min()):.4f}, {float(cf.max()):.4f}]")
    print(f"  CF mean: {float(cf.mean()):.4f}")
    print(f"  Time range: {str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]}")
    ds.close()
