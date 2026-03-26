"""Build the canonical full wet-fleet mapping artifact locally.

This materializes a single mapping table for all wet-cooled units with
provenance preserved in `coordinate_source`:

- `source`: exact unit coordinates present in the source fleet
- `inherited_location`: safe same-LOCATIONID coordinate inheritance
- `imputed_unit_donor`: remaining wet units assigned to donor cells by the
  explicit fallback ladder in `impacts.thermoelectric.mapping`

Usage:
    ./.venv/bin/python validation/build_imputed_wet_mapping.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impacts.thermoelectric import write_imputed_wet_unit_cell_mapping

THERMAL_PLANTS = Path("/home/raghunathan/Li_et_al_2025/thermal_plants.parquet")
COOLING_MAPPING = REPO_ROOT / "data" / "unit_cooling_mapping.parquet"
OUTPUT_MAPPING = REPO_ROOT / "data" / "imputed_wet_unit_cell_mapping.parquet"
OUTPUT_SUMMARY = REPO_ROOT / "output" / "imputed_wet_mapping_summary.json"
RANDOM_SEED = 0


def main() -> None:
    summary = write_imputed_wet_unit_cell_mapping(
        THERMAL_PLANTS,
        COOLING_MAPPING,
        OUTPUT_MAPPING,
        random_seed=RANDOM_SEED,
    )
    mapping = pd.read_parquet(OUTPUT_MAPPING)
    provenance = (
        mapping.groupby("coordinate_source", as_index=False)
        .agg(units=("UNITID", "size"), mw_gw=("mw", lambda s: float(s.sum() / 1000)))
        .sort_values("mw_gw", ascending=False, kind="stable")
    )

    report = {
        "random_seed": RANDOM_SEED,
        "output_mapping": str(OUTPUT_MAPPING),
        "summary": {
            "coordinate_units": summary.coordinate_units,
            "coordinate_mw_gw": summary.coordinate_mw / 1000,
            "occupied_cells": summary.occupied_cells,
            "median_units_per_cell": summary.median_units_per_cell,
            "p95_units_per_cell": summary.p95_units_per_cell,
            "max_units_per_cell": summary.max_units_per_cell,
        },
        "provenance_breakdown": provenance.to_dict(orient="records"),
    }

    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY.write_text(json.dumps(report, indent=2))

    print(f"Wrote {OUTPUT_MAPPING}")
    print(f"Wrote {OUTPUT_SUMMARY}")
    print(provenance.to_string(index=False))


if __name__ == "__main__":
    main()
