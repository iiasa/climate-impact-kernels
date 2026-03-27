"""Aggregate one monthly thermoelectric production file to country time series.

Usage:
    uv run --no-sync python scripts/run_monthly_country_aggregation.py \
        --source-path /pdrive/projects/watxene/IMPACTS/thermo-electric/production_monthly_imputed/dry/GFDL-ESM4_historical.nc \
        --output-root /pdrive/projects/watxene/IMPACTS/thermo-electric/production_monthly_country
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from impacts.thermoelectric.aggregation import write_aggregated_thermoelectric_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate one thermoelectric monthly production NetCDF to countries."
    )
    parser.add_argument(
        "--source-path",
        required=True,
        type=Path,
        help="Path to one wet or dry monthly production NetCDF.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Root directory for aggregated outputs. Cooling subdirectories are preserved.",
    )
    parser.add_argument(
        "--region-set",
        default="country",
        choices=["country", "message_r12"],
        help="Target region set. UniCC production currently uses country only.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    source_path = args.source_path.resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"source file not found: {source_path}")

    cooling_mode = source_path.parent.name
    target_path = args.output_root / cooling_mode / source_path.name

    logging.info("Aggregating %s to %s", source_path, target_path)
    write_aggregated_thermoelectric_file(
        source_path=source_path,
        target_path=target_path,
        region_set=args.region_set,
    )
    logging.info("Wrote %s", target_path)


if __name__ == "__main__":
    main()
