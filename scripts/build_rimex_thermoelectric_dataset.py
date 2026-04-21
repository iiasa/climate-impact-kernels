"""Build a monthly GMT-binned thermoelectric UCR surface for RIME-X.

Usage:
    uv run --project /home/raghunathan/impacts --no-sync python \
        scripts/build_rimex_thermoelectric_dataset.py \
        --source-path output/thermoelectric_country_monthly.parquet \
        --output-monthly-path output/thermoelectric_country_monthly_gwl.nc \
        --output-quantiles-path output/thermoelectric_ucr_quantiles_by_gwl.nc
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from climateresponse import (  # noqa: E402
    attach_warming_levels,
    build_warming_level_assignments,
    build_warming_level_quantile_surface,
)

from impacts.isimip import load_isimip_gmt  # noqa: E402
from impacts.thermoelectric.rimex import load_thermoelectric_ucr  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a monthly warming-level-binned thermoelectric dataset for RIME-X."
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        required=True,
        help="Monthly country-level thermoelectric parquet.",
    )
    parser.add_argument(
        "--gmt-root",
        type=Path,
        default=Path("/home/raghunathan/Ed_Jones/raw_data/climate"),
        help="Directory containing ISIMIP monthly temperature CSVs.",
    )
    parser.add_argument(
        "--output-monthly-path",
        type=Path,
        required=True,
        help="NetCDF path for the monthly UCR store with warming-level coordinates.",
    )
    parser.add_argument(
        "--output-quantiles-path",
        type=Path,
        required=True,
        help="NetCDF path for the warming-level quantile surface.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=0.1,
        help="Warming-level grid spacing in K.",
    )
    parser.add_argument(
        "--min-warming-level",
        type=float,
        default=0.0,
        help="Lowest warming level to include in the target grid.",
    )
    parser.add_argument(
        "--max-warming-level",
        type=float,
        default=None,
        help="Optional upper bound for the target warming-level grid.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    monthly_ucr = load_thermoelectric_ucr(args.source_path)
    gmt_by_path = {
        (str(gcm), str(scenario)): load_isimip_gmt(
            gmt_root=args.gmt_root,
            gcm=str(gcm),
            scenario=str(scenario),
        )
        for gcm in monthly_ucr.coords["gcm"].values
        for scenario in monthly_ucr.coords["scenario"].values
    }
    assignments = build_warming_level_assignments(
        gmt_by_path,
        path_columns=("gcm", "scenario"),
        min_warming_level=args.min_warming_level,
        max_warming_level=args.max_warming_level,
        bin_width=args.bin_width,
    )
    labelled = attach_warming_levels(monthly_ucr, assignments).to_dataset(name="ucr")
    quantile_surface = build_warming_level_quantile_surface(
        labelled["ucr"],
        sample_dims=("gcm", "scenario", "time"),
    ).assign_attrs({"ucr_definition": "usable_capacity_mw / total_mw"})

    args.output_monthly_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_quantiles_path.parent.mkdir(parents=True, exist_ok=True)

    labelled.to_netcdf(args.output_monthly_path)
    quantile_surface.to_netcdf(args.output_quantiles_path)

    logging.info(
        "Wrote monthly UCR store with %s warming levels to %s",
        len(assignments["target_warming_level"]),
        args.output_monthly_path,
    )
    logging.info("Wrote monthly quantile surface to %s", args.output_quantiles_path)


if __name__ == "__main__":
    main()
