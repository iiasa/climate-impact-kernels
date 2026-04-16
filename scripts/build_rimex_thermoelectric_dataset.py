"""Build a GMT-binned thermoelectric UCR dataset and quantile map for RIME-X.

Usage:
    uv run --project /home/raghunathan/impacts --no-sync python \
        scripts/build_rimex_thermoelectric_dataset.py \
        --source-path output/thermoelectric_country_monthly.parquet \
        --output-monthly-path output/thermoelectric_country_monthly_gwl.parquet \
        --output-quantiles-path output/thermoelectric_ucr_quantiles_by_gwl.nc
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from climateresponse import (
    attach_warming_levels,
    build_warming_level_assignments,
    build_warming_level_quantile_map,
)
from impacts.isimip import load_isimip_gmt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a warming-level-binned thermoelectric dataset for RIME-X."
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
        help="Parquet path for the monthly table with UCR and warming levels.",
    )
    parser.add_argument(
        "--output-quantiles-path",
        type=Path,
        required=True,
        help="NetCDF path for the warming-level quantile map.",
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

    monthly = pd.read_parquet(
        args.source_path,
        columns=[
            "cooling",
            "gcm",
            "scenario",
            "time",
            "country",
            "usable_capacity_mw",
            "total_mw",
        ],
    )
    if monthly["total_mw"].le(0).any():
        raise ValueError("total_mw must be strictly positive")
    monthly = monthly.assign(ucr=lambda data: data["usable_capacity_mw"] / data["total_mw"])

    gmt_by_path = {
        (row.gcm, row.scenario): load_isimip_gmt(
            gmt_root=args.gmt_root, gcm=row.gcm, scenario=row.scenario
        )
        for row in monthly.loc[:, ["gcm", "scenario"]]
        .drop_duplicates()
        .sort_values(["gcm", "scenario"])
        .itertuples(index=False)
    }
    assignments = build_warming_level_assignments(
        gmt_by_path,
        path_columns=("gcm", "scenario"),
        min_warming_level=args.min_warming_level,
        max_warming_level=args.max_warming_level,
        bin_width=args.bin_width,
    )
    enriched = attach_warming_levels(monthly, assignments)
    quantile_map = build_warming_level_quantile_map(
        enriched,
        value_column="ucr",
        group_columns=("cooling", "country"),
    ).assign_attrs({"ucr_definition": "usable_capacity_mw / total_mw"})

    args.output_monthly_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_quantiles_path.parent.mkdir(parents=True, exist_ok=True)

    enriched.to_parquet(args.output_monthly_path, index=False)
    quantile_map.to_netcdf(args.output_quantiles_path)

    logging.info(
        "Wrote %s rows with %s warming levels to %s",
        len(enriched),
        len(assignments.target_levels),
        args.output_monthly_path,
    )
    logging.info("Wrote quantile map to %s", args.output_quantiles_path)


if __name__ == "__main__":
    main()
