"""Build seasonal country-level CHILLED quantile surfaces for RIME-X."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from climateresponse import (  # noqa: E402
    attach_warming_levels,
    build_warming_level_assignments,
    build_warming_level_quantile_surface,
)

from impacts.buildings.rimex import (  # noqa: E402
    CHILLED_VARIABLE_REDUCTIONS,
    aggregate_chilled_to_country,
    expand_chilled_metadata,
    load_country_raster,
    parse_chilled_matrix_filename,
)
from impacts.isimip import load_isimip_gmt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build seasonal CHILLED warming-level quantile surfaces for RIME-X."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--inventory-path",
        type=Path,
        required=True,
        help="JSON inventory/sanity output for the generated indicator files.",
    )
    parser.add_argument(
        "--gmt-root",
        type=Path,
        default=Path("/home/raghunathan/Ed_Jones/raw_data/climate"),
    )
    parser.add_argument("--country-column", type=str, default="ISO3")
    parser.add_argument("--bin-width", type=float, default=0.1)
    parser.add_argument("--min-warming-level", type=float, default=0.0)
    parser.add_argument("--max-warming-level", type=float, default=None)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Restrict to these scenarios. Default: use every scenario present under --source-root.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    source_paths = sorted(args.source_root.glob("seasonal_*.nc"))
    if not source_paths:
        raise FileNotFoundError(f"no CHILLED seasonal files found in {args.source_root}")
    if args.scenarios is not None:
        allowed = set(args.scenarios)
        source_paths = [
            path
            for path in source_paths
            if parse_chilled_matrix_filename(path)["scenario"] in allowed
        ]
        if not source_paths:
            raise FileNotFoundError(
                f"no CHILLED seasonal files match --scenarios={sorted(allowed)}"
            )

    country_raster = load_country_raster(country_column=args.country_column)
    aggregated_datasets: list[xr.Dataset] = []
    for path in source_paths:
        metadata = parse_chilled_matrix_filename(path)
        with xr.open_dataset(path) as dataset:
            aggregated = aggregate_chilled_to_country(dataset, country_raster)
            aggregated = expand_chilled_metadata(aggregated, metadata).load()
        aggregated_datasets.append(aggregated)

    combined = xr.combine_by_coords(
        aggregated_datasets,
        join="outer",
        combine_attrs="drop_conflicts",
    )
    assignments = build_warming_level_assignments(
        {
            (str(gcm), str(scenario)): load_isimip_gmt(
                gmt_root=args.gmt_root,
                gcm=str(gcm),
                scenario=str(scenario),
            )
            for gcm in combined.coords["gcm"].values
            for scenario in combined.coords["scenario"].values
        },
        path_columns=("gcm", "scenario"),
        min_warming_level=args.min_warming_level,
        max_warming_level=args.max_warming_level,
        bin_width=args.bin_width,
    )
    labelled = attach_warming_levels(combined, assignments)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.inventory_path.parent.mkdir(parents=True, exist_ok=True)

    inventory = {
        "source_files": len(source_paths),
        "warming_levels": len(assignments["target_warming_level"]),
        "indicators": {},
    }
    for variable in CHILLED_VARIABLE_REDUCTIONS:
        if variable not in labelled.data_vars:
            continue

        surface = build_warming_level_quantile_surface(
            labelled[variable],
            sample_dims=("gcm", "scenario", "year"),
        )
        output_path = args.output_root / f"rimex_chilled_{variable}_quantiles_by_gwl.nc"
        surface.to_netcdf(output_path)
        inventory["indicators"][variable] = {
            "path": str(output_path),
            "dims": {dim: int(size) for dim, size in surface.sizes.items()},
            "mean_sample_size": float(surface["sample_size"].mean().item()),
        }
        logging.info("Wrote %s", output_path)

    args.inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True))
    logging.info("Wrote inventory to %s", args.inventory_path)


if __name__ == "__main__":
    main()
