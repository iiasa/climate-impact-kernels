"""Build seasonal country-level CHILLED quantile surfaces for RIME-X.

Vectorized + parallel. The 500 source files share the same entity grid, so
the entity -> country assignment and cell-area weights are pre-computed once
and expressed as sparse aggregation matrices. Per-file aggregation is then
four sparse matmuls (one per target variable), dispatched across a worker
pool. File I/O dominates runtime; pdrive native throughput and process-level
parallelism keep wall-clock roughly at max(n_files / n_workers × file_read_time).
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impacts.climatecore import (  # noqa: E402
    attach_gridded_dataset,
    attach_warming_levels,
    build_warming_level_assignments,
    build_warming_level_quantile_surface,
    grid_cell_area,
)

from impacts.domain.buildings.rimex import (  # noqa: E402
    CHILLED_VARIABLE_REDUCTIONS,
    load_country_raster,
    parse_chilled_matrix_filename,
)
from impacts.isimip import load_isimip_gmt  # noqa: E402


_WORKER: dict = {}


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
    parser.add_argument(
        "--gaul-raster-path",
        type=Path,
        default=None,
        help="Override the GAUL country raster path; defaults to impacts.domain.buildings.chilled_config.DEFAULT_GAUL_RASTER_PATH.",
    )
    parser.add_argument(
        "--gaul-csv-path",
        type=Path,
        default=None,
        help="Override the GAUL lookup CSV path; defaults to impacts.domain.buildings.chilled_config.DEFAULT_GAUL_CSV_PATH.",
    )
    parser.add_argument("--bin-width", type=float, default=0.1)
    parser.add_argument("--min-warming-level", type=float, default=0.0)
    parser.add_argument("--max-warming-level", type=float, default=None)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Restrict to these scenarios. Default: use every scenario present under --source-root.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of worker processes for per-file aggregation.",
    )
    return parser.parse_args()


def _build_aggregation_matrices(
    country_raster: xr.DataArray,
    sample_path: Path,
) -> tuple[csr_matrix, csr_matrix, list[str]]:
    """Derive sum + weighted-mean sparse matrices keyed on a shared entity grid."""

    with xr.open_dataset(sample_path) as sample:
        entity_ids = sample["entity"].values
        entity_lat = sample.coords["lat"].values
        entity_lon = sample.coords["lon"].values

    entity_mapping = pd.DataFrame(
        {
            "entity": entity_ids,
            "grid_lat": entity_lat,
            "grid_lon": entity_lon,
        }
    )
    lookup = xr.Dataset(
        data_vars={
            "country": country_raster,
            "cell_area": grid_cell_area(country_raster["lat"], country_raster["lon"]),
        }
    )
    attached = attach_gridded_dataset(
        entity_mapping,
        lookup,
        entity_id_column="entity",
        metadata_columns=("entity",),
    )
    entity_country = attached["country"].values
    entity_area = np.asarray(attached["cell_area"].values, dtype=np.float64)

    n_entities = len(entity_ids)
    valid_mask = entity_country != ""
    valid_rows = np.where(valid_mask)[0]
    valid_countries = entity_country[valid_mask]
    valid_area = entity_area[valid_mask]

    unique_countries = sorted(np.unique(valid_countries).tolist())
    country_to_idx = {name: idx for idx, name in enumerate(unique_countries)}
    valid_country_idx = np.asarray(
        [country_to_idx[name] for name in valid_countries], dtype=np.int64
    )

    sum_mat = csr_matrix(
        (np.ones(len(valid_rows), dtype=np.float64), (valid_country_idx, valid_rows)),
        shape=(len(unique_countries), n_entities),
    )
    country_totals = np.bincount(
        valid_country_idx, weights=valid_area, minlength=len(unique_countries)
    )
    mean_weights = valid_area / country_totals[valid_country_idx]
    mean_mat = csr_matrix(
        (mean_weights, (valid_country_idx, valid_rows)),
        shape=(len(unique_countries), n_entities),
    )

    return sum_mat, mean_mat, unique_countries


def _init_worker(
    sum_mat: csr_matrix,
    mean_mat: csr_matrix,
    countries: list[str],
    target_variables: tuple[str, ...],
    reductions: dict[str, str],
) -> None:
    _WORKER["sum_mat"] = sum_mat
    _WORKER["mean_mat"] = mean_mat
    _WORKER["countries"] = countries
    _WORKER["target_variables"] = target_variables
    _WORKER["reductions"] = reductions


def _aggregate_one(path: Path) -> xr.Dataset:
    metadata = parse_chilled_matrix_filename(path)
    target_variables = _WORKER["target_variables"]
    reductions = _WORKER["reductions"]
    sum_mat = _WORKER["sum_mat"]
    mean_mat = _WORKER["mean_mat"]
    countries = _WORKER["countries"]
    n_countries = len(countries)

    with xr.open_dataset(path) as dataset:
        subset = dataset[list(target_variables)].load()
        seasons = subset["season"].values
        years = np.asarray(subset["year"].values, dtype=int)

    data_vars: dict[str, tuple[tuple[str, str, str], np.ndarray]] = {}
    for name in target_variables:
        values = np.asarray(subset[name].values, dtype=np.float32)
        season_size, year_size, entity_size = values.shape
        flat = values.reshape(season_size * year_size, entity_size).T
        matrix = sum_mat if reductions[name] == "sum" else mean_mat
        aggregated = matrix @ flat
        aggregated = np.asarray(aggregated).reshape(n_countries, season_size, year_size)
        data_vars[name] = (
            ("season", "year", "country"),
            aggregated.transpose(1, 2, 0),
        )

    per_file = xr.Dataset(
        data_vars=data_vars,
        coords={
            "season": seasons,
            "year": years,
            "country": countries,
        },
    )
    return per_file.expand_dims(
        gcm=[str(metadata["gcm"])],
        scenario=[str(metadata["scenario"])],
        building_type=[str(metadata["building_type"])],
        urt=[str(metadata["urt"])],
        vintage=[str(metadata["vintage"])],
    ).transpose(
        "gcm",
        "scenario",
        "building_type",
        "urt",
        "vintage",
        "season",
        "year",
        "country",
    )


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

    raster_kwargs = {"country_column": args.country_column}
    if args.gaul_raster_path is not None:
        raster_kwargs["gaul_raster_path"] = args.gaul_raster_path
    if args.gaul_csv_path is not None:
        raster_kwargs["gaul_csv_path"] = args.gaul_csv_path
    country_raster = load_country_raster(**raster_kwargs)

    sum_mat, mean_mat, countries = _build_aggregation_matrices(
        country_raster, source_paths[0]
    )
    target_variables = tuple(
        name for name in CHILLED_VARIABLE_REDUCTIONS
    )
    logging.info(
        "Prepared sparse aggregation matrices: %s countries, %s entities, %s sum-nnz, %s mean-nnz",
        sum_mat.shape[0],
        sum_mat.shape[1],
        sum_mat.nnz,
        mean_mat.nnz,
    )
    logging.info("Processing %s files with %s workers", len(source_paths), args.workers)

    with mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(
            sum_mat,
            mean_mat,
            countries,
            target_variables,
            dict(CHILLED_VARIABLE_REDUCTIONS),
        ),
    ) as pool:
        aggregated_datasets: list[xr.Dataset] = []
        for idx, dataset in enumerate(pool.imap_unordered(_aggregate_one, source_paths), start=1):
            aggregated_datasets.append(dataset)
            if idx % 25 == 0:
                logging.info("Aggregated %s/%s files", idx, len(source_paths))

    logging.info("All files aggregated; combining and attaching warming levels")
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
