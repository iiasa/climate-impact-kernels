"""Generic impact kernel runner across GCM x scenario combinations.

The runner is kernel-agnostic: it accepts any callable with the signature
``(mapping: pd.DataFrame, climate: xr.Dataset) -> xr.Dataset`` and a
loader callable ``(gcm: str, scenario: str) -> xr.Dataset``.

Parallelization happens at the (gcm, scenario) level via
concurrent.futures, not within a single kernel evaluation.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TypeAlias

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

KernelFn: TypeAlias = Callable[[pd.DataFrame, xr.Dataset], xr.Dataset]
LoaderFn: TypeAlias = Callable[[str, str], xr.Dataset]


def run_impact_matrix(
    kernel: KernelFn,
    mapping: pd.DataFrame,
    loader: LoaderFn,
    gcms: Sequence[str],
    scenarios: Sequence[str],
    output_dir: Path,
    *,
    post_process: Callable[[xr.Dataset], xr.Dataset] | None = None,
    max_workers: int = 1,
) -> list[Path]:
    """Run a kernel across the GCM x scenario matrix.

    For each (gcm, scenario) pair:
    1. Load climate data via ``loader(gcm, scenario)``
    2. Run ``kernel(mapping, climate)``
    3. Optionally apply ``post_process`` (e.g. seasonal aggregation)
    4. Write output to ``output_dir/{gcm}_{scenario}.nc``

    Existing output files are skipped (checkpoint-resume).

    Returns the list of output file paths (including pre-existing ones).
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    combinations = [(gcm, scenario) for gcm in gcms for scenario in scenarios]
    output_paths: list[Path] = []

    if max_workers <= 1:
        for gcm, scenario in combinations:
            path = _run_single(
                kernel, mapping, loader, gcm, scenario, output_dir,
                post_process=post_process,
            )
            output_paths.append(path)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_single,
                    kernel, mapping, loader, gcm, scenario, output_dir,
                    post_process=post_process,
                ): (gcm, scenario)
                for gcm, scenario in combinations
            }
            for future in as_completed(futures):
                output_paths.append(future.result())

    return sorted(output_paths)


def _run_single(
    kernel: KernelFn,
    mapping: pd.DataFrame,
    loader: LoaderFn,
    gcm: str,
    scenario: str,
    output_dir: Path,
    *,
    post_process: Callable[[xr.Dataset], xr.Dataset] | None = None,
) -> Path:
    output_path = output_dir / f"{gcm}_{scenario}.nc"

    if output_path.exists():
        if _is_valid_checkpoint(output_path):
            logger.info("Skipping %s/%s — output exists: %s", gcm, scenario, output_path)
            return output_path
        invalid_path = _move_invalid_checkpoint(output_path)
        logger.warning(
            "Unreadable checkpoint for %s/%s moved aside: %s -> %s",
            gcm,
            scenario,
            output_path,
            invalid_path,
        )

    logger.info("Loading climate: %s/%s", gcm, scenario)
    climate = loader(gcm, scenario)

    logger.info("Running kernel: %s/%s", gcm, scenario)
    result = kernel(mapping, climate)

    if post_process is not None:
        logger.info("Post-processing: %s/%s", gcm, scenario)
        result = post_process(result)

    temp_path = _temporary_output_path(output_path)
    logger.info("Writing: %s", output_path)
    try:
        result.to_netcdf(temp_path)
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)

    return output_path


def _is_valid_checkpoint(path: Path) -> bool:
    try:
        with xr.open_dataset(path) as dataset:
            dataset.sizes
            tuple(dataset.data_vars)
    except (OSError, ValueError):
        return False
    return True


def _move_invalid_checkpoint(path: Path) -> Path:
    candidate = path.with_suffix(f"{path.suffix}.invalid")
    counter = 1
    while candidate.exists():
        candidate = path.with_suffix(f"{path.suffix}.invalid.{counter}")
        counter += 1
    path.rename(candidate)
    return candidate


def _temporary_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.{os.getpid()}.tmp")
