"""Generic mapped-kernel evaluation and matrix runner.

The runner operates on kernel objects that declare how climate inputs should be
normalized, how mapped climate should be evaluated, and which metadata columns
should be carried into the output. Temporal resolution is part of the
evaluation contract via ``evaluate_kernel(..., duration=...)`` rather than a
separate post-hoc aggregation step.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, TypeAlias

import pandas as pd
import xarray as xr
from climateresponse import EvaluationResult, aggregate_seasonal, evaluate_mapped_dataset

logger = logging.getLogger(__name__)

LoaderFn: TypeAlias = Callable[[str, str], xr.Dataset]
Duration: TypeAlias = Literal["monthly", "seasonal"]
NormalizeInputsFn: TypeAlias = Callable[[xr.Dataset], xr.Dataset]
PhysicsKernelFn: TypeAlias = Callable[[xr.Dataset], xr.Dataset | EvaluationResult]


@dataclass(frozen=True)
class ImpactKernel:
    """Pickle-safe kernel contract for mapped climate evaluation."""

    name: str
    normalize_inputs: NormalizeInputsFn
    physics_kernel: PhysicsKernelFn
    metadata_columns: tuple[str, ...] = ()
    clips: Mapping[str, tuple[float | None, float | None]] | None = None


def evaluate_kernel(
    kernel: ImpactKernel,
    mapping: pd.DataFrame,
    climate: xr.Dataset,
    duration: Duration,
    *,
    seasons: Mapping[str, tuple[int, ...]] | None = None,
) -> xr.Dataset:
    """Evaluate a mapped kernel at the requested temporal duration."""

    monthly = evaluate_mapped_dataset(
        mapping,
        climate,
        normalize_inputs=kernel.normalize_inputs,
        physics_kernel=kernel.physics_kernel,
        metadata_columns=kernel.metadata_columns,
        clips=kernel.clips,
    )

    match duration:
        case "monthly":
            return monthly
        case "seasonal":
            return aggregate_seasonal(monthly, seasons=seasons)
        case _:
            raise ValueError(f"unsupported duration: {duration!r}")


def run_impact_matrix(
    kernel: ImpactKernel,
    mapping: pd.DataFrame,
    loader: LoaderFn,
    gcms: Sequence[str],
    scenarios: Sequence[str],
    output_dir: Path,
    *,
    duration: Duration = "monthly",
    seasons: Mapping[str, tuple[int, ...]] | None = None,
    max_workers: int = 1,
) -> list[Path]:
    """Run a kernel across the GCM x scenario matrix.

    For each (gcm, scenario) pair:
    1. Load climate data via ``loader(gcm, scenario)``
    2. Run ``evaluate_kernel(kernel, mapping, climate, duration=...)``
    3. Write output to ``output_dir/{gcm}_{scenario}.nc``

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
                duration=duration,
                seasons=seasons,
            )
            output_paths.append(path)
    else:
        evaluate_single = partial(_run_single, duration=duration, seasons=seasons)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_single,
                    kernel, mapping, loader, gcm, scenario, output_dir,
                ): (gcm, scenario)
                for gcm, scenario in combinations
            }
            for future in as_completed(futures):
                output_paths.append(future.result())

    return sorted(output_paths)


def _run_single(
    kernel: ImpactKernel,
    mapping: pd.DataFrame,
    loader: LoaderFn,
    gcm: str,
    scenario: str,
    output_dir: Path,
    *,
    duration: Duration,
    seasons: Mapping[str, tuple[int, ...]] | None = None,
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

    logger.info("Running kernel: %s/%s (%s)", gcm, scenario, kernel.name)
    result = evaluate_kernel(
        kernel,
        mapping,
        climate,
        duration=duration,
        seasons=seasons,
    )

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
