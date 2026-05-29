"""Kernel contract and evaluation entry point.

A kernel declares how climate inputs are normalized, how attached climate is
turned into impact variables (its physics), and which metadata columns are
carried into the output. Temporal resolution is part of the evaluation contract
via ``evaluate_mapped_kernel(..., duration=...)`` rather than a separate
post-hoc aggregation step.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

import pandas as pd
import xarray as xr
from impacts.climatecore import (
    EvaluationResult,
    aggregate_monthly,
    aggregate_seasonal,
    resolve_unit,
)
from impacts.climatecore.evaluation import evaluate_gridded_dataset, evaluate_mapped_dataset

Duration: TypeAlias = Literal["native", "daily", "monthly", "seasonal"]
NormalizeFn: TypeAlias = Callable[[xr.Dataset], xr.Dataset]
PhysicsFn: TypeAlias = Callable[[xr.Dataset], xr.Dataset | EvaluationResult]
NormalizationSpec: TypeAlias = tuple[str, str, Callable]


@dataclass(frozen=True)
class ImpactKernel:
    """Pickle-safe kernel contract for impact evaluation."""

    name: str
    normalize: NormalizeFn
    physics: PhysicsFn
    metadata_columns: tuple[str, ...] = ()
    clips: Mapping[str, tuple[float | None, float | None]] | None = None
    entry_chunks: Mapping[str, int] | None = None


def normalize_climate_inputs(
    dataset: xr.Dataset,
    *,
    default_variable_names: Mapping[str, str],
    normalizations: tuple[NormalizationSpec, ...],
    variable_names: Mapping[str, str] | None = None,
    variable_units: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Select, rename, and unit-normalize climate variables.

    Each normalization spec is (internal_name, target_unit, normalize_fn) where
    normalize_fn has the signature of ``impacts.climatecore.normalize_temperature``.
    """

    variables = dict(default_variable_names)
    if variable_names is not None:
        variables.update(variable_names)

    missing = [source for source in variables.values() if source not in dataset.data_vars]
    if missing:
        raise ValueError(f"dataset missing required climate variables: {', '.join(sorted(missing))}")

    normalized = dataset[list(variables.values())].rename(
        {source: target for target, source in variables.items()}
    )
    for internal_name, target_unit, normalize_fn in normalizations:
        normalized[internal_name] = normalize_fn(
            normalized[internal_name],
            target_unit=target_unit,
            unit=resolve_unit(normalized[internal_name], variable_units=variable_units),
        )
    return normalized


def evaluate_mapped_kernel(
    kernel: ImpactKernel,
    mapping: pd.DataFrame,
    climate: xr.Dataset,
    duration: Duration,
    *,
    gridded_coords: xr.Dataset | None = None,
    seasons: Mapping[str, tuple[int, ...]] | None = None,
) -> xr.Dataset:
    """Evaluate a kernel over mapped entities at the requested temporal duration."""

    result = evaluate_mapped_dataset(
        mapping,
        climate,
        normalize=kernel.normalize,
        physics=kernel.physics,
        metadata_columns=kernel.metadata_columns,
        gridded_coords=gridded_coords,
        entry_chunks=kernel.entry_chunks,
        clips=kernel.clips,
    )
    return _apply_duration(result, duration, seasons=seasons)


def evaluate_gridded_kernel(
    kernel: ImpactKernel,
    climate: xr.Dataset,
    parameters: xr.Dataset,
    duration: Duration,
    *,
    spatial_dims: tuple[str, ...] = ("lat", "lon"),
    seasons: Mapping[str, tuple[int, ...]] | None = None,
) -> xr.Dataset:
    """Evaluate a kernel on a gridded dataset where climate and parameters share spatial dims."""

    result = evaluate_gridded_dataset(
        climate,
        parameters,
        normalize=kernel.normalize,
        physics=kernel.physics,
        spatial_dims=spatial_dims,
        entry_chunks=kernel.entry_chunks,
        clips=kernel.clips,
    )
    return _apply_duration(result, duration, seasons=seasons)


def _apply_duration(
    result: xr.Dataset,
    duration: Duration,
    *,
    seasons: Mapping[str, tuple[int, ...]] | None,
) -> xr.Dataset:
    match duration:
        case "native" | "daily":
            return result
        case "monthly":
            return aggregate_monthly(result)
        case "seasonal":
            return aggregate_seasonal(result, seasons=seasons)
        case _:
            raise ValueError(f"unsupported duration: {duration!r}")
