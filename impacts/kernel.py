"""Kernel contract and evaluation entry point.

The kernel abstraction declares how climate inputs should be normalized, how
mapped climate should be evaluated, and which metadata columns should be
carried into the output. Temporal resolution is part of the evaluation
contract via ``evaluate_kernel(..., duration=...)`` rather than a separate
post-hoc aggregation step.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

import pandas as pd
import xarray as xr
from climateresponse import (
    EvaluationResult,
    aggregate_monthly,
    aggregate_seasonal,
    evaluate_gridded_dataset,
    evaluate_mapped_dataset,
    normalize_temperature,
    resolve_unit,
)

Duration: TypeAlias = Literal["native", "daily", "monthly", "seasonal"]
NormalizeInputsFn: TypeAlias = Callable[[xr.Dataset], xr.Dataset]
PhysicsKernelFn: TypeAlias = Callable[[xr.Dataset], xr.Dataset | EvaluationResult]
NormalizationSpec: TypeAlias = tuple[str, str, Callable]


@dataclass(frozen=True)
class ImpactKernel:
    """Pickle-safe kernel contract for mapped climate evaluation."""

    name: str
    normalize_inputs: NormalizeInputsFn
    physics_kernel: PhysicsKernelFn
    metadata_columns: tuple[str, ...] = ()
    clips: Mapping[str, tuple[float | None, float | None]] | None = None


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
    normalize_fn has the signature of ``climateresponse.normalize_temperature``.
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


def evaluate_kernel(
    kernel: ImpactKernel,
    mapping: pd.DataFrame,
    climate: xr.Dataset,
    duration: Duration,
    *,
    seasons: Mapping[str, tuple[int, ...]] | None = None,
) -> xr.Dataset:
    """Evaluate a mapped kernel at the requested temporal duration."""

    result = evaluate_mapped_dataset(
        mapping,
        climate,
        normalize_inputs=kernel.normalize_inputs,
        physics_kernel=kernel.physics_kernel,
        metadata_columns=kernel.metadata_columns,
        clips=kernel.clips,
    )

    match duration:
        case "native" | "daily":
            return result
        case "monthly":
            return aggregate_monthly(result)
        case "seasonal":
            return aggregate_seasonal(result, seasons=seasons)
        case _:
            raise ValueError(f"unsupported duration: {duration!r}")


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
        normalize_inputs=kernel.normalize_inputs,
        physics_kernel=kernel.physics_kernel,
        spatial_dims=spatial_dims,
        clips=kernel.clips,
    )

    match duration:
        case "native" | "daily":
            return result
        case "monthly":
            return aggregate_monthly(result)
        case "seasonal":
            return aggregate_seasonal(result, seasons=seasons)
        case _:
            raise ValueError(f"unsupported duration: {duration!r}")

