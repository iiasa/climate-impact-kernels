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
from climateresponse import EvaluationResult, aggregate_seasonal, evaluate_mapped_dataset

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


