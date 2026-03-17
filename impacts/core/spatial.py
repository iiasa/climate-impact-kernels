from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import xarray as xr

ArrayLike: TypeAlias = xr.DataArray | xr.Dataset

def weighted_average(
    values: ArrayLike,
    weights: xr.DataArray,
    dims: str | Sequence[str],
    mask: xr.DataArray | None = None,
) -> ArrayLike:
    """Compute a masked weighted average over dims."""

    dims_ = [dims] if isinstance(dims, str) else list(dims)
    masked_values = values.where(mask) if mask is not None else values
    masked_weights = weights.where(mask, 0) if mask is not None else weights

    total_weight = masked_weights.sum(dim=dims_)
    if bool((total_weight <= 0).any()):
        raise ValueError("total weight must be positive along the aggregation dims")

    return masked_values.weighted(masked_weights).mean(dim=dims_)
