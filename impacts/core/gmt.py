from __future__ import annotations

from typing import TypeAlias

import xarray as xr

ArrayLike: TypeAlias = xr.DataArray | xr.Dataset

def compute_running_mean(
    data: ArrayLike,
    window: int,
    dim: str = "year",
    min_periods: int | None = None,
    center: bool = True,
) -> ArrayLike:
    """Return a rolling mean along dim."""

    if window < 1:
        raise ValueError("window must be a positive integer")

    return data.rolling({dim: window}, center=center, min_periods=min_periods or window).mean()
