from __future__ import annotations

from typing import TypeAlias

import xarray as xr

ArrayLike: TypeAlias = xr.DataArray | xr.Dataset


def compute_anomaly(
    data: ArrayLike,
    baseline_period: tuple[int, int] | slice,
    dim: str = "year",
    offset: float = 0.0,
) -> ArrayLike:
    """Return anomalies relative to the baseline mean along dim."""

    baseline_slice = _normalize_period(baseline_period)
    baseline = data.sel({dim: baseline_slice})
    if baseline.sizes.get(dim, 0) == 0:
        raise ValueError(f"baseline_period selects no entries along {dim!r}")

    return data - baseline.mean(dim=dim) + offset


def _normalize_period(baseline_period: tuple[int, int] | slice) -> slice:
    match baseline_period:
        case slice():
            return baseline_period
        case (int(start), int(stop)):
            return slice(start, stop)
        case _:
            raise TypeError("baseline_period must be a slice or a two-year tuple")
