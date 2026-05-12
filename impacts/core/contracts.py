from __future__ import annotations

from dataclasses import dataclass

import xarray as xr


@dataclass(frozen=True, slots=True)
class GMTPoint:
    """Single year of global-mean-temperature anomaly."""

    year: int
    value: float


@dataclass(frozen=True, slots=True)
class GMTSeries:
    """Canonical GMT contract shared across workflows."""

    model: str
    scenario: str
    baseline_start_year: int | None
    baseline_end_year: int | None
    values: tuple[GMTPoint, ...]


@dataclass(frozen=True, slots=True)
class WarmingLevelMatch:
    """Nearest-year match for a target warming level."""

    level: float
    year: int
    actual_warming: float
    absolute_error: float
    warming_rate: float | None


@dataclass(frozen=True, slots=True)
class YearLevelBin:
    """Assignment from a simulation year to a target warming level."""

    year: int
    level: float
    actual_warming: float

@dataclass(frozen=True, slots=True)
class RegionLevelValue:
    """Region-aggregated response value at a warming level."""

    region: str
    level: float
    value: float
    scenario: str | None = None
    year: int | None = None


@dataclass(frozen=True, slots=True)
class RegionResponseSurface:
    """Canonical contract for region-aggregated response surfaces."""

    variable: str
    values: xr.DataArray

    def __post_init__(self) -> None:
        required_dims = {"region", "warming_level"}
        if not required_dims.issubset(self.values.dims):
            raise ValueError(
                f"region response surface must include dims {sorted(required_dims)}"
            )


@dataclass(frozen=True, slots=True)
class GridQuantileResponseSurface:
    """Canonical contract for gridded quantile response surfaces."""

    variable: str
    values: xr.DataArray

    def __post_init__(self) -> None:
        required_dims = {"warming_level", "quantile", "lat", "lon"}
        if not required_dims.issubset(self.values.dims):
            raise ValueError(
                f"grid quantile response surface must include dims {sorted(required_dims)}"
            )
