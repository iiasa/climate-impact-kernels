"""Shared computational core for climate-response workflows."""

from climateresponse.baseline import compute_anomaly
from climateresponse.contracts import (
    GMTPoint,
    GMTSeries,
    GridQuantileResponseSurface,
    RegionLevelValue,
    RegionResponseSurface,
    WarmingLevelMatch,
    YearLevelBin,
)
from climateresponse.gmt import compute_running_mean
from climateresponse.response import interpolate_along_warming_level
from climateresponse.spatial import weighted_average
from climateresponse.stats import deterministic_resampling, weighted_quantiles
from climateresponse.temporal import (
    CLIMATOLOGICAL_SEASONS,
    aggregate_annual,
    aggregate_seasonal,
)
from climateresponse.warming_levels import (
    assign_years_to_levels,
    bin_by_warming_level,
    match_warming_levels,
)

__all__ = [
    "CLIMATOLOGICAL_SEASONS",
    "GMTPoint",
    "GMTSeries",
    "GridQuantileResponseSurface",
    "RegionLevelValue",
    "RegionResponseSurface",
    "WarmingLevelMatch",
    "YearLevelBin",
    "aggregate_annual",
    "aggregate_seasonal",
    "assign_years_to_levels",
    "bin_by_warming_level",
    "compute_anomaly",
    "compute_running_mean",
    "deterministic_resampling",
    "interpolate_along_warming_level",
    "match_warming_levels",
    "weighted_average",
    "weighted_quantiles",
]
