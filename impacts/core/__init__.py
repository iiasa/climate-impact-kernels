"""Shared computational core for climate-response workflows."""

from climateresponse.baseline import compute_anomaly
from climateresponse.evaluation import EvaluationResult, evaluate_gridded_dataset, evaluate_mapped_dataset
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
from climateresponse.spatial import (
    aggregate_by_region,
    attach_gridded_dataarray,
    attach_gridded_dataset,
    grid_cell_area,
    weighted_average,
)
from climateresponse.stats import deterministic_resampling, weighted_quantiles
from climateresponse.temporal import (
    CLIMATOLOGICAL_SEASONS,
    aggregate_annual,
    aggregate_monthly,
    aggregate_seasonal,
)
from climateresponse.warming_levels import (
    assign_years_to_levels,
    attach_warming_levels,
    bin_by_warming_level,
    build_warming_level_assignments,
    build_warming_level_quantile_surface,
    match_warming_levels,
)
from climateresponse.units import normalize_pressure, normalize_temperature, resolve_unit

__all__ = [
    "CLIMATOLOGICAL_SEASONS",
    "EvaluationResult",
    "GMTPoint",
    "GMTSeries",
    "GridQuantileResponseSurface",
    "RegionLevelValue",
    "RegionResponseSurface",
    "WarmingLevelMatch",
    "YearLevelBin",
    "aggregate_by_region",
    "aggregate_annual",
    "aggregate_monthly",
    "aggregate_seasonal",
    "attach_gridded_dataarray",
    "attach_gridded_dataset",
    "attach_warming_levels",
    "assign_years_to_levels",
    "bin_by_warming_level",
    "build_warming_level_assignments",
    "build_warming_level_quantile_surface",
    "compute_anomaly",
    "compute_running_mean",
    "deterministic_resampling",
    "evaluate_gridded_dataset",
    "evaluate_mapped_dataset",
    "grid_cell_area",
    "interpolate_along_warming_level",
    "match_warming_levels",
    "normalize_pressure",
    "normalize_temperature",
    "resolve_unit",
    "weighted_average",
    "weighted_quantiles",
]
