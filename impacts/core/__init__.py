"""Shared computational core for climate-response workflows."""

from .baseline import compute_anomaly
from .evaluation import EvaluationResult, evaluate_gridded_dataset, evaluate_mapped_dataset
from .contracts import (
    GMTPoint,
    GMTSeries,
    GridQuantileResponseSurface,
    RegionLevelValue,
    RegionResponseSurface,
    WarmingLevelMatch,
    YearLevelBin,
)
from .gmt import compute_running_mean
from .response import interpolate_along_warming_level
from .spatial import (
    aggregate_by_region,
    attach_gridded_dataarray,
    attach_gridded_dataset,
    grid_cell_area,
    weighted_average,
)
from .stats import deterministic_resampling, weighted_quantiles
from .temporal import (
    CLIMATOLOGICAL_SEASONS,
    aggregate_annual,
    aggregate_monthly,
    aggregate_seasonal,
)
from .warming_levels import (
    assign_years_to_levels,
    attach_warming_levels,
    bin_by_warming_level,
    build_warming_level_assignments,
    build_warming_level_quantile_surface,
    match_warming_levels,
)
from .units import normalize_pressure, normalize_temperature, resolve_unit

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
