"""Shared computational core for the impacts package."""

from .space import (
    attach_gridded_dataarray,
    attach_gridded_dataset,
    grid_cell_area,
    nearest_valid_cells,
    refine_to_grid,
)
from .time import (
    CLIMATOLOGICAL_SEASONS,
    aggregate_monthly,
    aggregate_seasonal,
)
from .units import (
    convert_units,
    normalize_pressure,
    normalize_temperature,
    resolve_unit,
)
from .warming import (
    assign_years_to_levels,
    bin_by_warming_level,
    build_warming_level_assignments,
    pool_by_warming_level,
    weighted_group_mean,
)

__all__ = [
    "CLIMATOLOGICAL_SEASONS",
    "aggregate_monthly",
    "aggregate_seasonal",
    "assign_years_to_levels",
    "attach_gridded_dataarray",
    "attach_gridded_dataset",
    "bin_by_warming_level",
    "build_warming_level_assignments",
    "convert_units",
    "grid_cell_area",
    "nearest_valid_cells",
    "normalize_pressure",
    "normalize_temperature",
    "pool_by_warming_level",
    "refine_to_grid",
    "resolve_unit",
    "weighted_group_mean",
]
