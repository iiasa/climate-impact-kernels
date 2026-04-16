from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from climateresponse import assign_years_to_levels
from impacts.isimip import load_isimip_gmt

# NB: Keep this workflow in impacts until a second concrete consumer appears.
# At that point, move only the xarray-first warming-level/quantile core into
# climateresponse and leave table/file adapters in impacts.

DEFAULT_BIN_WIDTH = 0.1
DEFAULT_MIN_WARMING_LEVEL = 0.0
DEFAULT_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


@dataclass(frozen=True)
class WarmingLevelAssignments:
    """Year-level warming assignments for the GCM/scenario combinations in a table."""

    table: pd.DataFrame
    target_levels: tuple[float, ...]
    gcm_column: str
    scenario_column: str
    year_column: str


def load_impact_table(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load a parquet-backed impact table."""

    return pd.read_parquet(Path(path), columns=None if columns is None else list(columns))


def build_warming_level_assignments(
    frame: pd.DataFrame,
    *,
    gmt_root: str | Path,
    gcm_column: str = "gcm",
    scenario_column: str = "scenario",
    min_warming_level: float = DEFAULT_MIN_WARMING_LEVEL,
    bin_width: float = DEFAULT_BIN_WIDTH,
    max_warming_level: float | None = None,
) -> WarmingLevelAssignments:
    """Assign each year in each GCM/scenario path to the nearest warming level."""

    _validate_required_columns(frame, required_columns=(gcm_column, scenario_column))
    _validate_warming_grid(min_warming_level=min_warming_level, bin_width=bin_width)

    combinations = (
        frame.loc[:, [gcm_column, scenario_column]]
        .drop_duplicates()
        .sort_values([gcm_column, scenario_column])
    )
    gmt_by_path = [
        (
            getattr(row, gcm_column),
            getattr(row, scenario_column),
            load_isimip_gmt(
                gmt_root=gmt_root,
                gcm=getattr(row, gcm_column),
                scenario=getattr(row, scenario_column),
            ),
        )
        for row in combinations.itertuples(index=False)
    ]
    resolved_max = (
        max_warming_level
        if max_warming_level is not None
        else max(float(np.nanmax(series.values)) for _, _, series in gmt_by_path)
    )
    target_levels = _build_target_levels(
        min_warming_level=min_warming_level,
        max_warming_level=resolved_max,
        bin_width=bin_width,
    )
    assignment_frames = [
        _assignment_frame(
            gcm=gcm,
            scenario=scenario,
            gmt=gmt,
            target_levels=target_levels,
            gcm_column=gcm_column,
            scenario_column=scenario_column,
        )
        for gcm, scenario, gmt in gmt_by_path
    ]
    return WarmingLevelAssignments(
        table=pd.concat(assignment_frames, ignore_index=True),
        target_levels=target_levels,
        gcm_column=gcm_column,
        scenario_column=scenario_column,
        year_column="year",
    )


def attach_warming_levels(
    frame: pd.DataFrame,
    assignments: WarmingLevelAssignments,
    *,
    time_column: str = "time",
) -> pd.DataFrame:
    """Join warming-level assignments onto an impact table using GCM/scenario/year."""

    _validate_required_columns(
        frame,
        required_columns=(assignments.gcm_column, assignments.scenario_column, time_column),
    )
    enriched = (
        frame.assign(**{time_column: lambda data: pd.to_datetime(data[time_column])})
        .assign(year=lambda data: data[time_column].dt.year)
        .merge(
            assignments.table,
            on=[assignments.gcm_column, assignments.scenario_column, assignments.year_column],
            how="left",
            validate="many_to_one",
        )
    )
    if enriched["warming_level"].isna().any():
        missing = (
            enriched.loc[
                enriched["warming_level"].isna(),
                [assignments.gcm_column, assignments.scenario_column, "year"],
            ]
            .drop_duplicates()
            .sort_values([assignments.gcm_column, assignments.scenario_column, "year"])
        )
        raise ValueError(
            "missing warming-level assignments for "
            f"{missing.to_dict(orient='records')[:10]}"
        )
    return enriched


def build_quantile_map(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_columns: Sequence[str],
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    warming_level_column: str = "warming_level",
) -> xr.Dataset:
    """Build a warming-level quantile map for a chosen value column."""

    quantiles = tuple(float(value) for value in quantiles)
    if not quantiles:
        raise ValueError("quantiles must not be empty")

    required_columns = tuple(group_columns) + (warming_level_column, value_column)
    _validate_required_columns(frame, required_columns=required_columns)

    grouping = [*group_columns, warming_level_column]
    quantile_frame = (
        frame.groupby(grouping, observed=True)[value_column]
        .quantile(quantiles)
        .rename(value_column)
        .reset_index()
        .rename(columns={f"level_{len(grouping)}": "quantile"})
    )
    count_frame = (
        frame.groupby(grouping, observed=True)
        .size()
        .rename("sample_size")
        .reset_index()
    )

    quantile_map = quantile_frame.set_index(
        [*group_columns, warming_level_column, "quantile"]
    ).to_xarray()
    sample_size = count_frame.set_index([*group_columns, warming_level_column]).to_xarray()

    return (
        xr.merge([quantile_map, sample_size])
        .sortby(warming_level_column)
        .sortby("quantile")
        .assign_attrs(
            {
                "description": f"Quantile map for {value_column} conditioned on annual-mean GMT",
                "warming_level_units": "K",
                "value_column": value_column,
                "group_columns": ",".join(group_columns),
            }
        )
    )


def _assignment_frame(
    *,
    gcm: str,
    scenario: str,
    gmt: xr.DataArray,
    target_levels: Sequence[float],
    gcm_column: str,
    scenario_column: str,
) -> pd.DataFrame:
    assignments = assign_years_to_levels(gmt=gmt, target_levels=target_levels)
    return pd.DataFrame(
        {
            gcm_column: gcm,
            scenario_column: scenario,
            "year": np.asarray(assignments["year"].values, dtype=int),
            "warming_level": np.asarray(assignments["warming_level"].values, dtype=float),
            "actual_warming": np.asarray(assignments["actual_warming"].values, dtype=float),
            "absolute_error": np.asarray(assignments["absolute_error"].values, dtype=float),
        }
    )


def _build_target_levels(
    *,
    min_warming_level: float,
    max_warming_level: float,
    bin_width: float,
) -> tuple[float, ...]:
    upper = max(max_warming_level, min_warming_level)
    level_count = int(np.ceil((upper - min_warming_level) / bin_width)) + 1
    return tuple(
        np.round(
            min_warming_level + np.arange(level_count, dtype=float) * bin_width,
            decimals=6,
        ).tolist()
    )


def _validate_required_columns(
    frame: pd.DataFrame,
    *,
    required_columns: Sequence[str],
) -> None:
    missing = set(required_columns).difference(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"frame missing required columns: {missing_list}")


def _validate_warming_grid(*, min_warming_level: float, bin_width: float) -> None:
    if min_warming_level < 0:
        raise ValueError("min_warming_level must be non-negative")
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
