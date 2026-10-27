"""Aggregation of thermoelectric unit and cell outputs to countries and MESSAGE R12 regions.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
import xarray as xr

from impacts.climatecore.regions import (
    aggregate_site_dataarray,
    aggregate_site_dataset,
    message_r12_region_labels,
    resolve_region_labels,
)

RegionSet: TypeAlias = Literal["country", "message_r12"]

_WET_WEIGHTED_MEAN_VARIABLES = frozenset(
    {
        "water_temperature",
        "discharge",
        "air_temperature",
        "relative_humidity",
        "surface_pressure",
        "intake_temperature",
        "wet_bulb_temperature",
        "saturation_vapor_pressure_pa",
        "actual_vapor_pressure_pa",
        "humidity_ratio_out_g_per_kg",
        "humidity_ratio_in_g_per_kg",
        "enthalpy_out_kj_per_kg",
        "enthalpy_in_kj_per_kg",
        "temperature_headroom_c",
        "withdrawal_intensity_m3_per_mwh",
        "capacity_factor",
        "capacity_loss",
        "thermal_efficiency_fraction",
        "heat_rejection_coefficient_fraction",
        "evaporation_coefficient_fraction",
        "discharge_constraint_applies",
    }
)
_WET_SUM_VARIABLES = frozenset(
    {
        "available_discharge_m3_per_s",
        "unit_available_discharge_m3_per_s",
        "usable_capacity_mw",
    }
)
# A property of the grid cell, repeated on every unit in the cell: a regional
# total counts each cell once.
_CELL_LEVEL_SUM_VARIABLE = "available_discharge_m3_per_s"
_DRY_WEIGHTED_MEAN_VARIABLES = frozenset(
    {
        "air_temperature_k",
        "backpressure_inhg",
        "efficiency_loss",
        "parasitic_load",
        "capacity_factor",
        "capacity_loss",
    }
)
_DRY_SUM_VARIABLES = frozenset({"usable_capacity_mw"})


def aggregate_thermoelectric_regions(
    dataset: xr.Dataset,
    *,
    region_set: RegionSet,
    site_dim: str = "entity",
) -> xr.Dataset:
    """Aggregate per-unit thermoelectric outputs (along ``site_dim``) to countries or MESSAGE regions."""

    if site_dim not in dataset.dims:
        raise ValueError(f"{site_dim!r} is not a dimension of the input dataset")
    if "mw" not in dataset.coords:
        raise ValueError("'mw' coordinate is required for thermoelectric aggregation")
    if "country" not in dataset.coords:
        raise ValueError("'country' coordinate is required for thermoelectric aggregation")

    region_labels = resolve_region_labels(
        dataset.coords["country"].values.tolist(),
        region_set=region_set,
    )
    aggregated = aggregate_site_dataset(
        dataset,
        region_labels=region_labels,
        variable_reductions=_variable_reductions(dataset, site_dim=site_dim),
        weights=dataset.coords["mw"],
        site_dim=site_dim,
    )
    if _CELL_LEVEL_SUM_VARIABLE in aggregated.data_vars:
        aggregated[_CELL_LEVEL_SUM_VARIABLE] = _sum_over_distinct_cells(
            dataset[_CELL_LEVEL_SUM_VARIABLE],
            region_labels,
            site_dim=site_dim,
        )
    total_mw = (
        xr.DataArray(
            dataset.coords["mw"].values,
            dims=(site_dim,),
            coords={site_dim: dataset.coords[site_dim].values},
        )
        .assign_coords(region=(site_dim, list(region_labels)))
        .groupby("region")
        .sum(dim=site_dim)
    )
    aggregated = aggregated.assign_coords(total_mw=("region", total_mw.values))
    if "usable_capacity_mw" in dataset.data_vars:
        # MW of units whose usable capacity is known: total_mw minus the units that
        # drop out of usable_capacity_mw (a sum that skips them) for want of inputs.
        usable = dataset["usable_capacity_mw"]
        valid_mw = xr.where(usable.notnull(), dataset.coords["mw"], 0.0).transpose(*usable.dims)
        aggregated["valid_mw"] = aggregate_site_dataarray(
            valid_mw,
            region_labels,
            reduction="sum",
            site_dim=site_dim,
        ).assign_attrs(units="MW")
    aggregated = aggregated.assign_attrs(dataset.attrs | {"region_set": region_set})

    if region_set == "message_r12":
        labels = message_r12_region_labels()
        aggregated = aggregated.assign_coords(
            region_name=("region", [labels[str(code)] for code in aggregated.coords["region"].values]),
        )

    return aggregated


def aggregate_thermoelectric_file(
    source_path: str | Path,
    *,
    region_set: RegionSet,
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Open and aggregate a thermoelectric NetCDF file to the requested regions."""

    with xr.open_dataset(Path(source_path), chunks=chunks) as dataset:
        # Load only the reduced result so callers do not keep the full source file in memory.
        return aggregate_thermoelectric_regions(
            dataset,
            region_set=region_set,
        ).load()


def write_aggregated_thermoelectric_file(
    source_path: str | Path,
    target_path: str | Path,
    *,
    region_set: RegionSet,
    chunks: Mapping[str, int] | None = None,
) -> Path:
    """Aggregate a thermoelectric NetCDF file and write the reduced result."""

    aggregated = aggregate_thermoelectric_file(
        source_path,
        region_set=region_set,
        chunks=chunks,
    )
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_netcdf(target)
    return target


def _sum_over_distinct_cells(
    values: xr.DataArray,
    region_labels: tuple[str, ...],
    *,
    site_dim: str,
) -> xr.DataArray:
    """Sum a per-cell quantity over the distinct cells of each region."""

    for coordinate in ("grid_lat", "grid_lon"):
        if coordinate not in values.coords:
            raise ValueError(
                f"{coordinate!r} coordinate is required to sum {values.name!r} over distinct cells"
            )
    cells = pd.DataFrame(
        {
            "region": region_labels,
            "grid_lat": values.coords["grid_lat"].values,
            "grid_lon": values.coords["grid_lon"].values,
        }
    )
    first_unit = np.flatnonzero(~cells.duplicated().to_numpy())
    return aggregate_site_dataarray(
        values.isel({site_dim: first_unit}),
        [region_labels[index] for index in first_unit],
        reduction="sum",
        site_dim=site_dim,
    )


def _variable_reductions(
    dataset: xr.Dataset,
    *,
    site_dim: str,
) -> dict[str, Literal["sum", "weighted_mean"]]:
    weighted = _WET_WEIGHTED_MEAN_VARIABLES | _DRY_WEIGHTED_MEAN_VARIABLES
    summed = _WET_SUM_VARIABLES | _DRY_SUM_VARIABLES

    reductions: dict[str, Literal["sum", "weighted_mean"]] = {}
    unknown: list[str] = []
    for name, variable in dataset.data_vars.items():
        if site_dim not in variable.dims:
            continue
        if name in summed:
            reductions[name] = "sum"
        elif name in weighted:
            reductions[name] = "weighted_mean"
        else:
            unknown.append(str(name))

    if unknown:
        raise ValueError(
            f"no regional reduction is defined for entity variables {sorted(unknown)}; "
            "add each to the sum or weighted-mean tables in impacts.thermoelectric.regions "
            "or drop it from the dataset before aggregating"
        )
    return reductions
