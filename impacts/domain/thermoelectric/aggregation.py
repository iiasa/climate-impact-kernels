from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypeAlias

import xarray as xr

from impacts.climatecore.regions import (
    aggregate_entity_dataset,
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
    strict: bool = True,
    entity_dim: str = "entity",
) -> xr.Dataset:
    """Aggregate entity-level thermoelectric outputs to countries or MESSAGE regions."""

    if entity_dim not in dataset.dims:
        raise ValueError(f"{entity_dim!r} is not a dimension of the input dataset")
    if "mw" not in dataset.coords:
        raise ValueError("'mw' coordinate is required for thermoelectric aggregation")
    if "country" not in dataset.coords:
        raise ValueError("'country' coordinate is required for thermoelectric aggregation")

    region_labels = resolve_region_labels(
        dataset.coords["country"].values.tolist(),
        region_set=region_set,
        strict=strict,
    )
    aggregated = aggregate_entity_dataset(
        dataset,
        region_labels=region_labels,
        variable_reductions=_variable_reductions(dataset, entity_dim=entity_dim),
        weights=dataset.coords["mw"],
        entity_dim=entity_dim,
    )
    total_mw = (
        xr.DataArray(
            dataset.coords["mw"].values,
            dims=(entity_dim,),
            coords={entity_dim: dataset.coords[entity_dim].values},
        )
        .assign_coords(region=("entity", list(region_labels)))
        .groupby("region")
        .sum(dim=entity_dim)
    )
    aggregated = aggregated.assign_coords(total_mw=("region", total_mw.values))
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
    strict: bool = True,
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Open and aggregate a thermoelectric NetCDF file to the requested regions."""

    with xr.open_dataset(Path(source_path), chunks=chunks) as dataset:
        # Load only the reduced result so callers do not keep the full source file in memory.
        return aggregate_thermoelectric_regions(
            dataset,
            region_set=region_set,
            strict=strict,
        ).load()


def write_aggregated_thermoelectric_file(
    source_path: str | Path,
    target_path: str | Path,
    *,
    region_set: RegionSet,
    strict: bool = True,
    chunks: Mapping[str, int] | None = None,
) -> Path:
    """Aggregate a thermoelectric NetCDF file and write the reduced result."""

    aggregated = aggregate_thermoelectric_file(
        source_path,
        region_set=region_set,
        strict=strict,
        chunks=chunks,
    )
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_netcdf(target)
    return target


def _variable_reductions(
    dataset: xr.Dataset,
    *,
    entity_dim: str,
) -> dict[str, Literal["sum", "weighted_mean"]]:
    weighted = _WET_WEIGHTED_MEAN_VARIABLES | _DRY_WEIGHTED_MEAN_VARIABLES
    summed = _WET_SUM_VARIABLES | _DRY_SUM_VARIABLES

    reductions: dict[str, Literal["sum", "weighted_mean"]] = {}
    for name, variable in dataset.data_vars.items():
        if entity_dim not in variable.dims:
            continue
        if name in summed:
            reductions[name] = "sum"
            continue
        if name in weighted:
            reductions[name] = "weighted_mean"
            continue
        # NB: defaulting unknown entity-level variables to MW-weighted means is permissive.
        # Tighten this to an explicit contract once the production variable set is fixed.
        reductions[name] = "weighted_mean"

    return reductions
