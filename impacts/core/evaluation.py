from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import pandas as pd
import xarray as xr

from climateresponse.spatial import attach_gridded_dataset


@dataclass(frozen=True)
class EvaluationResult:
    """Structured result for mapped-entity evaluation."""

    data_vars: Mapping[str, xr.DataArray]
    attrs: Mapping[str, object] | None = None


def evaluate_mapped_dataset(
    mapped_entities: pd.DataFrame,
    climate_inputs: xr.Dataset,
    *,
    normalize_inputs: Callable[[xr.Dataset], xr.Dataset],
    physics_kernel: Callable[[xr.Dataset], xr.Dataset | EvaluationResult],
    metadata_columns: Iterable[str] | None = None,
    clips: Mapping[str, tuple[float | None, float | None]] | None = None,
    transpose_order: tuple[str, ...] = ("time", "entity"),
) -> xr.Dataset:
    """Normalize, spatially attach, evaluate, clip, and package mapped outputs."""

    normalized_inputs = normalize_inputs(climate_inputs)
    attached = attach_gridded_dataset(
        mapped_entities,
        normalized_inputs,
        metadata_columns=metadata_columns,
    )
    evaluated = _coerce_result(
        result=physics_kernel(attached),
        coords=attached.coords,
    )

    for variable_name, (lower, upper) in (clips or {}).items():
        if variable_name not in evaluated.data_vars:
            raise ValueError(f"{variable_name!r} not returned by physics_kernel")
        evaluated[variable_name] = evaluated[variable_name].clip(min=lower, max=upper)

    if all(dim in evaluated.dims for dim in transpose_order):
        return evaluated.transpose(*transpose_order)
    return evaluated


def evaluate_gridded_dataset(
    climate_inputs: xr.Dataset,
    parameters: xr.Dataset,
    *,
    normalize_inputs: Callable[[xr.Dataset], xr.Dataset],
    physics_kernel: Callable[[xr.Dataset], xr.Dataset | EvaluationResult],
    spatial_dims: tuple[str, ...] = ("lat", "lon"),
    entity_dim: str = "entity",
    clips: Mapping[str, tuple[float | None, float | None]] | None = None,
    transpose_order: tuple[str, ...] = ("time", "entity"),
) -> xr.Dataset:
    """Normalize, stack spatial dims to entity, attach parameters, evaluate, clip."""

    normalized = normalize_inputs(climate_inputs)
    stacked = normalized.stack({entity_dim: spatial_dims})

    param_coords: dict[str, xr.DataArray] = {}
    for name in parameters.data_vars:
        param = parameters[name]
        if set(spatial_dims).issubset(param.dims):
            param_coords[name] = param.stack({entity_dim: spatial_dims})
        else:
            param_coords[name] = param

    attached = stacked.assign_coords(param_coords)
    evaluated = _coerce_result(
        result=physics_kernel(attached),
        coords=attached.coords,
    )

    for variable_name, (lower, upper) in (clips or {}).items():
        if variable_name not in evaluated.data_vars:
            raise ValueError(f"{variable_name!r} not returned by physics_kernel")
        evaluated[variable_name] = evaluated[variable_name].clip(min=lower, max=upper)

    if all(dim in evaluated.dims for dim in transpose_order):
        return evaluated.transpose(*transpose_order)
    return evaluated


def _coerce_result(
    *,
    result: xr.Dataset | EvaluationResult,
    coords: xr.core.coordinates.DatasetCoordinates,
) -> xr.Dataset:
    if isinstance(result, xr.Dataset):
        return result.assign_coords(coords)

    return xr.Dataset(
        data_vars=dict(result.data_vars),
        coords=coords,
        attrs={} if result.attrs is None else dict(result.attrs),
    )
