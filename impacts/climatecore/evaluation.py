from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import pandas as pd
import xarray as xr

from .transforms.spatial import attach_gridded_dataset


@dataclass(frozen=True)
class EvaluationResult:
    """Structured result for mapped-entity evaluation."""

    data_vars: Mapping[str, xr.DataArray]
    attrs: Mapping[str, object] | None = None


def evaluate_mapped_dataset(
    mapped_entities: pd.DataFrame,
    climate_inputs: xr.Dataset,
    *,
    normalize: Callable[[xr.Dataset], xr.Dataset],
    physics: Callable[[xr.Dataset], xr.Dataset | EvaluationResult],
    metadata_columns: Iterable[str] | None = None,
    gridded_coords: xr.Dataset | None = None,
    entry_chunks: Mapping[str, int] | None = None,
    clips: Mapping[str, tuple[float | None, float | None]] | None = None,
    transpose_order: tuple[str, ...] = ("time", "entity"),
) -> xr.Dataset:
    """Normalize, spatially attach, evaluate, clip, and package mapped outputs.

    ``gridded_coords`` carries additional gridded fields (e.g. a monthly solar
    climatology) that are sampled at the same entity points as the climate and
    attached as coordinates, letting a kernel read a per-entity profile that does
    not fit in the mapping DataFrame.
    """

    attached = attach_gridded_dataset(
        mapped_entities,
        normalize(climate_inputs),
        metadata_columns=metadata_columns,
    )
    if gridded_coords is not None:
        sampled = attach_gridded_dataset(mapped_entities, gridded_coords, metadata_columns=())
        attached = attached.assign_coords({name: sampled[name] for name in gridded_coords.data_vars})

    attached = _rechunk_for_entry(attached, entry_chunks)
    evaluated = _coerce_result(result=physics(attached), coords=attached.coords)
    evaluated = _clip_variables(evaluated, clips)
    evaluated = _drop_unused_dim_coords(evaluated)

    if all(dim in evaluated.dims for dim in transpose_order):
        return evaluated.transpose(*transpose_order)
    return evaluated


def evaluate_gridded_dataset(
    climate_inputs: xr.Dataset,
    parameters: xr.Dataset,
    *,
    normalize: Callable[[xr.Dataset], xr.Dataset],
    physics: Callable[[xr.Dataset], xr.Dataset | EvaluationResult],
    spatial_dims: tuple[str, ...] = ("lat", "lon"),
    entity_dim: str = "entity",
    entry_chunks: Mapping[str, int] | None = None,
    clips: Mapping[str, tuple[float | None, float | None]] | None = None,
    transpose_order: tuple[str, ...] = ("time", "entity"),
) -> xr.Dataset:
    """Normalize, stack spatial dims to entity, attach parameters, evaluate, clip.

    Parameter variables that carry the spatial dims (e.g. a ``(month, lat, lon)``
    solar field) are stacked to ``(..., entity)`` and attached as coordinates;
    spatially-constant parameters are attached as-is.
    """

    stacked = normalize(climate_inputs).stack({entity_dim: spatial_dims})

    param_coords: dict[str, xr.DataArray] = {}
    for name in parameters.data_vars:
        param = parameters[name]
        if set(spatial_dims).issubset(param.dims):
            param_coords[name] = param.stack({entity_dim: spatial_dims})
        else:
            param_coords[name] = param

    attached = _rechunk_for_entry(stacked.assign_coords(param_coords), entry_chunks)
    evaluated = _coerce_result(result=physics(attached), coords=attached.coords)
    evaluated = _clip_variables(evaluated, clips)
    evaluated = _drop_unused_dim_coords(evaluated)

    if all(dim in evaluated.dims for dim in transpose_order):
        return evaluated.transpose(*transpose_order)
    return evaluated


def _clip_variables(
    dataset: xr.Dataset,
    clips: Mapping[str, tuple[float | None, float | None]] | None,
) -> xr.Dataset:
    for variable_name, (lower, upper) in (clips or {}).items():
        if variable_name not in dataset.data_vars:
            raise ValueError(f"{variable_name!r} not returned by physics")
        dataset[variable_name] = dataset[variable_name].clip(min=lower, max=upper)
    return dataset


def _rechunk_for_entry(
    dataset: xr.Dataset,
    entry_chunks: Mapping[str, int] | None,
) -> xr.Dataset:
    """Rechunk on entry so the kernel sees declared chunk widths, not loader chunks."""

    if not entry_chunks:
        return dataset
    sizes = {
        dim: min(width, dataset.sizes[dim])
        for dim, width in entry_chunks.items()
        if dim in dataset.sizes
    }
    return dataset.chunk(sizes) if sizes else dataset


def _drop_unused_dim_coords(dataset: xr.Dataset) -> xr.Dataset:
    """Drop coordinates whose dimensions no output variable uses.

    A gridded field attached for the kernel to read — a ``(month, entity)`` solar
    climatology — would otherwise leave a stray ``month`` dimension on the result.
    """

    if not dataset.data_vars:
        return dataset
    data_dims = set().union(*(set(var.dims) for var in dataset.data_vars.values()))
    unused = [name for name, coord in dataset.coords.items() if not set(coord.dims) <= data_dims]
    return dataset.drop_vars(unused) if unused else dataset


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
