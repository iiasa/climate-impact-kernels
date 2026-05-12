from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import xarray as xr

ArrayLike: TypeAlias = xr.DataArray | xr.Dataset

def interpolate_along_warming_level(
    values: ArrayLike,
    target_levels: float | Sequence[float],
    dim: str = "warming_level",
) -> ArrayLike:
    """Linearly interpolate along the warming-level axis."""

    if dim not in values.dims:
        raise ValueError(f"{dim!r} is not a dimension of the input data")

    requested = np.atleast_1d(np.asarray(target_levels, dtype=float))
    available = np.asarray(values[dim].values, dtype=float)
    if requested.min() < available.min() or requested.max() > available.max():
        raise ValueError("target_levels lie outside the available warming-level range")

    interpolated = _interp_arraylike(values=values, target_levels=requested, dim=dim)
    if np.isscalar(target_levels):
        return interpolated.squeeze(dim=dim, drop=True)
    return interpolated


def _interp_arraylike(values: ArrayLike, target_levels: np.ndarray, dim: str) -> ArrayLike:
    match values:
        case xr.DataArray():
            return _interp_dataarray(values=values, target_levels=target_levels, dim=dim)
        case xr.Dataset():
            return values.map(
                lambda array: _interp_dataarray(values=array, target_levels=target_levels, dim=dim)
            )
        case _:
            raise TypeError("values must be an xarray DataArray or Dataset")


def _interp_dataarray(values: xr.DataArray, target_levels: np.ndarray, dim: str) -> xr.DataArray:
    source_levels = np.asarray(values[dim].values, dtype=float)
    other_dims = [name for name in values.dims if name != dim]
    transposed = values.transpose(dim, *other_dims)
    target_dim = "__target_warming_level"

    interpolated = xr.apply_ufunc(
        lambda array: np.interp(target_levels, source_levels, array),
        transposed,
        input_core_dims=[[dim]],
        output_core_dims=[[target_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[values.dtype],
        dask_gufunc_kwargs={"output_sizes": {target_dim: target_levels.size}},
    )
    interpolated = interpolated.rename({target_dim: dim}).assign_coords({dim: target_levels})
    return interpolated.transpose(*values.dims)
