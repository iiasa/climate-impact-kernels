from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeAlias

import numpy as np
import pandas as pd
import xarray as xr

ArrayLike: TypeAlias = xr.DataArray | xr.Dataset
GRID_COLUMNS = frozenset({"grid_lat", "grid_lon"})


def weighted_average(
    values: ArrayLike,
    weights: xr.DataArray,
    dims: str | Sequence[str],
    mask: xr.DataArray | None = None,
) -> ArrayLike:
    """Compute a masked weighted average over dims."""

    dims_ = [dims] if isinstance(dims, str) else list(dims)
    masked_values = values.where(mask) if mask is not None else values
    masked_weights = weights.where(mask, 0) if mask is not None else weights

    total_weight = masked_weights.sum(dim=dims_)
    if bool((total_weight <= 0).any()):
        raise ValueError("total weight must be positive along the aggregation dims")

    return masked_values.weighted(masked_weights).mean(dim=dims_)


def attach_gridded_dataset(
    mapped_entities: pd.DataFrame,
    dataset: xr.Dataset,
    *,
    lat_coord: str = "lat",
    lon_coord: str = "lon",
    entity_dim: str = "entity",
    entity_id_column: str | None = None,
    metadata_columns: Iterable[str] | None = None,
) -> xr.Dataset:
    """Sample a gridded dataset onto mapped entities."""

    missing = GRID_COLUMNS.difference(mapped_entities.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"mapped_entities missing required columns: {missing_list}")

    if lat_coord not in dataset.coords:
        raise ValueError(f"{lat_coord!r} is not a coordinate of the dataset")
    if lon_coord not in dataset.coords:
        raise ValueError(f"{lon_coord!r} is not a coordinate of the dataset")

    mapping = mapped_entities.reset_index(drop=True).copy()
    labels = _entity_labels(mapping, entity_id_column=entity_id_column)
    lon_mapping = _normalized_longitudes_for_dataset(
        mapping["grid_lon"],
        dataset=dataset,
        lon_coord=lon_coord,
    )
    _validate_cells_exist(
        mapping=mapping,
        lon_mapping=lon_mapping,
        dataset=dataset,
        lat_coord=lat_coord,
        lon_coord=lon_coord,
    )

    lat_indexer = xr.DataArray(
        mapping["grid_lat"].to_numpy(),
        dims=(entity_dim,),
        coords={entity_dim: labels},
    )
    lon_indexer = xr.DataArray(
        lon_mapping.to_numpy(),
        dims=(entity_dim,),
        coords={entity_dim: labels},
    )
    attached = dataset.sel({lat_coord: lat_indexer, lon_coord: lon_indexer})

    coords = _metadata_coords(
        mapping=mapping,
        entity_dim=entity_dim,
        labels=labels,
        metadata_columns=metadata_columns,
    )
    coords["grid_lat"] = (entity_dim, mapping["grid_lat"].to_numpy())
    coords["grid_lon"] = (entity_dim, mapping["grid_lon"].to_numpy())
    return attached.assign_coords(coords)


def attach_gridded_dataarray(
    mapped_entities: pd.DataFrame,
    data: xr.DataArray,
    **kwargs,
) -> xr.DataArray:
    """DataArray wrapper around :func:`attach_gridded_dataset`."""

    dataset = data.to_dataset(name=data.name or "__value__")
    attached = attach_gridded_dataset(mapped_entities, dataset, **kwargs)
    return attached[data.name or "__value__"]


def _entity_labels(mapping: pd.DataFrame, entity_id_column: str | None) -> list:
    if entity_id_column is not None:
        if entity_id_column not in mapping.columns:
            raise ValueError(f"{entity_id_column!r} is not a column in mapped_entities")
        return mapping[entity_id_column].tolist()

    for candidate in ("UNITID", "LOCATIONID"):
        if candidate in mapping.columns and mapping[candidate].is_unique:
            return mapping[candidate].tolist()

    return mapping.index.tolist()


def _metadata_coords(
    *,
    mapping: pd.DataFrame,
    entity_dim: str,
    labels: list,
    metadata_columns: Iterable[str] | None,
) -> dict[str, tuple[str, object]]:
    if metadata_columns is None:
        columns = [column for column in mapping.columns if column not in {"grid_lat", "grid_lon"}]
    else:
        columns = list(metadata_columns)

    coords: dict[str, tuple[str, object]] = {entity_dim: labels}
    for column in columns:
        if column not in mapping.columns:
            raise ValueError(f"{column!r} is not a column in mapped_entities")
        coords[column] = (entity_dim, mapping[column].to_numpy())
    return coords


def _validate_cells_exist(
    *,
    mapping: pd.DataFrame,
    lon_mapping: pd.Series,
    dataset: xr.Dataset,
    lat_coord: str,
    lon_coord: str,
) -> None:
    available_lat = set(dataset.coords[lat_coord].values.tolist())
    available_lon = set(dataset.coords[lon_coord].values.tolist())

    missing_lat = sorted(set(mapping["grid_lat"].tolist()) - available_lat)
    missing_lon = sorted(set(lon_mapping.tolist()) - available_lon)
    if missing_lat or missing_lon:
        parts: list[str] = []
        if missing_lat:
            parts.append(f"missing {lat_coord} values: {missing_lat[:5]}")
        if missing_lon:
            parts.append(f"missing {lon_coord} values: {missing_lon[:5]}")
        raise ValueError("; ".join(parts))


def _normalized_longitudes_for_dataset(
    longitudes: pd.Series,
    *,
    dataset: xr.Dataset,
    lon_coord: str,
) -> pd.Series:
    available_lon = pd.Series(dataset.coords[lon_coord].values)
    if available_lon.empty:
        return longitudes.astype(float)

    if available_lon.min() >= 0 and available_lon.max() > 180:
        normalized = np.mod(longitudes.astype(float), 360.0)
        normalized = normalized.where(normalized != 0.0, 0.0)
        return normalized

    return longitudes.astype(float)
