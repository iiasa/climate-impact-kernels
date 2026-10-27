from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import xarray as xr

GRID_COLUMNS = frozenset({"grid_lat", "grid_lon"})
EARTH_RADIUS_M = 6_371_000.0


def grid_cell_area(
    lat: xr.DataArray,
    lon: xr.DataArray,
    *,
    earth_radius_m: float = EARTH_RADIUS_M,
    resolution_deg: float | None = None,
) -> xr.DataArray:
    """Compute spherical grid-cell area weights for a lat/lon grid.

    Cell edges lie midway between neighbouring centres, so a coordinate with a
    single cell has no width of its own: ``resolution_deg`` then gives the cell
    width in degrees, and omitting it raises.
    """

    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("lat and lon must be one-dimensional coordinates")

    lat_edges = _coordinate_edges(
        np.deg2rad(np.asarray(lat.values, dtype=float)), "lat", resolution_deg
    )
    lon_edges = _coordinate_edges(
        np.deg2rad(np.asarray(lon.values, dtype=float)), "lon", resolution_deg
    )
    lat_edges = np.clip(lat_edges, -np.pi / 2, np.pi / 2)

    lat_band = np.abs(np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    lon_width = np.abs(lon_edges[1:] - lon_edges[:-1])
    area = earth_radius_m**2 * lat_band[:, None] * lon_width[None, :]

    return xr.DataArray(
        area,
        dims=(lat.dims[0], lon.dims[0]),
        coords={lat.dims[0]: lat, lon.dims[0]: lon},
        name="cell_area",
        attrs={"units": "m2"},
    )


def attach_gridded_dataset(
    mapped_sites: pd.DataFrame,
    dataset: xr.Dataset,
    *,
    lat_coord: str = "lat",
    lon_coord: str = "lon",
    site_id_column: str,
    site_dim: str = "entity",
    metadata_columns: Iterable[str] | None = None,
) -> xr.Dataset:
    """Sample a gridded dataset at the sites of a mapping table.

    A site is one row of ``mapped_sites``. ``site_id_column`` names the column
    of ``mapped_sites`` whose values label the sites along ``site_dim``. It must exist and hold unique values.
    """

    missing = GRID_COLUMNS.difference(mapped_sites.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"mapped_sites missing required columns: {missing_list}")

    if lat_coord not in dataset.coords:
        raise ValueError(f"{lat_coord!r} is not a coordinate of the dataset")
    if lon_coord not in dataset.coords:
        raise ValueError(f"{lon_coord!r} is not a coordinate of the dataset")

    mapping = mapped_sites.reset_index(drop=True).copy()
    labels = _entity_labels(mapping, site_id_column=site_id_column)
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
        dims=(site_dim,),
        coords={site_dim: labels},
    )
    lon_indexer = xr.DataArray(
        lon_mapping.to_numpy(),
        dims=(site_dim,),
        coords={site_dim: labels},
    )
    attached = dataset.sel({lat_coord: lat_indexer, lon_coord: lon_indexer})

    coords = _metadata_coords(
        mapping=mapping,
        site_dim=site_dim,
        labels=labels,
        metadata_columns=metadata_columns,
    )
    coords["grid_lat"] = (site_dim, mapping["grid_lat"].to_numpy())
    coords["grid_lon"] = (site_dim, mapping["grid_lon"].to_numpy())
    return attached.assign_coords(coords)


def attach_gridded_dataarray(
    mapped_sites: pd.DataFrame,
    data: xr.DataArray,
    **kwargs,
) -> xr.DataArray:
    """DataArray wrapper around :func:`attach_gridded_dataset`."""

    dataset = data.to_dataset(name=data.name or "__value__")
    attached = attach_gridded_dataset(mapped_sites, dataset, **kwargs)
    return attached[data.name or "__value__"]


def refine_to_grid(
    data: xr.DataArray | xr.Dataset,
    *,
    lat: xr.DataArray,
    lon: xr.DataArray,
    lat_coord: str = "lat",
    lon_coord: str = "lon",
) -> xr.DataArray | xr.Dataset:
    """Copy a coarse regular grid onto a finer one nested inside it.

    Each target cell takes the value of the one source cell whose bounds contain
    its centre, so a 1-degree field is placed on a 0.5-degree grid with every source
    value repeated in the four cells inside it and no interpolation. Raises when a
    target centre falls on a source cell boundary or outside the source grid.
    """

    for coord, target in ((lat_coord, lat), (lon_coord, lon)):
        source = np.asarray(data[coord].values, dtype=float)
        if source.size < 2:
            raise ValueError(f"{coord!r} needs at least two source cells to define their bounds")
        half_width = float(np.abs(np.diff(source)).max()) / 2
        target_values = np.asarray(target.values, dtype=float)
        nearest = source[np.abs(source[None, :] - target_values[:, None]).argmin(axis=1)]
        offset = np.abs(nearest - target_values)
        if not (offset < half_width).all():
            bad = target_values[offset >= half_width][:5].tolist()
            raise ValueError(f"target {coord} centres are not inside one source cell: {bad}")

    refined = data.sel({lat_coord: lat.values, lon_coord: lon.values}, method="nearest")
    return refined.assign_coords({lat_coord: lat.values, lon_coord: lon.values})


def nearest_valid_cells(
    lat: np.ndarray | pd.Series,
    lon: np.ndarray | pd.Series,
    valid: xr.DataArray,
    *,
    max_distance_km: float,
    lat_coord: str = "lat",
    lon_coord: str = "lon",
    earth_radius_m: float = EARTH_RADIUS_M,
) -> pd.DataFrame:
    """Find, for each point, the nearest grid cell where ``valid`` is true.

    Distance is great-circle between the point and the cell centre. Returns one
    row per point with ``cell_lat``, ``cell_lon`` and ``distance_km``; the cell
    columns are NaN where the nearest valid cell lies beyond ``max_distance_km``,
    and ``distance_km`` is always reported.
    """

    if set(valid.dims) != {lat_coord, lon_coord}:
        raise ValueError(f"valid must have exactly the dims ({lat_coord!r}, {lon_coord!r})")
    mask = valid.transpose(lat_coord, lon_coord).values.astype(bool)
    if not mask.any():
        raise ValueError("valid has no true cells")
    lat_index, lon_index = np.nonzero(mask)
    cell_lat = np.asarray(valid[lat_coord].values, dtype=float)[lat_index]
    cell_lon = np.asarray(valid[lon_coord].values, dtype=float)[lon_index]
    cells = _unit_vectors(cell_lat, cell_lon)

    points = _unit_vectors(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))
    nearest = np.empty(len(points), dtype=int)
    cosine = np.empty(len(points), dtype=float)
    for start in range(0, len(points), 64):
        dots = points[start : start + 64] @ cells.T
        best = dots.argmax(axis=1)
        nearest[start : start + 64] = best
        cosine[start : start + 64] = dots[np.arange(len(best)), best]
    distance_km = earth_radius_m / 1000.0 * np.arccos(np.clip(cosine, -1.0, 1.0))

    within = distance_km <= max_distance_km
    return pd.DataFrame(
        {
            "cell_lat": np.where(within, cell_lat[nearest], np.nan),
            "cell_lon": np.where(within, cell_lon[nearest], np.nan),
            "distance_km": distance_km,
        }
    )


def _unit_vectors(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)


def _entity_labels(mapping: pd.DataFrame, site_id_column: str) -> list:
    if site_id_column not in mapping.columns:
        raise ValueError(
            f"{site_id_column!r} is not a column in mapped_sites; "
            "pass site_id_column naming the column that identifies each site"
        )
    labels = mapping[site_id_column]
    if not labels.is_unique:
        duplicated = labels[labels.duplicated()].unique().tolist()[:5]
        raise ValueError(
            f"site_id_column {site_id_column!r} must hold unique values; "
            f"duplicated: {duplicated}"
        )
    return labels.tolist()


def _metadata_coords(
    *,
    mapping: pd.DataFrame,
    site_dim: str,
    labels: list,
    metadata_columns: Iterable[str] | None,
) -> dict[str, tuple[str, object]]:
    if metadata_columns is None:
        columns = [column for column in mapping.columns if column not in {"grid_lat", "grid_lon"}]
    else:
        columns = list(metadata_columns)

    coords: dict[str, tuple[str, object]] = {site_dim: labels}
    for column in columns:
        if column not in mapping.columns:
            raise ValueError(f"{column!r} is not a column in mapped_sites")
        coords[column] = (site_dim, mapping[column].to_numpy())
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


def _coordinate_edges(values: np.ndarray, name: str, resolution_deg: float | None) -> np.ndarray:
    """Cell edges (radians) of centres given in radians."""

    if values.ndim != 1:
        raise ValueError("coordinate values must be one-dimensional")
    if values.size == 1:
        if resolution_deg is None:
            raise ValueError(
                f"{name!r} has a single cell, whose width cannot be inferred; "
                "pass resolution_deg with the cell width in degrees"
            )
        if not resolution_deg > 0:
            raise ValueError(f"resolution_deg must be positive, got {resolution_deg!r}")
        delta = np.deg2rad(resolution_deg)
        return np.asarray([values[0] - delta / 2, values[0] + delta / 2], dtype=float)

    midpoints = (values[:-1] + values[1:]) / 2
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]])
