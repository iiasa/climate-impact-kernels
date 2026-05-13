"""Country-label rasters and entity→country aggregation.

The country raster maps each grid cell to an ISO3 (or other lookup column)
label via the GAUL country raster + lookup CSV that the CHILLED workflow
also uses for the R12 region raster. Any domain analysis that needs to
attribute gridded entity outputs to countries can consume these primitives;
they are not specific to the RIME-X downstream.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from impacts.climatecore import attach_gridded_dataset, grid_cell_area
from impacts.climatecore.regions import aggregate_entity_dataset

from impacts.domain.buildings.chilled_config import (
    DEFAULT_GAUL_CSV_PATH,
    DEFAULT_GAUL_RASTER_PATH,
)

_COUNTRY_COLUMN_CANDIDATES = (
    "ISO3",
    "iso3",
    "country",
    "COUNTRY",
    "NAME_ENGLI",
    "ADM0_NAME",
    "NAME",
)


def load_country_raster(
    *,
    gaul_raster_path: Path | str = DEFAULT_GAUL_RASTER_PATH,
    gaul_csv_path: Path | str = DEFAULT_GAUL_CSV_PATH,
    country_column: str | None = "ISO3",
) -> xr.DataArray:
    """Build a country-label raster from the GAUL country raster and lookup CSV."""

    gaul_csv = pd.read_csv(gaul_csv_path)
    label_column = _resolve_country_column(gaul_csv.columns, country_column=country_column)

    gaul_to_country: dict[int, str] = {}
    for _, row in gaul_csv.iterrows():
        country = str(row[label_column]).strip()
        raster_id = int(row["ID"])
        if country:
            gaul_to_country[raster_id] = country

    gaul_raster = xr.open_dataarray(gaul_raster_path).sortby("lat")
    country_array = np.full(gaul_raster.shape, "", dtype="U32")
    for raster_id, country in gaul_to_country.items():
        country_array[gaul_raster.values == raster_id] = country

    return xr.DataArray(
        country_array,
        dims=gaul_raster.dims,
        coords=gaul_raster.coords,
        name="country",
    )


def aggregate_entity_to_country(
    dataset: xr.Dataset,
    country_raster: xr.DataArray,
    *,
    variable_reductions: Mapping[str, str],
) -> xr.Dataset:
    """Area-weighted entity→country reduction for an entity-stacked Dataset.

    Caller supplies the per-variable reduction map (typically ``sum`` for
    extensive quantities and ``weighted_mean`` for rates or indicator-day
    counts). This primitive attaches lat/lon coordinates to country labels
    and cell areas, then dispatches the reduction through
    ``impacts.climatecore.regions.aggregate_entity_dataset``.
    """

    if "entity" not in dataset.dims:
        raise ValueError("dataset must define an entity dimension")
    for coordinate in ("lat", "lon"):
        if coordinate not in dataset.coords:
            raise ValueError(f"dataset must define {coordinate!r} coordinates on entity")

    entity_mapping = pd.DataFrame(
        {
            "entity": dataset["entity"].values,
            "grid_lat": dataset.coords["lat"].values,
            "grid_lon": dataset.coords["lon"].values,
        }
    )
    lookup_dataset = xr.Dataset(
        data_vars={
            "country": country_raster,
            "cell_area": grid_cell_area(country_raster["lat"], country_raster["lon"]),
        }
    )
    attached_lookup = attach_gridded_dataset(
        entity_mapping,
        lookup_dataset,
        entity_id_column="entity",
        metadata_columns=("entity",),
    )
    region_labels = attached_lookup["country"].values.tolist()
    return aggregate_entity_dataset(
        dataset,
        region_labels,
        variable_reductions=variable_reductions,
        entity_dim="entity",
        region_dim="country",
        weights=attached_lookup["cell_area"],
    )


def _resolve_country_column(
    columns: pd.Index,
    *,
    country_column: str | None,
) -> str:
    if country_column is not None and country_column in columns:
        return country_column

    for candidate in _COUNTRY_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate

    raise ValueError("could not find a country label column in the GAUL lookup CSV")
