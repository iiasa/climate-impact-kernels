from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import xarray as xr
from impacts.climatecore import attach_gridded_dataset, grid_cell_area
from impacts.climatecore.regions import aggregate_entity_dataset

from impacts.domain.buildings.chilled_config import DEFAULT_GAUL_CSV_PATH, DEFAULT_GAUL_RASTER_PATH

CHILLED_VARIABLE_REDUCTIONS = {
    "cooling_energy_total": "sum",
    "heating_energy": "sum",
    "Nd": "weighted_mean",
    "Nf": "weighted_mean",
}

_FILENAME_PATTERN = re.compile(
    r"^seasonal_(?P<gcm>.+)_(?P<scenario>historical|ssp126|ssp245|ssp370|ssp585)"
    r"_(?P<building_type>sfh|mfh)_(?P<urt>rur|urb)_(?P<vintage>s1|s2|s3|s51|s52)"
    r"_(?P<year_start>\d{4})_(?P<year_end>\d{4})\.nc$"
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


def parse_chilled_matrix_filename(path: str | Path) -> dict[str, str | int]:
    """Parse one production CHILLED seasonal filename."""

    match = _FILENAME_PATTERN.match(Path(path).name)
    if match is None:
        raise ValueError(f"unexpected CHILLED seasonal filename: {path}")

    return {
        "gcm": match.group("gcm").replace("_", "-").upper(),
        "scenario": match.group("scenario"),
        "building_type": match.group("building_type"),
        "urt": match.group("urt"),
        "vintage": match.group("vintage"),
        "year_start": int(match.group("year_start")),
        "year_end": int(match.group("year_end")),
    }


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


def aggregate_chilled_to_country(
    dataset: xr.Dataset,
    country_raster: xr.DataArray,
) -> xr.Dataset:
    """Aggregate one CHILLED seasonal entity dataset to countries."""

    if "entity" not in dataset.dims:
        raise ValueError("dataset must define an entity dimension")
    for coordinate in ("lat", "lon"):
        if coordinate not in dataset.coords:
            raise ValueError(f"dataset must define {coordinate!r} coordinates on entity")

    target_variables = [
        name for name in CHILLED_VARIABLE_REDUCTIONS if name in dataset.data_vars
    ]
    if not target_variables:
        raise ValueError("dataset does not contain any supported CHILLED output variables")

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
    aggregated = aggregate_entity_dataset(
        dataset[target_variables],
        region_labels,
        variable_reductions=CHILLED_VARIABLE_REDUCTIONS,
        entity_dim="entity",
        region_dim="country",
        weights=attached_lookup["cell_area"],
    )
    return aggregated.transpose("season", "year", "country", missing_dims="ignore")


def expand_chilled_metadata(
    dataset: xr.Dataset,
    metadata: dict[str, str | int],
) -> xr.Dataset:
    """Attach filename-derived CHILLED matrix metadata as singleton dims."""

    expanded = dataset.expand_dims(
        gcm=[str(metadata["gcm"])],
        scenario=[str(metadata["scenario"])],
        building_type=[str(metadata["building_type"])],
        urt=[str(metadata["urt"])],
        vintage=[str(metadata["vintage"])],
    )
    return expanded.transpose(
        "gcm",
        "scenario",
        "building_type",
        "urt",
        "vintage",
        "season",
        "year",
        "country",
        missing_dims="ignore",
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
