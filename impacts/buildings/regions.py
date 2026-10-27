"""Country-label rasters and site-to-country aggregation.

The country raster maps each grid cell to an ISO3 (or other lookup column)
label, using the GAUL country raster and lookup CSV that the CHILLED workflow
also uses for the R12 region raster. Any analysis that needs to assign gridded
site outputs to countries can use these functions; they are not specific to
RIME-X. A site is one grid cell, held on the ``entity`` dimension.

The functions for CHILLED outputs follow. They convert one CHILLED matrix-runner
output (a seasonal NetCDF with sites stacked on ``entity``) into the country-level
form that the warming-level summaries for RIME-X are built from. They cover only
what is specific to CHILLED: filename parsing, the reduction for each CHILLED
output variable, and expanding the matrix-index metadata into dimensions.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from impacts.buildings._config import (
    ALPS_DATA_VARIABLE,
    GAUL_CSV_FILE,
    GAUL_RASTER_FILE,
    resolve_input_path,
)
from impacts.climatecore import attach_gridded_dataset, grid_cell_area
from impacts.climatecore.regions import aggregate_site_dataset


def load_country_raster(
    *,
    gaul_raster_path: Path | str | None = None,
    gaul_csv_path: Path | str | None = None,
    country_column: str = "ISO3",
) -> xr.DataArray:
    """Build a country-label raster from the GAUL country raster and lookup CSV.

    Unset paths are looked up in the ``ALPS_DATA_PATH`` directory.
    ``country_column`` names the lookup CSV column holding the labels. It must
    exist and hold a label for every row: a missing column or a null label raises.
    """

    gaul_raster_path = resolve_input_path(gaul_raster_path, ALPS_DATA_VARIABLE, GAUL_RASTER_FILE)
    gaul_csv = pd.read_csv(resolve_input_path(gaul_csv_path, ALPS_DATA_VARIABLE, GAUL_CSV_FILE))
    if country_column not in gaul_csv.columns:
        raise ValueError(
            f"country column {country_column!r} is not in the GAUL lookup CSV "
            f"(columns: {', '.join(map(str, gaul_csv.columns))}); "
            "pass country_column naming the column that holds the country labels"
        )
    unlabelled = gaul_csv[country_column].isna()
    if unlabelled.any():
        ids = gaul_csv.loc[unlabelled, "ID"].tolist()[:5]
        raise ValueError(
            f"GAUL lookup CSV column {country_column!r} is null for {int(unlabelled.sum())} rows "
            f"(first IDs: {ids}); fill the labels or pass another country_column"
        )

    gaul_to_country: dict[int, str] = {}
    for _, row in gaul_csv.iterrows():
        country = str(row[country_column]).strip()
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


def aggregate_site_to_country(
    dataset: xr.Dataset,
    country_raster: xr.DataArray,
    *,
    variable_reductions: Mapping[str, str],
) -> xr.Dataset:
    """Area-weighted site-to-country reduction for a Dataset with sites stacked on ``entity``.

    Caller supplies the per-variable reduction map (typically ``sum`` for
    extensive quantities and ``weighted_mean`` for rates or indicator-day
    counts). This function attaches lat/lon coordinates to country labels
    and cell areas, then dispatches the reduction through
    ``impacts.climatecore.regions.aggregate_site_dataset``.
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
        site_id_column="entity",
        metadata_columns=("entity",),
    )
    region_labels = attached_lookup["country"].values.tolist()
    return aggregate_site_dataset(
        dataset,
        region_labels,
        variable_reductions=variable_reductions,
        site_dim="entity",
        region_dim="country",
        weights=attached_lookup["cell_area"],
    )


CHILLED_VARIABLE_REDUCTIONS = {
    "cooling_energy_total": "weighted_mean",
    "heating_energy": "weighted_mean",
    "Nd": "weighted_mean",
    "Nf": "weighted_mean",
}

_FILENAME_PATTERN = re.compile(
    r"^seasonal_(?P<gcm>.+)_(?P<scenario>historical|ssp126|ssp245|ssp370|ssp585)"
    r"_(?P<building_type>sfh|mfh)_(?P<urt>rur|urb)_(?P<vintage>s1|s2|s3|s51|s52)"
    r"_(?P<year_start>\d{4})_(?P<year_end>\d{4})\.nc$"
)


def parse_chilled_matrix_filename(path: str | Path) -> dict[str, str | int]:
    """Parse one CHILLED seasonal matrix-runner filename."""

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


def aggregate_chilled_to_country(
    dataset: xr.Dataset,
    country_raster: xr.DataArray,
) -> xr.Dataset:
    """Aggregate one CHILLED seasonal dataset with sites on ``entity`` to countries."""

    target_variables = [
        name for name in CHILLED_VARIABLE_REDUCTIONS if name in dataset.data_vars
    ]
    if not target_variables:
        raise ValueError("dataset does not contain any supported CHILLED output variables")

    aggregated = aggregate_site_to_country(
        dataset[target_variables],
        country_raster,
        variable_reductions=CHILLED_VARIABLE_REDUCTIONS,
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
