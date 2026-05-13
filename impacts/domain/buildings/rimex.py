"""CHILLED → RIME-X shape adapter.

Handles the per-file work of converting one CHILLED matrix-runner output
(entity-stacked seasonal NetCDF) into the country-level shape that the
RIME-X warming-level surface construction in
``impacts.climatecore.warming`` consumes. Country labelling and the
generic entity→country reduction live in
``impacts.domain.buildings.country``; this module is only the
CHILLED-specific orchestration: filename parsing, the reduction map for
CHILLED outputs, and matrix-index metadata expansion.
"""

from __future__ import annotations

from pathlib import Path
import re

import xarray as xr

from impacts.domain.buildings.country import aggregate_entity_to_country

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


def aggregate_chilled_to_country(
    dataset: xr.Dataset,
    country_raster: xr.DataArray,
) -> xr.Dataset:
    """Aggregate one CHILLED seasonal entity dataset to countries."""

    target_variables = [
        name for name in CHILLED_VARIABLE_REDUCTIONS if name in dataset.data_vars
    ]
    if not target_variables:
        raise ValueError("dataset does not contain any supported CHILLED output variables")

    aggregated = aggregate_entity_to_country(
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
