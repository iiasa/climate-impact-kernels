from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = frozenset(
    {
        "UNITID",
        "LOCATIONID",
        "lat",
        "lon",
        "mw",
        "has_coords",
    }
)


@dataclass(frozen=True)
class LocationAmbiguity:
    """Aggregate ambiguity statistics for LOCATIONID-level cell consistency."""

    ambiguous_locations: int
    ambiguous_units: int
    ambiguous_mw: float
    share_of_coordinate_units: float
    share_of_coordinate_mw: float


@dataclass(frozen=True)
class CellMappingSummary:
    """High-level summary of the unit-to-cell mapping."""

    coordinate_units: int
    coordinate_mw: float
    occupied_cells: int
    median_units_per_cell: float
    p95_units_per_cell: float
    max_units_per_cell: int
    ambiguity: LocationAmbiguity


DEFAULT_MAPPING_COLUMNS = [
    "UNITID",
    "LOCATIONID",
    "country",
    "lat",
    "lon",
    "grid_lat",
    "grid_lon",
    "cell_id",
    "mw",
    "location_cell_count",
    "location_is_single_cell",
]

DEFAULT_RECOVERED_MAPPING_COLUMNS = [
    "UNITID",
    "LOCATIONID",
    "country",
    "lat",
    "lon",
    "grid_lat",
    "grid_lon",
    "cell_id",
    "mw",
    "coordinate_source",
    "location_cell_count",
    "location_is_single_cell",
]

DEFAULT_LOCATION_COLUMNS = [
    "LOCATIONID",
    "country",
    "grid_lat",
    "grid_lon",
    "cell_id",
    "units",
    "mw",
    "location_is_single_cell",
    "source_resolution",
]

DEFAULT_COUNTRY_COLUMNS = [
    "country",
    "entities",
    "mw",
    "source_resolution",
]


def snap_to_half_degree_cell(values: pd.Series) -> pd.Series:
    """Snap coordinates to the 0.5-degree grid centered on .25/.75."""

    values = values.astype(float)
    floored = np.floor(values)
    centers = np.where(values - floored < 0.5, 0.25, 0.75)
    return pd.Series(floored + centers, index=values.index, dtype="float64")


def build_unit_cell_mapping(thermal_plants: pd.DataFrame) -> pd.DataFrame:
    """Build a unit-level mapping from plants to snapped climate cells."""

    missing = REQUIRED_COLUMNS.difference(thermal_plants.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"missing required columns: {missing_list}")

    coord_mask = thermal_plants["has_coords"].fillna(False).eq(True)
    mapped = thermal_plants.loc[coord_mask].copy()
    mapped["coordinate_source"] = "source"
    mapped["grid_lat"] = snap_to_half_degree_cell(mapped["lat"])
    mapped["grid_lon"] = snap_to_half_degree_cell(mapped["lon"])
    mapped["cell_id"] = (
        mapped["grid_lat"].map("{:.2f}".format) + "," + mapped["grid_lon"].map("{:.2f}".format)
    )
    cell_counts = mapped.groupby("LOCATIONID").apply(
        lambda frame: frame[["grid_lat", "grid_lon"]].drop_duplicates().shape[0],
        include_groups=False,
    )
    mapped["location_cell_count"] = mapped["LOCATIONID"].map(cell_counts).astype("int64")
    mapped["location_is_single_cell"] = mapped["location_cell_count"].eq(1)
    return mapped


def build_recovered_unit_cell_mapping(thermal_plants: pd.DataFrame) -> pd.DataFrame:
    """Build a unit-level mapping with safe LOCATIONID-based coordinate inheritance."""

    missing = REQUIRED_COLUMNS.difference(thermal_plants.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"missing required columns: {missing_list}")

    exact = build_unit_cell_mapping(thermal_plants)

    coord_mask = thermal_plants["has_coords"].fillna(False).eq(True)
    missing_coords = thermal_plants.loc[~coord_mask].copy()
    if missing_coords.empty:
        return exact.loc[:, DEFAULT_RECOVERED_MAPPING_COLUMNS]
    missing_coords = missing_coords.drop(columns=["lat", "lon"], errors="ignore")

    inheritable = (
        exact.groupby("LOCATIONID", as_index=False)
        .agg(
            lat=("lat", "first"),
            lon=("lon", "first"),
            grid_lat=("grid_lat", "first"),
            grid_lon=("grid_lon", "first"),
            cell_id=("cell_id", "first"),
            location_cell_count=("location_cell_count", "first"),
            location_is_single_cell=("location_is_single_cell", "first"),
            lat_nunique=("lat", "nunique"),
            lon_nunique=("lon", "nunique"),
        )
    )
    inheritable = inheritable[
        inheritable["lat_nunique"].eq(1) & inheritable["lon_nunique"].eq(1)
    ].drop(columns=["lat_nunique", "lon_nunique"])

    recovered = missing_coords.merge(inheritable, on="LOCATIONID", how="inner")
    recovered["coordinate_source"] = "inherited_location"

    combined = pd.concat(
        [
            exact.loc[:, DEFAULT_RECOVERED_MAPPING_COLUMNS],
            recovered.loc[:, DEFAULT_RECOVERED_MAPPING_COLUMNS],
        ],
        ignore_index=True,
    )
    return combined.sort_values(["country", "LOCATIONID", "UNITID"], kind="stable").reset_index(
        drop=True
    )


def summarize_location_ambiguity(mapped_units: pd.DataFrame) -> LocationAmbiguity:
    """Summarize how often LOCATIONIDs span more than one snapped cell."""

    if mapped_units.empty:
        return LocationAmbiguity(0, 0, 0.0, 0.0, 0.0)

    grouped = mapped_units.groupby("LOCATIONID").agg(
        units=("UNITID", "size"),
        mw=("mw", "sum"),
        grid_lat_nunique=("grid_lat", "nunique"),
        grid_lon_nunique=("grid_lon", "nunique"),
    )
    ambiguous = grouped[
        (grouped["grid_lat_nunique"] > 1) | (grouped["grid_lon_nunique"] > 1)
    ]

    coordinate_units = int(len(mapped_units))
    coordinate_mw = float(mapped_units["mw"].sum())
    ambiguous_units = int(ambiguous["units"].sum())
    ambiguous_mw = float(ambiguous["mw"].sum())

    return LocationAmbiguity(
        ambiguous_locations=int(len(ambiguous)),
        ambiguous_units=ambiguous_units,
        ambiguous_mw=ambiguous_mw,
        share_of_coordinate_units=ambiguous_units / coordinate_units,
        share_of_coordinate_mw=ambiguous_mw / coordinate_mw if coordinate_mw else 0.0,
    )


def summarize_mapping(mapped_units: pd.DataFrame) -> CellMappingSummary:
    """Summarize the occupied climate cells after unit-level snapping."""

    if mapped_units.empty:
        return CellMappingSummary(
            coordinate_units=0,
            coordinate_mw=0.0,
            occupied_cells=0,
            median_units_per_cell=0.0,
            p95_units_per_cell=0.0,
            max_units_per_cell=0,
            ambiguity=LocationAmbiguity(0, 0, 0.0, 0.0, 0.0),
        )

    cell_counts = mapped_units.groupby(["grid_lat", "grid_lon"]).size()
    ambiguity = summarize_location_ambiguity(mapped_units)

    return CellMappingSummary(
        coordinate_units=int(len(mapped_units)),
        coordinate_mw=float(mapped_units["mw"].sum()),
        occupied_cells=int(len(cell_counts)),
        median_units_per_cell=float(cell_counts.median()),
        p95_units_per_cell=float(cell_counts.quantile(0.95)),
        max_units_per_cell=int(cell_counts.max()),
        ambiguity=ambiguity,
    )


def build_location_cell_mapping(mapped_units: pd.DataFrame) -> pd.DataFrame:
    """Build a location-normalized mapping with ambiguity preserved explicitly."""

    if mapped_units.empty:
        return pd.DataFrame(columns=DEFAULT_LOCATION_COLUMNS)

    location_cell = (
        mapped_units.groupby(
            [
                "LOCATIONID",
                "country",
                "grid_lat",
                "grid_lon",
                "cell_id",
                "location_is_single_cell",
            ],
            as_index=False,
        )
        .agg(units=("UNITID", "size"), mw=("mw", "sum"))
        .sort_values(["country", "LOCATIONID", "grid_lat", "grid_lon"], kind="stable")
    )
    location_cell["source_resolution"] = np.where(
        location_cell["location_is_single_cell"],
        "location",
        "location_cell_split",
    )
    return location_cell.loc[:, DEFAULT_LOCATION_COLUMNS]


def build_country_rollup(
    mapped_entities: pd.DataFrame,
    *,
    source_resolution: str,
) -> pd.DataFrame:
    """Aggregate mapped entities to country-level capacity totals."""

    if mapped_entities.empty:
        return pd.DataFrame(columns=DEFAULT_COUNTRY_COLUMNS)

    if "country" not in mapped_entities.columns:
        raise ValueError("'country' is required to build a country rollup")

    entity_column = "UNITID" if "UNITID" in mapped_entities.columns else "LOCATIONID"
    rollup = (
        mapped_entities.groupby("country", as_index=False)
        .agg(entities=(entity_column, "size"), mw=("mw", "sum"))
        .sort_values("country", kind="stable")
    )
    rollup["source_resolution"] = source_resolution
    return rollup.loc[:, DEFAULT_COUNTRY_COLUMNS]


def load_and_map_thermal_plants(path: str | Path) -> tuple[pd.DataFrame, CellMappingSummary]:
    """Read the thermal plant parquet and return unit-level cell mappings."""

    mapped = build_unit_cell_mapping(pd.read_parquet(Path(path)))
    return mapped, summarize_mapping(mapped)


def write_unit_cell_mapping(
    source_path: str | Path,
    output_path: str | Path,
    columns: list[str] | None = None,
) -> CellMappingSummary:
    """Build and write the normalized unit-to-cell lookup artifact."""

    mapped, summary = load_and_map_thermal_plants(source_path)
    selected = mapped.loc[:, columns or DEFAULT_MAPPING_COLUMNS]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    match destination.suffix.lower():
        case ".parquet":
            selected.to_parquet(destination, index=False)
        case ".csv":
            selected.to_csv(destination, index=False)
        case _:
            raise ValueError("output_path must end with .parquet or .csv")

    return summary


def write_recovered_unit_cell_mapping(
    source_path: str | Path,
    output_path: str | Path,
    columns: list[str] | None = None,
) -> CellMappingSummary:
    """Build and write the unit mapping with safe inherited coordinates."""

    thermal_plants = pd.read_parquet(Path(source_path))
    mapped = build_recovered_unit_cell_mapping(thermal_plants)
    selected = mapped.loc[:, columns or DEFAULT_RECOVERED_MAPPING_COLUMNS]
    summary = summarize_mapping(mapped)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    match destination.suffix.lower():
        case ".parquet":
            selected.to_parquet(destination, index=False)
        case ".csv":
            selected.to_csv(destination, index=False)
        case _:
            raise ValueError("output_path must end with .parquet or .csv")

    return summary


def write_location_cell_mapping(
    source_path: str | Path,
    output_path: str | Path,
    columns: list[str] | None = None,
) -> CellMappingSummary:
    """Build and write the location-normalized lookup artifact."""

    mapped, summary = load_and_map_thermal_plants(source_path)
    location_mapping = build_location_cell_mapping(mapped)
    selected = location_mapping.loc[:, columns or DEFAULT_LOCATION_COLUMNS]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    match destination.suffix.lower():
        case ".parquet":
            selected.to_parquet(destination, index=False)
        case ".csv":
            selected.to_csv(destination, index=False)
        case _:
            raise ValueError("output_path must end with .parquet or .csv")

    return summary


def write_country_rollup(
    source_path: str | Path,
    output_path: str | Path,
    *,
    from_resolution: str = "location",
    columns: list[str] | None = None,
) -> CellMappingSummary:
    """Build and write the country rollup artifact."""

    mapped, summary = load_and_map_thermal_plants(source_path)
    source = build_location_cell_mapping(mapped) if from_resolution == "location" else mapped
    country_rollup = build_country_rollup(source, source_resolution=from_resolution)
    selected = country_rollup.loc[:, columns or DEFAULT_COUNTRY_COLUMNS]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    match destination.suffix.lower():
        case ".parquet":
            selected.to_parquet(destination, index=False)
        case ".csv":
            selected.to_csv(destination, index=False)
        case _:
            raise ValueError("output_path must end with .parquet or .csv")

    return summary
