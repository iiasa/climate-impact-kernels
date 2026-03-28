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

DEFAULT_IMPUTED_WET_MAPPING_COLUMNS = [
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
    "imputation_rule",
    "donor_UNITID",
    "donor_LOCATIONID",
    "donor_country",
    "location_cell_count",
    "location_is_single_cell",
]

DEFAULT_IMPUTED_DRY_MAPPING_COLUMNS = [
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
    "imputation_rule",
    "donor_UNITID",
    "donor_LOCATIONID",
    "donor_country",
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

WET_IMPUTATION_BACKOFF_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("country_cooling_size", ("country", "cooling_physics", "size_class")),
    ("country_cooling", ("country", "cooling_physics")),
    ("country_size", ("country", "size_class")),
    ("country", ("country",)),
    ("cooling_size", ("cooling_physics", "size_class")),
    ("cooling", ("cooling_physics",)),
    ("size", ("size_class",)),
    ("global", ()),
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
    cell_counts = (
        mapped[["LOCATIONID", "grid_lat", "grid_lon"]]
        .drop_duplicates()
        .groupby("LOCATIONID")
        .size()
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


def build_imputed_wet_unit_cell_mapping(
    thermal_plants: pd.DataFrame,
    cooling_mapping: pd.DataFrame,
    *,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Build a full wet-fleet mapping with explicit exact, inherited, and imputed provenance."""

    missing = REQUIRED_COLUMNS.difference(thermal_plants.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"missing required columns: {missing_list}")

    required_cooling = {"UNITID", "LOCATIONID", "country", "mw", "requires_water", "cooling_physics"}
    missing_cooling = required_cooling.difference(cooling_mapping.columns)
    if missing_cooling:
        missing_list = ", ".join(sorted(missing_cooling))
        raise ValueError(f"cooling_mapping missing required columns: {missing_list}")

    wet_population = (
        cooling_mapping.loc[cooling_mapping["requires_water"].fillna(False)].copy().pipe(
            _prepare_imputation_population
        )
    )
    return _build_imputed_population_mapping(
        thermal_plants,
        wet_population,
        columns=DEFAULT_IMPUTED_WET_MAPPING_COLUMNS,
        random_seed=random_seed,
    )


def build_imputed_dry_unit_cell_mapping(
    thermal_plants: pd.DataFrame,
    cooling_mapping: pd.DataFrame,
    *,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Build a full Qin-supported dry-fleet mapping with explicit provenance."""

    missing = REQUIRED_COLUMNS.difference(thermal_plants.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"missing required columns: {missing_list}")

    required_cooling = {
        "UNITID",
        "LOCATIONID",
        "country",
        "mw",
        "fuel",
        "technology_family",
        "cooling_physics",
    }
    missing_cooling = required_cooling.difference(cooling_mapping.columns)
    if missing_cooling:
        missing_list = ", ".join(sorted(missing_cooling))
        raise ValueError(f"cooling_mapping missing required columns: {missing_list}")

    from impacts.thermoelectric.dry import classify_dry_cooling_engine_type

    dry_population = cooling_mapping.loc[cooling_mapping["cooling_physics"].eq("dry")].copy()
    support_flags = dry_population.apply(
        lambda row: classify_dry_cooling_engine_type(
            technology_family=str(row["technology_family"]),
            fuel=str(row["fuel"]),
        )[1],
        axis=1,
    )
    supported_dry_population = dry_population.loc[support_flags].copy().pipe(_prepare_imputation_population)
    return _build_imputed_population_mapping(
        thermal_plants,
        supported_dry_population,
        columns=DEFAULT_IMPUTED_DRY_MAPPING_COLUMNS,
        random_seed=random_seed,
    )


def _build_imputed_population_mapping(
    thermal_plants: pd.DataFrame,
    population: pd.DataFrame,
    *,
    columns: list[str],
    random_seed: int,
) -> pd.DataFrame:
    recovered = build_recovered_unit_cell_mapping(thermal_plants)
    recovered_population = (
        recovered.loc[recovered["UNITID"].isin(population["UNITID"])]
        .copy()
        .assign(
            imputation_rule=pd.NA,
            donor_UNITID=pd.NA,
            donor_LOCATIONID=pd.NA,
            donor_country=pd.NA,
        )
    )

    unresolved = population.loc[~population["UNITID"].isin(recovered_population["UNITID"])].copy()
    if unresolved.empty:
        return (
            recovered_population.loc[:, columns]
            .sort_values(["country", "LOCATIONID", "UNITID"], kind="stable")
            .reset_index(drop=True)
        )

    donor_units = (
        recovered_population.merge(
            population.loc[:, ["UNITID", "LOCATIONID", "country", "cooling_physics", "size_class"]],
            on=["UNITID", "LOCATIONID", "country"],
            how="inner",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )
    donor_lookup = _build_donor_lookup(donor_units)

    unresolved_locations = _summarize_unresolved_locations(unresolved)
    assignments = _assign_imputed_locations(
        unresolved_locations,
        donor_units,
        donor_lookup,
        random_seed=random_seed,
    )

    unresolved_with_meta = unresolved.merge(
        thermal_plants.loc[:, ["UNITID", "LOCATIONID", "lat", "lon"]],
        on=["UNITID", "LOCATIONID"],
        how="left",
        validate="one_to_one",
    )
    imputed = unresolved_with_meta.merge(
        assignments,
        on="LOCATIONID",
        how="left",
        validate="many_to_one",
    )
    imputed["coordinate_source"] = "imputed_unit_donor"
    imputed["location_cell_count"] = 1
    imputed["location_is_single_cell"] = True

    combined = pd.concat(
        [
            recovered_population.loc[:, columns],
            imputed.loc[:, columns],
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


def _classify_size_class(mw: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(mw.to_numpy(dtype=float) >= 100.0, "large", "small"),
        index=mw.index,
        dtype="object",
    )


def _prepare_imputation_population(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["size_class"] = _classify_size_class(prepared["mw"])
    return prepared.sort_values(["country", "LOCATIONID", "UNITID"], kind="stable")


def _build_donor_lookup(donor_units: pd.DataFrame) -> dict[str, dict[tuple[object, ...], np.ndarray]]:
    lookups: dict[str, dict[tuple[object, ...], np.ndarray]] = {}

    for rule_name, columns in WET_IMPUTATION_BACKOFF_RULES:
        if columns:
            grouped = donor_units.reset_index().groupby(list(columns))["index"]
            lookups[rule_name] = {
                key if isinstance(key, tuple) else (key,): values.to_numpy(dtype=int)
                for key, values in grouped
            }
        else:
            lookups[rule_name] = {(): donor_units.index.to_numpy(dtype=int)}

    return lookups


def _resolve_imputation_candidates(
    row: pd.Series,
    donor_lookup: dict[str, dict[tuple[object, ...], np.ndarray]],
) -> tuple[str, np.ndarray]:
    for rule_name, columns in WET_IMPUTATION_BACKOFF_RULES:
        key = tuple(row[column] for column in columns)
        candidates = donor_lookup[rule_name].get(key, np.array([], dtype=int))
        if len(candidates) > 0:
            return rule_name, candidates

    raise ValueError("no donor candidates available for wet-fleet imputation")


def _dominant_value(frame: pd.DataFrame, column: str) -> str:
    weights = (
        frame.groupby(column, as_index=False)
        .agg(mw=("mw", "sum"))
        .sort_values(["mw", column], ascending=[False, True], kind="stable")
    )
    return str(weights.iloc[0][column])


def _summarize_unresolved_locations(unresolved: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for location_id, group in unresolved.groupby("LOCATIONID", sort=False):
        rows.append(
            {
                "LOCATIONID": location_id,
                "country": str(group["country"].iloc[0]),
                "cooling_physics": _dominant_value(group, "cooling_physics"),
                "size_class": _dominant_value(group.assign(size_class=group["size_class"]), "size_class"),
            }
        )

    return pd.DataFrame(rows).sort_values(["country", "LOCATIONID"], kind="stable").reset_index(drop=True)


def _assign_imputed_locations(
    unresolved_locations: pd.DataFrame,
    donor_units: pd.DataFrame,
    donor_lookup: dict[str, dict[tuple[object, ...], np.ndarray]],
    *,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []

    for location in unresolved_locations.itertuples(index=False):
        row = pd.Series(location._asdict())
        rule_name, candidates = _resolve_imputation_candidates(row, donor_lookup)
        donor = donor_units.iloc[int(rng.choice(candidates))]
        rows.append(
            {
                "LOCATIONID": int(location.LOCATIONID),
                "grid_lat": float(donor["grid_lat"]),
                "grid_lon": float(donor["grid_lon"]),
                "cell_id": str(donor["cell_id"]),
                "imputation_rule": rule_name,
                "donor_UNITID": int(donor["UNITID"]),
                "donor_LOCATIONID": int(donor["LOCATIONID"]),
                "donor_country": str(donor["country"]),
            }
        )

    return pd.DataFrame(rows)


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


def write_imputed_wet_unit_cell_mapping(
    source_path: str | Path,
    cooling_path: str | Path,
    output_path: str | Path,
    columns: list[str] | None = None,
    *,
    random_seed: int = 0,
) -> CellMappingSummary:
    """Build and write the full wet-fleet mapping with provenance-preserving imputation."""

    thermal_plants = pd.read_parquet(Path(source_path))
    cooling_mapping = pd.read_parquet(Path(cooling_path))
    mapped = build_imputed_wet_unit_cell_mapping(
        thermal_plants,
        cooling_mapping,
        random_seed=random_seed,
    )
    selected = mapped.loc[:, columns or DEFAULT_IMPUTED_WET_MAPPING_COLUMNS]
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


def write_imputed_dry_unit_cell_mapping(
    source_path: str | Path,
    cooling_path: str | Path,
    output_path: str | Path,
    columns: list[str] | None = None,
    *,
    random_seed: int = 0,
) -> CellMappingSummary:
    """Build and write the full Qin-supported dry-fleet mapping with provenance."""

    thermal_plants = pd.read_parquet(Path(source_path))
    cooling_mapping = pd.read_parquet(Path(cooling_path))
    mapped = build_imputed_dry_unit_cell_mapping(
        thermal_plants,
        cooling_mapping,
        random_seed=random_seed,
    )
    selected = mapped.loc[:, columns or DEFAULT_IMPUTED_DRY_MAPPING_COLUMNS]
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
