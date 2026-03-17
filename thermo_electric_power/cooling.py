from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COOLING_COLUMNS = frozenset(
    {
        "UNITID",
        "LOCATIONID",
        "country",
        "fuel",
        "utype",
        "mw",
        "cool_imputed",
        "cool_source",
    }
)

DEFAULT_COOLING_COLUMNS = [
    "UNITID",
    "LOCATIONID",
    "country",
    "fuel",
    "utype",
    "technology_family",
    "cooling_category",
    "cooling_source",
    "requires_water",
    "cooling_physics",
    "mw",
]

_CC_UTYPES = frozenset({"GT/C", "GT/T"})

COOLING_PHYSICS = {
    "air": "dry",
    "cl_fresh": "recirculating",
    "ot_fresh": "once_through",
    "ot_saline": "once_through",
}

REQUIRES_WATER = {
    "air": False,
    "cl_fresh": True,
    "ot_fresh": True,
    "ot_saline": True,
}


def classify_technology_family(utype: str) -> str:
    """Map WEPP unit type codes to a small technology family taxonomy."""

    if utype in _CC_UTYPES or utype.startswith("CC"):
        return "combined_cycle"
    if utype.startswith("ST") or utype == "ORC":
        return "steam"
    if utype.startswith("GT"):
        return "gas_turbine"
    if utype.startswith("IC") or utype in {"ECE", "FC"}:
        return "internal_combustion"
    return "other"


def build_unit_cooling_mapping(thermal_plants: pd.DataFrame) -> pd.DataFrame:
    """Build the normalized technology-to-cooling contract for each unit."""

    missing = REQUIRED_COOLING_COLUMNS.difference(thermal_plants.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"missing required columns: {missing_list}")

    mapping = thermal_plants.copy()
    mapping["technology_family"] = mapping["utype"].map(classify_technology_family)
    mapping["cooling_category"] = mapping["cool_imputed"]
    mapping["cooling_source"] = mapping["cool_source"]
    mapping["requires_water"] = mapping["cooling_category"].map(REQUIRES_WATER)
    mapping["cooling_physics"] = mapping["cooling_category"].map(COOLING_PHYSICS)

    if mapping["cooling_category"].isna().any():
        raise ValueError("cooling_category contains missing values after normalization")
    if mapping["cooling_physics"].isna().any():
        unknown = sorted(mapping.loc[mapping["cooling_physics"].isna(), "cooling_category"].unique())
        raise ValueError(f"unknown cooling categories: {unknown}")

    return mapping.loc[:, DEFAULT_COOLING_COLUMNS]


def write_unit_cooling_mapping(
    source_path: str | Path,
    output_path: str | Path,
    columns: list[str] | None = None,
) -> None:
    """Build and write the per-unit technology-to-cooling artifact."""

    mapping = build_unit_cooling_mapping(pd.read_parquet(Path(source_path)))
    selected = mapping.loc[:, columns or DEFAULT_COOLING_COLUMNS]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    match destination.suffix.lower():
        case ".parquet":
            selected.to_parquet(destination, index=False)
        case ".csv":
            selected.to_csv(destination, index=False)
        case _:
            raise ValueError("output_path must end with .parquet or .csv")
