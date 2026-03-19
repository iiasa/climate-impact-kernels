from __future__ import annotations

import pandas as pd


def validate_unique_units(frame: pd.DataFrame, *, frame_name: str) -> None:
    duplicates = frame["UNITID"].duplicated(keep=False)
    if duplicates.any():
        duplicate_units = frame.loc[duplicates, "UNITID"].tolist()[:5]
        raise ValueError(f"{frame_name} must have unique UNITID values; duplicates include {duplicate_units}")


def join_country_from_cooling(
    unit_cell_mapping: pd.DataFrame,
    cooling_mapping: pd.DataFrame,
) -> pd.DataFrame:
    if "country" in unit_cell_mapping.columns:
        return unit_cell_mapping
    if "country" not in cooling_mapping.columns:
        raise ValueError("country must be present in unit_cell_mapping or cooling_mapping")
    return unit_cell_mapping.merge(
        cooling_mapping[["UNITID", "LOCATIONID", "country"]],
        on=["UNITID", "LOCATIONID"],
        how="left",
        validate="one_to_one",
    )
