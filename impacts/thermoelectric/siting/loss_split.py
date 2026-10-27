"""Split the wet-cooling capacity loss into thermal loss and water-availability loss.

The input is a site evaluation (in the code, a census) that carries the monthly
drivers ``available_discharge_m3_per_s`` and ``withdrawal_intensity_m3_per_mwh``.
Thermal loss is the capacity lost because each MWh needs more water than in the
reference month. Water-availability loss is the capacity lost because the river
carries less water than the plant needs. The code calls them the thermal and
withdrawal channels (``thermal_loss_mw``, ``withdrawal_loss_mw``).

For one row (set of cells, cell, synthetic plant, month), with nameplate capacity
``N`` (MW), available discharge ``Q`` (m3 s-1) and withdrawal intensity ``I``
(m3 MWh-1):

1. Capacity per unit flow ``c = 3600 / I``: the MW of capacity that 1 m3 s-1 of
   water can cool (code name ``thermal_conductance``). It is infinite at
   ``I = 0``, 0 for ``I < 0`` and unknown for an unknown ``I``.
2. Reference capacity per unit flow ``c_ref``: the largest finite, positive ``c``
   of that set of cells, cell, synthetic plant and calendar month in the baseline
   site evaluation (:func:`build_low_warming_reference`). Where a cell has none, the
   largest over all cells of the same set, plant and month is used.
3. Thermally adjusted capacity ``T = min(N, N c / c_ref)``.
4. Water limit ``R = min(N, max(Q c, 0))``, and ``R = 0`` when ``Q = 0``. It is the
   capacity that the river's available water can cool.
5. Thermal loss ``N - T``; water-availability loss ``max(T - R, 0)``; usable
   capacity ``max(T - water-availability loss, 0)``. The three add up to ``N``.

Months enter the season as follows. A month with ``Q = 0`` counts: an unknown ``I``
is taken at the reference (no thermal loss). The water limit is 0, so the whole
thermally adjusted capacity is lost to water-availability loss and usable capacity
is 0. A month with an unknown driver (NaN ``Q``, or NaN ``I``
with water in the river) is left out. The seasonal value is the mean over the
counted months (``month_count``); a season with none is omitted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from impacts.thermoelectric.siting._io import (
    refuse_existing_outputs,
    staged_outputs,
)
from impacts.thermoelectric.siting.cells import (
    SyntheticPlant,
    _wet_archetypes_named,
)
from impacts.thermoelectric.wet import water_limited_capacity_mw

MONTHLY_RESPONSES = (
    "available_discharge_m3_per_s",
    "withdrawal_intensity_m3_per_mwh",
)

MONTHLY_INDEX = (
    "population",
    "region_id",
    "country",
    "cell_id",
    "grid_lat",
    "grid_lon",
    "cell_area_m2",
    "archetype",
    "gcm",
    "scenario",
    "year",
    "season",
    "month",
    "warming_level",
)

REFERENCE_INDEX = ("population", "cell_id", "archetype", "month")
REFERENCE_FALLBACK_INDEX = ("population", "archetype", "month")

SEASONAL_INDEX = (
    "population",
    "region_id",
    "country",
    "cell_id",
    "grid_lat",
    "grid_lon",
    "cell_area_m2",
    "archetype",
    "gcm",
    "scenario",
    "year",
    "season",
    "warming_level",
)

OUTPUT_COLUMNS = (
    *SEASONAL_INDEX,
    "nameplate_mw",
    "thermal_loss_mw",
    "withdrawal_loss_mw",
    "usable_capacity_mw",
    "thermal_loss_fraction",
    "withdrawal_loss_fraction",
    "capacity_factor",
    "baseline_reference",
)


def build_low_warming_reference(paths: list[str | Path]) -> pd.DataFrame:
    """Return the reference capacity per unit flow by set of cells, cell, synthetic plant and calendar month.

    ``baseline_thermal_conductance`` is the largest finite, positive capacity per
    unit flow in the baseline site evaluations (``paths``).
    ``fallback_thermal_conductance`` is the largest of those over all cells of the
    same set, plant and month; it is used where a cell has no reference month.
    """

    parts = []
    for month in range(1, 13):
        data = _read_monthly(paths, month=month)
        if data.empty:
            continue
        data["thermal_conductance"] = _finite_positive_conductance(
            data["withdrawal_intensity_m3_per_mwh"]
        )
        data = data.dropna(subset=["thermal_conductance"])
        parts.append(
            data.groupby(list(REFERENCE_INDEX), observed=True)["thermal_conductance"]
            .max()
            .rename("baseline_thermal_conductance")
            .reset_index()
        )
    if not parts:
        raise ValueError("low-warming-level reference has no monthly rows")
    reference = (
        pd.concat(parts, ignore_index=True)
        .groupby(list(REFERENCE_INDEX), observed=True)["baseline_thermal_conductance"]
        .max()
        .reset_index()
    )
    if reference.empty:
        raise ValueError("low-warming-level reference has no rows with a finite capacity per unit flow")
    fallback = (
        reference.groupby(list(REFERENCE_FALLBACK_INDEX), observed=True)[
            "baseline_thermal_conductance"
        ]
        .max()
        .rename("fallback_thermal_conductance")
        .reset_index()
    )
    return reference.merge(fallback, on=list(REFERENCE_FALLBACK_INDEX), how="left")


# The label written to ``baseline_reference`` records which site evaluation gave
# the reference capacity per unit flow. The two calculations take it from
# different evaluations:
#
# * ``DEFAULT_REFERENCE``: the warming-level calculation, which takes its
#   reference from the low-warming-level evaluation, a different file from the
#   evaluations it splits.
# * ``PRESENT_DAY_REFERENCE``: the present-day reference calculation, which
#   takes its reference from the 1995-2014 present-day reference evaluation. That
#   evaluation is its own baseline: it is passed both as the baseline and as the
#   one to split.
#
# Nothing checks the label against the files, so a wrong one is recorded
# silently; use the constant that matches the baseline evaluation.
DEFAULT_REFERENCE = "max_thermal_conductance_over_low_gwl"
PRESENT_DAY_REFERENCE = "max_thermal_conductance_over_ref_pd_1995_2014"


@dataclass(frozen=True)
class WetLossSplit:
    """A written split of the wet capacity loss and the reference it was measured against.

    Attributes:
        output: The parquet of the split.
        low_gwl_reference: Per set of cells, cell, synthetic plant and month, the
            largest capacity per unit flow (``baseline_thermal_conductance``) in
            the baseline site evaluation (the low-warming evaluation for the
            warming-level calculation, the present-day reference evaluation for
            the present-day calculation), with the fallback over all cells of the
            same set, plant and month (``fallback_thermal_conductance``).
    """

    output: Path
    low_gwl_reference: pd.DataFrame


def split_wet_losses(
    low_warming_evaluation_paths: list[str | Path],
    census_paths: list[str | Path],
    output: str | Path,
    *,
    reference: str,
    overwrite: bool = False,
    wet_synthetic_plants: Sequence[SyntheticPlant] | None = None,
) -> WetLossSplit:
    """Split the wet capacity loss of each site evaluation into thermal loss and water-availability loss.

    The reference capacity per unit flow is built from the baseline site
    evaluations (:func:`build_low_warming_reference`); each evaluation in
    ``census_paths`` is then split against it into seasonal losses in MW that add
    up to nameplate capacity (:func:`split_site_evaluations`), written to ``output``.
    The evaluations carry the monthly ``available_discharge_m3_per_s`` and
    ``withdrawal_intensity_m3_per_mwh`` drivers. The method is in the module
    docstring.

    ``reference`` has no default: it labels the output and must match
    the baseline site evaluation, which nothing checks. For the warming-level
    calculation pass :data:`DEFAULT_REFERENCE` with the low-warming-level
    evaluation as ``low_warming_evaluation_paths``. For the present-day reference
    calculation pass :data:`PRESENT_DAY_REFERENCE` with the present-day
    reference evaluation (``REF-PD``) as both ``low_warming_evaluation_paths`` and
    ``census_paths``, so that evaluation is its own baseline.

    ``wet_synthetic_plants`` gives each row's nameplate by synthetic plant name (the
    ``archetype`` column); the default wet plants when ``None``. Pass the set the
    site evaluation was run with when it was not the default one.

    Raises:
        FileExistsError: if ``output`` exists and ``overwrite`` is false.
    """

    if not overwrite:
        refuse_existing_outputs([output])
    low_warming_reference = build_low_warming_reference(low_warming_evaluation_paths)
    written = split_site_evaluations(
        census_paths,
        low_warming_reference,
        output,
        reference=reference,
        wet_synthetic_plants=wet_synthetic_plants,
    )
    return WetLossSplit(output=written, low_gwl_reference=low_warming_reference)


def split_site_evaluations(
    paths: list[str | Path],
    low_warming_reference: pd.DataFrame,
    output: str | Path,
    *,
    reference: str = DEFAULT_REFERENCE,
    wet_synthetic_plants: Sequence[SyntheticPlant] | None = None,
) -> Path:
    """Write the seasonal thermal loss and water-availability loss, in MW adding up to nameplate.

    A dry month (zero discharge) loses all its capacity to water-availability
    loss. A month with an unknown driver is excluded from its season, which is
    averaged over the months with known drivers only; a season with none is
    omitted.

    ``wet_synthetic_plants`` gives each row's nameplate by synthetic plant name (the
    ``archetype`` column); the default wet plants when ``None``. Pass the set the
    site evaluation was run with when it was not the default one.
    """

    low_warming_reference = _reference_with_fallback(low_warming_reference)
    output = Path(output)
    with staged_outputs([output], require_all=False) as staged:
        writer: pq.ParquetWriter | None = None
        try:
            for path in paths:
                decomposed = _decompose_path(
                    path, low_warming_reference, reference, wet_synthetic_plants=wet_synthetic_plants
                )
                table = pa.Table.from_pandas(decomposed, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(staged[output], table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
    return output


def _read_monthly(paths: list[str | Path], *, month: int | None = None) -> pd.DataFrame:
    dataset = pads.dataset([str(path) for path in paths])
    predicate = pc.field("response").isin(list(MONTHLY_RESPONSES))
    if month is not None:
        predicate = predicate & (pc.field("month") == month)
    columns = [*MONTHLY_INDEX, "response", "value"]
    table = dataset.to_table(columns=columns, filter=predicate)
    frame = table.to_pandas(self_destruct=True, split_blocks=True)
    wide = frame.pivot_table(
        index=list(MONTHLY_INDEX),
        columns="response",
        values="value",
        observed=True,
        aggfunc="first",
    ).reset_index()
    # pivot_table drops a response whose values are all NaN in this chunk.
    for response in MONTHLY_RESPONSES:
        if response not in wide.columns:
            wide[response] = np.nan
    return wide


def _decompose_path(
    path: str | Path,
    low_warming_reference: pd.DataFrame,
    reference: str,
    *,
    wet_synthetic_plants: Sequence[SyntheticPlant] | None = None,
) -> pd.DataFrame:
    parts = [
        _decompose_monthly_to_sums(
            _read_monthly([path], month=month), low_warming_reference, wet_synthetic_plants=wet_synthetic_plants
        )
        for month in range(1, 13)
    ]
    sums = (
        pd.concat(parts, ignore_index=True)
        .groupby(list(SEASONAL_INDEX), observed=True)
        .agg(
            nameplate_mw=("nameplate_mw", "first"),
            thermal_loss_sum=("thermal_loss_sum", "sum"),
            withdrawal_loss_sum=("withdrawal_loss_sum", "sum"),
            usable_capacity_sum=("usable_capacity_sum", "sum"),
            month_count=("month_count", "sum"),
        )
        .reset_index()
    )
    return _seasonal_components(sums, reference)


def _seasonal_components(sums: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Seasonal mean losses and fractions from per-season sums of monthly losses."""

    # A season with no month of known drivers has nothing to split.
    sums = sums[sums["month_count"] > 0]
    seasonal = sums.assign(
        thermal_loss_mw=sums["thermal_loss_sum"] / sums["month_count"],
        withdrawal_loss_mw=sums["withdrawal_loss_sum"] / sums["month_count"],
        usable_capacity_mw=sums["usable_capacity_sum"] / sums["month_count"],
    )
    seasonal["thermal_loss_fraction"] = seasonal["thermal_loss_mw"] / seasonal["nameplate_mw"]
    seasonal["withdrawal_loss_fraction"] = (
        seasonal["withdrawal_loss_mw"] / seasonal["nameplate_mw"]
    )
    seasonal["capacity_factor"] = seasonal["usable_capacity_mw"] / seasonal["nameplate_mw"]
    seasonal["baseline_reference"] = reference
    return seasonal[list(OUTPUT_COLUMNS)]


def _decompose_monthly_to_sums(
    monthly: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    wet_synthetic_plants: Sequence[SyntheticPlant] | None = None,
) -> pd.DataFrame:
    sum_columns = [
        *SEASONAL_INDEX,
        "nameplate_mw",
        "thermal_loss_sum",
        "withdrawal_loss_sum",
        "usable_capacity_sum",
        "month_count",
    ]
    if monthly.empty:
        return pd.DataFrame(columns=sum_columns)

    reference = _reference_with_fallback(reference)
    cell_reference = reference[
        [*REFERENCE_INDEX, "baseline_thermal_conductance"]
    ].drop_duplicates()
    data = monthly.merge(cell_reference, on=list(REFERENCE_INDEX), how="left", validate="many_to_one")
    fallback = reference[
        [*REFERENCE_FALLBACK_INDEX, "fallback_thermal_conductance"]
    ].drop_duplicates()
    data = data.merge(
        fallback,
        on=list(REFERENCE_FALLBACK_INDEX),
        how="left",
        validate="many_to_one",
    )
    data["baseline_thermal_conductance"] = data["baseline_thermal_conductance"].fillna(
        data["fallback_thermal_conductance"]
    )
    if data["baseline_thermal_conductance"].isna().any():
        missing = data.loc[
            data["baseline_thermal_conductance"].isna(),
            ["population", "cell_id", "archetype", "month"],
        ].drop_duplicates()
        raise ValueError(
            "baseline reference missing for monthly rows: "
            f"{missing.head(5).to_dict(orient='records')}"
        )

    synthetic_plants = _wet_archetypes_named(pd.unique(data["archetype"]), wet_synthetic_plants)
    nameplate = np.array(
        [synthetic_plants[name].parameters["mw"] for name in data["archetype"]], dtype=float
    )
    reference_conductance = data["baseline_thermal_conductance"].to_numpy(dtype=float)
    discharge = data["available_discharge_m3_per_s"].to_numpy(dtype=float)
    conductance = _thermal_conductance(data["withdrawal_intensity_m3_per_mwh"])

    # A dry month loses its capacity to water-availability loss: the river
    # carries nothing whatever the intensity, so an unknown intensity is taken at
    # the reference state and costs no thermal loss. Any other month with an
    # unknown driver (NaN discharge, or NaN intensity with water in the river) is
    # not split; it drops out of the season, as it does from the water limit of the
    # site evaluation.
    dry = discharge == 0.0
    conductance = np.where(dry & np.isnan(conductance), reference_conductance, conductance)
    known = ~np.isnan(discharge) & ~np.isnan(conductance)

    with np.errstate(invalid="ignore"):
        realized_ceiling = np.where(
            dry,
            0.0,
            np.minimum(nameplate, np.maximum(discharge * conductance, 0.0)),
        )
        thermal_adjusted = np.minimum(
            nameplate,
            nameplate * conductance / reference_conductance,
        )
    thermal_loss = nameplate - thermal_adjusted
    withdrawal_loss = np.maximum(thermal_adjusted - realized_ceiling, 0.0)
    usable = np.maximum(thermal_adjusted - withdrawal_loss, 0.0)

    data = data.assign(
        nameplate_mw=nameplate,
        thermal_loss_mw=np.where(known, thermal_loss, np.nan),
        withdrawal_loss_mw=np.where(known, withdrawal_loss, np.nan),
        usable_capacity_mw=np.where(known, usable, np.nan),
        month_known=known.astype(int),
    )

    return (
        data.groupby(list(SEASONAL_INDEX), observed=True)
        .agg(
            nameplate_mw=("nameplate_mw", "first"),
            thermal_loss_sum=("thermal_loss_mw", "sum"),
            withdrawal_loss_sum=("withdrawal_loss_mw", "sum"),
            usable_capacity_sum=("usable_capacity_mw", "sum"),
            month_count=("month_known", "sum"),
        )
        .reset_index()
    )


def _reference_with_fallback(reference: pd.DataFrame) -> pd.DataFrame:
    if "fallback_thermal_conductance" in reference.columns:
        return reference
    fallback = (
        reference.groupby(list(REFERENCE_FALLBACK_INDEX), observed=True)[
            "baseline_thermal_conductance"
        ]
        .max()
        .rename("fallback_thermal_conductance")
        .reset_index()
    )
    return reference.merge(fallback, on=list(REFERENCE_FALLBACK_INDEX), how="left")


def _finite_positive_conductance(intensity: pd.Series) -> np.ndarray:
    """Capacity per unit flow (MW per m3 s-1) from withdrawal intensity where finite and positive, else NaN.

    The reference keeps only months whose capacity per unit flow is usable as a
    reference: a zero, negative, infinite or unknown intensity gives NaN.
    """

    return water_limited_capacity_mw(
        1.0,
        intensity.to_numpy(dtype=float),
        zero_intensity=np.nan,
        negative_intensity=np.nan,
        finite_positive_only=True,
    )


def _thermal_conductance(intensity: pd.Series) -> np.ndarray:
    """Capacity per unit flow (MW per m3 s-1) from withdrawal intensity: infinite at zero, 0 if negative, NaN if unknown."""

    return water_limited_capacity_mw(
        1.0,
        intensity.to_numpy(dtype=float),
        zero_intensity=np.inf,
        negative_intensity=0.0,
    )
