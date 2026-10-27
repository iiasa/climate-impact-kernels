"""Capacity curves of wet-cooled plants for MESSAGE, built from the loss-split tables.

A loss-split table is a parquet file that splits each cell's wet capacity loss
into thermal loss and water-availability loss (written by
``split_wet_losses``). A capacity curve adds the cells of one region and
synthetic plant in a fixed order and records, at each step, the cumulative
nameplate capacity, thermal loss, water-availability loss and usable capacity.
This module builds the curves of existing sites and of potential sites (with
river water shared along the river network), takes their median across GCMs, and
checks the existing-site curves against the direct sum over cells.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as pads

from impacts.thermoelectric.siting._io import staged_outputs
from impacts.thermoelectric.siting.cells import (
    SyntheticPlant,
    _wet_archetypes_named,
)
from impacts.thermoelectric.siting.river_network import (
    DOWNSTREAM_COLUMN,
    remaining_water_limits,
    river_network_capped_allocation,
)

LOSS_SPLIT_COLUMNS = (
    "population",
    "region_id",
    "country",
    "cell_id",
    "grid_lat",
    "grid_lon",
    "archetype",
    "gcm",
    "scenario",
    "year",
    "season",
    "warming_level",
    "nameplate_mw",
    "thermal_loss_mw",
    "withdrawal_loss_mw",
    "usable_capacity_mw",
)

CELL_KEYS = (
    "population",
    "region_id",
    "country",
    "cell_id",
    "grid_lat",
    "grid_lon",
    "archetype",
    "gcm",
    "season",
    "warming_level",
)

CURVE_KEYS = (
    "population",
    "placement_rule",
    "gcm",
    "region_id",
    "plant_type",
    "cooling_type",
    "archetype",
    "season",
    "warming_level",
)

CURVE_COLUMNS = (
    *CURVE_KEYS,
    "point_index",
    "capacity_axis_mw",
    "marginal_cell_id",
    "marginal_country",
    "marginal_capacity_mw",
    "marginal_usable_fraction",
    "thermal_loss_mw",
    "withdrawal_loss_mw",
    "usable_capacity_mw",
    "thermal_loss_fraction",
    "withdrawal_loss_fraction",
    "usable_fraction",
)

SAMPLE_COLUMNS = ("scenario", "year")

SAMPLE_CURVE_COLUMNS = (
    *CURVE_KEYS[:3],
    *SAMPLE_COLUMNS,
    *CURVE_KEYS[3:],
    *CURVE_COLUMNS[len(CURVE_KEYS) :],
)

POOLED_COLUMNS = (
    "population",
    "placement_rule",
    "statistic",
    "region_id",
    "plant_type",
    "cooling_type",
    "archetype",
    "season",
    "warming_level",
    "point_index",
    "capacity_axis_mw",
    "gcm_count",
    "marginal_capacity_mw",
    "marginal_usable_fraction",
    "thermal_loss_mw",
    "withdrawal_loss_mw",
    "usable_capacity_mw",
    "thermal_loss_fraction",
    "withdrawal_loss_fraction",
    "usable_fraction",
)

ADDITIVITY_CHECK_COLUMNS = (
    "population",
    "placement_rule",
    "gcm",
    "region_id",
    "season",
    "warming_level",
    "direct_nameplate_mw",
    "curve_nameplate_mw",
    "direct_thermal_loss_mw",
    "curve_thermal_loss_mw",
    "direct_withdrawal_loss_mw",
    "curve_withdrawal_loss_mw",
    "direct_usable_capacity_mw",
    "curve_usable_capacity_mw",
    "max_abs_error_mw",
)

WATER_LIMIT_RESPONSE = "water_limited_capacity_ceiling_mw"

WATER_LIMIT_COLUMNS = (
    "population",
    "region_id",
    "country",
    "cell_id",
    "grid_lat",
    "grid_lon",
    "archetype",
    "gcm",
    "scenario",
    "year",
    "season",
    "warming_level",
    "response",
    "value",
)

RIVER_NETWORK_COLUMNS = ("population", "scenario", "cell_id", DOWNSTREAM_COLUMN)

ORDERING_KEYS = ("region_id", "archetype", "cell_id")
ORDERING_COLUMNS = (*ORDERING_KEYS, "ordering_usable_fraction", "fixed_order")

# One climate sample (GCM, scenario and year) for one synthetic plant, season and
# warming level: the unit over which river water is shared, across all cells at once.
CLIMATE_SAMPLE_KEYS = (
    "gcm",
    "scenario",
    "year",
    "archetype",
    "season",
    "warming_level",
)
EXISTING_SITE_WATER_LOSS_COLUMN = "incumbent_water_loss_mw"

# The cell order of the existing-site curves, which ranks the cells being curved on
# their own results: ``build_fixed_cell_ordering`` of the table being curved, that
# is each region and synthetic plant's cells by their mean usable fraction over
# every row of that table (all its GCMs, seasons and warming levels), highest
# first, ties broken by cell_id.
OWN_MEAN_USABLE_FRACTION = "own_mean_usable_fraction"
IncumbentOrdering = Literal["own_mean_usable_fraction"]


def export_wet_usable_capacity_curves(
    loss_split_paths: list[str | Path],
    output_dir: str | Path,
    *,
    population: str = "incumbent",
    placement_rule: str = "existing_fleet",
    fixed_ordering_path: str | Path | IncumbentOrdering = OWN_MEAN_USABLE_FRACTION,
    wet_synthetic_plants: Sequence[SyntheticPlant] | None = None,
) -> tuple[Path, Path, Path]:
    """Write the per-GCM capacity curves, their median across GCMs, and the additivity check.

    The additivity check (``wet_ucr_curve_additivity_oracle.parquet``) compares
    the end point of the curves with the direct sum over cells; the export raises
    ``ValueError`` if they differ by more than 1e-6 MW. Returns the three file
    paths, in that order: ``wet_ucr_curves_per_gcm.parquet``,
    ``wet_ucr_curves_pooled_median.parquet`` and the additivity check.

    ``fixed_ordering_path`` is the path of a parquet table giving the cell order,
    or the string ``"own_mean_usable_fraction"`` (the default) to rank the curved
    cells on themselves, as :func:`build_capacity_curves` describes.

    ``wet_synthetic_plants`` gives each synthetic plant name its plant type and cooling
    type; ``None`` uses the default wet synthetic plants.
    """

    if fixed_ordering_path is None:
        raise TypeError(
            "fixed_ordering_path must be a parquet ordering table or "
            f"{OWN_MEAN_USABLE_FRACTION!r}; None is not accepted"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_table = _load_cell_means(loss_split_paths, population=population)
    fixed_ordering = (
        OWN_MEAN_USABLE_FRACTION
        if isinstance(fixed_ordering_path, str)
        and fixed_ordering_path == OWN_MEAN_USABLE_FRACTION
        else pd.read_parquet(fixed_ordering_path)
    )
    curves = build_capacity_curves(
        cell_table,
        population=population,
        placement_rule=placement_rule,
        fixed_ordering=fixed_ordering,
        wet_synthetic_plants=wet_synthetic_plants,
    )
    pooled = build_pooled_median_curves(curves)
    additivity_check = reference_fleet_additivity_check(
        cell_table,
        curves,
        population=population,
        placement_rule=placement_rule,
    )
    validate_reference_fleet_additivity_check(additivity_check)

    per_gcm_path = output_dir / "wet_ucr_curves_per_gcm.parquet"
    pooled_path = output_dir / "wet_ucr_curves_pooled_median.parquet"
    oracle_path = output_dir / "wet_ucr_curve_additivity_oracle.parquet"
    with staged_outputs([per_gcm_path, pooled_path, oracle_path]) as staged:
        curves.to_parquet(staged[per_gcm_path], index=False)
        pooled.to_parquet(staged[pooled_path], index=False)
        additivity_check.to_parquet(staged[oracle_path], index=False)
    return per_gcm_path, pooled_path, oracle_path


def build_potential_site_curves_with_river_network(
    loss_split_samples: pd.DataFrame,
    water_limits: pd.DataFrame,
    river_network: pd.DataFrame,
    *,
    fixed_ordering: pd.DataFrame,
    population: str = "candidate",
    placement_rule: str = "candidate_reach_aware",
    capacity_grid_points: int = 61,
    existing_site_samples: pd.DataFrame | None = None,
    wet_synthetic_plants: Sequence[SyntheticPlant] | None = None,
) -> pd.DataFrame:
    """Return capacity curves of potential sites in which the water-availability loss reflects water shared along the river network.

    Thermal loss stays local to each cell and scales with the nameplate capacity
    placed. The water-availability loss is recomputed by placing the capacity left
    after thermal loss on the river network and applying the cumulative upstream
    water limits. Cells are placed in ``fixed_ordering`` (``region_id``,
    ``archetype``, ``cell_id``, ``ordering_usable_fraction``, ``fixed_order``), the
    one order shared by every GCM, season, warming level and both calculations
    (warming-level and present-day reference), from
    :func:`build_fixed_cell_ordering` or :func:`build_residual_water_ordering`.

    Pass ``existing_site_samples`` to subtract the water that existing sites withdraw
    from those water limits first, so the water left for potential sites is what
    remains rather than full flow that existing sites already use.

    ``wet_synthetic_plants`` gives each synthetic plant name its plant type and cooling
    type; ``None`` uses the default wet synthetic plants.
    """

    if capacity_grid_points < 2:
        raise ValueError("capacity_grid_points must be at least 2")
    sampled = build_potential_site_sample_curves_with_river_network(
        loss_split_samples,
        water_limits,
        river_network,
        population=population,
        placement_rule=placement_rule,
        capacity_grid_points=capacity_grid_points,
        fixed_ordering=fixed_ordering,
        existing_site_samples=existing_site_samples,
        wet_synthetic_plants=wet_synthetic_plants,
    )
    return average_potential_site_sample_curves_with_river_network(sampled)


def average_potential_site_sample_curves_with_river_network(
    sampled: pd.DataFrame,
) -> pd.DataFrame:
    """Average the per-sample potential-site curves over the scenarios and years within each GCM, region, synthetic plant, season and warming level."""

    value_columns = [
        "marginal_capacity_mw",
        "marginal_usable_fraction",
        "thermal_loss_mw",
        "withdrawal_loss_mw",
        "usable_capacity_mw",
    ]
    averaged = (
        sampled.groupby(
            [*CURVE_KEYS, "point_index", "capacity_axis_mw"],
            observed=True,
            sort=True,
        )
        .agg(
            marginal_cell_id=("marginal_cell_id", "first"),
            marginal_country=("marginal_country", "first"),
            **{column: (column, "mean") for column in value_columns},
        )
        .reset_index()
    )
    _add_loss_fractions(averaged)
    return averaged[list(CURVE_COLUMNS)]


def build_potential_site_sample_curves_with_river_network(
    loss_split_samples: pd.DataFrame,
    water_limits: pd.DataFrame,
    river_network: pd.DataFrame,
    *,
    fixed_ordering: pd.DataFrame,
    population: str = "candidate",
    placement_rule: str = "candidate_reach_aware",
    capacity_grid_points: int = 61,
    existing_site_samples: pd.DataFrame | None = None,
    wet_synthetic_plants: Sequence[SyntheticPlant] | None = None,
) -> pd.DataFrame:
    """Return one potential-site capacity curve for each climate sample, synthetic plant, season, warming level and region.

    Cells are placed in ``fixed_ordering``, as
    :func:`build_potential_site_curves_with_river_network` describes.

    If ``water_limits`` keeps ``year``, each result stays tied to the climate
    sample that produced both its thermal loss and its water limit. Without it,
    the water limit has no year and is applied to every year.

    Pass ``existing_site_samples`` to share the river between existing and potential
    sites: the withdrawal of the existing sites, routed down the river network, is
    subtracted from the potential-site water limits before placement, so the water
    left for potential sites is what remains rather than the full flow. Without
    it, potential and existing sites are each given the whole river independently,
    which counts the same water twice wherever they sit on the same stretch of
    river.

    ``wet_synthetic_plants`` gives each synthetic plant name its plant type and cooling
    type; ``None`` uses the default wet synthetic plants.
    """

    if capacity_grid_points < 2:
        raise ValueError("capacity_grid_points must be at least 2")
    data = _candidate_reach_inputs(loss_split_samples, water_limits, river_network)
    if existing_site_samples is not None:
        data = attach_existing_site_water_demand(data, existing_site_samples)
    data = _attach_fixed_ordering(data, fixed_ordering)
    data = _with_archetype_parts(data, wet_synthetic_plants).assign(
        local_usable_fraction=lambda df: df["usable_capacity_mw"] / df["nameplate_mw"],
        thermal_available_fraction=lambda df: (
            (df["nameplate_mw"] - df["thermal_loss_mw"]) / df["nameplate_mw"]
        ),
        thermal_loss_fraction_local=lambda df: (
            df["thermal_loss_mw"] / df["nameplate_mw"]
        ),
        population=population,
        placement_rule=placement_rule,
    )

    rows = []
    for _, sample in data.groupby(list(CLIMATE_SAMPLE_KEYS), observed=True, sort=True):
        rows.extend(
            _sample_reach_candidate_curves(
                sample,
                capacity_grid_points=capacity_grid_points,
                placement_rule=placement_rule,
            )
        )
    if not rows:
        raise ValueError("potential-site export produced no sample curves")
    sampled = pd.concat(rows, ignore_index=True)
    _add_loss_fractions(sampled)
    return sampled[list(SAMPLE_CURVE_COLUMNS)]


def _add_loss_fractions(curves: pd.DataFrame) -> None:
    """Add each loss and the usable capacity as a fraction of the capacity at that point, in place.

    The zero-capacity first point has no loss and is fully usable.
    """

    axis = curves["capacity_axis_mw"]
    curves["thermal_loss_fraction"] = curves["thermal_loss_mw"] / axis
    curves["withdrawal_loss_fraction"] = curves["withdrawal_loss_mw"] / axis
    curves["usable_fraction"] = curves["usable_capacity_mw"] / axis
    zero = axis == 0.0
    curves.loc[zero, ["thermal_loss_fraction", "withdrawal_loss_fraction"]] = 0.0
    curves.loc[zero, "usable_fraction"] = 1.0


def build_fixed_cell_ordering(reference_samples: pd.DataFrame) -> pd.DataFrame:
    """Rank cells once, by their mean usable fraction over the reference samples.

    The usable fraction is usable capacity divided by nameplate. Cells are ranked
    within each region and synthetic plant, highest mean usable fraction first,
    ties broken by ``cell_id``, and ``fixed_order`` numbers them from 0.
    """

    required = {*ORDERING_KEYS, "nameplate_mw", "usable_capacity_mw"}
    missing = required.difference(reference_samples.columns)
    if missing:
        raise ValueError(f"ordering reference missing columns: {sorted(missing)}")
    if reference_samples.empty:
        raise ValueError("cannot build a fixed ordering from no reference samples")
    quality = reference_samples.assign(
        ordering_usable_fraction=lambda df: (
            df["usable_capacity_mw"] / df["nameplate_mw"]
        )
    )
    if not np.isfinite(quality["ordering_usable_fraction"]).all():
        raise ValueError("ordering reference contains non-finite usable fractions")
    ordering = (
        quality.groupby(list(ORDERING_KEYS), observed=True, sort=True)
        .agg(ordering_usable_fraction=("ordering_usable_fraction", "mean"))
        .reset_index()
        .sort_values(
            ["region_id", "archetype", "ordering_usable_fraction", "cell_id"],
            ascending=[True, True, False, True],
            kind="stable",
        )
    )
    ordering["fixed_order"] = ordering.groupby(
        ["region_id", "archetype"], observed=True, sort=False
    ).cumcount()
    return ordering[list(ORDERING_COLUMNS)].reset_index(drop=True)


def build_residual_water_cell_ordering(
    reference_samples: pd.DataFrame,
    water_limits: pd.DataFrame,
    river_network: pd.DataFrame,
    *,
    existing_site_samples: pd.DataFrame | None,
) -> pd.DataFrame:
    """Rank cells once on the reference usable capacity left after existing sites upstream take their water.

    This is the residual-water order: cells ranked on the water left after
    existing plants upstream take theirs. The ranking, ties and output columns are
    those of :func:`build_fixed_cell_ordering`; only the per-sample usable
    capacity it averages is replaced by :func:`residual_water_reference_samples`.
    Pass ``existing_site_samples=None`` explicitly to rank on the full flow (the
    full-flow order).
    """

    return build_fixed_cell_ordering(
        residual_water_reference_samples(
            reference_samples,
            water_limits,
            river_network,
            existing_site_samples=existing_site_samples,
        )
    )


def residual_water_reference_samples(
    reference_samples: pd.DataFrame,
    water_limits: pd.DataFrame,
    river_network: pd.DataFrame,
    *,
    existing_site_samples: pd.DataFrame | None,
) -> pd.DataFrame:
    """Reduce each potential-site sample's usable capacity by what the water taken by existing sites removes.

    The remaining water limits are computed exactly as the potential-site curve
    build computes them: per climate sample, over all cells at once, with the
    existing sites' withdrawal after thermal loss routed down the river network
    and subtracted. A cell then loses ``min(usable, full_flow) - min(usable,
    remaining)`` of its usable capacity: the part its own full-flow water limit
    supports and its remaining water limit does not. Where the full-flow limit is
    at least the usable capacity this leaves ``min(usable, remaining)``. Writing
    it as a loss keeps a cell with no existing-site water upstream at exactly its
    usable capacity, so without existing sites the order is the full-flow order
    rather than one reshuffled among tied cells by float32 rounding of the water
    limit in the site evaluation.

    The reduction depends only on the cell's own water limit, as the full-flow
    order does: sharing among potential sites and limits downstream are left to
    the curve build. A NaN water limit marks a cell that passes water on without a
    limit, and reduces nothing.

    Returns one row per input sample, keyed by ``CLIMATE_SAMPLE_KEYS``,
    ``region_id`` and ``cell_id``, with ``nameplate_mw``, the reduced
    ``usable_capacity_mw`` and the reduction itself in
    ``incumbent_water_loss_mw``.
    """

    data = _candidate_reach_inputs(reference_samples, water_limits, river_network)
    if existing_site_samples is not None:
        data = attach_existing_site_water_demand(data, existing_site_samples)
    parts = []
    for _, sample in data.groupby(list(CLIMATE_SAMPLE_KEYS), observed=True, sort=True):
        domain = sample.sort_values("cell_id", kind="stable").reset_index(drop=True)
        residual = _joint_ceilings(domain, _downstream_indices(domain))
        virgin = domain["water_ceiling_mw"].to_numpy(dtype=float)
        usable = domain["usable_capacity_mw"].to_numpy(dtype=float)
        loss = np.fmin(usable, virgin) - np.fmin(usable, residual)
        parts.append(
            domain[
                [*CLIMATE_SAMPLE_KEYS, "region_id", "cell_id", "nameplate_mw"]
            ].assign(
                usable_capacity_mw=usable - loss,
                **{EXISTING_SITE_WATER_LOSS_COLUMN: loss},
            )
        )
    if not parts:
        raise ValueError("residual-water ordering received no reference samples")
    return pd.concat(parts, ignore_index=True)


def _attach_fixed_ordering(
    data: pd.DataFrame, fixed_ordering: pd.DataFrame
) -> pd.DataFrame:
    if not isinstance(fixed_ordering, pd.DataFrame):
        raise TypeError(
            f"fixed ordering must be an ordering table, got {type(fixed_ordering).__name__}"
        )
    missing = set(ORDERING_COLUMNS).difference(fixed_ordering.columns)
    if missing:
        raise ValueError(f"fixed ordering missing columns: {sorted(missing)}")
    if fixed_ordering.duplicated(list(ORDERING_KEYS)).any():
        raise ValueError("fixed ordering contains duplicate physical cells")
    merged = data.merge(
        fixed_ordering[list(ORDERING_COLUMNS)],
        on=list(ORDERING_KEYS),
        how="left",
        validate="many_to_one",
    )
    missing_order = merged["fixed_order"].isna()
    if missing_order.any():
        sample = merged.loc[missing_order, list(ORDERING_KEYS)].drop_duplicates().head(5)
        raise ValueError(
            "fixed ordering does not cover the cells of the loss-split table: "
            f"{sample.to_dict(orient='records')}"
        )
    return merged


def build_capacity_curves(
    cell_table: pd.DataFrame,
    *,
    population: str = "incumbent",
    placement_rule: str = "existing_fleet",
    fixed_ordering: pd.DataFrame | IncumbentOrdering = OWN_MEAN_USABLE_FRACTION,
    wet_synthetic_plants: Sequence[SyntheticPlant] | None = None,
) -> pd.DataFrame:
    """Return capacity curves: cumulative nameplate, losses and usable capacity as cells are added in a fixed order.

    Each curve adds the cells of one region and synthetic plant in
    ``fixed_ordering``: a table (``region_id``, ``archetype``, ``cell_id``,
    ``ordering_usable_fraction``, ``fixed_order``) such as one shared with the
    potential-site curves, or ``"own_mean_usable_fraction"`` (the default, for
    existing sites) to rank the cells of ``cell_table`` on themselves with
    :func:`build_fixed_cell_ordering`: by their mean usable fraction over every
    row of ``cell_table``, highest first, ties broken by ``cell_id``.

    ``wet_synthetic_plants`` gives each synthetic plant name its plant type and cooling
    type; ``None`` uses the default wet synthetic plants.

    Raises:
        ValueError: if ``cell_table`` is empty or ``fixed_ordering`` is another
            string.
    """

    if cell_table.empty:
        raise ValueError("cannot build curves from an empty loss-split table")
    data = _with_archetype_parts(cell_table, wet_synthetic_plants).assign(
        usable_fraction=lambda df: df["usable_capacity_mw"] / df["nameplate_mw"],
        population=population,
        placement_rule=placement_rule,
    )
    if isinstance(fixed_ordering, str):
        if fixed_ordering != OWN_MEAN_USABLE_FRACTION:
            raise ValueError(
                f"unknown fixed ordering {fixed_ordering!r}; pass an ordering table "
                f"or {OWN_MEAN_USABLE_FRACTION!r}"
            )
        fixed_ordering = build_fixed_cell_ordering(data)
    data = _attach_fixed_ordering(data, fixed_ordering)
    rows = []
    for keys, group in data.groupby(list(CURVE_KEYS), sort=True, observed=True):
        ordered = group.sort_values("fixed_order", kind="stable").reset_index(drop=True)
        capacity_axis = ordered["nameplate_mw"].cumsum()
        thermal = ordered["thermal_loss_mw"].cumsum()
        withdrawal = ordered["withdrawal_loss_mw"].cumsum()
        usable = ordered["usable_capacity_mw"].cumsum()
        curve = pd.DataFrame(
            {
                **dict(zip(CURVE_KEYS, keys, strict=True)),
                "point_index": np.arange(1, len(ordered) + 1, dtype=np.int64),
                "capacity_axis_mw": capacity_axis,
                "marginal_cell_id": ordered["cell_id"].to_numpy(),
                "marginal_country": ordered["country"].to_numpy(),
                "marginal_capacity_mw": ordered["nameplate_mw"].to_numpy(),
                "marginal_usable_fraction": ordered["usable_fraction"].to_numpy(),
                "thermal_loss_mw": thermal,
                "withdrawal_loss_mw": withdrawal,
                "usable_capacity_mw": usable,
                "thermal_loss_fraction": thermal / capacity_axis,
                "withdrawal_loss_fraction": withdrawal / capacity_axis,
                "usable_fraction": usable / capacity_axis,
            }
        )
        rows.append(curve)
    if not rows:
        raise ValueError("loss-split table produced no curve groups")
    return pd.concat(rows, ignore_index=True)[list(CURVE_COLUMNS)]


def build_pooled_median_curves(curves: pd.DataFrame) -> pd.DataFrame:
    """Return the median of the curves across GCMs, with the GCM count behind each point."""

    pool_keys = [key for key in CURVE_KEYS if key != "gcm"] + [
        "point_index",
        "capacity_axis_mw",
    ]
    value_columns = [
        "marginal_capacity_mw",
        "marginal_usable_fraction",
        "thermal_loss_mw",
        "withdrawal_loss_mw",
        "usable_capacity_mw",
        "thermal_loss_fraction",
        "withdrawal_loss_fraction",
        "usable_fraction",
    ]
    # Every GCM must place the same capacity at every point; otherwise a point's
    # median would silently pool only the GCMs that share its capacity value.
    gcm_total = curves["gcm"].nunique()
    per_point = curves.groupby(pool_keys[:-1], observed=True).agg(
        axis_values=("capacity_axis_mw", "nunique"), gcms=("gcm", "nunique")
    )
    split = per_point[(per_point["axis_values"] > 1) | (per_point["gcms"] < gcm_total)]
    if len(split):
        first = split.reset_index().iloc[0].to_dict()
        raise ValueError(
            f"{len(split)} curve points differ in capacity across GCMs or are missing for "
            f"some of the {gcm_total} GCMs, so their median would pool fewer GCMs; "
            f"first: {first}"
        )
    pooled = (
        curves.groupby(pool_keys, observed=True, sort=True)
        .agg(
            gcm_count=("gcm", "nunique"),
            **{column: (column, "median") for column in value_columns},
        )
        .reset_index()
    )
    pooled["statistic"] = "pooled_median"
    return pooled[list(POOLED_COLUMNS)]


def reference_fleet_additivity_check(
    cell_table: pd.DataFrame,
    curves: pd.DataFrame,
    *,
    population: str = "incumbent",
    placement_rule: str = "existing_fleet",
) -> pd.DataFrame:
    """Compare the end points of the curves with the direct sum over the cells of ``cell_table`` (the additivity check).

    The end points of all synthetic-plant curves are summed per GCM, region,
    season and warming level and compared with the direct sums of nameplate,
    thermal loss, water-availability loss and usable capacity. ``max_abs_error_mw``
    is the largest absolute difference (MW), infinite where either side is
    missing.
    """

    direct = (
        cell_table.groupby(
            ["gcm", "region_id", "season", "warming_level"], observed=True
        )
        .agg(
            direct_nameplate_mw=("nameplate_mw", "sum"),
            direct_thermal_loss_mw=("thermal_loss_mw", "sum"),
            direct_withdrawal_loss_mw=("withdrawal_loss_mw", "sum"),
            direct_usable_capacity_mw=("usable_capacity_mw", "sum"),
        )
        .reset_index()
    )
    comparable = curves.loc[curves["placement_rule"] == placement_rule]
    final_points = (
        comparable.sort_values("point_index", kind="stable")
        .groupby(list(CURVE_KEYS), observed=True, sort=True)
        .tail(1)
    )
    reconstructed = (
        final_points.groupby(
            ["gcm", "region_id", "season", "warming_level"], observed=True
        )
        .agg(
            curve_nameplate_mw=("capacity_axis_mw", "sum"),
            curve_thermal_loss_mw=("thermal_loss_mw", "sum"),
            curve_withdrawal_loss_mw=("withdrawal_loss_mw", "sum"),
            curve_usable_capacity_mw=("usable_capacity_mw", "sum"),
        )
        .reset_index()
    )
    additivity_check = direct.merge(
        reconstructed,
        on=["gcm", "region_id", "season", "warming_level"],
        how="outer",
        validate="one_to_one",
    )
    error_columns = [
        ("direct_nameplate_mw", "curve_nameplate_mw"),
        ("direct_thermal_loss_mw", "curve_thermal_loss_mw"),
        ("direct_withdrawal_loss_mw", "curve_withdrawal_loss_mw"),
        ("direct_usable_capacity_mw", "curve_usable_capacity_mw"),
    ]
    errors = [
        (additivity_check[left] - additivity_check[right]).abs().rename(f"{left}_error")
        for left, right in error_columns
    ]
    additivity_check["max_abs_error_mw"] = pd.concat(errors, axis=1).max(axis=1)
    compared_columns = [column for pair in error_columns for column in pair]
    additivity_check.loc[additivity_check[compared_columns].isna().any(axis=1), "max_abs_error_mw"] = np.inf
    additivity_check["population"] = population
    additivity_check["placement_rule"] = placement_rule
    return additivity_check[list(ADDITIVITY_CHECK_COLUMNS)]


def validate_reference_fleet_additivity_check(
    additivity_check: pd.DataFrame,
    *,
    atol_mw: float = 1e-6,
) -> None:
    """Raise ``ValueError`` when the curve end points differ from the direct sum over cells by more than ``atol_mw`` (MW)."""

    bad = additivity_check.loc[additivity_check["max_abs_error_mw"] > atol_mw]
    if not bad.empty:
        sample = bad.head(5).to_dict(orient="records")
        raise ValueError(
            "the end points of the curves do not add up to the direct sum over cells: "
            f"max_abs_error_mw={additivity_check['max_abs_error_mw'].max():.6g}; sample={sample}"
        )


def _load_cell_means(paths: list[str | Path], *, population: str) -> pd.DataFrame:
    parts = [
        _load_decomposition_samples([path], population=population) for path in paths
    ]
    data = pd.concat(parts, ignore_index=True)
    if data.empty:
        raise ValueError(f"no rows found for population={population!r}")
    return (
        data.groupby(list(CELL_KEYS), observed=True, sort=True)
        .agg(
            nameplate_mw=("nameplate_mw", "first"),
            thermal_loss_mw=("thermal_loss_mw", "mean"),
            withdrawal_loss_mw=("withdrawal_loss_mw", "mean"),
            usable_capacity_mw=("usable_capacity_mw", "mean"),
        )
        .reset_index()
    )


def _load_decomposition_samples(
    paths: list[str | Path], *, population: str
) -> pd.DataFrame:
    dataset = pads.dataset([str(path) for path in paths], format="parquet")
    table = dataset.to_table(
        columns=list(LOSS_SPLIT_COLUMNS),
        filter=pc.field("population") == population,
    )
    frame = table.to_pandas(self_destruct=True, split_blocks=True)
    missing = set(LOSS_SPLIT_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"loss-split inputs missing columns: {sorted(missing)}")
    return frame


def _load_reach_water_ceilings(
    paths: list[str | Path],
    *,
    population: str,
    preserve_year: bool = False,
) -> pd.DataFrame:
    dataset = pads.dataset([str(path) for path in paths], format="parquet")
    predicate = (pc.field("population") == population) & (
        pc.field("response") == WATER_LIMIT_RESPONSE
    )
    table = dataset.to_table(columns=list(WATER_LIMIT_COLUMNS), filter=predicate)
    frame = table.to_pandas(self_destruct=True, split_blocks=True)
    if frame.empty:
        raise ValueError(
            f"no {WATER_LIMIT_RESPONSE!r} rows for population={population!r}"
        )
    group_columns = [
        "population",
        "region_id",
        "country",
        "cell_id",
        "grid_lat",
        "grid_lon",
        "archetype",
        "gcm",
        "scenario",
        *(("year",) if preserve_year else ()),
        "season",
        "warming_level",
    ]
    return (
        frame.drop(columns=["response"])
        .rename(columns={"value": "water_ceiling_mw"})
        .groupby(group_columns, observed=True, sort=True)["water_ceiling_mw"]
        .mean()
        .reset_index()
    )


def _load_topology(paths: list[str | Path], *, population: str) -> pd.DataFrame:
    parts = []
    for path in paths:
        frame = pd.read_parquet(
            path, columns=["population", "cell_id", DOWNSTREAM_COLUMN]
        )
        frame["scenario"] = _scenario_from_path(Path(path))
        parts.append(frame[list(RIVER_NETWORK_COLUMNS)])
    river_network = pd.concat(parts, ignore_index=True)
    river_network = river_network.loc[
        river_network["population"] == population, list(RIVER_NETWORK_COLUMNS)
    ]
    river_network = river_network.drop_duplicates()
    duplicated = river_network[["scenario", "cell_id"]].duplicated(keep=False)
    if duplicated.any():
        sample = river_network.loc[duplicated, ["scenario", "cell_id"]].head(5)
        raise ValueError(
            "river network has conflicting duplicate scenario/cell_id rows: "
            f"{sample.to_dict(orient='records')}"
        )
    if river_network.empty:
        raise ValueError(f"river network has no rows for population={population!r}")
    return river_network


def _scenario_from_path(path: Path) -> str:
    match = re.search(r"(ssp\d{3}|historical)", path.name)
    if match is None:
        raise ValueError(f"cannot infer scenario from river-network file path: {path}")
    return match.group(1)


def _candidate_reach_inputs(
    loss_split_samples: pd.DataFrame,
    water_limits: pd.DataFrame,
    river_network: pd.DataFrame,
) -> pd.DataFrame:
    merge_keys = [
        "population",
        "region_id",
        "country",
        "cell_id",
        "grid_lat",
        "grid_lon",
        "archetype",
        "gcm",
        "scenario",
        *(["year"] if "year" in water_limits else []),
        "season",
        "warming_level",
    ]
    ceiling_values = water_limits[[*merge_keys, "water_ceiling_mw"]].assign(
        _ceiling_key_present=True
    )
    merged = loss_split_samples.merge(
        ceiling_values,
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )
    missing_ceiling_key = merged["_ceiling_key_present"].isna()
    if missing_ceiling_key.any():
        sample = merged.loc[
            missing_ceiling_key,
            [
                "cell_id",
                "archetype",
                "gcm",
                "scenario",
                "year",
                "season",
                "warming_level",
            ],
        ].head(5)
        raise ValueError(
            "missing river-network water limits for rows of the loss-split table: "
            f"{sample.to_dict(orient='records')}"
        )
    merged = merged.drop(columns=["_ceiling_key_present"])
    merged = merged.merge(
        river_network[["scenario", "cell_id", DOWNSTREAM_COLUMN]],
        on=["scenario", "cell_id"],
        how="left",
        validate="many_to_one",
    )
    topology_keys = pd.MultiIndex.from_frame(river_network[["scenario", "cell_id"]])
    row_keys = pd.MultiIndex.from_frame(merged[["scenario", "cell_id"]])
    missing_topology = merged[DOWNSTREAM_COLUMN].isna() & ~row_keys.isin(topology_keys)
    if missing_topology.any():
        sample = merged.loc[missing_topology, ["scenario", "cell_id"]].head(5)
        raise ValueError(
            "missing river-network rows for scenario/cell_id pairs: "
            f"{sample.to_dict(orient='records')}"
        )
    return merged


EXISTING_SITE_DEMAND_COLUMN = "incumbent_thermal_mw"


def attach_existing_site_water_demand(
    potential_site_inputs: pd.DataFrame,
    existing_site_samples: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the existing sites' water demand, after thermal loss, to the potential sites.

    Without this, potential and existing sites are each given the whole river
    independently: water is shared along the river network among potential sites,
    but nothing subtracts the water that existing sites already withdraw, so the
    water left for potential sites counts full flow that is in fact already used.
    The demand attached here is ``nameplate - thermal_loss`` (at least 0), the
    capacity that actually reaches the water limit, matched on the same climate
    sample, synthetic plant and cell as the water limit it will be subtracted
    from.

    Cells with no existing site get zero, so this changes nothing wherever the
    two sets of cells do not overlap.
    """

    keys = ["gcm", "scenario", "archetype", "season", "warming_level", "cell_id"]
    if "year" in potential_site_inputs.columns and "year" in existing_site_samples.columns:
        keys.insert(2, "year")
    missing = [k for k in keys if k not in existing_site_samples.columns]
    if missing:
        raise ValueError(f"existing-site samples are missing join keys: {missing}")

    demand = (
        existing_site_samples.assign(
            **{
                EXISTING_SITE_DEMAND_COLUMN: lambda df: (
                    df["nameplate_mw"] - df["thermal_loss_mw"]
                ).clip(lower=0.0)
            }
        )
        .groupby(keys, observed=True, as_index=False)[EXISTING_SITE_DEMAND_COLUMN]
        .sum()
    )
    merged = potential_site_inputs.merge(demand, on=keys, how="left", validate="many_to_one")
    merged[EXISTING_SITE_DEMAND_COLUMN] = merged[EXISTING_SITE_DEMAND_COLUMN].fillna(0.0)
    return merged


def _joint_ceilings(domain: pd.DataFrame, downstream: np.ndarray) -> np.ndarray:
    """Potential-site water limits after the existing sites' usage, routed down the river network.

    Computed once per climate sample over all cells rather than per region, so an
    existing site upstream in one region reduces the water left for potential
    sites downstream in another.
    """

    ceiling = domain["water_ceiling_mw"].to_numpy(dtype=float)
    if EXISTING_SITE_DEMAND_COLUMN not in domain.columns:
        return ceiling
    demand = domain[EXISTING_SITE_DEMAND_COLUMN].to_numpy(dtype=float)
    if not demand.any():
        return ceiling
    used = river_network_capped_allocation(ceiling, demand, downstream)
    return remaining_water_limits(ceiling, used, downstream)


def _sample_reach_candidate_curves(
    sample: pd.DataFrame,
    *,
    capacity_grid_points: int,
    placement_rule: str,
) -> list[pd.DataFrame]:
    domain = sample.sort_values("cell_id", kind="stable").reset_index(drop=True)
    downstream = _downstream_indices(domain)
    ceilings = _joint_ceilings(domain, downstream)
    rows = []
    for region_id, target in domain.groupby("region_id", observed=True, sort=True):
        ordered = target.sort_values("fixed_order", kind="stable")
        total_capacity = float(ordered["nameplate_mw"].sum())
        grid = np.linspace(0.0, total_capacity, capacity_grid_points)
        target_positions = ordered.index.to_numpy(dtype=int)
        placed_nameplate = _placed_nameplate_matrix(domain, target_positions, grid)
        thermal_placed = (
            placed_nameplate
            * domain["thermal_available_fraction"].to_numpy(dtype=float)[:, None]
        )
        allocation = river_network_capped_allocation(
            ceilings,
            thermal_placed,
            downstream,
        )
        thermal_loss = (
            placed_nameplate
            * domain["thermal_loss_fraction_local"].to_numpy(dtype=float)[:, None]
        ).sum(axis=0)
        thermal_capacity = thermal_placed.sum(axis=0)
        usable = allocation.sum(axis=0)
        withdrawal_loss = thermal_capacity - usable
        marginal = _marginal_rows(ordered, grid)
        curve = pd.DataFrame(
            {
                "population": domain["population"].iloc[0],
                "placement_rule": placement_rule,
                "gcm": domain["gcm"].iloc[0],
                "scenario": domain["scenario"].iloc[0],
                "year": int(domain["year"].iloc[0]),
                "region_id": region_id,
                "plant_type": domain["plant_type"].iloc[0],
                "cooling_type": domain["cooling_type"].iloc[0],
                "archetype": domain["archetype"].iloc[0],
                "season": domain["season"].iloc[0],
                "warming_level": domain["warming_level"].iloc[0],
                "point_index": np.arange(1, grid.size + 1, dtype=np.int64),
                "capacity_axis_mw": grid,
                "marginal_cell_id": marginal["cell_id"].to_numpy(),
                "marginal_country": marginal["country"].to_numpy(),
                "marginal_capacity_mw": marginal["placed_mw"].to_numpy(),
                "marginal_usable_fraction": marginal[
                    "local_usable_fraction"
                ].to_numpy(),
                "thermal_loss_mw": thermal_loss,
                "withdrawal_loss_mw": withdrawal_loss,
                "usable_capacity_mw": usable,
            }
        )
        rows.append(curve)
    return rows


def _downstream_indices(cells: pd.DataFrame) -> np.ndarray:
    index = {cell_id: position for position, cell_id in enumerate(cells["cell_id"])}
    downstream = np.full(len(cells), -1, dtype=int)
    for position, target in enumerate(cells[DOWNSTREAM_COLUMN]):
        if pd.isna(target):
            continue
        if target not in index:
            continue
        downstream[position] = index[target]
    return downstream


def _placed_nameplate_matrix(
    domain: pd.DataFrame,
    target_positions: np.ndarray,
    capacity_grid: np.ndarray,
) -> np.ndarray:
    placed = np.zeros((len(domain), capacity_grid.size), dtype=float)
    nameplate = domain.loc[target_positions, "nameplate_mw"].to_numpy(dtype=float)
    starts = np.concatenate([[0.0], np.cumsum(nameplate)[:-1]])
    amounts = np.clip(capacity_grid[None, :] - starts[:, None], 0.0, nameplate[:, None])
    placed[target_positions, :] = amounts
    return placed


def _marginal_rows(ordered: pd.DataFrame, capacity_grid: np.ndarray) -> pd.DataFrame:
    nameplate = ordered["nameplate_mw"].to_numpy(dtype=float)
    cumulative = np.cumsum(nameplate)
    starts = np.concatenate([[0.0], cumulative[:-1]])
    positions = np.searchsorted(cumulative, capacity_grid, side="left")
    positions = np.minimum(positions, len(ordered) - 1)
    marginal = ordered.iloc[positions][
        ["cell_id", "country", "local_usable_fraction"]
    ].copy()
    placed = np.clip(capacity_grid - starts[positions], 0.0, nameplate[positions])
    marginal["placed_mw"] = np.where(capacity_grid > 0.0, placed, 0.0)
    marginal.loc[
        capacity_grid == 0.0, ["cell_id", "country", "local_usable_fraction"]
    ] = np.nan
    return marginal


def _with_archetype_parts(
    data: pd.DataFrame, wet_synthetic_plants: Sequence[SyntheticPlant] | None = None
) -> pd.DataFrame:
    """Attach ``plant_type`` and ``cooling_type`` from the named wet synthetic plants.

    The two values are the definition's ``wet_fuel_type`` and ``cooling_physics``;
    a synthetic plant name the definitions do not know raises ``ValueError``.
    """

    names = data["archetype"]
    synthetic_plants = _wet_archetypes_named(pd.unique(names), wet_synthetic_plants)
    return data.assign(
        plant_type=[synthetic_plants[name].parameters["wet_fuel_type"] for name in names],
        cooling_type=[synthetic_plants[name].parameters["cooling_physics"] for name in names],
    )
