"""Output files for the siting of new wet-cooled plants, for use in MESSAGE scenario runs.

Two builders make the outputs from loss-split tables (parquet files that split
each cell's wet capacity loss into thermal loss and water-availability loss),
water limits and river-network files, and return them as in-memory tables:

* :func:`build_residual_water_ordering` ranks potential sites on the water that
  remains after existing sites withdraw theirs, pooled over all GCMs.
* :func:`build_wet_siting_outputs` builds the capacity curves of potential
  sites, their median across GCMs, the additivity check and the RIME-X
  warming-level samples (RIME-X is IIASA's emulator of regional climate impacts
  by warming level), with the water of potential sites shared with existing
  sites.

A third step pools the outputs of several GCMs, each written to its own
directory by :func:`write_wet_siting_outputs`. It combines the two
calculations that the outputs come from: the warming-level calculation and the
present-day reference calculation.

* :func:`pool_wet_siting_outputs` pools the per-GCM outputs of both
  calculations: the curves of each with their median across GCMs, the check that
  both place the same physical cells, and the warming-level-bin percentile
  dataset.
* :func:`finalize_wet_siting_outputs` reads the per-GCM output directories,
  pools them and writes the result with a ``validation.json`` of the checks.

The writers only save what the builders return.
"""

from __future__ import annotations

import gc
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pandas as pd
import xarray as xr
from pandas.api.types import union_categoricals

from impacts.thermoelectric.siting._io import (
    refuse_existing_outputs,
    staged_outputs,
)
from impacts.thermoelectric.siting.cells import SyntheticPlant
from impacts.thermoelectric.siting.curves import (
    EXISTING_SITE_WATER_LOSS_COLUMN,
    ORDERING_KEYS,
    _load_cell_means,
    _load_decomposition_samples,
    _load_reach_water_ceilings,
    _load_topology,
    average_potential_site_sample_curves_with_river_network,
    build_fixed_cell_ordering,
    build_pooled_median_curves,
    build_potential_site_sample_curves_with_river_network,
    reference_fleet_additivity_check,
    residual_water_reference_samples,
)
from impacts.thermoelectric.siting.rime import (
    DEFAULT_NEW_BUILD_RATIOS,
    build_warming_level_bin_percentiles,
    build_wet_rime_samples_from_curves,
    percentiles_to_dataset,
)

PER_GCM_CURVES_FILE = "wet_ucr_curves_per_gcm.parquet"
POOLED_MEDIAN_CURVES_FILE = "wet_ucr_curves_pooled_median.parquet"
ADDITIVITY_ORACLE_FILE = "wet_ucr_curve_additivity_oracle.parquet"
RIME_SAMPLES_FILE = "wet_ucr_rime_samples.parquet"
GWL_ARM_DIRECTORY = "gwl"
REFERENCE_ARM_DIRECTORY = "reference_pd"
RIME_DATASET_FILE = "r12_thermoelectric_wet_gwl.nc"
VALIDATION_FILE = "validation.json"

# Default dimension sizes the pooled RIME-X dataset is checked against: R12 regions,
# wet plant types, cooling types, seasons, warming levels and new-build ratios.
# ``gwl: 6`` is the six warming levels 1.5-4.0 K. The default ``GWLSampling`` gives 7
# levels (1.0-4.0 K), so callers with other sampling pass ``expected_rime_dimensions``.
DEFAULT_RIME_DIMENSIONS = {
    "region_id": 12,
    "plant_type": 2,
    "cooling_type": 2,
    "season": 4,
    "gwl": 6,
    "depth_ratio": 3,
}


def existing_potential_site_overlap(
    existing: pd.DataFrame, potential: pd.DataFrame
) -> pd.DataFrame:
    """Share of existing-site nameplate capacity on cells that are also potential sites, by region.

    The water that existing sites withdraw is subtracted only where an existing
    site's cell is also a potential site of the same synthetic plant; demand
    anywhere else is not subtracted. Columns: ``region_id``, ``nameplate_mw``
    (existing-site nameplate, MW), ``matched_mw`` (the part on a potential-site
    cell, MW) and ``matched_share`` (their ratio).
    """

    keys = ["cell_id", "archetype"]
    plants = existing[["region_id", *keys, "nameplate_mw"]].drop_duplicates(keys)
    potential_cells = pd.MultiIndex.from_frame(potential[keys].drop_duplicates())
    on_potential = pd.MultiIndex.from_frame(plants[keys]).isin(potential_cells)
    plants = plants.assign(matched_mw=plants["nameplate_mw"].where(on_potential, 0.0))
    overlap = plants.groupby("region_id", observed=True, sort=True)[
        ["nameplate_mw", "matched_mw"]
    ].sum()
    overlap["matched_share"] = overlap["matched_mw"] / overlap["nameplate_mw"]
    return overlap.reset_index()


@dataclass(frozen=True)
class WetSitingOutputs:
    """Output tables for potential wet-cooled sites, one table per output file.

    Attributes:
        per_gcm_curves: Capacity curve of the potential sites per GCM, region,
            synthetic plant, season and warming level, averaged over the sampled
            years. The curve maps new capacity (``capacity_axis_mw``, MW, cells
            filled in the fixed order) to the capacity still usable after
            thermal loss and water-availability loss (``usable_capacity_mw``,
            MW).
        pooled_median_curves: Median of those curves across GCMs, with the GCM
            count behind each point.
        additivity_oracle: The additivity check. Per GCM, region, season and
            warming level, the end point of the curves against the direct sum
            over the cells' own results (MW); ``max_abs_error_mw`` is the largest
            absolute difference. For potential sites it is written but not
            checked, and it would not pass: sharing water along the river
            network makes the curve's usable capacity smaller than the sum of
            each cell's own usable capacity, by the water given up to upstream
            cells.
        rime_samples: Per climate sample and new-build ratio, the existing-site
            usable fraction and the potential-site usable fraction at
            ``depth_ratio`` times the existing-site capacity, with the usable
            capacity ratio of the two together (``usable_capacity_ratio``).
        incumbent_overlap: Per region, the existing-site nameplate (MW) on cells
            that are also potential sites, which is therefore subtracted from the
            potential-site water.
    """

    per_gcm_curves: pd.DataFrame
    pooled_median_curves: pd.DataFrame
    additivity_oracle: pd.DataFrame
    rime_samples: pd.DataFrame
    incumbent_overlap: pd.DataFrame


def build_wet_siting_outputs(
    loss_split_paths: Sequence[str | Path],
    water_limit_paths: Sequence[str | Path],
    river_network_paths: Sequence[str | Path],
    fixed_ordering: pd.DataFrame,
    *,
    capacity_grid_points: int = 61,
    new_build_ratios: Iterable[float] = DEFAULT_NEW_BUILD_RATIOS,
    wet_synthetic_plants: Sequence[SyntheticPlant] | None = None,
) -> WetSitingOutputs:
    """Build the capacity curves of potential sites and the RIME-X samples from file inputs.

    Water is shared between potential and existing sites: the existing sites'
    withdrawal after thermal loss is routed down the river network and
    subtracted from the potential-site water limits before cells are placed in
    ``fixed_ordering``. The sample curves are solved once and reused for the
    averaged curves, the pooling across GCMs, the additivity check and the RIME-X
    samples.

    Parameters:
        loss_split_paths: Loss-split parquet files (wet capacity loss split
            into thermal loss and water-availability loss), holding
            potential-site and existing-site samples for every GCM and scenario
            built.
        water_limit_paths: Site-evaluation parquet files with the water
            limit for each year (``water_limited_capacity_ceiling_mw``, MW).
        river_network_paths: River-network parquet files, one per scenario; the
            scenario is read from the file name.
        fixed_ordering: Cell order with ``region_id``, ``archetype``, ``cell_id``,
            ``ordering_usable_fraction`` and ``fixed_order``.
        capacity_grid_points: Number of capacity steps on each curve, from 0 to
            the region's total potential-site nameplate (at least 2).
        new_build_ratios: New-build ratios: new capacity as multiples of the
            existing-site capacity.
        wet_synthetic_plants: Definitions of the synthetic plant names in the inputs;
            the default wet synthetic plants when ``None``.

    Raises:
        ValueError: if ``capacity_grid_points`` is below 2, or if no existing
            site is also a potential site, since sharing water between them would
            then change nothing (the overlap table is appended to the message).
    """

    loss_split_paths = list(loss_split_paths)
    potential = _load_decomposition_samples(loss_split_paths, population="candidate")
    existing = _load_decomposition_samples(loss_split_paths, population="incumbent")
    overlap = existing_potential_site_overlap(existing, potential)
    if not (overlap["matched_mw"] > 0.0).any():
        raise ValueError(
            "no existing site is also a potential site; sharing water between "
            "them would be a no-op\n"
            "existing-site nameplate subtracted from potential-site water limits, by region:\n"
            f"{overlap.to_string(index=False)}"
        )
    ceilings = _load_reach_water_ceilings(
        list(water_limit_paths), population="candidate", preserve_year=True
    )
    river_network = _load_topology(list(river_network_paths), population="candidate")
    sample_curves = build_potential_site_sample_curves_with_river_network(
        potential,
        ceilings,
        river_network,
        capacity_grid_points=capacity_grid_points,
        fixed_ordering=fixed_ordering,
        existing_site_samples=existing,
        wet_synthetic_plants=wet_synthetic_plants,
    )
    curves = average_potential_site_sample_curves_with_river_network(sample_curves)
    cell_means = _load_cell_means(loss_split_paths, population="candidate")
    return WetSitingOutputs(
        per_gcm_curves=curves,
        pooled_median_curves=build_pooled_median_curves(curves),
        additivity_oracle=reference_fleet_additivity_check(
            cell_means,
            curves,
            population="candidate",
            placement_rule="candidate_reach_aware",
        ),
        rime_samples=build_wet_rime_samples_from_curves(
            existing, sample_curves, new_build_ratios=new_build_ratios, wet_synthetic_plants=wet_synthetic_plants
        ),
        incumbent_overlap=overlap,
    )


def write_wet_siting_outputs(
    products: WetSitingOutputs, directory: str | Path, *, overwrite: bool = False
) -> list[Path]:
    """Write the four output tables to ``directory`` as parquet; return the paths.

    Raises:
        FileExistsError: if any of the four files exists and ``overwrite`` is false;
            nothing is written then.
    """

    root = Path(directory)
    tables = {
        root / PER_GCM_CURVES_FILE: products.per_gcm_curves,
        root / POOLED_MEDIAN_CURVES_FILE: products.pooled_median_curves,
        root / ADDITIVITY_ORACLE_FILE: products.additivity_oracle,
        root / RIME_SAMPLES_FILE: products.rime_samples,
    }
    if not overwrite:
        refuse_existing_outputs(tables)
    with staged_outputs(tables) as staged:
        for path, table in tables.items():
            table.to_parquet(staged[path], index=False)
    return list(tables)


@dataclass(frozen=True)
class ResidualWaterOrdering:
    """Potential sites ranked on the water left after existing sites take theirs, with diagnostics.

    Attributes:
        ordering: One row per region, synthetic plant and cell: mean usable
            fraction of nameplate after the water taken by existing sites
            (``ordering_usable_fraction``) and the resulting rank within region
            and synthetic plant (``fixed_order``, 0 is filled first).
        charged_share: Per GCM and region, the share of potential-site samples
            whose usable capacity was reduced by the water of existing sites.
        moves: Per region and synthetic plant, the number of cells and of those
            whose rank differs from the baseline ordering; ``None`` without a
            baseline.
    """

    ordering: pd.DataFrame
    charged_share: pd.DataFrame
    moves: pd.DataFrame | None


def build_residual_water_ordering(
    loss_split_paths: Sequence[str | Path],
    water_limit_paths: Sequence[str | Path],
    river_network_paths: Sequence[str | Path],
    *,
    baseline_ordering: pd.DataFrame | None = None,
) -> ResidualWaterOrdering:
    """Rank potential sites on the water left after existing sites take theirs (the residual-water order).

    Each GCM's potential-site samples are reduced by the usable capacity that the
    existing sites' withdrawal, routed down the river network, removes; the
    reduced samples of all GCMs are then pooled and ranked together, by mean
    usable fraction, highest first, with ties broken by ``cell_id``, as in
    :func:`build_fixed_cell_ordering`. The three path sequences hold one file per
    GCM, in the same GCM order; the loss-split table of each GCM must hold one GCM
    and match its river-network file name.

    Parameters:
        loss_split_paths: Loss-split parquet file (wet capacity loss split into
            thermal loss and water-availability loss), one per GCM.
        water_limit_paths: Site-evaluation parquet file with the water
            limit (MW), one per GCM.
        river_network_paths: River-network parquet file, one per GCM.
        baseline_ordering: Full-flow ordering (ranked before any existing plant
            takes water) to report rank changes against. It carries the ordering
            key columns (``region_id``, ``archetype``, ``cell_id``) plus
            ``fixed_order``, filled from 0 upward within each region and synthetic
            plant.

    Raises:
        ValueError: if the path sequences differ in length, if a loss-split table
            does not pair with its river-network file on one GCM, if the baseline
            covers other cells, or if the result equals the baseline ranking (the
            water taken by existing sites then changes nothing; the moves table
            is appended to the message).
    """

    if not (len(loss_split_paths) == len(water_limit_paths) == len(river_network_paths)):
        raise ValueError("give one loss-split table, one water-limit file and one river-network file per GCM")

    parts = []
    shares = []
    for decomposition, ceiling, river_network in zip(
        loss_split_paths, water_limit_paths, river_network_paths, strict=True
    ):
        part, share = _charged_samples(Path(decomposition), Path(ceiling), Path(river_network))
        parts.append(part)
        shares.append(share)
        gc.collect()
    ordering = build_fixed_cell_ordering(_pool(parts))
    for key in ORDERING_KEYS:
        ordering[key] = ordering[key].astype(str)

    moves = None
    if baseline_ordering is not None:
        moves = _moves(ordering, baseline_ordering)
        if not moves["moved"].any():
            raise ValueError(
                "residual-water ordering equals the baseline; the water taken by existing "
                "sites never changes a rank, so re-exporting on it would be a no-op\n"
                "cells whose rank differs from the baseline ordering:\n"
                f"{moves.to_string(index=False)}"
            )
    return ResidualWaterOrdering(
        ordering=ordering,
        charged_share=pd.concat(shares, ignore_index=True),
        moves=moves,
    )


def write_residual_water_ordering(
    result: ResidualWaterOrdering, path: str | Path, *, overwrite: bool = False
) -> Path:
    """Write the ordering to ``path`` as parquet and return the path.

    Raises:
        FileExistsError: if ``path`` exists and ``overwrite`` is false.
    """

    output = Path(path)
    if not overwrite:
        refuse_existing_outputs([output])
    with staged_outputs([output]) as staged:
        result.ordering.to_parquet(staged[output], index=False)
    return output


def _charged_samples(
    decomposition: Path, ceiling: Path, river_network: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = _load_decomposition_samples([decomposition], population="candidate")
    gcms = samples["gcm"].unique().tolist()
    if len(gcms) != 1 or gcms[0] not in river_network.name:
        raise ValueError(
            f"inputs do not pair on one GCM: loss-split table {gcms}, river-network file {river_network.name}"
        )
    existing = _load_decomposition_samples([decomposition], population="incumbent")
    charged = residual_water_reference_samples(
        samples,
        _load_reach_water_ceilings([ceiling], population="candidate", preserve_year=True),
        _load_topology([river_network], population="candidate"),
        existing_site_samples=existing,
    )
    share = (
        (charged[EXISTING_SITE_WATER_LOSS_COLUMN] > 0.0)
        .groupby(charged["region_id"])
        .mean()
        .rename("charged_share")
        .reset_index()
        .assign(gcm=gcms[0])[["gcm", "region_id", "charged_share"]]
    )
    kept = charged[[*ORDERING_KEYS, "nameplate_mw", "usable_capacity_mw"]]
    return kept.astype({key: "category" for key in ORDERING_KEYS}), share


def _pool(parts: list[pd.DataFrame]) -> pd.DataFrame:
    # Categorical keys keep the samples of many GCMs small; sorted categories keep the
    # ranking's cell_id tie-break lexicographic, as it is on strings.
    columns = {
        key: union_categoricals([part[key] for part in parts], sort_categories=True)
        for key in ORDERING_KEYS
    }
    pooled = pd.DataFrame(columns)
    for value in ("nameplate_mw", "usable_capacity_mw"):
        pooled[value] = pd.concat([part[value] for part in parts], ignore_index=True)
    return pooled


def _moves(ordering: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    joined = ordering.merge(
        baseline[[*ORDERING_KEYS, "fixed_order"]],
        on=list(ORDERING_KEYS),
        how="outer",
        suffixes=("", "_baseline"),
        validate="one_to_one",
        indicator=True,
    )
    if (joined["_merge"] != "both").any():
        raise ValueError("new and baseline orderings cover different cells")
    joined["moved"] = joined["fixed_order"] != joined["fixed_order_baseline"]
    return (
        joined.groupby(["region_id", "archetype"], observed=True)["moved"]
        .agg(cells="size", moved="sum")
        .reset_index()
    )


class PooledValidation(TypedDict):
    """Counts of the pooled outputs that were checked or recorded.

    A plain ``dict`` at run time, so it prints and serialises as one:

    * ``fixed_membership_rows``: curve points (GCM, region, synthetic plant,
      point index) whose physical cell both calculations share.
    * ``gcms``, ``regions``, ``archetypes``, ``seasons``, ``gwl_levels`` and
      ``curve_points``: distinct values in the curves of the warming-level
      calculation.
    * ``rime_dimensions``: sizes of the RIME-X dataset's dimensions.
    """

    fixed_membership_rows: int
    gcms: int
    regions: int
    archetypes: int
    seasons: int
    gwl_levels: int
    curve_points: int
    rime_dimensions: dict[str, int]


@dataclass(frozen=True)
class PooledWetOutputs:
    """The wet-cooling outputs pooled over GCMs, with their validation.

    Attributes:
        gwl_per_gcm_curves: Potential-site curves of the warming-level
            calculation, all GCMs.
        gwl_pooled_median_curves: Their median across GCMs.
        reference_per_gcm_curves: Potential-site curves of the present-day
            reference calculation.
        reference_pooled_median_curves: Their median across GCMs.
        rime_percentiles: Percentiles of the RIME-X samples within each
            warming-level bin, as the dataset that is written (usable fractions
            are fractions of the existing-site nameplate capacity).
        validation: The counts that were checked or recorded, as
            :class:`PooledValidation` lists them.
    """

    gwl_per_gcm_curves: pd.DataFrame
    gwl_pooled_median_curves: pd.DataFrame
    reference_per_gcm_curves: pd.DataFrame
    reference_pooled_median_curves: pd.DataFrame
    rime_percentiles: xr.Dataset
    validation: PooledValidation


def read_gcm_outputs(root: str | Path, name: str, *, gcm_count: int) -> pd.DataFrame:
    """Concatenate ``name`` from every ``<root>/<gcm directory>/``, in path order.

    Each GCM has its own output directory under ``root``.

    Raises:
        ValueError: if ``name`` is found in other than ``gcm_count`` directories.
    """

    paths = sorted(Path(root).glob(f"*/{name}"))
    if len(paths) != gcm_count:
        raise ValueError(
            f"expected {name} in {gcm_count} per-GCM output directories, found {len(paths)}"
        )
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def pool_wet_siting_outputs(
    gwl_curves: pd.DataFrame,
    reference_curves: pd.DataFrame,
    gwl_rime_samples: pd.DataFrame,
    *,
    expected_rime_dimensions: Mapping[str, int] | None = DEFAULT_RIME_DIMENSIONS,
) -> PooledWetOutputs:
    """Pool the per-GCM outputs of both calculations and package the RIME-X percentiles.

    The warming-level calculation and the present-day reference calculation must
    place the same physical cells: the cell added at every curve point (GCM,
    region, synthetic plant, point index; the zero-capacity first point excluded)
    must not vary with season or warming level and must be the same in both
    calculations. The warming-level-bin percentiles are taken over the pooled
    RIME-X samples of the warming-level calculation only.

    Parameters:
        gwl_curves: ``wet_ucr_curves_per_gcm`` of all GCMs, warming-level
            calculation.
        reference_curves: The same for the present-day reference calculation.
        gwl_rime_samples: ``wet_ucr_rime_samples`` of all GCMs, warming-level
            calculation.
        expected_rime_dimensions: Required sizes of the dataset's dimensions
            (``DEFAULT_RIME_DIMENSIONS`` by default); ``None`` skips the check.

    Raises:
        ValueError: if a cell's membership changes across season or warming level,
            if the two calculations cover different curve points or different
            cells, or if the RIME-X dataset has other dimension sizes than
            expected or any missing value.
        pandas.errors.MergeError: if a curve point repeats within a calculation.
    """

    gwl_membership = _membership(gwl_curves)
    reference_membership = _membership(reference_curves)
    identity = gwl_membership.merge(
        reference_membership,
        on=["gcm", "region_id", "archetype", "point_index"],
        suffixes=("_gwl", "_reference"),
        validate="one_to_one",
    )
    if len(identity) != len(gwl_membership) or len(identity) != len(reference_membership):
        raise ValueError(
            "the warming-level and present-day reference calculations do not cover "
            "identical curve membership"
        )
    if not (identity["marginal_cell_id_gwl"] == identity["marginal_cell_id_reference"]).all():
        raise ValueError(
            "the warming-level and present-day reference calculations use different "
            "physical cells"
        )

    dataset = percentiles_to_dataset(build_warming_level_bin_percentiles(gwl_rime_samples))
    if expected_rime_dimensions is not None and dict(dataset.sizes) != dict(
        expected_rime_dimensions
    ):
        raise ValueError(f"unexpected RIME dimensions: {dict(dataset.sizes)}")
    if dataset.to_array().isnull().any().item():
        raise ValueError("packaged RIME dataset contains missing values")

    validation = PooledValidation(
        fixed_membership_rows=len(identity),
        gcms=int(gwl_curves["gcm"].nunique()),
        regions=int(gwl_curves["region_id"].nunique()),
        archetypes=int(gwl_curves["archetype"].nunique()),
        seasons=int(gwl_curves["season"].nunique()),
        gwl_levels=int(gwl_curves["warming_level"].nunique()),
        curve_points=int(gwl_curves["point_index"].nunique()),
        rime_dimensions=dict(dataset.sizes),
    )
    return PooledWetOutputs(
        gwl_per_gcm_curves=gwl_curves,
        gwl_pooled_median_curves=build_pooled_median_curves(gwl_curves),
        reference_per_gcm_curves=reference_curves,
        reference_pooled_median_curves=build_pooled_median_curves(reference_curves),
        rime_percentiles=dataset,
        validation=validation,
    )


def pooled_output_paths(directory: str | Path) -> list[Path]:
    """The files :func:`write_pooled_wet_outputs` writes under ``directory``, in its order."""

    root = Path(directory)
    gwl_directory = root / GWL_ARM_DIRECTORY
    reference_directory = root / REFERENCE_ARM_DIRECTORY
    return [
        gwl_directory / PER_GCM_CURVES_FILE,
        gwl_directory / POOLED_MEDIAN_CURVES_FILE,
        reference_directory / PER_GCM_CURVES_FILE,
        reference_directory / POOLED_MEDIAN_CURVES_FILE,
        gwl_directory / RIME_DATASET_FILE,
        root / VALIDATION_FILE,
    ]


def write_pooled_wet_outputs(
    products: PooledWetOutputs, directory: str | Path, *, overwrite: bool = False
) -> list[Path]:
    """Write the pooled output files under ``directory``; return the paths.

    ``gwl/`` holds the curves of the warming-level calculation, their median and
    the RIME-X NetCDF; ``reference_pd/`` the curves of the present-day reference
    calculation and their median; ``validation.json`` the validation counts.

    Raises:
        FileExistsError: if any of the files exists and ``overwrite`` is false;
            nothing is written then.
    """

    paths = pooled_output_paths(directory)
    if not overwrite:
        refuse_existing_outputs(paths)
    gwl_curves, gwl_median, reference_curves, reference_median, dataset_path, validation_path = paths
    tables = {
        gwl_curves: products.gwl_per_gcm_curves,
        gwl_median: products.gwl_pooled_median_curves,
        reference_curves: products.reference_per_gcm_curves,
        reference_median: products.reference_pooled_median_curves,
    }
    with staged_outputs(paths) as staged:
        for path, table in tables.items():
            table.to_parquet(staged[path], index=False)
        products.rime_percentiles.to_netcdf(staged[dataset_path])
        staged[validation_path].write_text(json.dumps(products.validation, indent=2) + "\n")
    return [*tables, dataset_path, validation_path]


def finalize_wet_siting_outputs(
    gwl_root: str | Path,
    reference_root: str | Path,
    output_directory: str | Path,
    *,
    gcm_count: int,
    expected_rime_dimensions: Mapping[str, int] | None = DEFAULT_RIME_DIMENSIONS,
    overwrite: bool = False,
) -> PooledWetOutputs:
    """Pool the per-GCM outputs of both calculations and write the pooled output files.

    Each root holds one directory per GCM with the files that
    :func:`write_wet_siting_outputs` writes; the RIME-X samples of the
    warming-level root feed the dataset. Nothing is written unless every check in
    :func:`pool_wet_siting_outputs` passes.

    Parameters:
        gwl_root: Per-GCM output directories of the warming-level calculation.
        reference_root: Per-GCM output directories of the present-day reference
            calculation.
        output_directory: Where the pooled output files (see
            :func:`write_pooled_wet_outputs`) are written.
        gcm_count: Number of per-GCM output directories each calculation must
            hold.
        expected_rime_dimensions: Required sizes of the RIME-X dataset's
            dimensions: R12 regions, plant types, cooling types, seasons, warming
            levels and new-build ratios; ``None`` skips the check. The dimensions do
            not include the GCM, so ``gcm_count`` does not affect them, but a
            product that differs from the defaults in any of them (for example a
            subset of the regions) is refused unless this is adjusted to match,
            or ``None``.
        overwrite: Replace existing output files. By default the call refuses,
            before reading anything, when any of the files exists.

    Raises:
        FileExistsError: if any of the files exists and ``overwrite`` is false.
        ValueError: as :func:`read_gcm_outputs` and :func:`pool_wet_siting_outputs`.
    """

    if not overwrite:
        refuse_existing_outputs(pooled_output_paths(output_directory))
    gwl_curves = read_gcm_outputs(gwl_root, PER_GCM_CURVES_FILE, gcm_count=gcm_count)
    reference_curves = read_gcm_outputs(reference_root, PER_GCM_CURVES_FILE, gcm_count=gcm_count)
    rime_samples = read_gcm_outputs(gwl_root, RIME_SAMPLES_FILE, gcm_count=gcm_count)
    products = pool_wet_siting_outputs(
        gwl_curves,
        reference_curves,
        rime_samples,
        expected_rime_dimensions=expected_rime_dimensions,
    )
    write_pooled_wet_outputs(products, output_directory, overwrite=overwrite)
    return products


def _membership(curves: pd.DataFrame) -> pd.DataFrame:
    keys = ["gcm", "region_id", "archetype", "point_index"]
    nonzero = curves.loc[curves["point_index"] > 1]
    inconsistent = nonzero.groupby(keys, observed=True)["marginal_cell_id"].nunique()
    if (inconsistent != 1).any():
        raise ValueError("cell membership changes across season or warming level")
    return nonzero[[*keys, "marginal_cell_id"]].drop_duplicates(keys)
