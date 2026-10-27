"""Site evaluation for the thermoelectric cooling kernels: potential sites against existing sites.

A potential site is a grid cell where a new plant could be built. An existing site
is a grid cell where today's fleet has plants. The site evaluation runs the
cooling models at every potential and existing site for each sampled year and
writes the results as one long table. In the code this evaluation is called the
census (``run_site_evaluation``, ``build_site_cells``, ``write_site_evaluation``), and
the two sets of cells are the values ``candidate`` (potential sites) and
``incumbent`` (existing sites) of the ``population`` column.

The evaluation answers a single question: are the cells of today's thermoelectric
fleet a defensible basis for letting the energy model choose where to expand
capacity, or are they a selection-biased subset of the cells a policy could
actually build on? It evaluates the physical cooling kernels over an
independently listed set of potential sites and compares the per-R12-region
distribution of results with the distribution over the existing sites, holding GCM,
season, warming level and synthetic plant fixed. A synthetic plant is a made-up plant
of fixed type and size, placed on every cell (code name ``archetype``).

Design:

- **Dense in space.** Every potential site is evaluated, so the distribution of
  results over cells *is* the distribution over all potential sites. Cells are
  weighted by spherical cell area; there are no inclusion probabilities, analysis
  weights, or strata. The two kernels consider different cells (see ``cells``).
- **Sparse in time, at daily resolution.** The dry kernel's efficiency loss is a
  sum of exponentials in air temperature, so it is run on *daily* climate and
  aggregated to seasonal **after** the nonlinearity. This avoids the Jensen gap
  (the bias from applying a nonlinear function to averaged inputs) that taking
  monthly means first would introduce in the hot tail. Time is thinned by sampling
  whole years within global warming level (GWL) bins; sampling uncertainty is a
  bootstrap over the sampled years. The wet kernel stays monthly because river
  temperature (triver) is monthly on disk, like the other hydrology inputs.
- **One GCM per site evaluation.**

This module samples the years, runs the site evaluation and writes it
(:func:`run_site_evaluation`). The cells considered, the synthetic plants and the kernel
adapters live in ``impacts.thermoelectric.siting.cells``; ``summary`` reduces the
evaluation to per-region statistics.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from impacts.climatecore import build_warming_level_assignments
from impacts.climatecore.warming import assign_years_to_levels
from impacts.isimip import (
    DEFAULT_WATERGAP_SOC,
    ISIMIPPaths,
    ISIMIPVariable,
    isimip3a_counterclim_paths,
    isimip3b_paths,
    load_isimip_dataset,
    load_isimip_gmt,
)
from impacts.thermoelectric.siting._io import (
    refuse_existing_outputs,
    staged_outputs,
)
from impacts.thermoelectric.siting.cells import (
    CELLS_CONSIDERED_COLUMNS,
    RegionRasterPaths,
    SyntheticPlant,
    build_synthetic_sites,
    build_valid_cells,
    default_dry_synthetic_plants,
    default_wet_synthetic_plants,
    evaluate_dry_potential_sites,
    evaluate_wet_potential_sites_seasonal_and_monthly,
    extract_existing_site_cells,
)
from impacts.thermoelectric.siting.summary import (
    DEFAULT_SAMPLING_SEED,
    SUMMARY_INPUT_COLUMNS,
    compare_potential_to_existing_sites,
    summarize_site_evaluation,
)
from impacts.thermoelectric.wet import WET_COOLING_VARIABLES

TAS_DAILY = (ISIMIPVariable("tas", "climate", "daily"),)


# Seasonal responses (`wet`, `dry`) are aggregated over the season before export.
# `wet_monthly` responses are the raw kernel drivers, written at the native
# monthly resolution (a `month` key on the long schema) so that the split of the
# wet capacity loss into thermal and water-availability parts can be computed month
# by month and reproduce the mean-of-ratios water limit; it is empty by default.
DEFAULT_RESPONSES = {
    "wet": {"capacity_factor": "1", "water_limited_capacity_ceiling_mw": "MW"},
    "wet_monthly": {},
    "dry": {"capacity_factor": "1"},
}


CELL_METADATA = (
    "cell_id",
    "region_id",
    "country",
    "grid_lat",
    "grid_lon",
    "cell_area_m2",
    "population",
)


# Number of sites (synthetic plants at cells) per block for the daily dry
# evaluation, which builds an (entity, day) array before averaging to seasons.
DRY_SITE_CHUNK = 30_000


# Columns of the long result table. The kernel output carries many plant
# attributes (fuel, utype, mw, ...) that the site evaluation does not use; keeping
# them would inflate a table with a row per cell, plant, year and response, so the
# table keeps only the comparison keys, the cell-area weight, the response, and
# the source of each synthetic plant's parameters (`kernel_provenance`).
LONG_COLUMNS = (
    "cell_id",
    "region_id",
    "country",
    "grid_lat",
    "grid_lon",
    "cell_area_m2",
    "population",
    "archetype",
    "gcm",
    "scenario",
    "year",
    "season",
    "month",
    "warming_level",
    "response",
    "value",
    "unit",
    "kernel_provenance",
)


@dataclass(frozen=True)
class GWLSampling:
    """Whole-year sampling within warming-level bins."""

    bin_width: float = 0.5
    min_level: float = 1.0
    max_level: float = 4.0
    years_per_bin: int = 3
    seed: int = DEFAULT_SAMPLING_SEED
    target_levels: tuple[float, ...] | None = None
    # Per-(gcm, level) half-width overrides for bins with few years in the
    # target_levels path, e.g. [{"gcm": "<GCM name>", "level": 1.2, "half_width": 0.075}].
    # Scoped to one GCM and level so that widening a bin with few years does not
    # change which years are drawn for bins that already have enough years at the
    # shared width.
    level_tolerances: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class SampledYear:
    """A whole year drawn for a warming-level bin; all its days are kept for seasonal means."""

    scenario: str
    year: int
    warming_level: float | str


@dataclass(frozen=True)
class ReferenceSampling:
    """A fixed reference period, written on the ``warming_level`` axis.

    ``mode`` is ``REF-PD``, the present-day reference period (1995-2014, each
    GCM's historical run), or ``REF-PI``, the pre-industrial reference climate
    (ISIMIP3a ``counterclim``, 1901-2019: observed weather with the warming trend
    removed).

    ``soc`` is the WaterGAP water-use setting of ``REF-PI``. The default holds
    human water use at 2015 levels, which makes ``REF-PI`` a climate-only
    counterfactual rather than a reconstruction of 1901-2019 conditions; changing
    it changes what the ``REF-PI`` result estimates.
    """

    mode: Literal["REF-PD", "REF-PI"]
    soc: str = DEFAULT_WATERGAP_SOC

    def __post_init__(self) -> None:
        if self.mode not in REFERENCE_WINDOWS:
            raise ValueError(
                f"reference mode must be one of {', '.join(sorted(REFERENCE_WINDOWS))}; "
                f"got {self.mode!r}"
            )


# Scenario and last year of each reference period (REF-PD 1995-2014, REF-PI
# 1901-2019).
REFERENCE_WINDOWS = {
    "REF-PD": ("historical", 2014),
    "REF-PI": ("counterclim", 2019),
}


# First season-year of each reference period. REF-PI begins one season-year after
# the start of its source period because DJF needs the preceding December, and
# counterclim has no 1900 input.
REFERENCE_SAMPLE_STARTS = {"REF-PD": 1995, "REF-PI": 1902}


COUNTERCLIM_GCM = "GSWP3-W5E5"


@dataclass(frozen=True)
class SiteCells:
    """The wet and dry cell tables of a site evaluation, keyed by set of cells.

    ``wet_cells`` and ``dry_cells`` are the arguments of the same names of
    :func:`evaluate_sites`: each maps ``candidate`` (potential sites:
    the cells considered) and ``incumbent`` (existing sites: the fleet cells inside
    them) to a table tagged with
    ``cell_id``, ``region_id``, ``country``, ``grid_lat``, ``grid_lon``,
    ``cell_area_m2`` and ``population``.
    """

    wet_cells: dict[str, pd.DataFrame]
    dry_cells: dict[str, pd.DataFrame]


def build_site_cells(
    paths: ISIMIPPaths,
    gcm: str,
    scenarios: Sequence[str],
    region_paths: RegionRasterPaths,
    *,
    fleet_wet: str | Path,
    fleet_dry: str | Path,
    cells_considered_scenarios: str | Sequence[str] | None = None,
    dry_cache: str | Path | None = None,
    wet_cache: str | Path | None = None,
) -> SiteCells:
    """Build the potential and existing sites that :func:`run_site_evaluation` evaluates.

    The two kernels take different domains. The dry potential sites are every
    land cell in an R12 region, listed on the grid of the **first** of
    ``scenarios`` (``require_hydrology=False``). The wet potential sites are the
    cells with river temperature in **any** of ``cells_considered_scenarios``
    (``require_hydrology=True``), which defaults to all of ``scenarios``. The
    existing sites of each kernel are the cells of today's fleet that lie inside
    that kernel's potential sites, so potential and existing sites are compared on
    the same domain.

    Parameters:
        paths: ISIMIP files that define the grid and the hydrology mask.
        gcm: Climate model name.
        scenarios: Scenarios of the run; the first fixes the dry potential sites. A
            run on a fixed reference period passes its single reference scenario
            here (``historical`` for REF-PD) and lists the scenarios whose rivers
            the wet potential sites should cover in ``cells_considered_scenarios``.
        region_paths: GAUL raster, GAUL lookup and R12 definition.
        fleet_wet: Per-plant NetCDF of the wet fleet, which gives its cells.
        fleet_dry: Per-plant NetCDF of the dry fleet, which gives its cells.
        cells_considered_scenarios: Scenarios whose river temperature defines the wet
            potential sites; ``scenarios`` when ``None``.
        dry_cache: Optional parquet path that holds the dry potential sites of
            ``gcm``; written on first use and read afterwards, whatever grid or
            scenarios it was built from, so give each GCM its own path.
        wet_cache: The same for the wet potential sites.

    Raises:
        ValueError: if ``scenarios`` or ``cells_considered_scenarios`` is empty, if a cache
            path is not a non-empty string or path, holds a ``{`` or ``}``
            (a template left unfilled) or the cache lacks the cell columns, or if
            no fleet cell falls inside the potential sites.
    """

    scenarios = list(scenarios)
    if not scenarios:
        raise ValueError("scenarios must not be empty")
    if cells_considered_scenarios is None:
        wet_scenarios = scenarios
    elif isinstance(cells_considered_scenarios, str):
        wet_scenarios = [cells_considered_scenarios]
    else:
        wet_scenarios = list(cells_considered_scenarios)
        if not wet_scenarios or not all(isinstance(item, str) for item in wet_scenarios):
            raise ValueError(
                "cells_considered_scenarios must be a non-empty string or sequence of strings"
            )
    # Refuse a templated cache before either set of cells is built or written.
    _refuse_template(dry_cache)
    _refuse_template(wet_cache)
    dry_universe = _load_or_build_universe(
        paths, gcm, scenarios[0], region_paths, require_hydrology=False, cache_path=dry_cache
    )
    dry_universe["population"] = "candidate"
    wet_universe = _load_or_build_universe(
        paths, gcm, wet_scenarios, region_paths, require_hydrology=True, cache_path=wet_cache
    )
    wet_universe["population"] = "candidate"
    return SiteCells(
        wet_cells={
            "candidate": wet_universe,
            "incumbent": _fleet_cells(fleet_wet, fleet_dry, region_paths, wet_universe),
        },
        dry_cells={
            "candidate": dry_universe,
            "incumbent": _fleet_cells(fleet_wet, fleet_dry, region_paths, dry_universe),
        },
    )


def _fleet_cells(
    fleet_wet: str | Path,
    fleet_dry: str | Path,
    region_paths: RegionRasterPaths,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    fleet = extract_existing_site_cells(
        Path(fleet_wet), Path(fleet_dry), region_paths, restrict_to=universe
    )
    fleet["population"] = "incumbent"
    return fleet


def evaluate_sites(
    *,
    paths: ISIMIPPaths,
    gcm: str,
    sampled_years: Sequence[SampledYear],
    wet_cells: Mapping[str, pd.DataFrame],
    dry_cells: Mapping[str, pd.DataFrame] | None = None,
    responses: Mapping[str, Mapping[str, str]] = DEFAULT_RESPONSES,
    wet_synthetic_plants: Iterable[SyntheticPlant] | None = None,
    dry_synthetic_plants: Iterable[SyntheticPlant] | None = None,
) -> Iterator[pd.DataFrame]:
    """Evaluate the cooling kernels on cells and sampled years; yield long rows.

    The cells of each set (potential sites, existing sites) get every synthetic
    plant and are evaluated against one climate load per sampled year. The rows of
    one (year, set of cells, kernel) block are yielded as one frame in the
    ``LONG_COLUMNS`` schema, in the order they are written: for each year the dry
    blocks (``dry_cells`` given), then the wet seasonal block and, when
    ``responses["wet_monthly"]`` is non-empty, the wet monthly block of each set of
    cells. :func:`write_site_evaluation` writes them incrementally.

    This is the site evaluation only: the long table of per-cell results. It
    carries no summary and no comparison; :func:`summarize_site_evaluation` reduces the
    written rows to per-region statistics and :func:`compare_potential_to_existing_sites` turns
    that summary into a comparison of potential sites with existing sites;
    :func:`run_site_evaluation` writes the evaluation and, when asked, both of those. The
    cells come from :func:`build_site_cells`.

    Parameters:
        sampled_years: Whole years to evaluate, from :func:`sample_years_within_gwl`
            or :func:`sample_years_within_reference`.
        wet_cells: Name of a set of cells (``candidate`` or ``incumbent``) -> its
            cells, with ``cell_id``, ``region_id``, ``country``, ``grid_lat``,
            ``grid_lon`` and ``cell_area_m2``, as :func:`build_valid_cells`
            and :func:`extract_existing_site_cells` return them. The name becomes the
            ``population`` column; blocks follow the mapping's order.
        dry_cells: The same for the dry kernel; ``None`` skips it.
        responses: Kernel -> {response: unit}: ``wet`` (seasonal), ``wet_monthly``
            (raw monthly drivers; may be empty or absent) and ``dry``.
        wet_synthetic_plants: Synthetic wet plants; the default set when ``None``.
        dry_synthetic_plants: Synthetic dry plants; the default set when ``None``.

    Raises:
        ValueError: if ``sampled_years`` is empty, if a kernel's cells or synthetic plants
            are empty, or (while iterating) if a configured response is missing
            from the kernel output.
    """

    if not sampled_years:
        raise ValueError("climate sampling selected no years")
    wet_set = tuple(default_wet_synthetic_plants() if wet_synthetic_plants is None else wet_synthetic_plants)
    wet_entities = {
        population: build_synthetic_sites(cells.assign(population=population), wet_set)
        for population, cells in wet_cells.items()
    }
    dry_entities = {}
    if dry_cells is not None:
        dry_set = tuple(default_dry_synthetic_plants() if dry_synthetic_plants is None else dry_synthetic_plants)
        dry_entities = {
            population: build_synthetic_sites(
                cells.assign(population=population), dry_set
            )
            for population, cells in dry_cells.items()
        }
    return _census_windows(
        paths, gcm, sampled_years, wet_entities, dry_entities, responses
    )


def write_site_evaluation(
    windows: Iterable[pd.DataFrame], output: str | Path, *, overwrite: bool = False
) -> Path:
    """Write the site evaluation's windows (blocks of rows) to one parquet file as they arrive; return its path.

    A site evaluation over every potential site and many sampled years is too large
    to hold in one frame, so the windows are appended to a temporary file in the
    directory of ``output`` and the file takes its name only once every window is
    written. An evaluation that fails midway therefore leaves neither a partial
    file under the final name nor the temporary one, and an earlier evaluation is
    never replaced by part of a later one.

    Raises:
        FileExistsError: if ``output`` exists and ``overwrite`` is false; checked
            before the first window is drawn from ``windows``.
        ValueError: if ``windows`` yields nothing. An existing ``output`` is left
            as it was.
    """

    output = Path(output)
    if not overwrite:
        refuse_existing_outputs([output])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    writer: pq.ParquetWriter | None = None
    try:
        for frame in windows:
            writer = _append_long(writer, temporary, frame)
        if writer is None:
            raise ValueError(f"no blocks of site-evaluation rows to write to {output}")
        writer.close()
        writer = None
        temporary.replace(output)
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return output


@dataclass(frozen=True)
class SiteEvaluationSummary:
    """Settings of the per-region summary and the comparison of potential with existing sites that :func:`run_site_evaluation` writes beside a site evaluation.

    Attributes:
        low_value_thresholds: Response -> threshold at or below which a cell counts in
            the bad tail (the share of cell area whose result is that low, metric
            ``bad_tail_area_fraction``); responses without one get no bad-tail
            fraction.
        bootstrap_replicates: Replicates of the cluster bootstrap over sampled
            years.
        seed: Seed of that bootstrap.
    """

    low_value_thresholds: Mapping[str, float] = field(default_factory=dict)
    bootstrap_replicates: int = 200
    seed: int = DEFAULT_SAMPLING_SEED


# The summary written with a site evaluation unless told otherwise.
DEFAULT_SITE_EVALUATION_SUMMARY = SiteEvaluationSummary()


def run_site_evaluation(
    cells: SiteCells,
    sampled_years: Sequence[SampledYear],
    *,
    paths: ISIMIPPaths,
    gcm: str,
    output: str | Path,
    responses: Mapping[str, Mapping[str, str]] = DEFAULT_RESPONSES,
    wet_synthetic_plants: Iterable[SyntheticPlant] | None = None,
    dry_synthetic_plants: Iterable[SyntheticPlant] | None = None,
    summary: SiteEvaluationSummary | None = DEFAULT_SITE_EVALUATION_SUMMARY,
    overwrite: bool = False,
) -> list[Path]:
    """Evaluate potential and existing sites for one GCM and write the site evaluation; return the written paths.

    The site evaluation is :func:`evaluate_sites` over both kernels'
    cells, written window by window to ``output`` as :func:`write_site_evaluation`
    does, so it never has to fit in memory. Unless ``summary`` is ``None``, the
    per-region summary (:func:`summarize_site_evaluation`, read back from the written
    evaluation) and the comparison of potential with existing sites
    (:func:`compare_potential_to_existing_sites`) are written beside it as
    ``<stem>_summary<suffix>`` and ``<stem>_verdict<suffix>``, and either all the
    files appear or none does. The summary groups on season: in an evaluation
    that includes monthly drivers (non-empty ``responses["wet_monthly"]``) it
    summarises each driver response over the months of a season pooled, and the
    bootstrap draws of the seasonal responses differ from those of an evaluation
    without the drivers, since every group draws in turn from one random
    generator.

    Parameters:
        cells: Wet and dry potential and existing sites, from :func:`build_site_cells`.
        sampled_years: Whole years to evaluate, from :func:`sample_years_within_gwl`
            or :func:`sample_years_within_reference`.
        paths: ISIMIP files to load the climate from: ``isimip3b_paths`` for a
            warming-level or REF-PD run, :func:`reference_climate_paths` for
            either reference period.
        output: Parquet path of the site evaluation.
        responses, wet_synthetic_plants, dry_synthetic_plants: As for
            :func:`evaluate_sites`.
        summary: Summary settings, by default :class:`SiteEvaluationSummary` defaults
            (no bad-tail thresholds, 200 bootstrap replicates, ``DEFAULT_SAMPLING_SEED``), or
            ``None`` to write the site evaluation only.
        overwrite: Replace existing files; by default the call refuses before
            evaluating anything when any of them exists.

    Returns:
        The path of the site evaluation, followed by the paths of the summary and of the comparison file when written.

    Raises:
        FileExistsError: if a file to write exists and ``overwrite`` is false.
        ValueError: as :func:`evaluate_sites`, or if ``summary`` is set
            and the cells lack the ``candidate`` or the ``incumbent`` set.
    """

    output = Path(output)
    written = [output]
    if summary is not None:
        written += [
            output.with_name(f"{output.stem}_summary{output.suffix}"),
            output.with_name(f"{output.stem}_verdict{output.suffix}"),
        ]
    if not overwrite:
        refuse_existing_outputs(written)
    windows = evaluate_sites(
        paths=paths,
        gcm=gcm,
        sampled_years=sampled_years,
        wet_cells=cells.wet_cells,
        dry_cells=cells.dry_cells,
        responses=responses,
        wet_synthetic_plants=wet_synthetic_plants,
        dry_synthetic_plants=dry_synthetic_plants,
    )
    if summary is not None:
        # The comparison needs both sets of cells; refuse a site evaluation of one
        # set before any year is evaluated rather than after.
        populations = {
            population
            for frames in (cells.wet_cells, cells.dry_cells)
            for population, frame in frames.items()
            if len(frame)
        }
        absent = [name for name in ("candidate", "incumbent") if name not in populations]
        if absent:
            raise ValueError(
                f"the cells have no {' or '.join(absent)} population, so potential sites "
                "cannot be compared with existing sites; pass summary=None to write the "
                "site evaluation only"
            )
    with staged_outputs(written) as staged:
        write_site_evaluation(windows, staged[output], overwrite=True)
        if summary is not None:
            summary_path, verdict_path = written[1:]
            summarized = summarize_site_evaluation(
                pd.read_parquet(staged[output], columns=list(SUMMARY_INPUT_COLUMNS)),
                low_value_thresholds=dict(summary.low_value_thresholds),
                bootstrap_replicates=summary.bootstrap_replicates,
                seed=summary.seed,
            )
            summarized.to_parquet(staged[summary_path], index=False)
            compare_potential_to_existing_sites(summarized).to_parquet(staged[verdict_path], index=False)
    return written


def reference_climate_paths(
    reference: ReferenceSampling,
    gcm: str,
    watxene_root: str | Path | None = None,
) -> ISIMIPPaths:
    """Return the ISIMIP files that a site evaluation on a fixed reference period loads its climate from.

    REF-PD (the present-day reference period) reads the ISIMIP3b historical run of
    ``gcm``. REF-PI (the pre-industrial reference climate) reads the single
    ISIMIP3a ``counterclim`` member, which only ``GSWP3-W5E5`` provides, with the
    WaterGAP water-use setting ``reference.soc``.

    Raises:
        ValueError: if ``reference`` is REF-PI and ``gcm`` is not ``GSWP3-W5E5`` or
            ``reference.soc`` is empty.
    """

    root = None if watxene_root is None else Path(watxene_root)
    if reference.mode == "REF-PI":
        if gcm != COUNTERCLIM_GCM:
            raise ValueError(f"REF-PI requires gcm={COUNTERCLIM_GCM!r}; got {gcm!r}")
        if not isinstance(reference.soc, str) or not reference.soc:
            raise ValueError("REF-PI water-use forcing (soc) must be a non-empty string")
        return isimip3a_counterclim_paths(root, soc=reference.soc)
    return isimip3b_paths(root)


def _census_windows(
    paths: ISIMIPPaths,
    gcm: str,
    sampled_years: Sequence[SampledYear],
    wet_entities: Mapping[str, pd.DataFrame],
    dry_entities: Mapping[str, pd.DataFrame],
    responses: Mapping[str, Mapping[str, str]],
) -> Iterator[pd.DataFrame]:
    for sample in sampled_years:
        if dry_entities:
            dry_climate = _load_dry_climate(sample, paths, gcm)
            for sites in dry_entities.values():
                evaluated = _evaluate_chunked(
                    evaluate_dry_potential_sites, sites, dry_climate, DRY_SITE_CHUNK
                )
                yield _to_long(evaluated, sites, sample, gcm, dict(responses["dry"]))
        wet_climate = _load_wet_climate(sample, paths, gcm)
        wet_monthly_responses = dict(responses.get("wet_monthly", {}))
        for sites in wet_entities.values():
            seasonal, monthly = evaluate_wet_potential_sites_seasonal_and_monthly(
                sites, wet_climate
            )
            yield _to_long(seasonal, sites, sample, gcm, dict(responses["wet"]))
            if wet_monthly_responses:
                yield _to_long(
                    monthly,
                    sites,
                    sample,
                    gcm,
                    wet_monthly_responses,
                    monthly=True,
                )


def _refuse_template(cache_path: Any) -> None:
    if cache_path is not None and ("{" in str(cache_path) or "}" in str(cache_path)):
        raise ValueError(
            f"cache path {str(cache_path)!r} looks like a template; pass one "
            "concrete path per GCM (the path is not formatted with the GCM name)"
        )


def _load_or_build_universe(
    paths: ISIMIPPaths,
    gcm: str,
    scenarios: str | list[str],
    region_paths: RegionRasterPaths,
    *,
    require_hydrology: bool,
    cache_path: Any = None,
) -> pd.DataFrame:
    if cache_path is None:
        return build_valid_cells(
            paths, gcm, scenarios, region_paths, require_hydrology=require_hydrology
        )
    if not isinstance(cache_path, Path) and (not isinstance(cache_path, str) or not cache_path):
        raise ValueError("cache path of the cells must be a non-empty string")
    path = Path(cache_path)
    _refuse_template(path)
    if path.exists():
        cached = pd.read_parquet(path)
        missing = set(CELLS_CONSIDERED_COLUMNS) - set(cached.columns)
        if missing:
            raise ValueError(
                f"cell cache {path} is missing columns: {sorted(missing)}"
            )
        return cached[list(CELLS_CONSIDERED_COLUMNS)].copy()

    universe = build_valid_cells(
        paths, gcm, scenarios, region_paths, require_hydrology=require_hydrology
    )
    universe = universe[list(CELLS_CONSIDERED_COLUMNS)].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    universe.to_parquet(tmp, index=False)
    tmp.replace(path)
    return universe


def sample_years_within_reference(reference: ReferenceSampling) -> list[SampledYear]:
    """Return every whole year of a fixed reference period."""

    scenario, end_year = REFERENCE_WINDOWS[reference.mode]
    sample_start = REFERENCE_SAMPLE_STARTS[reference.mode]
    return [
        SampledYear(scenario=scenario, year=year, warming_level=reference.mode)
        for year in range(sample_start, end_year + 1)
    ]


def sample_years_within_gwl(
    gmt_root: str | Path,
    gcm: str,
    scenarios: list[str],
    sampling: GWLSampling,
) -> list[SampledYear]:
    """Draw whole years per warming-level bin, pooled across the GCM's scenarios.

    Each (scenario, year) is assigned to its nearest target warming level; within
    a level bin a fixed number of (scenario, year) units are drawn without
    replacement. A year farther than half a bin width from its level (warmer than
    the top bin, or cooler than the bottom one) is not sampled. The earliest year
    allowed leaves a preceding December for a complete DJF: 2016 for ssp
    scenarios (ISIMIP3b future files start at 2015), 1851 for ``historical``
    (ISIMIP3b historical files run back to 1850).
    """

    gmt_by_path = {
        (gcm, scenario): load_isimip_gmt(gmt_root, gcm, scenario)
        for scenario in scenarios
    }
    if sampling.target_levels is None:
        assignments = build_warming_level_assignments(
            gmt_by_path,
            path_columns=("gcm", "scenario"),
            bin_width=sampling.bin_width,
            min_warming_level=sampling.min_level,
            max_warming_level=sampling.max_level,
        )
        table = (
            assignments[["warming_level", "absolute_error"]].to_dataframe().reset_index()
        )
        # Years beyond the level range are assigned to its end bins however far
        # away they are; keep only those within half a bin of the level they were
        # given.
        table = table[table["absolute_error"] <= sampling.bin_width / 2.0 + 1e-9]
        min_level = sampling.min_level
        max_level = sampling.max_level
    else:
        target_levels = tuple(float(level) for level in sampling.target_levels)
        if not target_levels:
            raise ValueError("target_levels must not be empty")
        rows = []
        for (_path_gcm, scenario), gmt in gmt_by_path.items():
            assigned = assign_years_to_levels(gmt, target_levels)
            frame = (
                assigned[["warming_level", "absolute_error", "actual_warming"]]
                .to_dataframe()
                .reset_index()
            )
            frame["gcm"] = gcm
            frame["scenario"] = scenario
            rows.append(frame)
        table = pd.concat(rows, ignore_index=True)
        ordered = np.sort(np.asarray(target_levels, dtype=float))
        half_width = (
            float(np.diff(ordered).min()) / 2.0
            if ordered.size > 1
            else sampling.bin_width / 2.0
        )
        # Nearest-level assignment hard-partitions years, so a point's
        # absolute_error is already bounded by half_width for any interior
        # target level -- raising the shared half_width cannot rescue an interior
        # bin with few years. Instead, for the explicitly listed (gcm, level)
        # overrides, borrow extra years directly by distance to that level
        # from the FULL unfiltered pool (independent of which level they
        # nearest-assign to, and independent of the default cut) and add them
        # under that level only. Rows for every other level, and every other
        # (gcm, level) pair, are untouched, so no other bin's pool of years
        # changes.
        full_table = table
        table = full_table[full_table["absolute_error"] <= half_width + 1e-9]
        borrowed = []
        for override in sampling.level_tolerances:
            if override.get("gcm") != gcm:
                continue
            level = float(override["level"])
            tolerance = float(override["half_width"])
            distance = (full_table["actual_warming"] - level).abs()
            candidates = full_table[distance <= tolerance + 1e-9].copy()
            candidates["warming_level"] = level
            candidates["absolute_error"] = (candidates["actual_warming"] - level).abs()
            borrowed.append(candidates)
        if borrowed:
            table = pd.concat([table, *borrowed], ignore_index=True)
        min_level = min(target_levels)
        max_level = max(target_levels)
    table = table.dropna(subset=["warming_level"])
    ssp_year_floor = 2016
    historical_year_floor = 1851
    year_floor = np.where(
        table["scenario"] == "historical", historical_year_floor, ssp_year_floor
    )
    table = table[(table["year"] >= year_floor) & (table["scenario"].isin(scenarios))]
    table = table[
        (table["warming_level"] >= min_level - 1e-9)
        & (table["warming_level"] <= max_level + 1e-9)
    ]

    rng = np.random.default_rng(sampling.seed)
    sampled: list[SampledYear] = []
    for level, group in table.groupby("warming_level", sort=True):
        units = group[["scenario", "year"]].drop_duplicates().to_numpy()
        take = min(sampling.years_per_bin, len(units))
        chosen = rng.choice(len(units), size=take, replace=False)
        for index in sorted(chosen):
            scenario, year = units[index]
            sampled.append(SampledYear(str(scenario), int(year), float(level)))
    return sampled


def _load_dry_climate(sample: SampledYear, paths: ISIMIPPaths, gcm: str) -> xr.Dataset:
    # Daily tas over (Y-1, Y) so the season-year Y carries a complete DJF; the
    # kernel runs per-day and aggregates to seasonal after the nonlinearity.
    return load_isimip_dataset(
        TAS_DAILY,
        gcm,
        sample.scenario,
        paths,
        year_range=(sample.year - 1, sample.year),
        target_frequency="native",
        chunks={"time": 120},
    )


def _load_wet_climate(sample: SampledYear, paths: ISIMIPPaths, gcm: str) -> xr.Dataset:
    # Monthly: river temperature is monthly on disk, so a native daily merge of
    # the mixed-frequency wet inputs fails. The Jensen gap (see the module
    # docstring) then applies only to the wet kernel's air-side nonlinear terms.
    return load_isimip_dataset(
        WET_COOLING_VARIABLES,
        gcm,
        sample.scenario,
        paths,
        year_range=(sample.year - 1, sample.year),
        target_frequency="monthly",
        chunks={"time": 120},
    )


def _evaluate_chunked(
    evaluate: Any,
    sites: pd.DataFrame,
    climate: xr.Dataset,
    chunk: int,
) -> xr.Dataset:
    """Evaluate a kernel over blocks of sites and concatenate along ``entity``.

    The daily dry path builds an (entity, day) array, so all potential sites are
    processed in blocks to bound peak memory.
    """

    if len(sites) <= chunk:
        return evaluate(sites, climate)
    parts = [
        evaluate(sites.iloc[start : start + chunk], climate)
        for start in range(0, len(sites), chunk)
    ]
    return xr.concat(parts, dim="entity")


def _append_long(
    writer: pq.ParquetWriter | None,
    output: Path,
    frame: pd.DataFrame,
) -> pq.ParquetWriter:
    """Append one window's long rows to the parquet, opening the writer once."""

    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(output, table.schema)
    writer.write_table(table)
    return writer


def _to_long(
    evaluated: xr.Dataset,
    sites: pd.DataFrame,
    sample: SampledYear,
    gcm: str,
    response_units: dict[str, str],
    *,
    monthly: bool = False,
) -> pd.DataFrame:
    missing = set(response_units).difference(evaluated.data_vars)
    if missing:
        raise ValueError(
            f"configured responses absent from kernel output: {sorted(missing)}"
        )
    if monthly:
        # Monthly drivers carry season/season-year/month as coords along the time
        # dimension; keep the months whose season-year is the sampled year (this
        # pulls in the prior December for a complete DJF, matching the seasonal
        # water limit's year labelling).
        selected = evaluated[list(response_units)]
        selected = selected.isel(time=(selected["year"] == sample.year).values)
    else:
        selected = evaluated[list(response_units)].sel(year=sample.year)
    frame = selected.to_dataframe().reset_index()
    frame["entity"] = frame["entity"].astype(str)

    metadata = sites[
        ["UNITID", *CELL_METADATA, "archetype", "archetype_provenance"]
    ].copy()
    metadata["entity"] = metadata["UNITID"].astype(str)
    duplicated = (
        set(metadata.columns).intersection(frame.columns).difference({"entity"})
    )
    frame = frame.drop(columns=sorted(duplicated))
    frame = frame.merge(
        metadata.drop(columns="UNITID"), on="entity", validate="many_to_one"
    )

    frame = frame.melt(
        id_vars=[column for column in frame.columns if column not in response_units],
        value_vars=list(response_units),
        var_name="response",
        value_name="value",
    )
    frame["unit"] = frame["response"].map(response_units)
    frame["gcm"] = gcm
    frame["scenario"] = sample.scenario
    if not monthly:
        # Seasonal rows carry the season-year directly; monthly rows already carry
        # it from the coord (== sample.year for every kept month) and have no
        # single month, so `month` is null on seasonal rows.
        frame["year"] = sample.year
    if "month" not in frame.columns:
        frame["month"] = pd.NA
    frame["month"] = frame["month"].astype("Int64")
    frame["warming_level"] = sample.warming_level
    frame["kernel_provenance"] = frame["archetype_provenance"]
    return frame[list(LONG_COLUMNS)]
