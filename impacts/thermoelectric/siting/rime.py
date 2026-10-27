"""Percentiles of wet thermoelectric results by warming-level bin, for RIME-X.

RIME-X is IIASA's emulator of regional climate impacts by warming level. For each
climate sample (GCM, scenario and year), region, plant type, cooling type, season
and warming level, a result pairs the totals of the existing sites with the
capacity curve of the potential sites:

* the existing-site capacity ``K`` (nameplate, MW) and the existing-site usable
  fraction (their usable capacity divided by ``K``);
* for each new-build ratio ``r`` (``depth_ratio``, new capacity as a multiple of
  ``K``; 0.5, 1 and 2 by default), the new capacity ``D = r K`` and the
  potential-site usable fraction ``U(D) / D``, where ``U`` is the usable capacity
  on the sample's curve, interpolated linearly;
* the usable capacity ratio, ``existing fraction + r * potential-site fraction``:
  the usable capacity of existing and new capacity together as a fraction of
  ``K``.

Percentiles over the samples of each warming-level bin are written as a NetCDF
dataset.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from impacts.thermoelectric.siting._io import staged_outputs
from impacts.thermoelectric.siting.cells import SyntheticPlant
from impacts.thermoelectric.siting.curves import (
    _load_decomposition_samples,
    _load_reach_water_ceilings,
    _load_topology,
    _with_archetype_parts,
    build_potential_site_sample_curves_with_river_network,
)

SAMPLE_KEYS = ("gcm", "scenario", "year")
RESPONSE_KEYS = (
    "region_id",
    "plant_type",
    "cooling_type",
    "season",
    "warming_level",
)
PAIR_KEYS = (*SAMPLE_KEYS, *RESPONSE_KEYS)
WARMING_LEVEL_SUMMARY_KEYS = (*RESPONSE_KEYS, "depth_ratio")

DEFAULT_NEW_BUILD_RATIOS = (0.5, 1.0, 2.0)
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)

RESPONSE_COLUMNS = (
    "incumbent_capacity_mw",
    "incumbent_usable_fraction",
    "candidate_usable_fraction",
    "usable_capacity_ratio",
)
SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")


def _positive_depth_ratios(new_build_ratios: Iterable[float]) -> tuple[float, ...]:
    ratios = tuple(float(value) for value in new_build_ratios)
    if not ratios or any(value <= 0.0 for value in ratios):
        raise ValueError("new-build ratios (depth ratios) must be positive")
    return ratios


def build_existing_site_totals(
    samples: pd.DataFrame,
    *,
    wet_synthetic_plants: Sequence[SyntheticPlant] | None = None,
) -> pd.DataFrame:
    """Total the existing sites separately for every climate sample.

    Returns, for each climate sample, region, plant type, cooling type, season and
    warming level, the existing-site nameplate capacity
    (``incumbent_capacity_mw``) and usable fraction (usable capacity divided by
    nameplate).

    ``wet_synthetic_plants`` gives each synthetic plant name its plant type and cooling
    type; ``None`` uses the default wet synthetic plants.
    """

    if samples.empty:
        raise ValueError("cannot build existing-site totals from no samples")
    data = _with_archetype_parts(samples, wet_synthetic_plants)
    endpoints = (
        data.groupby(list(PAIR_KEYS), observed=True, sort=True)
        .agg(
            incumbent_capacity_mw=("nameplate_mw", "sum"),
            incumbent_usable_capacity_mw=("usable_capacity_mw", "sum"),
        )
        .reset_index()
    )
    endpoints["incumbent_usable_fraction"] = (
        endpoints["incumbent_usable_capacity_mw"]
        / endpoints["incumbent_capacity_mw"]
    )
    return endpoints.drop(columns="incumbent_usable_capacity_mw")


def build_wet_rime_samples(
    existing_site_samples: pd.DataFrame,
    potential_site_samples: pd.DataFrame,
    water_limits: pd.DataFrame,
    river_network: pd.DataFrame,
    *,
    fixed_ordering: pd.DataFrame,
    new_build_ratios: Iterable[float] = DEFAULT_NEW_BUILD_RATIOS,
    capacity_grid_points: int = 61,
    wet_synthetic_plants: Sequence[SyntheticPlant] | None = None,
) -> pd.DataFrame:
    """Pair, for each climate sample, the wet-cooling results of existing and potential sites.

    The existing-site samples also set the water of the potential sites: their
    withdrawal after thermal loss is subtracted from the potential-site water
    limits wherever an existing site is also a potential site, so the two terms
    do not both use the same river water. Potential sites are filled in the order
    given by ``fixed_ordering``, as :func:`build_potential_site_curves_with_river_network`
    describes.
    ``wet_synthetic_plants`` gives each synthetic plant name its plant type and cooling
    type; ``None`` uses the default wet synthetic plants.
    """

    if "year" not in water_limits:
        raise ValueError("RIME samples require year-resolved water limits")
    ratios = _positive_depth_ratios(new_build_ratios)

    potential = build_potential_site_sample_curves_with_river_network(
        potential_site_samples,
        water_limits,
        river_network,
        capacity_grid_points=capacity_grid_points,
        fixed_ordering=fixed_ordering,
        existing_site_samples=existing_site_samples,
        wet_synthetic_plants=wet_synthetic_plants,
    )
    return build_wet_rime_samples_from_curves(
        existing_site_samples, potential, new_build_ratios=ratios, wet_synthetic_plants=wet_synthetic_plants
    )


def build_wet_rime_samples_from_curves(
    existing_site_samples: pd.DataFrame,
    potential_site_curves: pd.DataFrame,
    *,
    new_build_ratios: Iterable[float] = DEFAULT_NEW_BUILD_RATIOS,
    wet_synthetic_plants: Sequence[SyntheticPlant] | None = None,
) -> pd.DataFrame:
    """Read each sample's potential-site curve at the requested new capacities and pair it with the existing-site totals.

    The new capacity for a new-build ratio is the ratio times the existing-site
    capacity; the curve's usable capacity there is interpolated linearly. Raises
    ``ValueError`` if a curve ends below a requested new capacity, or if a
    climate sample of the existing sites has no curve.

    ``wet_synthetic_plants`` gives each synthetic plant name its plant type and cooling
    type; ``None`` uses the default wet synthetic plants.
    """

    ratios = _positive_depth_ratios(new_build_ratios)
    existing = build_existing_site_totals(
        existing_site_samples, wet_synthetic_plants=wet_synthetic_plants
    )
    candidate_groups = {
        keys: group.sort_values("capacity_axis_mw", kind="stable")
        for keys, group in potential_site_curves.groupby(
            list(PAIR_KEYS), observed=True, sort=True
        )
    }

    rows: list[dict] = []
    missing: list[tuple] = []
    for endpoint in existing.itertuples(index=False):
        pair = tuple(getattr(endpoint, key) for key in PAIR_KEYS)
        curve = candidate_groups.get(pair)
        if curve is None:
            missing.append(pair)
            continue
        axis = curve["capacity_axis_mw"].to_numpy(dtype=float)
        usable = curve["usable_capacity_mw"].to_numpy(dtype=float)
        for ratio in ratios:
            depth_mw = float(endpoint.incumbent_capacity_mw) * ratio
            if depth_mw > axis[-1] + 1e-9:
                raise ValueError(
                    f"potential-site curve {pair} ends at {axis[-1]:g} MW, below "
                    f"the requested {depth_mw:g} MW (new-build ratio {ratio:g})"
                )
            candidate_usable_mw = float(np.interp(depth_mw, axis, usable))
            candidate_fraction = candidate_usable_mw / depth_mw
            incumbent_fraction = float(endpoint.incumbent_usable_fraction)
            rows.append(
                {
                    **dict(zip(PAIR_KEYS, pair, strict=True)),
                    "depth_ratio": ratio,
                    "incumbent_capacity_mw": float(
                        endpoint.incumbent_capacity_mw
                    ),
                    "incumbent_usable_fraction": incumbent_fraction,
                    "candidate_usable_fraction": candidate_fraction,
                    "usable_capacity_ratio": (
                        incumbent_fraction + ratio * candidate_fraction
                    ),
                }
            )
    if missing:
        raise ValueError(
            "potential-site curves are missing for existing-site climate samples: "
            f"{missing[:5]} ({len(missing)} total)"
        )
    return pd.DataFrame(rows)[[*PAIR_KEYS, "depth_ratio", *RESPONSE_COLUMNS]]


def build_warming_level_bin_percentiles(
    samples: pd.DataFrame,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Calculate quantiles of each result over the samples in each warming-level bin.

    The quantiles are taken for each region, plant type, cooling type, season,
    warming level and new-build ratio, over the GCMs, scenarios and years. The
    table also gives the number of samples and of GCMs behind each quantile.
    """

    required = set(PAIR_KEYS) | {"depth_ratio", *RESPONSE_COLUMNS}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"RIME samples missing columns: {sorted(missing)}")
    values = tuple(float(value) for value in quantiles)
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("quantiles must lie in [0, 1]")

    grouped = samples.groupby(list(WARMING_LEVEL_SUMMARY_KEYS), observed=True, sort=True)
    coverage = (
        grouped.agg(sample_count=("year", "size"), gcm_count=("gcm", "nunique"))
        .reset_index()
    )
    parts = []
    for quantile in values:
        part = grouped[list(RESPONSE_COLUMNS)].quantile(quantile).reset_index()
        part["statistic"] = f"p{round(100 * quantile):02d}"
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    return result.merge(coverage, on=list(WARMING_LEVEL_SUMMARY_KEYS), validate="many_to_one")


def percentiles_to_dataset(percentiles: pd.DataFrame) -> xr.Dataset:
    """Convert a long percentile table to the NetCDF dataset that RIME-X reads.

    The dimensions are ``region_id``, ``plant_type``, ``cooling_type``,
    ``season``, ``gwl`` and ``depth_ratio``. There is one variable per result and
    percentile, plus the sample and GCM counts.
    """

    index = list(WARMING_LEVEL_SUMMARY_KEYS)
    data_vars = {}
    for statistic, group in percentiles.groupby("statistic", sort=True):
        indexed = group.set_index(index)
        for column in RESPONSE_COLUMNS:
            data_vars[f"{column}_{statistic}"] = indexed[column].to_xarray()
    coverage = percentiles.drop_duplicates(index).set_index(index)
    for column in ("sample_count", "gcm_count"):
        data_vars[column] = coverage[column].to_xarray()

    dataset = xr.Dataset(
        data_vars,
        attrs={
            "title": "R12 wet thermoelectric usable capacity by warming-level bin",
            "description": (
                "Wet-cooling results of existing sites and of potential sites, "
                "with water shared along the river network. "
                "Percentiles are calculated across climate samples assigned "
                "to each global warming level bin after the potential-site "
                "capacity curves are interpolated to common new-build ratios "
                "(new capacity as a multiple of existing-site capacity)."
            ),
            "units": "fraction of existing-site nameplate capacity",
            "percentile_population": "GCM, scenario, and sampled year within GWL",
        },
    ).rename({"warming_level": "gwl"})
    dataset = dataset.sortby(
        ["region_id", "plant_type", "cooling_type", "gwl", "depth_ratio"]
    )
    seasons = list(map(str, dataset.coords["season"].values))
    ordered_seasons = [season for season in SEASON_ORDER if season in seasons]
    ordered_seasons.extend(sorted(set(seasons).difference(SEASON_ORDER)))
    return dataset.reindex(season=ordered_seasons)


def export_wet_rime_samples(
    loss_split_paths: Sequence[str | Path],
    water_limit_paths: Sequence[str | Path],
    river_network_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    new_build_ratios: Iterable[float] = DEFAULT_NEW_BUILD_RATIOS,
    capacity_grid_points: int = 61,
    fixed_ordering_path: str | Path | None = None,
    wet_synthetic_plants: Sequence[SyntheticPlant] | None = None,
) -> Path:
    """Write the paired existing-site and potential-site results of every climate sample, for one or more GCMs.

    ``fixed_ordering_path`` is required: the parquet table that gives the cell
    order of the potential sites. ``wet_synthetic_plants`` gives each synthetic plant
    name its plant type and cooling type; ``None`` uses the default wet synthetic
    plants.
    """

    existing = _load_decomposition_samples(
        list(loss_split_paths), population="incumbent"
    )
    potential = _load_decomposition_samples(
        list(loss_split_paths), population="candidate"
    )
    ceilings = _load_reach_water_ceilings(
        list(water_limit_paths), population="candidate", preserve_year=True
    )
    river_network = _load_topology(list(river_network_paths), population="candidate")
    if fixed_ordering_path is None:
        raise ValueError("wet RIME export requires a fixed cell ordering of the potential sites")
    fixed_ordering = pd.read_parquet(fixed_ordering_path)
    result = build_wet_rime_samples(
        existing,
        potential,
        ceilings,
        river_network,
        new_build_ratios=new_build_ratios,
        capacity_grid_points=capacity_grid_points,
        fixed_ordering=fixed_ordering,
        wet_synthetic_plants=wet_synthetic_plants,
    )
    path = Path(output_path)
    with staged_outputs([path]) as staged:
        result.to_parquet(staged[path], index=False)
    return path


def export_wet_rime_dataset(
    sample_paths: Sequence[str | Path], output_path: str | Path
) -> Path:
    """Pool the sample files and write the warming-level-bin percentile dataset (NetCDF)."""

    samples = pd.concat(
        [pd.read_parquet(path) for path in sample_paths], ignore_index=True
    )
    dataset = percentiles_to_dataset(build_warming_level_bin_percentiles(samples))
    path = Path(output_path)
    with staged_outputs([path]) as staged:
        dataset.to_netcdf(staged[path])
    return path
