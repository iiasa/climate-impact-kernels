"""Per-region summary of the site evaluation, and the comparison of potential with existing sites.

The site evaluation is the long table that ``run_site_evaluation`` writes (in the code it
is called the census). Its ``population`` column holds ``candidate`` for potential
sites (grid cells where a new plant could be built) and ``incumbent`` for existing
sites (grid cells where today's fleet has plants).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Default seed of the year sampling and of the summary bootstrap.
DEFAULT_SAMPLING_SEED = 730

# Quantiles reported per region; p01 and p99 describe the tails, p50 the centre,
# p05 and p95 the body of the distribution.
QUANTILES = (0.01, 0.05, 0.50, 0.95, 0.99)

# Columns the per-region summary reads back from the long parquet. The spatial
# and source columns it does not group on are left out, to keep the read small.
SUMMARY_INPUT_COLUMNS = (
    "population",
    "region_id",
    "archetype",
    "gcm",
    "scenario",
    "year",
    "season",
    "warming_level",
    "response",
    "unit",
    "cell_id",
    "cell_area_m2",
    "value",
)


GROUP_COLUMNS = [
    "population",
    "region_id",
    "archetype",
    "gcm",
    "season",
    "warming_level",
    "response",
    "unit",
]


def summarize_site_evaluation(
    results: pd.DataFrame,
    *,
    low_value_thresholds: dict[str, float] | None = None,
    bootstrap_replicates: int = 200,
    seed: int = DEFAULT_SAMPLING_SEED,
) -> pd.DataFrame:
    """Per-region area-weighted quantiles and mean, with a standard error from a cluster bootstrap over years.

    Whole years are resampled together: within each group the sampled
    (scenario, year) pairs are drawn with replacement to carry the uncertainty
    from sampling years into a standard error for every metric. The standard error
    is NaN where a group has fewer than two (scenario, year) pairs (or the run
    fewer than two replicates).

    Where ``low_value_thresholds`` gives a threshold for a response, the metric
    ``bad_tail_area_fraction`` is the share of cell area whose value is at or
    below that threshold.
    """

    low_value_thresholds = low_value_thresholds or {}
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for keys, group in results.groupby(GROUP_COLUMNS, dropna=False, sort=True):
        base = dict(zip(GROUP_COLUMNS, keys, strict=True))
        data = group[np.isfinite(group["value"].to_numpy(dtype=float))]
        if data.empty:
            continue
        metrics = _census_metrics(data, base["response"], low_value_thresholds)
        # One (scenario, year) unit resamples to itself, so its replicates agree
        # and say nothing about sampling uncertainty.
        resampling_units = (
            data["scenario"].astype(str) + "|" + data["year"].astype(str)
        ).nunique()
        replicates = [
            _census_metrics(
                _resample_years(data, rng), base["response"], low_value_thresholds
            )
            for _ in range(bootstrap_replicates)
        ]
        for metric, value in metrics.items():
            errors = np.asarray(
                [row.get(metric, np.nan) for row in replicates], dtype=float
            )
            records.append(
                {
                    **base,
                    "metric": metric,
                    "value": value,
                    "sampling_standard_error": float(np.nanstd(errors, ddof=1))
                    if bootstrap_replicates >= 2 and resampling_units >= 2
                    else float("nan"),
                    "bootstrap_replicates": bootstrap_replicates,
                }
            )
    summary = pd.DataFrame.from_records(records)
    return _attach_between_region_spread(summary)


def _census_metrics(
    data: pd.DataFrame,
    response: str,
    low_value_thresholds: dict[str, float],
) -> dict[str, float]:
    values = data["value"].to_numpy(dtype=float)
    weights = data["cell_area_m2"].to_numpy(dtype=float)
    # Each cell recurs once per sampled year, so summing cell_area_m2 over the
    # cell-year rows would multiply land area by the year count. The area-weighted
    # mean, quantiles and `bad_tail_area_fraction` are ratios in which that
    # multiplicity cancels, but area_total_m2 is a raw sum, so it is taken over
    # distinct cells.
    distinct_area = data.drop_duplicates("cell_id")["cell_area_m2"].to_numpy(
        dtype=float
    )
    metrics: dict[str, float] = {
        "area_weighted_mean": float(np.average(values, weights=weights)),
        "n_cells": float(data["cell_id"].nunique()),
        "area_total_m2": float(distinct_area.sum()),
    }
    for quantile in QUANTILES:
        metrics[f"p{round(quantile * 100):02d}"] = _weighted_quantile(
            values, weights, quantile
        )
    if response in low_value_thresholds:
        bad = values <= low_value_thresholds[response]
        metrics["bad_tail_area_fraction"] = float(weights[bad].sum() / weights.sum())
    return metrics


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    total = weights.sum()
    if total <= 0:
        return float("nan")
    return float(np.interp(quantile * total, cumulative, values))


def _resample_years(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Cluster bootstrap: resample whole (scenario, year) units with replacement."""

    key = data["scenario"].astype(str) + "|" + data["year"].astype(str)
    blocks = {unit: block for unit, block in data.groupby(key, sort=False)}
    units = list(blocks)
    if len(units) <= 1:
        return data
    chosen = rng.integers(0, len(units), size=len(units))
    return pd.concat([blocks[units[index]] for index in chosen], ignore_index=True)


def _attach_between_region_spread(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    medians = summary[summary["metric"] == "p50"]
    spread_keys = [
        "population",
        "archetype",
        "gcm",
        "season",
        "warming_level",
        "response",
        "unit",
    ]
    spread = (
        medians.groupby(spread_keys)["value"]
        .agg(lambda values: float(values.max() - values.min()))
        .reset_index()
    )
    spread["region_id"] = "ALL"
    spread["metric"] = "between_region_p50_spread"
    spread["sampling_standard_error"] = float("nan")
    spread["bootstrap_replicates"] = 0
    return pd.concat([summary, spread], ignore_index=True)


def compare_potential_to_existing_sites(summary: pd.DataFrame) -> pd.DataFrame:
    """Pivot the summary into differences between existing and potential sites per region and metric.

    The ``grid`` column holds the potential sites (``candidate``) and the ``fleet``
    column the existing sites (``incumbent``). Positive ``fleet_minus_grid`` on a
    capacity-factor or water-limit metric means the existing sites sit on better
    cells than the potential sites as a whole. A ``p05`` for existing sites far
    above the ``p05`` for potential sites (or a much smaller
    ``bad_tail_area_fraction``) says the existing sites leave out the worse sites that the question
    of capacity expansion depends on.

    Needs both sets of cells in ``summary``: a site evaluation of one set has
    nothing to compare, so this raises instead of returning a table without
    ``fleet_minus_grid`` (write such an evaluation with ``summary=None``).
    """

    populations = set(summary["population"].unique())
    absent = [name for name in ("candidate", "incumbent") if name not in populations]
    if absent:
        raise ValueError(
            f"the summary has no {' or '.join(absent)} population (found {sorted(populations)}); "
            "comparing potential sites with existing sites needs both"
        )
    pivot_keys = [
        "region_id",
        "archetype",
        "gcm",
        "season",
        "warming_level",
        "response",
        "unit",
        "metric",
    ]
    wide = summary.pivot_table(
        index=pivot_keys, columns="population", values="value", aggfunc="first"
    ).reset_index()
    wide = wide.rename(columns={"candidate": "grid", "incumbent": "fleet"})
    wide["fleet_minus_grid"] = wide["fleet"] - wide["grid"]
    return wide
