from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LARGE_UNIT_THRESHOLD_MW = 100.0
MATERIAL_GAP_THRESHOLD_PP = 1.0
BACKOFF_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("country_cooling_size", ("country", "cooling_physics", "size_class")),
    ("country_size", ("country", "size_class")),
    ("country_cooling", ("country", "cooling_physics")),
    ("cooling_size", ("cooling_physics", "size_class")),
    ("country", ("country",)),
    ("cooling", ("cooling_physics",)),
    ("size", ("size_class",)),
    ("global", ()),
]


@dataclass(frozen=True)
class HeadlineMetrics:
    fleet_ucr: float
    large_ucr: float
    small_ucr: float
    high_risk_share: float
    future_decline_share: float


@dataclass(frozen=True)
class BenchmarkAssessment:
    benchmark_spanned: bool
    residual_gap_pp: float
    interval_width_pp: float
    synthetic_mae_pp: float | None
    tolerance_pp: float
    warrants_latent_location: bool
    direction_vs_benchmark: str


def classify_size_class(mw: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(mw.to_numpy(dtype=float) >= LARGE_UNIT_THRESHOLD_MW, "large", "small"),
        index=mw.index,
        dtype="object",
    )


def add_size_class(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["size_class"] = classify_size_class(result["mw"])
    return result


def compute_headline_metrics(frame: pd.DataFrame) -> HeadlineMetrics:
    total_mw = float(frame["mw"].sum())
    if total_mw <= 0:
        raise ValueError("headline metrics require positive total capacity")

    large_mask = frame["size_class"].eq("large")
    small_mask = frame["size_class"].eq("small")

    def weighted_mean(mask: pd.Series) -> float:
        subset = frame.loc[mask]
        return float((subset["historical_ucr"] * subset["mw"]).sum() / subset["mw"].sum())

    return HeadlineMetrics(
        fleet_ucr=float((frame["historical_ucr"] * frame["mw"]).sum() / total_mw),
        large_ucr=weighted_mean(large_mask),
        small_ucr=weighted_mean(small_mask),
        high_risk_share=float(frame.loc[frame["historical_ucr"] <= 0.2, "mw"].sum() / total_mw),
        future_decline_share=float(frame.loc[frame["future_decline"], "mw"].sum() / total_mw),
    )


def summarise_metrics(frame: pd.DataFrame) -> dict[str, float]:
    metrics = compute_headline_metrics(frame)
    return {
        "fleet_ucr": metrics.fleet_ucr,
        "large_ucr": metrics.large_ucr,
        "small_ucr": metrics.small_ucr,
        "high_risk_share": metrics.high_risk_share,
        "future_decline_share": metrics.future_decline_share,
    }


def build_peer_lookup(donors: pd.DataFrame) -> dict[str, dict[tuple[object, ...], np.ndarray]]:
    lookups = {}

    for rule_name, columns in BACKOFF_RULES:
        if columns:
            grouped = donors.reset_index().groupby(list(columns))["index"]
            lookups[rule_name] = {
                key if isinstance(key, tuple) else (key,): values.to_numpy(dtype=int)
                for key, values in grouped
            }
        else:
            lookups[rule_name] = {(): donors.index.to_numpy(dtype=int)}

    return lookups


def resolve_peer_rule(
    unit: pd.Series,
    peer_lookup: dict[str, dict[tuple[object, ...], np.ndarray]],
) -> tuple[str, np.ndarray]:
    for rule_name, columns in BACKOFF_RULES:
        key = tuple(unit[column] for column in columns)
        candidates = peer_lookup[rule_name].get(key, np.array([], dtype=int))
        if len(candidates) > 0:
            return rule_name, candidates
    raise ValueError("no donor candidates available")


def sample_missing_unit_outcomes(
    missing_units: pd.DataFrame,
    donors: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    donor_pool = donors.reset_index(drop=True)
    peer_lookup = build_peer_lookup(donor_pool)

    rows = []
    for unit in missing_units.itertuples(index=False):
        unit_series = pd.Series(unit._asdict())
        backoff_rule, candidates = resolve_peer_rule(unit_series, peer_lookup)
        donor_index = int(rng.choice(candidates))
        donor = donor_pool.iloc[donor_index]
        rows.append(
            {
                "UNITID": unit.UNITID,
                "LOCATIONID": unit.LOCATIONID,
                "country": unit.country,
                "cooling_physics": unit.cooling_physics,
                "mw": float(unit.mw),
                "size_class": unit.size_class,
                "historical_ucr": float(donor["historical_ucr"]),
                "future_decline": bool(donor["future_decline"]),
                "donor_unitid": int(donor["UNITID"]),
                "backoff_rule": backoff_rule,
            }
        )

    return pd.DataFrame(rows)


def synthetic_missingness_mask(
    mapped_units: pd.DataFrame,
    rng: np.random.Generator,
    mask_fraction: float = 0.2,
    min_group_size: int = 5,
) -> np.ndarray:
    mask = np.zeros(len(mapped_units), dtype=bool)
    grouped = mapped_units.groupby(["country", "cooling_physics", "size_class"]).indices

    for indices in grouped.values():
        if len(indices) < min_group_size:
            continue
        n_mask = max(1, int(round(len(indices) * mask_fraction)))
        n_mask = min(n_mask, len(indices) - 1)
        if n_mask <= 0:
            continue
        chosen = rng.choice(np.asarray(indices), size=n_mask, replace=False)
        mask[chosen] = True

    return mask


def assess_benchmark_interval(
    interval: dict[str, float],
    benchmark: float,
    synthetic_mae_pp: float | None = None,
    material_gap_threshold_pp: float = MATERIAL_GAP_THRESHOLD_PP,
) -> BenchmarkAssessment:
    residual_gap_pp = abs(interval["mean"] - benchmark) * 100
    interval_width_pp = (interval["p95"] - interval["p05"]) * 100
    benchmark_spanned = interval["p05"] <= benchmark <= interval["p95"]
    tolerance_pp = max(
        material_gap_threshold_pp,
        interval_width_pp / 2 + (synthetic_mae_pp or 0.0),
    )

    if interval["mean"] < benchmark:
        direction_vs_benchmark = "below"
    elif interval["mean"] > benchmark:
        direction_vs_benchmark = "above"
    else:
        direction_vs_benchmark = "at"

    warrants_latent_location = (not benchmark_spanned) and residual_gap_pp > tolerance_pp

    return BenchmarkAssessment(
        benchmark_spanned=benchmark_spanned,
        residual_gap_pp=residual_gap_pp,
        interval_width_pp=interval_width_pp,
        synthetic_mae_pp=synthetic_mae_pp,
        tolerance_pp=tolerance_pp,
        warrants_latent_location=warrants_latent_location,
        direction_vs_benchmark=direction_vs_benchmark,
    )
