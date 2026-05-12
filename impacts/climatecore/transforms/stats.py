from __future__ import annotations

from collections.abc import Sequence


def weighted_quantiles(
    values: Sequence[float],
    weights: Sequence[float],
    quantiles: Sequence[float],
) -> tuple[float, ...]:
    """Return weighted quantiles for quantiles in [0, 1]."""

    _validate_shapes(values=values, weights=weights)
    if not quantiles:
        raise ValueError("quantiles must not be empty")

    ordered = sorted(zip(values, weights, strict=True))
    cumulative = []
    total_weight = 0.0
    for _, weight in ordered:
        if weight < 0:
            raise ValueError("weights must be non-negative")
        total_weight += weight
        cumulative.append(total_weight)

    if total_weight <= 0:
        raise ValueError("total weight must be positive")

    return tuple(
        _weighted_quantile(
            ordered=ordered,
            cumulative_weights=cumulative,
            total_weight=total_weight,
            quantile=quantile,
        )
        for quantile in quantiles
    )


def deterministic_resampling(
    values: Sequence[float],
    n_samples: int,
) -> tuple[float, ...]:
    """Return deterministic resamples at evenly spaced empirical quantiles."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    quantiles = tuple((index + 0.5) / n_samples for index in range(n_samples))
    unit_weights = tuple(1.0 for _ in values)
    return weighted_quantiles(values=values, weights=unit_weights, quantiles=quantiles)


def _validate_shapes(values: Sequence[float], weights: Sequence[float]) -> None:
    if not values:
        raise ValueError("values must not be empty")
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")


def _weighted_quantile(
    ordered: Sequence[tuple[float, float]],
    cumulative_weights: Sequence[float],
    total_weight: float,
    quantile: float,
) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")

    threshold = quantile * total_weight
    for index, cumulative_weight in enumerate(cumulative_weights):
        if cumulative_weight >= threshold:
            return ordered[index][0]

    return ordered[-1][0]
