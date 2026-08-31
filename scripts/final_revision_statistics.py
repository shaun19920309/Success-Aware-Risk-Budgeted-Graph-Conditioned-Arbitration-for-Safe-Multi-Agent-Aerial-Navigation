#!/usr/bin/env python3
"""Dependency-light paired inference used by the final paper analysis."""

from __future__ import annotations

import math

import numpy as np


def paired_bootstrap_ci(
    deltas: np.ndarray,
    n_resamples: int,
    seed: int,
    chunk_size: int = 10_000,
) -> tuple[float, float]:
    """Return a percentile 95% CI for the mean paired difference."""
    rng = np.random.default_rng(seed)
    count = deltas.size
    if count == 0:
        raise ValueError("paired_bootstrap_ci requires at least one paired value")
    means = np.empty(n_resamples, dtype=np.float64)
    offset = 0
    while offset < n_resamples:
        size = min(chunk_size, n_resamples - offset)
        indices = rng.integers(0, count, size=(size, count))
        means[offset : offset + size] = deltas[indices].mean(axis=1)
        offset += size
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def monte_carlo_sign_flip_pvalue(
    deltas: np.ndarray,
    n_draws: int,
    seed: int,
    chunk_size: int = 10_000,
) -> float:
    """Return a two-sided Monte Carlo sign-flip p-value for paired data."""
    if deltas.size == 0:
        raise ValueError("monte_carlo_sign_flip_pvalue requires paired values")
    observed = abs(float(deltas.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    generated = 0
    tolerance = 1e-12
    while generated < n_draws:
        size = min(chunk_size, n_draws - generated)
        signs = rng.integers(0, 2, size=(size, deltas.size), dtype=np.int8)
        signs = signs * 2 - 1
        null_effects = np.abs((signs * deltas).mean(axis=1))
        extreme += int(np.count_nonzero(null_effects + tolerance >= observed))
        generated += size
    return (extreme + 1.0) / (n_draws + 1.0)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Apply Holm's family-wise error correction in input order."""
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [math.nan] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted
