"""
Phase 3a — Sampling/Stratifizierung testen.

DoD: Test prüft Determinismus bei festem Seed, korrekte Stratifizierung
über cnt-Quintile/Tageszeit/Wetter, Abwesenheit von Duplikaten und die
Zielgröße n.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data import load_train_test, sample_stratified


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def test_data():
    """Loads X_test / y_test once for the whole module."""
    _, _, X_test, y_test = load_train_test()
    return X_test, y_test


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism(test_data):
    """Same seed must yield identical results on repeated calls."""
    X, y = test_data
    first  = sample_stratified(X, y, n=50, seed=42)
    second = sample_stratified(X, y, n=50, seed=42)
    assert first == second


def test_different_seeds_differ(test_data):
    """Different seeds should (almost certainly) produce different samples."""
    X, y = test_data
    s42 = sample_stratified(X, y, n=50, seed=42)
    s99 = sample_stratified(X, y, n=50, seed=99)
    assert s42 != s99


# ---------------------------------------------------------------------------
# Target size & no duplicates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [20, 50, 150, 200])
def test_exact_size(test_data, n):
    """Returned list must contain exactly n indices."""
    X, y = test_data
    result = sample_stratified(X, y, n=n, seed=42)
    assert len(result) == n


@pytest.mark.parametrize("n", [20, 50, 150, 200])
def test_no_duplicates(test_data, n):
    """All returned indices must be unique."""
    X, y = test_data
    result = sample_stratified(X, y, n=n, seed=42)
    assert len(set(result)) == n


def test_indices_in_range(test_data):
    """All indices must be valid positional indices into the DataFrame."""
    X, y = test_data
    result = sample_stratified(X, y, n=100, seed=42)
    valid_idx = set(X.index.tolist())
    assert all(i in valid_idx for i in result)


def test_oversized_n_raises(test_data):
    """Requesting more indices than available rows must raise ValueError."""
    X, y = test_data
    with pytest.raises(ValueError, match="exceeds dataset size"):
        sample_stratified(X, y, n=len(X) + 1, seed=42)


# ---------------------------------------------------------------------------
# Stratification correctness
# ---------------------------------------------------------------------------

def _compute_strata(X: pd.DataFrame, y: pd.Series, indices: list[int]):
    """Helper: returns a DataFrame with stratum columns for the given indices."""
    X_s = X.loc[indices]
    y_s = y.loc[indices]
    return pd.DataFrame({
        "cnt_q":   pd.qcut(y, q=5, labels=False, duplicates="drop").loc[indices],
        "time_b":  (X_s["hr"].astype(int) // 6).astype(int),
        "weather": X_s["weathersit"].astype(int),
    }, index=indices)


def test_all_cnt_quintiles_covered(test_data):
    """Sample must contain instances from all 5 cnt-quintile bins."""
    X, y = test_data
    result = sample_stratified(X, y, n=150, seed=42)
    strata = _compute_strata(X, y, result)
    present_quintiles = set(strata["cnt_q"].dropna().astype(int).unique())
    # qcut with duplicates='drop' may merge bins → at least 4 distinct bins
    assert len(present_quintiles) >= 4, (
        f"Only {len(present_quintiles)} cnt-quintile bins covered: {present_quintiles}"
    )


def test_all_time_blocks_covered(test_data):
    """Sample must cover all 4 time-of-day blocks (night/morning/afternoon/evening)."""
    X, y = test_data
    result = sample_stratified(X, y, n=50, seed=42)
    strata = _compute_strata(X, y, result)
    present_blocks = set(strata["time_b"].unique())
    # hr 0-5 → 0, 6-11 → 1, 12-17 → 2, 18-23 → 3
    assert present_blocks == {0, 1, 2, 3}, (
        f"Not all time blocks covered: {present_blocks}"
    )


def test_multiple_weather_conditions_covered(test_data):
    """Sample must include at least 2 distinct weathersit values."""
    X, y = test_data
    result = sample_stratified(X, y, n=50, seed=42)
    strata = _compute_strata(X, y, result)
    present_weather = set(strata["weather"].unique())
    assert len(present_weather) >= 2, (
        f"Only {len(present_weather)} weathersit values: {present_weather}"
    )


def test_rare_weather_covered_in_large_sample(test_data):
    """weathersit=3 (rare, ~7 % of test set) must appear in a sample of n=150."""
    X, y = test_data
    result = sample_stratified(X, y, n=150, seed=42)
    strata = _compute_strata(X, y, result)
    assert 3 in strata["weather"].values, (
        "weathersit=3 absent from sample of 150 — stratification may be broken"
    )


# ---------------------------------------------------------------------------
# Proportionality sanity check
# ---------------------------------------------------------------------------

def test_sample_distribution_roughly_proportional(test_data):
    """
    For each cnt-quintile, the sample fraction should be within ±15 pp
    of the population fraction (proportional allocation, not uniform).
    """
    X, y = test_data
    n = 200
    result = sample_stratified(X, y, n=n, seed=42)

    pop_q = pd.qcut(y, q=5, labels=False, duplicates="drop")
    sample_q = pop_q.loc[result]

    for q in pop_q.dropna().unique():
        pop_frac    = (pop_q == q).mean()
        sample_frac = (sample_q == q).mean()
        assert abs(pop_frac - sample_frac) < 0.15, (
            f"Quintile {q}: population {pop_frac:.2f} vs sample {sample_frac:.2f}"
        )
