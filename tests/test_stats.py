"""
Phase 3a — Statistik-Funktionen testen.

DoD: Unit-Tests gegen Referenzfälle (CI-Abdeckung bei bekanntem Generator,
Wilcoxon gegen scipy, Cliff's-delta-Grenzfälle −1/0/+1) grün.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.stats import (
    _delta_magnitude,
    adjust_pvalues,
    bootstrap_ci,
    cliffs_delta,
    wilcoxon_pairwise,
)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

def test_bootstrap_ci_determinism():
    """Gleicher seed → identische Tripel (ci_lo, ci_hi, observed)."""
    data = list(range(1, 21))
    first  = bootstrap_ci(data, seed=7)
    second = bootstrap_ci(data, seed=7)
    assert first == second


def test_bootstrap_ci_monotone():
    """ci_lower ≤ observed ≤ ci_upper für zufällige Eingaben."""
    rng = np.random.default_rng(0)
    for seed in range(20):
        data = rng.normal(0, 1, size=50)
        lo, hi, obs = bootstrap_ci(data, seed=seed)
        assert lo <= obs <= hi, f"seed={seed}: {lo} ≤ {obs} ≤ {hi} verletzt"


def test_bootstrap_ci_coverage():
    """95%-CI muss den wahren Mittelwert in ≈95 % der Wiederholungen einschließen.

    Erzeuge 500 Stichproben à n=40 aus N(0,1); Coverage soll 90–100 % liegen.
    """
    rng = np.random.default_rng(42)
    n_trials  = 500
    true_mean = 0.0
    covered   = 0
    for seed in range(n_trials):
        sample = rng.normal(true_mean, 1.0, size=40)
        lo, hi, _ = bootstrap_ci(sample, n_boot=500, seed=seed)
        if lo <= true_mean <= hi:
            covered += 1
    coverage = covered / n_trials
    assert 0.90 <= coverage <= 1.00, f"Coverage {coverage:.3f} außerhalb [0.90, 1.00]"


def test_bootstrap_ci_empty_returns_nan():
    """Leere Eingabe → (nan, nan, nan)."""
    lo, hi, obs = bootstrap_ci([])
    assert all(np.isnan(v) for v in (lo, hi, obs))


def test_bootstrap_ci_nan_values_dropped():
    """NaN-Werte werden ignoriert; valide Einträge bleiben maßgeblich."""
    clean = [1.0, 2.0, 3.0, 4.0, 5.0]
    dirty = [1.0, np.nan, 2.0, np.nan, 3.0, 4.0, 5.0]
    lo_c, hi_c, obs_c = bootstrap_ci(clean, seed=0)
    lo_d, hi_d, obs_d = bootstrap_ci(dirty, seed=0)
    assert obs_c == pytest.approx(obs_d)
    assert lo_c  == pytest.approx(lo_d)
    assert hi_c  == pytest.approx(hi_d)


# ---------------------------------------------------------------------------
# cliffs_delta — Grenzfälle
# ---------------------------------------------------------------------------

def test_cliffs_delta_plus_one():
    """Alle x > alle y → d = +1.0."""
    assert cliffs_delta([10, 11, 12], [1, 2, 3]) == pytest.approx(1.0)


def test_cliffs_delta_minus_one():
    """Alle x < alle y → d = −1.0."""
    assert cliffs_delta([1, 2, 3], [10, 11, 12]) == pytest.approx(-1.0)


def test_cliffs_delta_zero():
    """x == y (identische Werte) → d = 0.0."""
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_cliffs_delta_antisymmetric():
    """cliffs_delta(x, y) == −cliffs_delta(y, x)."""
    x = [1, 3, 5, 7]
    y = [2, 4, 6, 8]
    assert cliffs_delta(x, y) == pytest.approx(-cliffs_delta(y, x))


def test_cliffs_delta_range():
    """Ergebnis liegt stets in [−1, +1]."""
    rng = np.random.default_rng(99)
    for _ in range(50):
        x = rng.normal(0, 1, size=10)
        y = rng.normal(0.5, 1, size=10)
        d = cliffs_delta(x, y)
        assert -1.0 <= d <= 1.0

def test_cliffs_delta_empty_returns_nan():
    """Leere Eingabe → nan."""
    assert np.isnan(cliffs_delta([], [1, 2, 3]))
    assert np.isnan(cliffs_delta([1, 2, 3], []))


# ---------------------------------------------------------------------------
# _delta_magnitude
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d, expected", [
    (0.00,  "negligible"),
    (0.146, "negligible"),
    (0.147, "small"),
    (0.329, "small"),
    (0.330, "medium"),
    (0.473, "medium"),
    (0.474, "large"),
    (1.00,  "large"),
    # Symmetrie: negatives d gleiche Magnitude
    (-0.50, "large"),
    (-0.20, "small"),
])
def test_delta_magnitude_thresholds(d, expected):
    assert _delta_magnitude(d) == expected


def test_delta_magnitude_nan():
    assert _delta_magnitude(float("nan")) == "n/a"


# ---------------------------------------------------------------------------
# wilcoxon_pairwise — Vergleich gegen scipy
# ---------------------------------------------------------------------------

@pytest.fixture
def paired_df():
    """Konstruiertes DataFrame mit 10 gematchten Paaren für zwei Pipelines."""
    rng = np.random.default_rng(123)
    n = 10
    scores_a = rng.uniform(2, 5, size=n).round(2)
    scores_b = (scores_a + rng.normal(0.3, 0.5, size=n)).clip(1, 5).round(2)
    return pd.DataFrame({
        "pipeline_label": ["A"] * n + ["B"] * n,
        "instance_id":    list(range(n)) * 2,
        "xai_model":      ["lime"] * n * 2,
        "score":          list(scores_a) + list(scores_b),
    })


def test_wilcoxon_pairwise_schema(paired_df):
    """Rückgabe enthält alle erwarteten Spalten."""
    result = wilcoxon_pairwise(paired_df, ["A", "B"], "score")
    expected_cols = {
        "pipeline_a", "pipeline_b", "n_pairs",
        "mean_a", "mean_b", "delta_mean",
        "statistic", "p_value", "cliffs_d", "magnitude",
    }
    assert expected_cols.issubset(set(result.columns))


def test_wilcoxon_pairwise_n_pairs(paired_df):
    """n_pairs muss der Anzahl gematchter Paare entsprechen."""
    result = wilcoxon_pairwise(paired_df, ["A", "B"], "score")
    assert result.loc[0, "n_pairs"] == 10


def test_wilcoxon_pairwise_vs_scipy(paired_df):
    """p_value und Statistik müssen mit scipy.stats.wilcoxon übereinstimmen."""
    result = wilcoxon_pairwise(paired_df, ["A", "B"], "score")

    xa = paired_df[paired_df["pipeline_label"] == "A"].set_index("instance_id")["score"].values
    xb = paired_df[paired_df["pipeline_label"] == "B"].set_index("instance_id")["score"].values
    ref_stat, ref_pval = scipy_stats.wilcoxon(xa, xb, alternative="two-sided")

    assert result.loc[0, "statistic"] == pytest.approx(ref_stat, abs=0.01)
    assert result.loc[0, "p_value"]   == pytest.approx(ref_pval, abs=1e-4)


def test_wilcoxon_pairwise_cliffs_d_consistent(paired_df):
    """cliffs_d in der Tabelle muss mit direktem cliffs_delta()-Aufruf übereinstimmen."""
    result = wilcoxon_pairwise(paired_df, ["A", "B"], "score")
    xa = paired_df[paired_df["pipeline_label"] == "A"].set_index("instance_id")["score"].values
    xb = paired_df[paired_df["pipeline_label"] == "B"].set_index("instance_id")["score"].values
    expected_d = round(cliffs_delta(xa, xb), 3)
    assert result.loc[0, "cliffs_d"] == pytest.approx(expected_d, abs=1e-3)


def test_wilcoxon_pairwise_too_few_pairs():
    """Weniger als 3 Paare → Zeile mit nan statt Fehler."""
    df = pd.DataFrame({
        "pipeline_label": ["A", "A", "B", "B"],
        "instance_id":    [0, 1, 0, 1],
        "xai_model":      ["lime"] * 4,
        "score":          [3.0, 4.0, 2.0, 5.0],
    })
    result = wilcoxon_pairwise(df, ["A", "B"], "score")
    assert result.loc[0, "n_pairs"] == 2
    assert np.isnan(result.loc[0, "p_value"])


# ---------------------------------------------------------------------------
# adjust_pvalues — Multiplizitätskorrektur (Holm / Benjamini-Hochberg)
# ---------------------------------------------------------------------------

def test_adjust_pvalues_holm_known():
    """Hand gerechnetes Holm-Beispiel (m=3)."""
    adj = adjust_pvalues([0.01, 0.04, 0.03], method="holm")
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06], atol=1e-9)


def test_adjust_pvalues_bh_known():
    """Hand gerechnetes Benjamini-Hochberg-Beispiel (m=3)."""
    adj = adjust_pvalues([0.01, 0.04, 0.03], method="fdr_bh")
    np.testing.assert_allclose(adj, [0.03, 0.04, 0.04], atol=1e-9)


def test_adjust_pvalues_nan_passthrough():
    """NaN-Tests (n < 3) zählen nicht zur Familiengröße und bleiben NaN."""
    adj = adjust_pvalues([np.nan, 0.01, 0.04, np.nan], method="holm")
    assert np.isnan(adj[0]) and np.isnan(adj[3])
    # verbleibende zwei werden als Familie der Größe 2 korrigiert
    np.testing.assert_allclose(adj[[1, 2]], [0.02, 0.04], atol=1e-9)


def test_adjust_pvalues_holm_at_least_bh():
    """Holm (FWER) ist nie kleiner als BH (FDR) — elementweise."""
    pv = [0.001, 0.01, 0.02, 0.03, 0.2]
    holm = adjust_pvalues(pv, "holm")
    bh = adjust_pvalues(pv, "fdr_bh")
    assert np.all(holm + 1e-12 >= bh)
    assert np.all(holm <= 1.0)


def test_adjust_pvalues_invalid_method():
    with pytest.raises(ValueError):
        adjust_pvalues([0.1, 0.2], method="bonferroni")


def test_adjust_pvalues_empty():
    assert adjust_pvalues([]).shape == (0,)


# ---------------------------------------------------------------------------
# wilcoxon_pairwise — Korrekturspalten
# ---------------------------------------------------------------------------

@pytest.fixture
def three_pipeline_df():
    """Drei Pipelines × 10 gematchte Paare → C(3,2)=3 paarweise Tests."""
    rng = np.random.default_rng(7)
    n = 10
    a = rng.uniform(2, 5, size=n).round(2)
    b = (a + rng.normal(0.4, 0.4, size=n)).clip(1, 5).round(2)
    c = (a + rng.normal(-0.4, 0.4, size=n)).clip(1, 5).round(2)
    return pd.DataFrame({
        "pipeline_label": ["A"] * n + ["B"] * n + ["C"] * n,
        "instance_id":    list(range(n)) * 3,
        "xai_model":      ["lime"] * n * 3,
        "score":          list(a) + list(b) + list(c),
    })


def test_wilcoxon_pairwise_adds_correction_columns(three_pipeline_df):
    res = wilcoxon_pairwise(three_pipeline_df, ["A", "B", "C"], "score")
    assert {"p_value_adj", "reject"}.issubset(res.columns)
    assert len(res) == 3  # C(3,2)
    valid = res.dropna(subset=["p_value"])
    # Holm-korrigiert ≥ roh; reject ist boolesch
    assert np.all(valid["p_value_adj"] >= valid["p_value"] - 1e-9)
    assert res["reject"].dtype == bool


def test_wilcoxon_pairwise_correction_none(three_pipeline_df):
    res = wilcoxon_pairwise(three_pipeline_df, ["A", "B", "C"], "score",
                            correction=None)
    assert "p_value_adj" not in res.columns
    assert "reject" not in res.columns
