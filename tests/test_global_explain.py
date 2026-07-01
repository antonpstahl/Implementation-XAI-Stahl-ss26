"""Tests for utils.global_explain (EBM-native beeswarm)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.global_explain import (
    ebm_local_contributions, feature_importance_order, coloring_directions,
    beeswarm_plot, order_concordance, direction_agreement,
)


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({
        "cat": pd.Series(rng.integers(0, 3, n), dtype="category"),
        "num": rng.normal(size=n).astype(float),
    })
    y = X["num"] * 3 + X["cat"].astype(int) + rng.normal(scale=0.1, size=n)
    from interpret.glassbox import ExplainableBoostingRegressor
    m = ExplainableBoostingRegressor(interactions=0, random_state=0).fit(X, y)
    return m, X


def test_contributions_shape_and_columns(toy):
    model, X = toy
    contrib = ebm_local_contributions(model, X)
    assert contrib.shape == (len(X), X.shape[1])          # n x main-effect features
    assert list(contrib.columns) == list(X.columns)       # no interaction terms
    assert contrib.index.equals(X.index)


def test_contributions_match_explain_local(toy):
    model, X = toy
    contrib = ebm_local_contributions(model, X)
    d0 = model.explain_local(X.iloc[:1]).data(0)
    ref = {n: s for n, s in zip(d0["names"], d0["scores"]) if " & " not in str(n)}
    for f in contrib.columns:
        assert contrib.iloc[0][f] == pytest.approx(ref[f], abs=1e-9)


def test_importance_order_and_direction(toy):
    model, X = toy
    contrib = ebm_local_contributions(model, X)
    order = feature_importance_order(contrib)
    assert order[0] == "num"                               # num dominates by construction
    dirs = coloring_directions(contrib, X)
    assert dirs["num"] == 1                                # y rises with num -> +1


def test_beeswarm_plot_writes_png(toy, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    model, X = toy
    contrib = ebm_local_contributions(model, X)
    p = beeswarm_plot(contrib, X, tmp_path / "bees.png")
    assert p.exists() and p.stat().st_size > 0


def test_concordance_helpers():
    a = ["hr", "temp", "yr", "hum"]
    assert order_concordance(a, a)["spearman"] == pytest.approx(1.0)
    assert order_concordance(a, a)["top3_overlap"] == 3
    da = direction_agreement({"a": 1, "b": -1, "c": 0}, {"a": 1, "b": 1, "c": 1})
    assert da["n_compared"] == 2                            # c dropped (0)
    assert da["agreement"] == pytest.approx(0.5)
    assert da["disagree"] == ["b"]
