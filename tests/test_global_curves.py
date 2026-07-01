"""Tests for build_global_curves (global shape / dependence extraction)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.explanations import build_global_curves, save_global_curves


@pytest.fixture(scope="module")
def toy_data():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "cat": pd.Series(rng.integers(0, 3, n), dtype="category"),
        "num": rng.normal(size=n).astype(float),
    })
    y = df["num"] * 2 + df["cat"].astype(int) + rng.normal(scale=0.1, size=n)
    return df, y


@pytest.fixture(scope="module")
def ebm_model(toy_data):
    from interpret.glassbox import ExplainableBoostingRegressor
    X, y = toy_data
    m = ExplainableBoostingRegressor(interactions=0, random_state=0)
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def xgb_model(toy_data):
    from xgboost import XGBRegressor
    X, y = toy_data
    m = XGBRegressor(n_estimators=10, max_depth=2, enable_categorical=True)
    m.fit(X, y)
    return m


def _check_schema(curves, features):
    assert {c["feature"] for c in curves} == set(features)
    for c in curves:
        assert set(c) == {"feature", "kind", "x", "y", "importance"}
        assert c["kind"] in ("continuous", "categorical")
        assert len(c["x"]) == len(c["y"]) >= 1          # x and y aligned
        assert isinstance(c["importance"], float)


def test_ebm_curves_schema_and_kinds(ebm_model, toy_data):
    X, _ = toy_data
    curves = build_global_curves(ebm_model, "ebm", X)
    _check_schema(curves, X.columns)
    kinds = {c["feature"]: c["kind"] for c in curves}
    assert kinds["cat"] == "categorical"
    assert kinds["num"] == "continuous"


def test_xgb_curves_one_point_per_row(xgb_model, toy_data):
    X, _ = toy_data
    curves = build_global_curves(xgb_model, "xgb", X)
    _check_schema(curves, X.columns)
    # dependence cloud: one (feature_value, shap_value) per training row
    for c in curves:
        assert len(c["x"]) == len(X)


def test_unknown_model_name_raises(toy_data):
    X, _ = toy_data
    with pytest.raises(ValueError):
        build_global_curves(None, "rf", X)


def test_save_global_curves_writes_per_feature(ebm_model, toy_data, tmp_path):
    X, _ = toy_data
    curves = build_global_curves(ebm_model, "ebm", X)
    paths = save_global_curves(curves, "ebm", out_dir=tmp_path)
    names = {p.name for p in paths}
    assert names == {f"global_curve_ebm_{f}.json" for f in X.columns}
    assert all(p.exists() for p in paths)
