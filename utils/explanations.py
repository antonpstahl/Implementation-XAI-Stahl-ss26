"""
utils/explanations.py - central feature schema and explanation builders.

The FEATURE_SCHEMA is passed to the LLM in the JSON interface. It is defined here
once so that all notebooks (03, 04, 05, 06) and the Tool Use pipeline (06) use
identical feature descriptions.
"""

from __future__ import annotations

import logging
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import EXPLANATIONS_DIR

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Denormalisierungs-Konstanten (einzige Quelle)
# -----------------------------------------------------------------------------
TEMP_FACTOR: int = 41
HUM_FACTOR:  int = 100
WIND_FACTOR: int = 67

WEEKDAY_NAMES: list[str] = [
    "Sunday", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday",
]
MONTH_NAMES: list[str] = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
WEATHER_NAMES: dict[int, str] = {
    1: "clear/few clouds",
    2: "mist/cloudy",
    3: "light rain/snow",
    4: "heavy rain/thunderstorm",
}

# -----------------------------------------------------------------------------
# Feature-Schema
# -----------------------------------------------------------------------------
FEATURE_SCHEMA: dict[str, dict[str, Any]] = {
    # --- Categorical features (category dtype) ---
    "weathersit": {
        "type": "categorical",
        "description": "Weather situation",
        "categories": WEATHER_NAMES,
    },
    "mnth": {
        "type": "categorical",
        "description": "Month (1=January, 12=December)",
        "range": [1, 12],
    },
    "hr": {
        "type": "categorical",
        "description": "Hour of the day (0 to 23)",
        "range": [0, 23],
    },
    "weekday": {
        "type": "categorical",
        "description": "Weekday (0=Sunday, 6=Saturday)",
        "range": [0, 6],
    },
    # --- Numerical features (float64) ---
    "yr": {
        "type": "binary",
        "description": "Year (0=2011, 1=2012)",
        "categories": {0: "2011", 1: "2012"},
    },
    "holiday": {
        "type": "binary",
        "description": "Holiday (0=no, 1=yes)",
        "categories": {0: "no holiday", 1: "holiday"},
    },
    "temp": {
        "type": "numerical",
        "description": "Normalised temperature in Celsius (divided by 41)",
        "range": [0.0, 1.0],
    },
    "hum": {
        "type": "numerical",
        "description": "Normalised humidity (divided by 100)",
        "range": [0.0, 1.0],
    },
    "windspeed": {
        "type": "numerical",
        "description": "Normalised wind speed (divided by 67)",
        "range": [0.0, 1.0],
    },
    # Removed features (redundant, not in the model):
    #   season     derivable from mnth
    #   workingday derivable from weekday + holiday
}

TARGET_DESCRIPTION: dict[str, Any] = {
    "name": "cnt",
    "description": (
        "Number of rented bikes per hour "
        "(sum of casual and registered users)"
    ),
    "type": "count",
}


# -----------------------------------------------------------------------------
# Public denormalisation helpers (single source)
# -----------------------------------------------------------------------------

def humanize_feature(feature: str, value: Any) -> str | None:
    """Convert raw feature values into readable strings for the LLM."""
    try:
        if feature == "temp":       return f"~{float(value) * TEMP_FACTOR:.1f} C"
        if feature == "hum":        return f"{float(value) * HUM_FACTOR:.0f} %"
        if feature == "windspeed":  return f"{float(value) * WIND_FACTOR:.1f} km/h"
        if feature == "hr":         return f"{int(value):02d}:00"
        if feature == "weekday":    return WEEKDAY_NAMES[int(value)]
        if feature == "mnth":       return MONTH_NAMES[int(value)]
        if feature == "weathersit": return WEATHER_NAMES.get(int(value))
        if feature == "yr":         return "2011" if int(value) == 0 else "2012"
        if feature == "holiday":    return "holiday" if int(value) == 1 else "no holiday"
    except (ValueError, TypeError, IndexError) as exc:
        logger.debug("humanize_feature failed for %s=%r: %s", feature, value, exc)
    return None


def build_context_string(fv: dict) -> str:
    """Human readable comma list of all feature values (NB04b JSON payload field)."""
    parts: list[str] = []
    if "hr" in fv:
        parts.append(f"{int(fv['hr']):02d}:00")
    if "weekday" in fv:
        parts.append(WEEKDAY_NAMES[int(fv["weekday"])])
    if "mnth" in fv:
        parts.append(MONTH_NAMES[int(fv["mnth"])])
    if "yr" in fv:
        parts.append("2011" if int(fv["yr"]) == 0 else "2012")
    if "weathersit" in fv:
        parts.append(WEATHER_NAMES.get(int(fv["weathersit"]), "unknown"))
    if "temp" in fv:
        parts.append(f"~{float(fv['temp']) * TEMP_FACTOR:.1f} C")
    if "hum" in fv:
        parts.append(f"{float(fv['hum']) * HUM_FACTOR:.0f} % humidity")
    if "windspeed" in fv:
        parts.append(f"wind {float(fv['windspeed']) * WIND_FACTOR:.1f} km/h")
    if "holiday" in fv and int(fv["holiday"]) == 1:
        parts.append("holiday")
    return ", ".join(parts)


# -----------------------------------------------------------------------------
# Internal helper functions
# -----------------------------------------------------------------------------

# Module level cache: model object to shap.TreeExplainer (avoid re-creating per call)
_shap_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _get_shap_explainer(model: Any) -> Any:
    if model not in _shap_cache:
        import shap
        _shap_cache[model] = shap.TreeExplainer(model)
    return _shap_cache[model]


def _feat_value(val: Any) -> Any:
    """Convert numpy scalars to Python native types for JSON."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def _global_xgb(model: Any, X_train: pd.DataFrame) -> tuple[list[str], list[float], float]:
    """SHAP mean |value| over the training set as global importance for XGBoost."""
    explainer = _get_shap_explainer(model)
    shap_vals = explainer.shap_values(X_train)
    importance = np.abs(shap_vals).mean(axis=0).tolist()
    base_value = float(explainer.expected_value)
    return X_train.columns.tolist(), importance, base_value


def _global_ebm(model: Any) -> tuple[list[str], list[float], float]:
    """EBM term importances (main effects only, no interactions)."""
    gexp = model.explain_global()
    gd = gexp.data()
    names = gd["names"]
    scores = [float(s) for s in gd["scores"]]
    # The intercept lives in explain_local; for global we use the mean prediction
    # as an approximation of the base value (passed in build_global).
    main_names, main_scores = [], []
    for n, s in zip(names, scores):
        if " & " not in n:  # exclude interaction terms
            main_names.append(n)
            main_scores.append(s)
    return main_names, main_scores, 0.0  # base_value over the training set separately


def _local_xgb(
    model: Any, X_train: pd.DataFrame, instance: pd.DataFrame
) -> tuple[dict[str, float], float, float]:
    """SHAP values (log space) + base value + prediction for one instance."""
    explainer = _get_shap_explainer(model)
    shap_vals = explainer.shap_values(instance)[0]
    contribs = {col: float(v) for col, v in zip(instance.columns, shap_vals)}
    base_value = float(explainer.expected_value)
    prediction = float(model.predict(instance)[0])
    return contribs, base_value, prediction


def _local_ebm(
    model: Any, instance: pd.DataFrame
) -> tuple[dict[str, float], float, float]:
    """EBM contributions (log space) + intercept + prediction for one instance."""
    lexp = model.explain_local(instance)
    d = lexp.data(0)
    contribs = {n: float(s) for n, s in zip(d["names"], d["scores"])}
    base_value = float(d["extra"]["scores"][0])  # intercept
    prediction = float(d["perf"]["predicted"])
    return contribs, base_value, prediction


# -----------------------------------------------------------------------------
# Public builders
# -----------------------------------------------------------------------------

def build_global(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    metrics: dict[str, float],
) -> dict:
    """
    Build a global explanation structure (JSON serialisable).

    Returns:
        {
          "model": str,
          "task": TARGET_DESCRIPTION,
          "feature_schema": FEATURE_SCHEMA,
          "metrics": {...},
          "base_value": float,          # log space (Poisson) or cnt space (RMSE)
          "global_importance": [
              {"feature": str, "importance": float, "rank": int}, ...
          ]
        }
    """
    if model_name == "xgb":
        names, scores, base_value = _global_xgb(model, X_train)
    elif model_name == "ebm":
        names, scores, _ = _global_ebm(model)
        # EBM base value: mean of the training predictions (log space)
        import numpy as np
        base_value = float(np.log(model.predict(X_train)).mean())
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    ranked = sorted(
        zip(names, scores), key=lambda x: -x[1]
    )
    return {
        "model": model_name,
        "task": TARGET_DESCRIPTION,
        "feature_schema": FEATURE_SCHEMA,
        "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
        "base_value": round(base_value, 6),
        "global_importance": [
            {"feature": f, "importance": round(imp, 6), "rank": i + 1}
            for i, (f, imp) in enumerate(ranked)
        ],
    }


def build_local(
    model: Any,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    instance_id: int,
) -> dict:
    """
    Build a local explanation structure for a single test instance.

    instance_id is the position index in the X_test DataFrame (iloc).

    Returns:
        {
          "model": str,
          "instance_id": int,
          "feature_values": {feature: value, ...},
          "y_true": float,
          "prediction": float,          # original scale (cnt)
          "base_value": float,          # log space
          "contribution_space": "log",
          "contributions": [
              {"feature": str, "value": ..., "contribution": float}, ...
          ]
        }
    """
    instance = X_test.iloc[[instance_id]]
    y_true = float(y_test.iloc[instance_id])

    if model_name == "xgb":
        contribs, base_value, prediction = _local_xgb(model, X_test, instance)
    elif model_name == "ebm":
        contribs, base_value, prediction = _local_ebm(model, instance)
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    feature_values = {
        col: _feat_value(instance.iloc[0][col])
        for col in instance.columns
    }

    contributions_list = sorted(
        [
            {
                "feature": f,
                "value": feature_values.get(f),
                "contribution": round(float(c), 6),
            }
            for f, c in contribs.items()
            if f in feature_values  # skip EBM interaction terms
        ],
        key=lambda x: -abs(x["contribution"]),
    )

    return {
        "model": model_name,
        "instance_id": instance_id,
        "feature_values": feature_values,
        "y_true": y_true,
        "prediction": round(prediction, 4),
        "base_value": round(base_value, 6),
        "contribution_space": "log",
        "contributions": contributions_list,
    }


def save_explanation(data: dict, filename: str, out_dir: Path | None = None) -> Path:
    """Save an explanation dict as JSON."""
    import json
    out_dir = Path(out_dir) if out_dir is not None else EXPLANATIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


# -----------------------------------------------------------------------------
# Global shape / dependence curves (for global XAI plots)
# -----------------------------------------------------------------------------

def _global_curves_ebm(model: Any) -> list[dict]:
    """Per-term shape functions from ``explain_global().data(i)`` (main effects)."""
    gexp = model.explain_global()
    gd = gexp.data()
    importance = {n: float(s) for n, s in zip(gd["names"], gd["scores"])}
    feat_names = list(model.feature_names_in_)
    feat_types = list(model.feature_types_in_)

    curves: list[dict] = []
    for i, term in enumerate(model.term_features_):
        if len(term) != 1:
            continue  # skip interaction terms
        fname = feat_names[term[0]]
        d = gexp.data(i)
        if feat_types[term[0]] == "continuous":
            # names = bin edges (n+1), scores = per-bin contribution (n)
            edges = [float(v) for v in d["names"]]
            y = [float(s) for s in d["scores"]]
            x = [round((edges[k] + edges[k + 1]) / 2, 6) for k in range(len(y))]
            kind = "continuous"
        else:
            # nominal: names = categories, scores = per-category contribution
            x = [str(v) for v in d["names"]]
            y = [float(s) for s in d["scores"]]
            kind = "categorical"
        curves.append({
            "feature": fname,
            "kind": kind,
            "x": x,
            "y": [round(v, 6) for v in y],
            "importance": round(importance.get(fname, 0.0), 6),
        })
    return curves


def _global_curves_xgb(model: Any, X_train: pd.DataFrame) -> list[dict]:
    """SHAP dependence cloud per feature (one point per training row, log space)."""
    explainer = _get_shap_explainer(model)
    sv = explainer(X_train)
    values = np.asarray(sv.values)
    data = np.asarray(sv.data)

    curves: list[dict] = []
    for j, feat in enumerate(X_train.columns):
        kind = "categorical" if str(X_train[feat].dtype) == "category" else "continuous"
        curves.append({
            "feature": str(feat),
            "kind": kind,
            "x": [_feat_value(v) for v in data[:, j]],
            "y": [round(float(v), 6) for v in values[:, j]],
            "importance": round(float(np.abs(values[:, j]).mean()), 6),
        })
    return curves


def build_global_curves(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
) -> list[dict]:
    """Per-feature global shape / dependence curves (JSON serialisable).

    Each entry::

        {"feature": str, "kind": "continuous" | "categorical",
         "x": [...], "y": [...], "importance": float}

    EBM: shape function per main-effect term. Continuous features give bin
    midpoints (x) and the per-bin contribution in log space (y); categorical
    features give the category label (x) and its contribution (y).
    XGB: SHAP dependence cloud per feature, one point per training row
    (x = feature value, y = shap value in log space); ``importance`` = mean |shap|.
    The full per-feature (x, y) columns together form the beeswarm raw matrix.
    """
    if model_name == "ebm":
        return _global_curves_ebm(model)
    if model_name == "xgb":
        return _global_curves_xgb(model, X_train)
    raise ValueError(f"Unknown model_name: {model_name!r}")


def save_global_curves(
    curves: list[dict], model_name: str, out_dir: Path | None = None
) -> list[Path]:
    """Write one JSON per feature: ``global_curve_{model}_{feature}.json``."""
    return [
        save_explanation(
            c, f"global_curve_{model_name}_{c['feature']}.json", out_dir=out_dir
        )
        for c in curves
    ]
