"""
utils/explanations.py - central feature schema and explanation builders.

The FEATURE_SCHEMA is passed to the LLM in the JSON interface. It is defined here
once so that all notebooks (03, 04, 05, 06) and the Tool Use pipeline (06) use
identical feature descriptions.
"""

from __future__ import annotations

import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import EXPLANATIONS_DIR

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
    except (ValueError, TypeError, IndexError):
        pass
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
