"""
utils/models.py - save and load the trained models.

Uses joblib. Models are stored under the naming scheme
``{model_type}_{loss_key}.pkl`` (for example ``xgb_poisson_log.pkl``) and
addressed only via ``save_model`` / ``load_model`` so that all notebooks access
identical model artefacts consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np

from . import MODELS_DIR

# Valid model types (prefix in the file name).
MODEL_TYPES: tuple[str, ...] = ("xgb", "ebm")


# ---------------------------------------------------------------------------
# Loss-Optionen
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LossOption:
    key: str
    label: str
    description: str
    ebm_objective: str
    xgb_objective: str
    contribution_space: str


LOSS_OPTIONS: Dict[str, LossOption] = {
    "squared_error": LossOption(
        key="squared_error",
        label="Option 1: Squared Error",
        description=(
            "Classic squared loss. Easy to interpret, but not ideal for right "
            "skewed count data, can produce negative predictions."
        ),
        ebm_objective="rmse",
        xgb_objective="reg:squarederror",
        contribution_space="native",
    ),
    "poisson_log": LossOption(
        key="poisson_log",
        label="Option 2: Poisson Deviance (contributions in log space)",
        description=(
            "Poisson deviance loss. Predictions strictly positive via exp(). "
            "Contributions are extracted and interpreted on the log scale."
        ),
        ebm_objective="poisson_deviance",
        xgb_objective="count:poisson",
        contribution_space="log",
    ),
    "poisson_native": LossOption(
        key="poisson_native",
        label="Option 3: Poisson Deviance (contributions approximately in rentals)",
        description=(
            "Identical model to option 2. Contributions are extracted approximately "
            "on the rental scale (XGBoost output_margin=False, EBM analogous)."
        ),
        ebm_objective="poisson_deviance",
        xgb_objective="count:poisson",
        contribution_space="native",
    ),
}


# ---------------------------------------------------------------------------
# Metriken
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics on the original scale."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residuals = y_true - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Mean Poisson deviance (only meaningful for y_pred > 0)
    eps = 1e-8
    pred_pos = np.clip(y_pred, eps, None)
    poisson_deviance = float(
        2.0 * np.mean(y_true * np.log((y_true + eps) / pred_pos) - (y_true - pred_pos))
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "poisson_deviance": poisson_deviance,
        "min_prediction": float(y_pred.min()),
        "n_negative_predictions": int((y_pred < 0).sum()),
    }


# ---------------------------------------------------------------------------
# Save / load  (uniform scheme: {model_type}_{loss_key}.pkl)
# ---------------------------------------------------------------------------

def model_path(model_type: str, loss_key: str,
               models_dir: Path | str | None = None) -> Path:
    """Path of a model artefact under ``{model_type}_{loss_key}.pkl``."""
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Allowed: {MODEL_TYPES}."
        )
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    return models_dir / f"{model_type}_{loss_key}.pkl"


def save_model(model: Any, model_type: str, loss_key: str,
               models_dir: Path | str | None = None) -> Path:
    """Save a model under ``{model_type}_{loss_key}.pkl``."""
    path = model_path(model_type, loss_key, models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(model_type: str, loss_key: str,
               models_dir: Path | str | None = None) -> Any:
    """Load a single model artefact (``xgb`` or ``ebm``)."""
    path = model_path(model_type, loss_key, models_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Please run notebook 02a_Modeling_AllOptions.ipynb first."
        )
    return joblib.load(path)


def load_models(loss_key: str,
                models_dir: Path | str | None = None) -> Tuple[Any, Any]:
    """Load both models of a loss variant. Returns: ``(xgb, ebm)``."""
    return (
        load_model("xgb", loss_key, models_dir),
        load_model("ebm", loss_key, models_dir),
    )
