"""
utils/models.py – Speichern und Laden der trainierten Modelle.

Verwendet joblib. Modelle werden unter dem Namensschema
``{model_type}_{loss_key}.pkl`` abgelegt (z. B. ``xgb_poisson_log.pkl``)
und ausschließlich über ``save_model`` / ``load_model`` adressiert, damit
alle Notebooks konsistent auf identische Modell-Artefakte zugreifen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np

from . import MODELS_DIR

# Gültige Modelltypen (Präfix im Dateinamen).
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
            "Klassische quadratische Verlustfunktion. Einfach zu interpretieren, "
            "aber nicht ideal für rechtsschiefe Count-Daten — kann negative "
            "Vorhersagen liefern."
        ),
        ebm_objective="rmse",
        xgb_objective="reg:squarederror",
        contribution_space="native",
    ),
    "poisson_log": LossOption(
        key="poisson_log",
        label="Option 2: Poisson-Deviance (Beitraege auf Log-Skala)",
        description=(
            "Poisson-Deviance-Verlust. Vorhersagen strikt positiv via exp(). "
            "Beiträge werden auf der Log-Skala extrahiert und interpretiert."
        ),
        ebm_objective="poisson_deviance",
        xgb_objective="count:poisson",
        contribution_space="log",
    ),
    "poisson_native": LossOption(
        key="poisson_native",
        label="Option 3: Poisson-Deviance (Beitraege approximativ auf Ausleihe-Skala)",
        description=(
            "Identisches Modell wie Option 2. Beiträge werden approximativ auf "
            "der Ausleihe-Skala extrahiert (XGBoost output_margin=False, EBM analog)."
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
    """Berechnet Regressionsmetriken auf der Original-Skala."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residuals = y_true - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Mean Poisson deviance (nur für y_pred > 0 sinnvoll)
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
# Speichern / Laden  (einheitliches Schema: {model_type}_{loss_key}.pkl)
# ---------------------------------------------------------------------------

def model_path(model_type: str, loss_key: str,
               models_dir: Path | str | None = None) -> Path:
    """Pfad eines Modell-Artefakts unter ``{model_type}_{loss_key}.pkl``."""
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"Unbekannter model_type {model_type!r}. Erlaubt: {MODEL_TYPES}."
        )
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    return models_dir / f"{model_type}_{loss_key}.pkl"


def save_model(model: Any, model_type: str, loss_key: str,
               models_dir: Path | str | None = None) -> Path:
    """Speichert ein Modell unter ``{model_type}_{loss_key}.pkl``."""
    path = model_path(model_type, loss_key, models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(model_type: str, loss_key: str,
               models_dir: Path | str | None = None) -> Any:
    """Lädt ein einzelnes Modell-Artefakt (``xgb`` oder ``ebm``)."""
    path = model_path(model_type, loss_key, models_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Modell nicht gefunden unter {path}. "
            "Bitte zuerst Notebook 02a_Modeling_AllOptions.ipynb ausführen."
        )
    return joblib.load(path)


def load_models(loss_key: str,
                models_dir: Path | str | None = None) -> Tuple[Any, Any]:
    """Lädt beide Modelle einer Loss-Variante. Returns: ``(xgb, ebm)``."""
    return (
        load_model("xgb", loss_key, models_dir),
        load_model("ebm", loss_key, models_dir),
    )
