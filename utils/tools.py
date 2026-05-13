"""
utils/tools.py – Tool-Definitionen und ToolBox für die Tool-Use-Pipeline (Notebook 06).

TOOL_DEFINITIONS: Schemas im Anthropic-Format, die dem Modell übergeben werden.
ToolBox:          Kapselt Modell + Test-Daten; führt Tool-Aufrufe aus.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data import CATEGORICAL_COLS, NUMERIC_COLS
from .explanations import FEATURE_SCHEMA


# -----------------------------------------------------------------------------
# Tool-Schemas
# -----------------------------------------------------------------------------
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "get_feature_schema",
        "description": (
            "Gibt Metadaten zu allen Features zurück (Typ, Beschreibung, "
            "Wertebereich, Kategorien-Mapping). Nützlich als erster Schritt, "
            "um zu verstehen, was die Eingabevariablen bedeuten."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_feature_importance",
        "description": (
            "Gibt die globale Feature-Importance des Modells zurück, sortiert "
            "absteigend nach Wichtigkeit. Für XGBoost: SHAP-basierte Importance "
            "(mean |SHAP|). Für EBM: Term-Importance aus den gelernten Funktionen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_k": {
                    "type": "integer",
                    "description": "Anzahl der zurückzugebenden Top-Features (Default: alle).",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_prediction",
        "description": (
            "Gibt die Modellvorhersage (Anzahl Fahrräder) für eine konkrete "
            "Feature-Kombination zurück. Nützlich für Was-wäre-wenn-Szenarien."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "object",
                    "description": (
                        "Mapping von Feature-Name zu Wert. Kategoriale Features "
                        "(mnth, hr, weekday, weathersit) als Integer, "
                        "binäre Features (yr, holiday) als Integer (0 oder 1), "
                        "numerische Features (temp, hum, windspeed) als Float."
                    ),
                }
            },
            "required": ["features"],
        },
    },
    {
        "name": "get_shap_values",
        "description": (
            "Gibt die lokalen Beiträge (SHAP für XGBoost, EBM-Terme) einer "
            "Testinstanz zurück — sortiert nach absolutem Betrag. "
            "Zeigt, welche Features die Vorhersage erhöhen oder senken."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "integer",
                    "description": "Positions-Index der Testinstanz im Test-Set (0-basiert).",
                }
            },
            "required": ["instance_id"],
        },
    },
    {
        "name": "get_partial_dependence",
        "description": (
            "Berechnet die Partial-Dependence-Kurve für ein einzelnes Feature: "
            "zeigt, wie sich die durchschnittliche Vorhersage ändert, wenn "
            "dieses Feature variiert wird (alle anderen Merkmale unverändert)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "feature": {
                    "type": "string",
                    "description": "Name des Features (z.B. 'hr', 'temp', 'weathersit').",
                },
                "n_grid_points": {
                    "type": "integer",
                    "description": "Stützstellen für numerische Features (Default: 20).",
                },
            },
            "required": ["feature"],
        },
    },
    {
        "name": "get_feature_value_context",
        "description": (
            "Gibt Kontext zum Wert eines Features bei einer Testinstanz: "
            "Percentile im Trainingsset, Minimum, Maximum, Mittelwert. "
            "Hilft einzuordnen, ob der Wert typisch oder außergewöhnlich ist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "integer",
                    "description": "Positions-Index der Testinstanz im Test-Set (0-basiert).",
                },
                "feature": {
                    "type": "string",
                    "description": "Name des Features (z.B. 'temp', 'hr', 'windspeed').",
                },
            },
            "required": ["instance_id", "feature"],
        },
    },
    {
        "name": "get_similar_instances",
        "description": (
            "Sucht die k ähnlichsten Trainingsinstanzen zu einer Testinstanz "
            "(euklidische Distanz auf normierten Features). "
            "Zeigt, wie das Modell bei vergleichbaren Situationen vorhersagt — "
            "nützlich für Plausibilitätsprüfung und Kontextualisierung."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "integer",
                    "description": "Positions-Index der Testinstanz im Test-Set (0-basiert).",
                },
                "k": {
                    "type": "integer",
                    "description": "Anzahl ähnlicher Instanzen (Default: 5).",
                },
            },
            "required": ["instance_id"],
        },
    },
    {
        "name": "get_counterfactual_prediction",
        "description": (
            "Berechnet eine kontrafaktische Vorhersage: Wie verändert sich die "
            "Vorhersage, wenn ein oder mehrere Features einen anderen Wert hätten? "
            "Nützlich für Was-wäre-wenn-Analysen (z.B. 'Was wäre bei 10°C mehr?')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "integer",
                    "description": "Positions-Index der Basis-Testinstanz im Test-Set (0-basiert).",
                },
                "changes": {
                    "type": "object",
                    "description": (
                        "Mapping von Feature-Name zu neuem Wert. "
                        "Nur die geänderten Features müssen angegeben werden. "
                        "Kategoriale Features als Integer, numerische als Float."
                    ),
                },
            },
            "required": ["instance_id", "changes"],
        },
    },
]


# -----------------------------------------------------------------------------
# ToolBox
# -----------------------------------------------------------------------------
class ToolBox:
    """
    Kapselt Modell + Test-Daten und führt Tool-Aufrufe aus.
    Wird in Notebook 06 mit dem geladenen Modell instanziiert.
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
    ):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.call_log: list[dict[str, Any]] = []
        self._shap_explainer: Any = None  # lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        """Führt einen Tool-Aufruf aus, loggt ihn und gibt das Ergebnis zurück."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            result = {"error": f"Unbekanntes Tool: '{name}'"}
        else:
            try:
                result = handler(**arguments)
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}

        self.call_log.append({
            "tool": name,
            "arguments": arguments,
            "result_preview": _preview(result),
        })
        return result

    # ------------------------------------------------------------------
    # Tool-Implementierungen
    # ------------------------------------------------------------------
    def _tool_get_feature_schema(self) -> dict:
        return FEATURE_SCHEMA

    def _tool_get_feature_importance(self, top_k: int | None = None) -> list[dict]:
        if self.model_name == "xgb":
            # SHAP-basierte Importance (mean |SHAP|) über Trainingsset
            explainer = self._get_shap_explainer()
            shap_vals = explainer.shap_values(self.X_train)
            scores = np.abs(shap_vals).mean(axis=0)
            names = self.X_train.columns.tolist()
        else:
            # EBM: nur Haupteffekte (keine Interaktionen)
            all_names = self.model.term_names_
            all_scores = self.model.term_importances()
            names, scores = [], []
            for n, s in zip(all_names, all_scores):
                if " & " not in n:
                    names.append(n)
                    scores.append(float(s))

        ranked = sorted(
            zip(names, scores), key=lambda x: -float(x[1])
        )
        if top_k is not None:
            ranked = ranked[:top_k]

        return [
            {"rank": i + 1, "feature": f, "importance": round(float(s), 5)}
            for i, (f, s) in enumerate(ranked)
        ]

    def _tool_get_prediction(self, features: dict[str, Any]) -> dict:
        df = self._build_input_df(features)
        pred = float(self.model.predict(df)[0])
        filled_by_mode = [c for c in self.X_test.columns if c not in features]
        result = {
            "prediction": round(pred, 2),
            "unit": "Fahrräder pro Stunde",
            "features_used": features,
        }
        if filled_by_mode:
            result["features_filled_by_mode"] = filled_by_mode
        return result

    def _tool_get_shap_values(self, instance_id: int) -> dict:
        if instance_id < 0 or instance_id >= len(self.X_test):
            return {"error": f"instance_id {instance_id} außerhalb des Test-Sets (0–{len(self.X_test)-1})."}

        instance = self.X_test.iloc[[instance_id]]
        y_true = float(self.y_test.iloc[instance_id])
        pred = float(self.model.predict(instance)[0])

        if self.model_name == "xgb":
            explainer = self._get_shap_explainer()
            shap_vals = explainer.shap_values(instance)[0]
            base_value = float(explainer.expected_value)
            contribs = {
                col: round(float(v), 5)
                for col, v in zip(self.X_test.columns, shap_vals)
            }
        else:
            exp = self.model.explain_local(instance)
            d = exp.data(0)
            base_value = float(d["extra"]["scores"][0])
            contribs = {
                n: round(float(s), 5)
                for n, s in zip(d["names"], d["scores"])
                if " & " not in n
            }

        feature_values = {
            col: _feat_val(instance.iloc[0][col])
            for col in self.X_test.columns
        }
        sorted_contribs = sorted(
            [
                {
                    "feature": f,
                    "value": feature_values.get(f),
                    "value_human": _humanize(f, feature_values.get(f)),
                    "contribution": v,
                }
                for f, v in contribs.items()
            ],
            key=lambda x: -abs(x["contribution"]),
        )

        return {
            "instance_id": instance_id,
            "y_true": y_true,
            "prediction": round(pred, 2),
            "base_value": round(base_value, 5),
            "contribution_space": "log",
            "note": "Positive Beiträge erhöhen, negative senken die Vorhersage (multiplikativ via exp).",
            "contributions": sorted_contribs,
        }

    def _tool_get_partial_dependence(
        self, feature: str, n_grid_points: int = 20
    ) -> dict:
        if feature not in self.X_test.columns:
            return {"error": f"Feature '{feature}' nicht bekannt. Verfügbar: {list(self.X_test.columns)}"}

        X_copy = self.X_test.copy()

        if feature in CATEGORICAL_COLS:
            grid = sorted(self.X_test[feature].cat.categories.tolist())
        else:
            lo, hi = self.X_test[feature].min(), self.X_test[feature].max()
            grid = np.linspace(lo, hi, n_grid_points).tolist()

        results = []
        for val in grid:
            X_copy[feature] = val
            if feature in CATEGORICAL_COLS:
                X_copy[feature] = X_copy[feature].astype(
                    self.X_test[feature].dtype
                )
            avg_pred = float(self.model.predict(X_copy).mean())
            raw_val = int(val) if feature in CATEGORICAL_COLS else round(float(val), 4)
            entry = {
                "value": raw_val,
                "avg_prediction": round(avg_pred, 3),
            }
            human = _humanize(feature, raw_val)
            if human is not None:
                entry["value_human"] = human
            results.append(entry)

        return {
            "feature": feature,
            "type": "categorical" if feature in CATEGORICAL_COLS else "numerical",
            "partial_dependence": results,
        }

    def _tool_get_feature_value_context(self, instance_id: int, feature: str) -> dict:
        if instance_id < 0 or instance_id >= len(self.X_test):
            return {"error": f"instance_id {instance_id} außerhalb des Test-Sets (0–{len(self.X_test)-1})."}
        if feature not in self.X_train.columns:
            return {"error": f"Feature '{feature}' nicht bekannt. Verfügbar: {list(self.X_train.columns)}"}

        val = self.X_test.iloc[instance_id][feature]
        train_col = self.X_train[feature]

        if feature in CATEGORICAL_COLS:
            counts = train_col.value_counts().sort_index()
            total = len(train_col)
            distribution = {int(k): {"count": int(v), "pct": round(100 * v / total, 1)}
                            for k, v in counts.items()}
            return {
                "feature": feature,
                "instance_value": int(val),
                "instance_value_human": _humanize(feature, val),
                "type": "categorical",
                "training_distribution": distribution,
            }
        else:
            train_vals = train_col.astype(float)
            percentile = float(np.mean(train_vals <= float(val)) * 100)
            return {
                "feature": feature,
                "instance_value": round(float(val), 4),
                "instance_value_human": _humanize(feature, val),
                "type": "numerical",
                "percentile_in_train": round(percentile, 1),
                "train_min": round(float(train_vals.min()), 4),
                "train_max": round(float(train_vals.max()), 4),
                "train_mean": round(float(train_vals.mean()), 4),
                "train_std": round(float(train_vals.std()), 4),
            }

    def _tool_get_similar_instances(self, instance_id: int, k: int = 5) -> dict:
        if instance_id < 0 or instance_id >= len(self.X_test):
            return {"error": f"instance_id {instance_id} außerhalb des Test-Sets (0–{len(self.X_test)-1})."}

        # Numerische Kodierung für Distanzberechnung
        def _encode(df: pd.DataFrame) -> np.ndarray:
            parts = []
            for col in df.columns:
                if col in CATEGORICAL_COLS:
                    parts.append(df[col].cat.codes.values.astype(float))
                else:
                    parts.append(df[col].astype(float).values)
            X = np.column_stack(parts)
            # Min-Max Normierung auf Trainingsset
            return X

        X_train_enc = _encode(self.X_train)
        x_test_enc = _encode(self.X_test.iloc[[instance_id]])[0]

        col_min = X_train_enc.min(axis=0)
        col_max = X_train_enc.max(axis=0)
        col_range = np.where(col_max > col_min, col_max - col_min, 1.0)

        X_train_norm = (X_train_enc - col_min) / col_range
        x_test_norm = (x_test_enc - col_min) / col_range

        dists = np.sqrt(((X_train_norm - x_test_norm) ** 2).sum(axis=1))
        top_k_idx = np.argsort(dists)[:k]

        results = []
        for idx in top_k_idx:
            row = self.X_train.iloc[idx]
            fv = {col: _feat_val(row[col]) for col in self.X_train.columns}
            fv_human = {col: h for col in fv if (h := _humanize(col, fv[col])) is not None}
            pred = float(self.model.predict(self.X_train.iloc[[idx]])[0])
            results.append({
                "train_index": int(idx),
                "distance": round(float(dists[idx]), 4),
                "feature_values": fv,
                "feature_values_human": fv_human,
                "prediction": round(pred, 2),
            })

        return {
            "instance_id": instance_id,
            "k": k,
            "similar_instances": results,
        }

    def _tool_get_counterfactual_prediction(
        self, instance_id: int, changes: dict[str, Any]
    ) -> dict:
        if instance_id < 0 or instance_id >= len(self.X_test):
            return {"error": f"instance_id {instance_id} außerhalb des Test-Sets (0–{len(self.X_test)-1})."}

        base_instance = self.X_test.iloc[[instance_id]].copy()
        base_pred = float(self.model.predict(base_instance)[0])

        cf_instance = base_instance.copy()
        applied_changes = {}
        for feat, new_val in changes.items():
            if feat not in cf_instance.columns:
                return {"error": f"Feature '{feat}' nicht bekannt."}
            if feat in CATEGORICAL_COLS:
                cf_instance[feat] = pd.Categorical(
                    [int(float(new_val))],
                    categories=self.X_test[feat].cat.categories,
                )
                applied_changes[feat] = int(float(new_val))
            else:
                cf_instance[feat] = float(new_val)
                applied_changes[feat] = float(new_val)

        cf_pred = float(self.model.predict(cf_instance)[0])
        delta = cf_pred - base_pred

        return {
            "instance_id": instance_id,
            "base_prediction": round(base_pred, 2),
            "counterfactual_prediction": round(cf_pred, 2),
            "delta": round(delta, 2),
            "applied_changes": applied_changes,
            "unit": "Fahrräder pro Stunde",
        }

    # ------------------------------------------------------------------
    # Interne Helfer
    # ------------------------------------------------------------------
    def _get_shap_explainer(self) -> Any:
        if self._shap_explainer is None:
            import shap
            self._shap_explainer = shap.TreeExplainer(self.model)
        return self._shap_explainer

    def _build_input_df(self, features: dict[str, Any]) -> pd.DataFrame:
        """Baut einen validen DataFrame mit korrekten Dtypes aus einem Feature-Dict."""
        row: dict[str, Any] = {}
        for col in self.X_test.columns:
            val = features.get(col)
            if val is None:
                modes = self.X_test[col].mode()
                val = modes.iloc[0] if len(modes) > 0 else 0
            if col in CATEGORICAL_COLS:
                row[col] = pd.Categorical(
                    [int(float(val))],
                    categories=self.X_test[col].cat.categories,
                )
            else:
                row[col] = float(val)
        return pd.DataFrame(row)


# ------------------------------------------------------------------
# Helfer
# ------------------------------------------------------------------
def _preview(obj: Any, max_len: int = 1000) -> Any:
    if isinstance(obj, (dict, list)):
        s = repr(obj)
        return s if len(s) <= max_len else s[:max_len] + "…"
    if isinstance(obj, np.ndarray):
        return f"ndarray(shape={obj.shape})"
    return obj


def _feat_val(val: Any) -> Any:
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def _humanize(feature: str, value: Any) -> str | None:
    """Konvertiert rohe Feature-Werte in lesbare Strings für das LLM."""
    try:
        if feature == "temp":       return f"~{float(value)*41:.1f} °C"
        if feature == "hum":        return f"{float(value)*100:.0f} %"
        if feature == "windspeed":  return f"{float(value)*67:.1f} km/h"
        if feature == "hr":         return f"{int(value):02d}:00 Uhr"
        if feature == "weekday":    return ["So","Mo","Di","Mi","Do","Fr","Sa"][int(value)]
        if feature == "mnth":       return ["","Jan","Feb","Mär","Apr","Mai","Jun",
                                            "Jul","Aug","Sep","Okt","Nov","Dez"][int(value)]
        if feature == "weathersit": return {1:"klar/wenige Wolken",2:"Nebel/bewölkt",
                                            3:"leichter Regen/Schnee",4:"Starkregen/Gewitter"}.get(int(value))
        if feature == "yr":         return "2011" if int(value) == 0 else "2012"
        if feature == "holiday":    return "Feiertag" if int(value) == 1 else "kein Feiertag"
    except (ValueError, TypeError, IndexError):
        pass
    return None
