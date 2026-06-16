"""
utils/data.py – Daten-Loading mit zentral definierten Datentypen.

Wichtig: train.csv / test.csv enthalten nach dem CSV-Roundtrip keine
category-Dtypes mehr. Diese Datei ist die *einzige* Stelle, an der
die Dtypes wiederhergestellt werden – damit EBM und XGBoost
(mit enable_categorical=True) konsistent dieselben Spalten als
kategorisch behandeln.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from . import DATA_DIR, RANDOM_STATE

# -----------------------------------------------------------------------------
# Spalten-Schema
# -----------------------------------------------------------------------------
# Nominale (ungeordnete) Kategorien
NOMINAL_COLS: list[str] = [
    "weathersit",  # 1..4 (Wetterlage)
]

# Ordinale Kategorien (haben natürliche Ordnung; für EBM/XGBoost
# als category ausreichend – Reihenfolge spiegelt sich in den Codes wider)
ORDINAL_COLS: list[str] = [
    "mnth",     # 1..12
    "hr",       # 0..23
    "weekday",  # 0..6
]

CATEGORICAL_COLS: list[str] = NOMINAL_COLS + ORDINAL_COLS

NUMERIC_COLS: list[str] = [
    "yr",        # 0: 2011 / 1: 2012 — binär, als 0/1 behandelt
    "holiday",   # 0: kein Feiertag / 1: Feiertag — binär
    "temp",      # normalisierte Temperatur
    "hum",       # Luftfeuchtigkeit (normalisiert)
    "windspeed", # Windgeschwindigkeit (normalisiert)
]
# Entfernte Features (redundant):
#   season     → vollständig aus mnth ableitbar (Monate 3–5 = Frühling etc.)
#   workingday → vollständig aus weekday + holiday ableitbar

TARGET_COL: str = "cnt"

# Spalten, die nicht als Feature verwendet werden sollen.
# Falls sie noch in train.csv/test.csv vorhanden sind, werden sie verworfen.
DROP_COLS: list[str] = [
    "instant",     # Zeilen-ID
    "dteday",      # Datum (Leakage-frei nur als Quelle für hr/yr/mnth/weekday verwendbar)
    "casual",      # Target-Leakage (Teil von cnt)
    "registered",  # Target-Leakage (Teil von cnt)
    "season",      # redundant (aus mnth ableitbar) — safety net für alte CSVs
    "workingday",  # redundant (aus weekday+holiday ableitbar) — safety net
    "cnt_log1p",   # abgeleitetes Target, kein Feature
    "atemp",       # nahezu perfekt korreliert mit temp (r≈0.99)
]

FEATURE_COLS: list[str] = CATEGORICAL_COLS + NUMERIC_COLS


# -----------------------------------------------------------------------------
# Dtype-Wiederherstellung
# -----------------------------------------------------------------------------
def _apply_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Stellt category-Dtypes für kategoriale Spalten wieder her."""
    df = df.copy()

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            # float -> int -> category: avoids '1.0' categories and is compatible
            # with XGBoost 3.x (Int64 nullable dtype breaks enable_categorical).
            df[col] = df[col].astype(float).astype(int).astype("category")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype("int64")

    return df


def _drop_unused(df: pd.DataFrame) -> pd.DataFrame:
    """Entfernt ID-/Leakage-Spalten falls noch vorhanden."""
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def load_train_test(
    data_dir: Path | str | None = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Lädt train.csv und test.csv mit korrekten Datentypen.

    Returns
    -------
    X_train, y_train, X_test, y_test
        Features als DataFrame mit category-Dtypes für kategoriale Spalten,
        Target als Series.
    """
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"train.csv nicht gefunden unter {train_path}. "
            "Bitte zuerst Notebook 01_Preprocessing.ipynb ausführen."
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"test.csv nicht gefunden unter {test_path}. "
            "Bitte zuerst Notebook 01_Preprocessing.ipynb ausführen."
        )

    train = _apply_dtypes(_drop_unused(pd.read_csv(train_path)))
    test = _apply_dtypes(_drop_unused(pd.read_csv(test_path)))

    if TARGET_COL not in train.columns or TARGET_COL not in test.columns:
        raise ValueError(
            f"Target-Spalte '{TARGET_COL}' fehlt in train.csv oder test.csv."
        )

    X_train = train[FEATURE_COLS].copy()
    y_train = train[TARGET_COL].copy()
    X_test = test[FEATURE_COLS].copy()
    y_test = test[TARGET_COL].copy()

    return X_train, y_train, X_test, y_test


def sample_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    n: int,
    seed: int = RANDOM_STATE,
) -> list[int]:
    """
    Returns n unique positional row-indices sampled from (X, y) with
    proportional stratification over cnt-quintile, time-of-day block
    (hr // 6), and weathersit.

    Each non-empty stratum receives at least 1 draw; the total is
    adjusted to hit exactly n.  Deterministic given the same seed.

    Parameters
    ----------
    X    : feature DataFrame (must contain 'hr' and 'weathersit')
    y    : target Series (cnt, used to compute quintile bins)
    n    : number of indices to draw (must be ≤ len(X))
    seed : RNG seed for reproducibility

    Returns
    -------
    Sorted list of n unique integer row-indices (positional, 0-based,
    matching X.index).
    """
    if n > len(X):
        raise ValueError(f"n={n} exceeds dataset size {len(X)}")

    work = pd.DataFrame({
        "cnt_q":   pd.qcut(y, q=5, labels=False, duplicates="drop"),
        "time_b":  (X["hr"].astype(int) // 6).astype(int),
        "weather": X["weathersit"].astype(int),
    }, index=X.index)
    work["stratum"] = (
        work["cnt_q"].astype(str) + "_"
        + work["time_b"].astype(str) + "_"
        + work["weather"].astype(str)
    )

    strata_sizes = work.groupby("stratum").size()
    raw = strata_sizes / len(work) * n

    # When n is large enough for at least 1 per stratum, enforce it.
    if n >= len(strata_sizes):
        raw = raw.clip(lower=1.0)

    # Hamilton largest-remainder method — guarantees total == n, all values >= 0
    alloc = np.floor(raw).astype(int)
    remainder = n - int(alloc.sum())
    for s in (raw - alloc).sort_values(ascending=False).index[:remainder]:
        alloc[s] += 1

    rng = np.random.default_rng(seed)
    result: list[int] = []
    for stratum, k in alloc.items():
        pool = work.index[work["stratum"] == stratum].tolist()
        k = min(k, len(pool))
        chosen = rng.choice(pool, size=k, replace=False).tolist()
        result.extend(chosen)

    return sorted(result)


def get_feature_lists() -> dict[str, list[str]]:
    """Gibt die zentrale Feature-Klassifikation zurück (für Notebooks/Plots)."""
    return {
        "categorical": list(CATEGORICAL_COLS),
        "nominal": list(NOMINAL_COLS),
        "ordinal": list(ORDINAL_COLS),
        "numeric": list(NUMERIC_COLS),
        "all_features": list(FEATURE_COLS),
        "target": TARGET_COL,
    }
