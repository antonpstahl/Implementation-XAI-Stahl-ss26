"""
utils/data.py - data loading with centrally defined dtypes.

Important: after the CSV roundtrip train.csv / test.csv no longer carry category
dtypes. This file is the only place where the dtypes are restored, so that EBM and
XGBoost (with enable_categorical=True) consistently treat the same columns as
categorical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from . import DATA_DIR

# -----------------------------------------------------------------------------
# Column schema
# -----------------------------------------------------------------------------
# Nominal (unordered) categories
NOMINAL_COLS: list[str] = [
    "weathersit",  # 1..4 (weather situation)
]

# Ordinal categories (have a natural order; for EBM/XGBoost category is enough,
# the order is reflected in the codes)
ORDINAL_COLS: list[str] = [
    "mnth",     # 1..12
    "hr",       # 0..23
    "weekday",  # 0..6
]

CATEGORICAL_COLS: list[str] = NOMINAL_COLS + ORDINAL_COLS

NUMERIC_COLS: list[str] = [
    "yr",        # 0: 2011 / 1: 2012, binary, treated as 0/1
    "holiday",   # 0: no holiday / 1: holiday, binary
    "temp",      # normalised temperature
    "hum",       # humidity (normalised)
    "windspeed", # wind speed (normalised)
]
# Removed features (redundant):
#   season     fully derivable from mnth (months 3 to 5 = spring etc.)
#   workingday fully derivable from weekday + holiday

TARGET_COL: str = "cnt"

# Columns that must not be used as a feature.
# If they are still present in train.csv/test.csv they are dropped.
DROP_COLS: list[str] = [
    "instant",     # row ID
    "dteday",      # date (leakage free only as a source for hr/yr/mnth/weekday)
    "casual",      # target leakage (part of cnt)
    "registered",  # target leakage (part of cnt)
    "season",      # redundant (derivable from mnth), safety net for old CSVs
    "workingday",  # redundant (derivable from weekday+holiday), safety net
    "cnt_log1p",   # derived target, not a feature
    "atemp",       # almost perfectly correlated with temp (r approx 0.99)
]

FEATURE_COLS: list[str] = CATEGORICAL_COLS + NUMERIC_COLS


# -----------------------------------------------------------------------------
# Dtype-Wiederherstellung
# -----------------------------------------------------------------------------
def _apply_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Restore category dtypes for the categorical columns."""
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
    """Drop ID / leakage columns if still present."""
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
    Load train.csv and test.csv with correct dtypes.

    Returns
    -------
    X_train, y_train, X_test, y_test
        Features as a DataFrame with category dtypes for the categorical columns,
        target as a Series.
    """
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"train.csv not found at {train_path}. "
            "Please run notebook 01_Data_Preprocessing.ipynb first."
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"test.csv not found at {test_path}. "
            "Please run notebook 01_Data_Preprocessing.ipynb first."
        )

    train = _apply_dtypes(_drop_unused(pd.read_csv(train_path)))
    test = _apply_dtypes(_drop_unused(pd.read_csv(test_path)))

    if TARGET_COL not in train.columns or TARGET_COL not in test.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' missing in train.csv or test.csv."
        )

    X_train = train[FEATURE_COLS].copy()
    y_train = train[TARGET_COL].copy()
    X_test = test[FEATURE_COLS].copy()
    y_test = test[TARGET_COL].copy()

    return X_train, y_train, X_test, y_test



def get_feature_lists() -> dict[str, list[str]]:
    """Return the central feature classification (for notebooks/plots)."""
    return {
        "categorical": list(CATEGORICAL_COLS),
        "nominal": list(NOMINAL_COLS),
        "ordinal": list(ORDINAL_COLS),
        "numeric": list(NUMERIC_COLS),
        "all_features": list(FEATURE_COLS),
        "target": TARGET_COL,
    }
