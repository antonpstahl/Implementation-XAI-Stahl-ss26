"""utils/global_explain.py - Global beeswarm of the EBM's exact term contributions.

An EBM is additive: ``prediction(x) = intercept + sum_j f_j(x_j)``. Each ``f_j`` is
the global shape function that ``explain_global()`` plots, and the per-instance
contribution ``f_j(x_ij)`` is simply that shape curve evaluated at the data point.
These term contributions are the model's *exact* internal additive components (not a
post-hoc approximation), which makes them the direct analogue of SHAP values - one
contribution per feature per instance - obtained without going through SHAP.

The beeswarm itself is *not* something the EBM provides: it is a reconstructed,
SHAP-style visualisation assembled here from those contributions (row order =
aggregated local importance, x = the shape-curve value per instance, colour = feature
value). So the *contributions* are native/exact; the *plot* is a rebuild. Contrast
with XGB, whose beeswarm relies on TreeSHAP - a post-hoc attribution.
Feasibility: arXiv 2603.17175v1.

`ebm_local_contributions` uses ``model.eval_terms(X)``, the vectorised equivalent of
the per-instance ``model.explain_local(X)`` term scores (verified byte-identical on
the main-effect terms), so it scales to the full training set. Interaction terms are
dropped so the matrix aligns 1:1 with the feature columns (the SHAP-analogue view).

The concordance helpers support the two-layer validation from the revision plan:
primary (confound-free) against the EBM's own global importances, secondary against
the XGB SHAP beeswarm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# EBM per-instance term contributions (SHAP analogue)
# -----------------------------------------------------------------------------

def ebm_local_contributions(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Per-instance main-effect term contributions as an ``[n x features]`` frame.

    The EBM equivalent of a SHAP-value matrix. Columns are the feature main effects
    (interaction terms excluded), aligned to and indexed like ``X``. Values are in
    the model's link (log) space.
    """
    terms = np.asarray(model.eval_terms(X))
    main = [i for i, t in enumerate(model.term_features_) if len(t) == 1]
    cols = [model.feature_names_in_[model.term_features_[i][0]] for i in main]
    return pd.DataFrame(terms[:, main], columns=cols, index=X.index)


# -----------------------------------------------------------------------------
# Ordering / direction (used by the beeswarm and the validation)
# -----------------------------------------------------------------------------

def feature_importance_order(contrib_df: pd.DataFrame) -> list[str]:
    """Features sorted by mean |contribution|, descending (beeswarm row order)."""
    return contrib_df.abs().mean().sort_values(ascending=False).index.tolist()


def _as_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def coloring_directions(
    contrib_df: pd.DataFrame, feature_values: pd.DataFrame
) -> dict[str, int]:
    """Sign of corr(feature value, contribution) per feature.

    +1 = higher feature value -> higher contribution (red-right in the beeswarm),
    -1 = inverse, 0 = no variation / undefined.
    """
    out: dict[str, int] = {}
    for feat in contrib_df.columns:
        v = _as_float(feature_values[feat])
        c = contrib_df[feat].to_numpy(dtype=float)
        if np.nanstd(v) == 0 or np.nanstd(c) == 0:
            out[feat] = 0
            continue
        r = np.corrcoef(v, c)[0, 1]
        out[feat] = int(np.sign(r)) if np.isfinite(r) else 0
    return out


# -----------------------------------------------------------------------------
# Beeswarm plot (SHAP-style, matplotlib)
# -----------------------------------------------------------------------------

def _norm_color(values: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1] clipped at the 5th/95th percentile (as SHAP does)."""
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return np.full_like(v, 0.5)
    lo, hi = np.percentile(finite, 5), np.percentile(finite, 95)
    if hi <= lo:
        return np.full_like(v, 0.5)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def _jitter(x: np.ndarray, width: float = 0.4, nbins: int = 100) -> np.ndarray:
    """Deterministic density-aware vertical jitter (beeswarm shape)."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or hi <= lo:
        return y
    bins = np.floor(nbins * (x - lo) / (hi - lo)).astype(int)
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        n = idx.size
        if n > 1:
            spread = width * min(1.0, n / (x.size / nbins + 1))
            y[idx] = np.linspace(-spread, spread, n)
    return y


def beeswarm_plot(
    contrib_df: pd.DataFrame,
    feature_values: pd.DataFrame,
    out_path: Path | str,
    *,
    max_display: int | None = None,
    title: str = "Beeswarm of EBM term contributions",
) -> Path:
    """SHAP-style beeswarm rebuilt from the EBM's exact term contributions.

    Rows = features sorted by mean |contribution| (most important on top), points =
    instances jittered by local density, coloured by the feature value (low -> high).
    Semantically identical to ``shap.plots.beeswarm``: importance (spread + order)
    plus direction (colour).
    """
    import matplotlib.pyplot as plt

    order = feature_importance_order(contrib_df)
    if max_display is not None:
        order = order[:max_display]
    rows = order[::-1]  # least important first -> most important ends up on top

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.5))
    cmap = plt.get_cmap("coolwarm")
    for row_y, feat in enumerate(rows):
        x = contrib_df[feat].to_numpy(dtype=float)
        colors = _norm_color(_as_float(feature_values[feat]))
        ax.scatter(
            x, row_y + _jitter(x), c=colors, cmap=cmap, vmin=0.0, vmax=1.0,
            s=6, alpha=0.7, linewidths=0, rasterized=True,
        )
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("EBM contribution (log space)")
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, 1.0))
    cbar = fig.colorbar(sm, ax=ax, ticks=[0.0, 1.0], pad=0.01, aspect=30)
    cbar.set_ticklabels(["low", "high"])
    cbar.set_label("Feature value")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# -----------------------------------------------------------------------------
# Concordance (validation)
# -----------------------------------------------------------------------------

def _spearman(rank_a: list[int], rank_b: list[int]) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(rank_a, rank_b).statistic)


def order_concordance(order_a: list[str], order_b: list[str]) -> dict:
    """Agreement of two feature orderings: Spearman rho + top-3 set overlap.

    Used both for the primary check (EBM beeswarm vs EBM ``explain_global``) and the
    secondary check (EBM vs XGB SHAP beeswarm).
    """
    feats = [f for f in order_a if f in order_b]
    ra = [order_a.index(f) for f in feats]
    rb = [order_b.index(f) for f in feats]
    top3 = set(order_a[:3]) & set(order_b[:3])
    return {
        "spearman": _spearman(ra, rb),
        "top3_overlap": len(top3),
        "n_features": len(feats),
    }


def direction_agreement(
    dirs_a: dict[str, int], dirs_b: dict[str, int]
) -> dict:
    """Fraction of features whose colouring direction (sign) agrees."""
    common = [f for f in dirs_a if f in dirs_b and dirs_a[f] != 0 and dirs_b[f] != 0]
    agree = sum(1 for f in common if dirs_a[f] == dirs_b[f])
    return {
        "agreement": agree / len(common) if common else None,
        "n_compared": len(common),
        "disagree": [f for f in common if dirs_a[f] != dirs_b[f]],
    }
