"""Stratified evaluation of global feature explanations (Phase G3).

Merges the deterministic rubric scores (:mod:`utils.rubric`, no LLM) with the
reference-based judge scores (:func:`utils.eval.run_global_judge`) and reports
them stratified by shape type (monotonic / non-monotonic / categorical /
near-flat) x modality (json / vision / tooluse / template) — the level the
supervisor asked results to be broken down at, rather than a single aggregate.

Also provides the judge-robustness figure for the global track: Krippendorff's
alpha (interval metric) across judge vendors, computed with the exact estimator
promoted from NB 05 (:func:`utils.stats.krippendorff_alpha_interval`).

Everything here is pure post-processing of files already on disk — no API calls.
The judge tables are optional: with only the rubric CSV present the report still
runs (rubric columns only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils import RESULTS_DIR
from utils.stats import krippendorff_alpha_interval

FORM_TYPES = ["monotonic", "non-monotonic", "categorical", "near-flat"]
MODALITY_ORDER = ["template", "json", "vision", "tooluse"]
JUDGE_CRITERIA = ["faithfulness", "clarity", "completeness"]
MERGE_KEYS = ["form_pipeline", "xai_model", "feature"]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_rubric(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Load the deterministic rubric scores (results/global_rubric.csv)."""
    return pd.read_csv(results_dir / "global_rubric.csv")


def load_judge(subdir: str = "global_judge",
               results_dir: Path = RESULTS_DIR) -> Optional[pd.DataFrame]:
    """Load reference-based judge scores from results/{subdir}/*.json.

    Returns None if the directory is absent or empty (judge not run yet).
    """
    d = results_dir / subdir
    files = sorted(d.glob("*.json")) if d.is_dir() else []
    if not files:
        return None
    rows = []
    for p in files:
        r = json.loads(p.read_text())
        rows.append({k: r.get(k) for k in
                     MERGE_KEYS + ["form_type", "judge_model"] + JUDGE_CRITERIA})
    return pd.DataFrame(rows)


def merge_scores(rubric_df: pd.DataFrame,
                 judge_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Join rubric and judge scores on (modality, model, feature)."""
    if judge_df is None:
        return rubric_df.copy()
    jcols = MERGE_KEYS + JUDGE_CRITERIA
    return rubric_df.merge(judge_df[jcols], on=MERGE_KEYS, how="left")


# ---------------------------------------------------------------------------
# stratified reporting
# ---------------------------------------------------------------------------

def stratified_table(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """Mean of ``value`` per modality (rows) x form type (columns), with an
    overall column and an n-count table alongside.

    Form types keep the fixed FORM_TYPES order; modalities the MODALITY_ORDER.
    """
    present_forms = [f for f in FORM_TYPES if f in set(df["form_type"])]
    pivot = df.pivot_table(index="form_pipeline", columns="form_type",
                           values=value, aggfunc="mean")
    pivot = pivot.reindex(index=[m for m in MODALITY_ORDER if m in pivot.index],
                          columns=present_forms)
    pivot["overall"] = df.groupby("form_pipeline")[value].mean().reindex(pivot.index)
    return pivot.round(3)


def counts_table(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """Non-NaN count of ``value`` per modality × form type (same layout as
    :func:`stratified_table`). Judge columns are sparse until the judge has run,
    so counts must be per-value, not borrowed from the rubric."""
    present_forms = [f for f in FORM_TYPES if f in set(df["form_type"])]
    sub = df[df[value].notna()]
    piv = sub.pivot_table(index="form_pipeline", columns="form_type",
                          values=value, aggfunc="count")
    return piv.reindex(index=[m for m in MODALITY_ORDER if m in piv.index],
                       columns=present_forms).fillna(0)


def report(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build every stratified table available given the columns in ``df``.

    Always includes the rubric ``total``; adds judge criteria when present.
    """
    tables = {"rubric_total": stratified_table(df, "total")}
    for crit in JUDGE_CRITERIA:
        if crit in df.columns and df[crit].notna().any():
            tables[f"judge_{crit}"] = stratified_table(df, crit)
    return tables


def rubric_judge_correlation(df: pd.DataFrame) -> dict:
    """Spearman correlation between the deterministic rubric total and the
    judge's faithfulness — a validity check that the LLM-free rubric and the
    reference-based judge rank explanations consistently.
    """
    if "faithfulness" not in df.columns:
        return {}
    from scipy.stats import spearmanr
    sub = df[["total", "faithfulness"]].dropna()
    if len(sub) < 3:
        return {"n": int(len(sub)), "spearman_rho": np.nan, "p_value": np.nan}
    rho, p = spearmanr(sub["total"], sub["faithfulness"])
    return {"n": int(len(sub)), "spearman_rho": round(float(rho), 3),
            "p_value": round(float(p), 4)}


# ---------------------------------------------------------------------------
# judge robustness: cross-vendor Krippendorff alpha
# ---------------------------------------------------------------------------

def cross_vendor_alpha(subdirs: list[str], criterion: str = "faithfulness",
                       results_dir: Path = RESULTS_DIR) -> float:
    """Krippendorff's alpha (interval) between judge vendors on one criterion.

    ``subdirs`` are the per-vendor judge output folders (e.g.
    ``["global_judge", "global_judge_openai"]``). Each explanation is a unit,
    each vendor a rater; the shared units are aligned on (modality, model,
    feature). Returns NaN if fewer than two vendors have overlapping scores.
    """
    frames = []
    for sub in subdirs:
        jd = load_judge(sub, results_dir=results_dir)
        if jd is None:
            continue
        frames.append(jd.set_index(MERGE_KEYS)[criterion].rename(sub))
    if len(frames) < 2:
        return float("nan")
    wide = pd.concat(frames, axis=1).dropna()
    if wide.empty:
        return float("nan")
    # reliability matrix: (n_raters=vendors, n_units=explanations)
    return krippendorff_alpha_interval(wide.to_numpy().T)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_table(name: str, tab: pd.DataFrame, counts: pd.DataFrame) -> None:
    print(f"\n### {name} (mean; n in parentheses)")
    forms = [c for c in tab.columns if c != "overall"]
    hdr = f"{'modality':10s} " + " ".join(f"{f:16s}" for f in forms) + " overall"
    print(hdr)
    print("-" * len(hdr))
    for mod in tab.index:
        cells = []
        for f in forms:
            v = tab.loc[mod, f]
            n = counts.loc[mod, f] if (mod in counts.index and f in counts.columns) else 0
            cells.append(f"{v:.2f} (n={int(n)})" if pd.notna(v) else "   -      ")
        ov = tab.loc[mod, "overall"]
        line = f"{mod:10s} " + " ".join(f"{c:16s}" for c in cells)
        print(line + (f" {ov:.2f}" if pd.notna(ov) else ""))


def _main() -> None:
    rubric = load_rubric()
    judge = load_judge()
    df = merge_scores(rubric, judge)

    # value column behind each report table (rubric_total -> "total", judge_x -> "x")
    value_of = {"rubric_total": "total",
                **{f"judge_{c}": c for c in JUDGE_CRITERIA}}
    for name, tab in report(df).items():
        _print_table(name, tab, counts_table(df, value_of[name]))

    if judge is not None:
        corr = rubric_judge_correlation(df)
        print(f"\nRubric-vs-judge (faithfulness) Spearman: rho={corr.get('spearman_rho')} "
              f"(p={corr.get('p_value')}, n={corr.get('n')})")
        vendors = sorted({p.name for p in RESULTS_DIR.glob("global_judge*") if p.is_dir()})
        if len(vendors) >= 2:
            print("\nCross-vendor Krippendorff alpha (interval):")
            for crit in JUDGE_CRITERIA:
                a = cross_vendor_alpha(vendors, crit)
                print(f"  {crit:13s}: {a:.3f}")
    else:
        print("\n(no judge output yet — run utils.eval.run_global_judge to add "
              "judge columns and robustness)")


if __name__ == "__main__":
    _main()
