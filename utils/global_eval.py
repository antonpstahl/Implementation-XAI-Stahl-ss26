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


def modality_significance(judge_df: pd.DataFrame,
                          metric: str = "faithfulness") -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests across modalities on a judge metric.

    Observations are paired on (feature, xai_model) — the same feature-curve
    judged under two modalities — so the test respects the matched design.
    Holm-corrected over the pairwise family; Cliff's delta as effect size.
    """
    from utils.stats import wilcoxon_pairwise
    pipelines = [m for m in MODALITY_ORDER if m in set(judge_df["form_pipeline"])]
    return wilcoxon_pairwise(judge_df, pipelines, metric,
                             group_col="form_pipeline",
                             id_cols=("feature", "xai_model"))


def ceiling_flags(df: pd.DataFrame, tol: float = 0.1) -> dict[str, str]:
    """Detect judge criteria with (near) zero variance — non-informative ceilings.

    Returns {criterion: message} only for the criteria that are constant or
    near-constant across all explanations (e.g. completeness pinned at 5).
    """
    flags = {}
    for crit in JUDGE_CRITERIA:
        if crit not in df.columns or not df[crit].notna().any():
            continue
        vals = df[crit].dropna()
        if vals.nunique() == 1:
            flags[crit] = (f"constant at {vals.iloc[0]:.0f} across all "
                           f"{len(vals)} explanations — non-informative "
                           f"(cross-vendor alpha undefined/chance)")
        elif vals.std() < tol:
            flags[crit] = (f"near-constant (mean={vals.mean():.2f}, "
                           f"std={vals.std():.2f}) — treat as ceiling effect")
    return flags


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
# Whole-model (Phase G2b) — the two comparison axes + beeswarm readability
# ---------------------------------------------------------------------------
WHOLE_CONDITION_ORDER = ["json_all", "vision_all", "tooluse_all",
                         "json_beeswarm", "vision_beeswarm"]
# GT sub-fields a representation can carry: the beeswarm shows ranking + direction only,
# so its `structure` (shape/peak) column is not a fair comparison (plan G2b).
_BEESWARM_FIELDS = ["direction", "rank"]
_ALL_FIELDS = ["direction", "rank", "structure"]


def _whole_fair_total(row: pd.Series) -> float:
    """Summary score over only the GT fields the representation can carry.

    ``all`` representations: the full rubric total (direction+rank+structure).
    ``beeswarm`` representations: mean of direction+rank only — scoring a beeswarm on
    shape/peak would penalise it for information it never conveys (plan G2b).
    """
    if row.get("representation") == "beeswarm":
        return round((row["direction"] + row["rank"]) / 2, 4)
    return row["total"]


def load_whole_rubric(results_dir: Path = RESULTS_DIR,
                      split_subdir: str = "global_whole_split") -> Optional[pd.DataFrame]:
    """Rubric-score every whole-model split record and attach its condition axes.

    Reads ``results/{split_subdir}/*.json`` (written by
    :func:`utils.global_whole.write_split_records`), scores each with the same
    deterministic rubric used for G2a, and adds ``modality``/``representation``/
    ``mechanism``/``dropped`` plus a ``fair_total`` (beeswarm scored only on the fields
    it can carry). A **dropped** feature (the whole-model answer omitted it) is recorded
    as a total miss (all sub-scores 0) — the "only 5 of 9 right" measurement. Returns
    None if the directory is absent or empty (04Ge not run with RUN_API yet).
    """
    from utils import rubric  # local import: rubric imports nothing heavy, avoids cycle

    src = Path(results_dir) / split_subdir
    files = sorted(src.glob("*.json")) if src.is_dir() else []
    if not files:
        return None

    rows = []
    for p in files:
        rec = json.loads(p.read_text())
        scored = rubric.score_result_file(p)          # gives form_type for both cases
        if rec.get("dropped"):
            scored.update({"direction": 0.0, "rank": 0.0, "structure": 0.0, "total": 0.0})
        scored.update({
            "condition":      rec["form"],
            "modality":       rec.get("modality"),
            "representation": rec.get("representation"),
            "mechanism":      rec.get("mechanism"),
            "dropped":        bool(rec.get("dropped", False)),
        })
        rows.append(scored)

    df = pd.DataFrame(rows)
    df["fair_total"] = df.apply(_whole_fair_total, axis=1)
    return df


def whole_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Features described (of 9) per condition x model — the "5 of 9" measurement."""
    cov = (df.assign(covered=~df["dropped"])
             .groupby(["condition", "xai_model"])["covered"].sum().unstack())
    return cov.reindex([c for c in WHOLE_CONDITION_ORDER if c in cov.index])


def axis1_representation(df: pd.DataFrame) -> pd.DataFrame:
    """Axis 1 — per GT field, all-plots/curves vs the beeswarm (no aggregate winner).

    Mean of each rubric sub-field (direction / rank / structure) per condition, so the
    reader sees *which fields survive which representation* rather than a single number.
    Read the beeswarm rows on ``direction``/``rank`` only (``structure`` is greyed out by
    ``fair_total``).
    """
    tab = (df.groupby("condition")[_ALL_FIELDS + ["fair_total"]].mean()
             .reindex([c for c in WHOLE_CONDITION_ORDER if c in df["condition"].unique()]))
    return tab.round(3)


def axis2_mechanism(df: pd.DataFrame, value: str = "total") -> pd.DataFrame:
    """Axis 2 — push (vision_all/json_all) vs pull (tooluse_all) at full information.

    Restricted to the ``all`` conditions (constant full information); the beeswarm
    conditions belong to Axis 1, not here. Returns mean ``value`` by mechanism x model.
    """
    allc = df[df["representation"] == "all"]
    return (allc.groupby(["mechanism", "xai_model"])[value].mean()
                .unstack().round(3))


def beeswarm_readability(df: pd.DataFrame, value: str = "fair_total") -> pd.DataFrame:
    """Beeswarm readability — vision_beeswarm vs json_beeswarm (info-matched).

    Both carry the same information (rank + direction + spread); the only difference is
    modality (swarm image vs equivalent numbers), so the gap isolates the pure
    visual-reading effect. Scored on ``fair_total`` (direction+rank) by default.
    """
    bee = df[df["representation"] == "beeswarm"]
    return (bee.groupby(["condition", "xai_model"])[value].mean()
               .unstack().round(3))


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
        for crit, msg in ceiling_flags(df).items():
            print(f"\n[ceiling] judge '{crit}': {msg}")

        corr = rubric_judge_correlation(df)
        print(f"\nRubric-vs-judge (faithfulness) Spearman: rho={corr.get('spearman_rho')} "
              f"(p={corr.get('p_value')}, n={corr.get('n')})")

        print("\nModality significance (Wilcoxon on judge faithfulness, "
              "paired on feature x model, Holm):")
        sig = modality_significance(judge, "faithfulness")
        print(sig[["pipeline_a", "pipeline_b", "n_pairs", "delta_mean",
                   "p_value_adj", "reject", "cliffs_d", "magnitude"]].to_string(index=False))

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
