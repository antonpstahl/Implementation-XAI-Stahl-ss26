"""Number ledger: every paper-reported quantity, computed from `results/`.

Dumps a flat ``{key: value}`` JSON so before/after runs can be diffed key by key.
Run it on run 1 (Step 0) and on run 2, then::

    python analyses/numbers_ledger.py                     # emits ledger.json in cwd
    python analyses/numbers_ledger.py --out before.json
    python analyses/numbers_ledger.py --out after.json
    python analyses/numbers_ledger.py --diff before.json after.json

Keys are grouped by paper table via dotted prefixes:

  ``g2a.rubric_total.<modality>.<form_type>``
  ``g2a.judge_faithfulness.<vendor>.<modality>.<form_type>``
  ``g2a.significance.<vendor>.<pair>.<field>``
  ``g2a.correlation.<vendor>.spearman_rho`` etc.
  ``g2a.interaction.<modality>.<field>``
  ``variance.sd_within.<modality>.<xai>.rubric_total`` (etc.)
  ``variance.judge_span.<modality>.<xai>``
  ``g2b.axis1.<condition>.<field>``
  ``g2b.axis2.<push_condition>.<xai>.<field>``
  ``g2b.beeswarm.<xai>.<condition>``
  ``g2b.coverage.<condition>.<xai>``
  ``g2b.stated_rank_count`` (integer over all conditions)
  ``alpha.g2a.<criterion>``, ``alpha.g2b.<criterion>``
  ``cost.<modality>.<field>``
  ``leak.count.<top>``
  ``monte_carlo.<variant>.<key>``  (populated by monte_carlo_orderings.py)
  ``plot_range.spearman.<xai>``    (title-independent sanity key)

Missing sub-tables (judge not run yet, whole-model not run, etc.) show up as
absent keys, so a run-2 ledger with fewer keys already tells the reader what did
not run.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import global_eval as GE
from utils.rubric import rubric_score, load_ground_truth, RESULTS_DIR as RUBRIC_RESULTS_DIR

RESULTS = ROOT / "results"

VENDORS = [
    ("anthropic", "global_judge",           "global_variance_judge",       "global_whole_judge"),
    ("openai",    "global_judge_openai",    "global_variance_judge_openai","global_whole_judge_openai"),
]


def _round(x, ndigits=4):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return round(v, ndigits)


def _dump_table(ledger: dict, prefix: str, tab: pd.DataFrame) -> None:
    for row_key, row in tab.iterrows():
        for col_key, val in row.items():
            ledger[f"{prefix}.{row_key}.{col_key}"] = _round(val)


# ---------------------------------------------------------------------------
# G2a
# ---------------------------------------------------------------------------

_ALL4 = ("template", "json", "vision", "tooluse")
_LLM3 = ("json", "vision", "tooluse")


def _modality_span(strat: pd.DataFrame) -> tuple[float | None, float | None]:
    """Return (span over all4, span over llm3) of the 'overall' column of a
    stratified table indexed by form_pipeline. None if not enough rows."""
    if "overall" not in strat.columns:
        return None, None
    ov = strat["overall"]
    def _span(keys):
        vals = [ov[k] for k in keys if k in ov.index]
        return max(vals) - min(vals) if len(vals) >= 2 else None
    return _span(_ALL4), _span(_LLM3)


def _g2a(ledger: dict) -> None:
    rubric = GE.load_rubric()
    ledger["g2a.n_records"] = int(len(rubric))
    strat_rubric = GE.stratified_table(rubric, "total")
    _dump_table(ledger, "g2a.rubric_total", strat_rubric)
    _dump_table(ledger, "g2a.interaction",
                GE.method_format_interaction(rubric, "total"))

    all4, llm3 = _modality_span(strat_rubric)
    if all4 is not None:
        ledger["g2a.modality_span.rubric_total.all4"] = _round(all4)
    if llm3 is not None:
        ledger["g2a.modality_span.rubric_total.llm3"] = _round(llm3)

    for vendor, judge_sub, _var_sub, _whole_sub in VENDORS:
        judge = GE.load_judge(judge_sub)
        if judge is None:
            continue
        merged = GE.merge_scores(rubric, judge)
        for crit in GE.JUDGE_CRITERIA:
            if crit in merged.columns:
                strat = GE.stratified_table(merged, crit)
                _dump_table(ledger, f"g2a.judge_{crit}.{vendor}", strat)
                if crit == "faithfulness":
                    all4, llm3 = _modality_span(strat)
                    if all4 is not None:
                        ledger[f"g2a.modality_span.judge_faithfulness.{vendor}.all4"] = _round(all4)
                    if llm3 is not None:
                        ledger[f"g2a.modality_span.judge_faithfulness.{vendor}.llm3"] = _round(llm3)
        corr = GE.rubric_judge_correlation(merged)
        for k, v in corr.items():
            ledger[f"g2a.correlation.{vendor}.{k}"] = _round(v)
        sig = GE.modality_significance(judge, "faithfulness")
        # sig columns: pair, statistic, p, cliffs_d, ...
        for _, r in sig.iterrows():
            pair = f"{r['pipeline_a']}_vs_{r['pipeline_b']}"
            for f in ("statistic", "p_value", "cliffs_d", "p_value_adj", "delta_mean"):
                if f in r:
                    ledger[f"g2a.sig.{vendor}.{pair}.{f}"] = _round(r[f])
        # method-format interaction table on the judge metric (this is where
        # e.g. the vision +0.556 cell lives, per plan tab:g2a-interaction).
        _dump_table(ledger, f"g2a.interaction_judge.{vendor}",
                    GE.method_format_interaction(merged, "faithfulness"))

    ledger["g2a.rubric.spread_ebm_xgb.rubric_total"] = _round(
        GE.method_format_interaction(rubric, "total")["delta"].max()
        - GE.method_format_interaction(rubric, "total")["delta"].min()
    )


# ---------------------------------------------------------------------------
# Variance
# ---------------------------------------------------------------------------

_EXCLUDED_VAR = {"json_xgb_weekday_gen1"}


_HR_CONTROL = "hr"


def _pool_within_cell_sd(df: pd.DataFrame, value: str, features: set) -> dict:
    """Mean within-cell SD across the given feature subset, per (form, xai).
    Returns {(form, xai): mean_sd_within}. Pooling the SDs across the four
    variance features (or the non-hr three) reproduces the paper's tab:variance
    numbers, which the per-cell breakdown in the earlier ledger could not diff."""
    sub = df[df["feature"].isin(features)]
    out: dict = {}
    for (f, x), g in sub.groupby(["form", "xai"]):
        per_cell = g.groupby(["feature"])[value].std().dropna()
        if len(per_cell):
            out[(f, x)] = per_cell.mean()
    return out


def _variance(ledger: dict) -> None:
    src = RESULTS / "global_variance"
    if not src.is_dir():
        return
    rows = []
    for p in sorted(src.glob("*.json")):
        stem = p.stem
        if stem in _EXCLUDED_VAR:
            continue
        m = re.match(r"^(\w+?)_(\w+?)_(\w+)_gen([012])$", stem)
        if not m:
            continue
        form, xai, feature, gen = m.groups()
        rec = json.loads(p.read_text())
        try:
            gt = load_ground_truth(rec["xai_model"], rec["feature"])
        except Exception:
            continue
        s = rubric_score(rec["explanation"], gt)
        rows.append({"form": form, "xai": xai, "feature": feature,
                     "gen": int(gen), "total": s["total"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    ledger["variance.n_draws"] = int(len(df))

    all_feats = set(df["feature"].unique())
    non_hr = all_feats - {_HR_CONTROL}

    # Per-cell keys (unchanged) — kept for backwards compatibility with earlier ledgers.
    for (f, x), g in df.groupby(["form", "xai"]):
        per_cell = g.groupby(["feature"])["total"].std().dropna()
        ledger[f"variance.sd_within.{f}.{x}.rubric_total"] = _round(per_cell.mean())
        ledger[f"variance.span_within.{f}.{x}.rubric_total"] = _round(
            g.groupby("feature")["total"].apply(lambda s: s.max() - s.min()).mean()
        )

    # Pooled non-hr SDs per (form, xai), then pooled across (form, xai) for the
    # paper's headline number (tab:variance: rubric 0.078). Also the hr control
    # SDs and the between-format span of the pooled values.
    def _emit_pool(prefix: str, value: str, source_df: pd.DataFrame) -> None:
        non_hr_pool = _pool_within_cell_sd(source_df, value, non_hr)
        hr_pool     = _pool_within_cell_sd(source_df, value, {_HR_CONTROL})
        # per (form, xai) pooled non-hr
        for (f, x), sd in non_hr_pool.items():
            ledger[f"{prefix}.pooled_non_hr.{f}.{x}"] = _round(sd)
        # global mean over cells (paper's headline number)
        if non_hr_pool:
            ledger[f"{prefix}.pooled_non_hr.overall"] = _round(
                sum(non_hr_pool.values()) / len(non_hr_pool)
            )
        # per (form, xai) hr control
        for (f, x), sd in hr_pool.items():
            ledger[f"{prefix}.hr_control.{f}.{x}"] = _round(sd)
        if hr_pool:
            ledger[f"{prefix}.hr_control.overall"] = _round(
                sum(hr_pool.values()) / len(hr_pool)
            )

    _emit_pool("variance.rubric", "total", df)

    # judge-side variance per (form, xai) — per-cell + pooled — if verdicts present
    for vendor, _judge, var_sub, _whole in VENDORS:
        vdir = RESULTS / var_sub
        if not vdir.is_dir():
            continue
        jrows = []
        for p in sorted(vdir.glob("*.json")):
            stem = p.stem
            if stem in _EXCLUDED_VAR:
                continue
            m = re.match(r"^(\w+?)_(\w+?)_(\w+)_gen([012])$", stem)
            if not m:
                continue
            form, xai, feature, gen = m.groups()
            rec = json.loads(p.read_text())
            jrows.append({"form": form, "xai": xai, "feature": feature,
                          "gen": int(gen),
                          "faithfulness": rec.get("faithfulness")})
        if not jrows:
            continue
        jdf = pd.DataFrame(jrows)
        for (f, x), g in jdf.groupby(["form", "xai"]):
            per_cell = g.groupby(["feature"])["faithfulness"].std().dropna()
            ledger[f"variance.sd_within.{f}.{x}.judge_{vendor}"] = _round(per_cell.mean())
            ledger[f"variance.span_within.{f}.{x}.judge_{vendor}"] = _round(
                g.groupby("feature")["faithfulness"]
                 .apply(lambda s: s.max() - s.min()).mean()
            )
        _emit_pool(f"variance.judge_{vendor}", "faithfulness", jdf)


# ---------------------------------------------------------------------------
# G2b whole
# ---------------------------------------------------------------------------

def _g2b(ledger: dict) -> None:
    df = GE.load_whole_rubric()
    if df is None:
        return
    ledger["g2b.n_splits"] = int(len(df))
    ledger["g2b.stated_rank_count"] = int(df["parsed_rank"].notna().sum())
    # Correct-when-stated
    correct = 0
    stated = 0
    for _, r in df.iterrows():
        if pd.isna(r["parsed_rank"]):
            continue
        stated += 1
        try:
            gt = load_ground_truth(r["xai_model"], r["feature"])
            if int(r["parsed_rank"]) == int(gt["importance_rank"]):
                correct += 1
        except Exception:
            pass
    ledger["g2b.stated_and_correct"] = correct
    ledger["g2b.stated_total"] = stated

    _dump_table(ledger, "g2b.axis1", GE.axis1_representation(df))

    pair = GE.axis2_mechanism_pairwise(df, "fair_total")
    for _, r in pair.iterrows():
        for f in ("pull", "push", "delta_pull_minus_push"):
            ledger[f"g2b.axis2.{r['push_condition']}.{r['xai_model']}.{f}"] = _round(r[f])

    bee = GE.beeswarm_readability(df, "fair_total")
    for xai, row in bee.iterrows():
        for cond, val in row.items():
            ledger[f"g2b.beeswarm.{xai}.{cond}"] = _round(val)

    cov = GE.whole_coverage(df)
    for cond, row in cov.iterrows():
        for xai, val in row.items():
            if pd.notna(val):
                ledger[f"g2b.coverage.{cond}.{xai}"] = int(val)

    # Whole-model judge means per condition × vendor. Reads the split-level
    # judge outputs (05Gb writes them under global_whole_judge*), pools
    # dropped=True as 1 (dropped features get the "miss" verdict). Plan
    # tab:g2b-axis1 references e.g. vision_all XGB Anthropic 3.44 vs 5.00.
    for vendor, _g2a_sub, _var_sub, whole_sub in VENDORS:
        wdir = RESULTS / whole_sub
        if not wdir.is_dir():
            continue
        wrows = []
        for p in sorted(wdir.glob("*.json")):
            rec = json.loads(p.read_text())
            wrows.append({
                "condition": rec.get("form_pipeline"),
                "xai": rec.get("xai_model"),
                "feature": rec.get("feature"),
                "faithfulness": rec.get("faithfulness"),
            })
        if not wrows:
            continue
        wdf = pd.DataFrame(wrows).dropna(subset=["faithfulness"])
        for (cond, xai), g in wdf.groupby(["condition", "xai"]):
            ledger[f"g2b.judge_faithfulness.{vendor}.{cond}.{xai}"] = _round(
                g["faithfulness"].mean()
            )


# ---------------------------------------------------------------------------
# Cost / tokens / tool sequence
# ---------------------------------------------------------------------------

# The paper's "importance -> curve -> plot" tool-use sequence, as the actual
# tool names appear in results/global/tooluse_*.json (`tool_calls[*]["tool"]`).
_IMP_CURVE_PLOT_SEQ = ("get_feature_importances", "get_feature_curve", "get_feature_plot")


def _has_seq(names: list[str], seq: tuple[str, ...]) -> bool:
    """Does the ordered `seq` appear as a subsequence of `names` (not
    necessarily contiguous)? Used for the "imp -> curve -> plot" tool count."""
    it = iter(names)
    return all(any(n == token for n in it) for token in seq)


def _cost_and_tokens(ledger: dict) -> None:
    """Mean input/output tokens, tool_calls, elapsed per (form, xai) on G2a
    records, plus the paper's XGB/EBM input-token asymmetry per format, and the
    ordered "importance -> curve -> plot" sequence count for tooluse."""
    src = RESULTS / "global"
    if not src.is_dir():
        return
    rows = []
    for p in sorted(src.glob("*.json")):
        rec = json.loads(p.read_text())
        usage = rec.get("usage") or {}
        rows.append({
            "form": rec.get("form"),
            "xai": rec.get("xai_model"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "elapsed_s": rec.get("elapsed_s"),
            "n_tool_calls": rec.get("n_tool_calls"),
            "tool_calls": rec.get("tool_calls") or [],
        })
    if not rows:
        return
    df = pd.DataFrame(rows)

    # Per-format aggregates (as before).
    for form, g in df.groupby("form"):
        for col in ("input_tokens", "output_tokens", "elapsed_s"):
            v = g[col].dropna()
            if len(v):
                ledger[f"cost.{form}.mean_{col}"] = _round(v.mean(), 3)
        n_tc = g["n_tool_calls"].dropna()
        if len(n_tc):
            ledger[f"cost.{form}.mean_n_tool_calls"] = _round(n_tc.mean(), 3)
            ledger[f"cost.{form}.records_with_tool_calls"] = int((n_tc > 0).sum())
            ledger[f"cost.{form}.records_total"] = int(len(g))

    # Per-(form, xai) mean input tokens + per-format xgb/ebm ratio. The
    # ~340x asymmetry the paper cites is on the *json* arm.
    for (form, xai), g in df.groupby(["form", "xai"]):
        v = g["input_tokens"].dropna()
        if len(v):
            ledger[f"cost.{form}.{xai}.mean_input_tokens"] = _round(v.mean(), 3)
    for form, g in df.groupby("form"):
        ebm = g[g["xai"] == "ebm"]["input_tokens"].dropna()
        xgb = g[g["xai"] == "xgb"]["input_tokens"].dropna()
        if len(ebm) and len(xgb) and ebm.mean() > 0:
            ledger[f"cost.asymmetry.{form}.xgb_over_ebm"] = _round(
                xgb.mean() / ebm.mean(), 2
            )

    # Ordered "importance -> curve -> plot" sequence count for tooluse.
    tool_rows = df[df["form"] == "tooluse"]
    hits = sum(
        1 for _, r in tool_rows.iterrows()
        if _has_seq([tc.get("tool") for tc in r["tool_calls"] if isinstance(tc, dict)],
                    _IMP_CURVE_PLOT_SEQ)
    )
    ledger["cost.tooluse.records_with_imp_curve_plot_seq"] = int(hits)


# ---------------------------------------------------------------------------
# Alpha, leak, plot range
# ---------------------------------------------------------------------------

def _alpha(ledger: dict) -> None:
    """Cross-vendor Krippendorff α per judge criterion, on G2a and G2b. Any
    per-criterion failure lands as ``_error.alpha.<scope>.<crit>`` so the diff
    sees it — bare ``except: pass`` was silently dropping keys."""
    for crit in GE.JUDGE_CRITERIA:
        try:
            ledger[f"alpha.g2a.{crit}"] = _round(
                GE.cross_vendor_alpha(["global_judge", "global_judge_openai"], crit)
            )
        except Exception as exc:
            ledger[f"_error.alpha.g2a.{crit}"] = f"{type(exc).__name__}: {exc}"
        try:
            ledger[f"alpha.g2b.{crit}"] = _round(
                GE.cross_vendor_alpha(
                    ["global_whole_judge", "global_whole_judge_openai"], crit,
                    results_dir=RESULTS,
                )
            )
        except Exception as exc:
            ledger[f"_error.alpha.g2b.{crit}"] = f"{type(exc).__name__}: {exc}"


_LEAK_OPEN = re.compile(r"<(analysis|thinking)>")  # case-sensitive: matches strip_scratchpad
_EXEMPT_LEAK = {"global_variance/json_xgb_weekday_gen1.json"}


def _leak(ledger: dict) -> None:
    for top in ("global", "global_variance", "global_whole"):
        d = RESULTS / top
        if not d.is_dir():
            continue
        n = 0
        for p in d.glob("*.json"):
            rel = p.relative_to(RESULTS).as_posix()
            if rel in _EXEMPT_LEAK:
                continue
            try:
                rec = json.loads(p.read_text())
            except Exception:
                continue
            expl = rec.get("explanation")
            if isinstance(expl, str) and _LEAK_OPEN.search(expl):
                n += 1
        ledger[f"leak.count.{top}"] = n


def _plot_range(ledger: dict) -> None:
    """Spearman(range of plotted y, ground-truth rank) per XAI.
    Title-independent sanity key: this must not change between run 1 and run 2.

    Sign convention: rank 1 is the most important feature, so a strong
    range↔importance relation gives a **negative** ρ here (large y-range ↔
    small rank number). The paper reports the same statistic as a positive ρ
    (range vs. importance). Compare magnitudes: EBM 0.90, XGB 0.23."""
    from scipy.stats import spearmanr
    for xai, jsn in (("ebm", "global_ebm_poisson_log.json"),
                     ("xgb", "global_xgb_poisson_log.json")):
        p = ROOT / "explanations" / jsn
        if not p.exists():
            continue
        entries = json.loads(p.read_text())["global_importance"]
        # Range of the curve values per feature = the plot's y-span
        ranges, ranks = [], []
        for e in entries:
            curve_path = ROOT / "explanations" / f"global_curve_{xai}_{e['feature']}.json"
            if not curve_path.exists():
                continue
            c = json.loads(curve_path.read_text())
            y = c.get("y", [])
            if not y:
                continue
            ranges.append(max(y) - min(y))
            ranks.append(e["rank"])
        if len(ranges) >= 3:
            rho, _p = spearmanr(ranges, ranks)
            ledger[f"plot_range.spearman.{xai}"] = _round(rho, 2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _merge_monte_carlo(ledger: dict, path: Path, variant_label: str) -> None:
    """Fold monte_carlo_orderings output into the ledger under monte_carlo.*."""
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    for k, v in payload.items():
        base = f"monte_carlo.{variant_label}.{k}"
        if isinstance(v, dict):
            for kk, vv in v.items():
                ledger[f"{base}.{kk}"] = _round(vv) if isinstance(vv, (int, float)) else vv
        elif isinstance(v, (int, float)):
            ledger[base] = _round(v)
        elif isinstance(v, str):
            ledger[base] = v


def build_ledger(monte_carlo: dict[str, Path] | None = None) -> dict:
    """Compute the ledger. ``monte_carlo`` optionally maps a variant label to
    a JSON file produced by ``analyses/monte_carlo_orderings.py --out ...`` so
    those numbers are folded in under ``monte_carlo.<label>.*``."""
    ledger: dict = {}
    for fn in (_g2a, _variance, _g2b, _cost_and_tokens, _alpha, _leak, _plot_range):
        try:
            fn(ledger)
        except Exception as exc:
            ledger[f"_error.{fn.__name__}"] = f"{type(exc).__name__}: {exc}"
    for label, path in (monte_carlo or {}).items():
        try:
            _merge_monte_carlo(ledger, path, label)
        except Exception as exc:
            ledger[f"_error.monte_carlo.{label}"] = f"{type(exc).__name__}: {exc}"
    return ledger


def _diff(before: dict, after: dict) -> int:
    all_keys = sorted(set(before) | set(after))
    changed = []
    added = []
    removed = []
    for k in all_keys:
        b = before.get(k)
        a = after.get(k)
        if k not in before:
            added.append((k, a))
        elif k not in after:
            removed.append((k, b))
        elif b != a:
            changed.append((k, b, a))
    print(f"# ledger diff: {len(changed)} changed, {len(added)} added, {len(removed)} removed")
    for k, b, a in changed:
        try:
            d = a - b
            ds = f"  Δ={d:+.4f}"
        except TypeError:
            ds = ""
        print(f"  ~ {k}: {b} -> {a}{ds}")
    for k, a in added:
        print(f"  + {k}: {a}")
    for k, b in removed:
        print(f"  - {k}: {b}")
    return 0 if not (changed or added or removed) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("numbers_ledger.json"),
                    help="Where to write the ledger JSON.")
    ap.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="Diff two existing ledgers.")
    ap.add_argument("--mc", action="append", default=[], metavar="LABEL=PATH",
                    help="Fold a monte_carlo_orderings JSON into the ledger. "
                         "Repeatable, e.g. --mc v18=mc18.json --mc v24=mc24.json")
    args = ap.parse_args()

    if args.diff:
        before = json.loads(Path(args.diff[0]).read_text())
        after = json.loads(Path(args.diff[1]).read_text())
        return _diff(before, after)

    mc: dict[str, Path] = {}
    for spec in args.mc:
        label, _, path = spec.partition("=")
        if not path:
            print(f"# ignoring malformed --mc spec: {spec!r}", file=sys.stderr)
            continue
        mc[label] = Path(path)

    ledger = build_ledger(monte_carlo=mc)
    args.out.write_text(json.dumps(ledger, indent=2, sort_keys=True))
    print(f"# ledger: {len(ledger)} keys -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
