#!/usr/bin/env python3
"""
Generate README result tables from results/ artefacts and update the READMEs in-place.

Each numeric table is wrapped with sentinel HTML comments in the Markdown source:

    <!-- AUTO-TABLE:name -->
    ...table...
    <!-- /AUTO-TABLE:name -->

Usage:
    python utils/update_readme_tables.py            # update in-place
    python utils/update_readme_tables.py --check    # exit 1 if any table is stale
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

_PIPELINE_ORDER = ["Template", "JSON to Text", "Vision", "Tool Use"]
_FAITH_ORDER = ["JSON to Text", "Tool Use", "Vision"]

# --- global track (G2a per-feature / G2b whole-model) ---
# Presentation order = the argument's order: the deterministic baseline first, then the
# three LLM handover formats, so the "what does the LLM add over a template?" comparison
# reads top-down.
_GLOBAL_FORM_ORDER = ["template", "json", "vision", "tooluse"]
_GLOBAL_FORM_LABEL = {"template": "Template", "json": "JSON", "vision": "Vision",
                      "tooluse": "Tool Use"}
# Ascending by mean faithfulness: near-flat is the failure mode, so it leads the table.
_FORM_TYPE_ORDER = ["near-flat", "categorical", "non-monotonic", "monotonic"]
# Axis 1 (all vs beeswarm) pairs first, the pull condition last.
_WHOLE_ORDER = ["json_all", "vision_all", "tooluse_all", "json_beeswarm", "vision_beeswarm"]
_JUDGE_METRICS = ("faithfulness", "clarity", "completeness")


# --- markdown helpers ---

def _padded_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a padded Markdown table (English style)."""
    widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]

    def _row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([_row(headers), sep] + [_row(r) for r in rows])


def _minimal_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a minimal Markdown table (German style, no column padding)."""
    header_row = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    data_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, sep] + data_rows)


# --- number formatters ---

def _de(v: float, decimals: int = 2) -> str:
    """German locale float: comma decimal separator."""
    return f"{v:.{decimals}f}".replace(".", ",")


def _round_half_up(v: float) -> int:
    """Round-half-up (avoids banker's rounding for .5 values)."""
    return int(v + 0.5) if v >= 0 else -int(-v + 0.5)


def _de_int(v: float) -> str:
    """German locale integer: space as thousands separator, round-half-up."""
    n = _round_half_up(v)
    if n >= 1000:
        high, low = divmod(n, 1000)
        return f"{high} {low:03d}"
    return str(n)


# --- data loaders ---

def _load_metrics(loss_key: str) -> dict:
    with open(RESULTS / f"model_metrics_{loss_key}.json", encoding="utf-8") as f:
        return json.load(f)


def _load_eval_summary() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(RESULTS / "eval_summary.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["pipeline_label"]] = row
    return rows


def _load_faithfulness() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    p = RESULTS / "eval06_ichmoukhamedov" / "faithfulness_summary.csv"
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["pipeline_label"]] = row
    return rows


def _load_judge_records(subdir: str) -> list[dict]:
    """All judge records under results/<subdir> (one JSON per scored explanation)."""
    out = []
    for path in sorted((RESULTS / subdir).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _group_means(records: list[dict], key: str) -> dict[str, dict[str, float]]:
    """Mean of each judge metric per value of *key* (e.g. form_pipeline, form_type)."""
    buckets: dict[str, dict[str, list[float]]] = {}
    for r in records:
        b = buckets.setdefault(r[key], {m: [] for m in _JUDGE_METRICS})
        for m in _JUDGE_METRICS:
            b[m].append(float(r[m]))
    return {k: {m: _mean(v[m]) for m in _JUDGE_METRICS} | {"n": len(v["faithfulness"])}
            for k, v in buckets.items()}


def _load_global_process() -> dict[str, dict[str, float]]:
    """Mean process metrics per G2a form: tokens, latency, tool calls (results/global).

    These come straight off the persisted generation records, so the cost/effort side of
    the comparison is reproducible from the same artefacts as the quality side.
    """
    acc: dict[str, dict[str, list[float]]] = {}
    for path in sorted((RESULTS / "global").glob("*.json")):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        u = r.get("usage", {})
        b = acc.setdefault(r["form"], {k: [] for k in
                                       ("tok_in", "cache", "tok_out", "latency", "calls")})
        b["tok_in"].append(float(u.get("input_tokens", 0)))
        b["cache"].append(float(u.get("cache_read_input_tokens", 0)))
        b["tok_out"].append(float(u.get("output_tokens", 0)))
        b["latency"].append(float(r.get("elapsed_s") or 0))
        b["calls"].append(float(r.get("n_tool_calls") or 0))
    return {form: {k: _mean(v) for k, v in d.items()} for form, d in acc.items()}


def _load_global_rubric() -> dict[str, float]:
    """Mean deterministic rubric total per form (results/global_rubric.csv)."""
    totals: dict[str, list[float]] = {}
    with open(RESULTS / "global_rubric.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            totals.setdefault(row["form_pipeline"], []).append(float(row["total"]))
    return {k: _mean(v) for k, v in totals.items()}


def _load_whole_coverage() -> dict[str, dict[str, int]]:
    """Features actually described per (condition, xai_model), out of 9.

    Read from the split records: a feature the whole-model answer omitted is written as
    ``dropped: true``, so coverage is just the non-dropped count.
    """
    cov: dict[str, dict[str, int]] = {}
    for path in sorted((RESULTS / "global_whole_split").glob("*.json")):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        per_model = cov.setdefault(r["condition"], {})
        per_model[r["xai_model"]] = per_model.get(r["xai_model"], 0) + (0 if r["dropped"] else 1)
    return cov


# --- English table generators ---

def gen_model_metrics_en() -> str:
    """Poisson-log model metrics (2 rows: XGB + EBM)."""
    m = _load_metrics("poisson_log")["metrics"]
    headers = ["Loss", "Model", "RMSE", "MAE", "R²", "Poisson dev.", "Neg. pred."]
    rows = []
    for key, label in [("xgb", "XGB"), ("ebm", "EBM")]:
        mm = m[key]
        rows.append([
            "Poisson-log",
            label,
            f"{mm['rmse']:.2f}",
            f"{mm['mae']:.2f}",
            f"{mm['r2']:.3f}",
            f"{mm['poisson_deviance']:.2f}",
            str(int(mm["n_negative_predictions"])),
        ])
    return _padded_table(headers, rows)


def gen_pipeline_eval_en() -> str:
    """Pipeline quantitative + LLM-judge summary (4 rows)."""
    data = _load_eval_summary()
    headers = [
        "Pipeline", "Avg words", "Input tok.¹", "Output tok.",
        "Cost (20 calls)", "Avg latency", "Judge Faith.", "Clarity", "Complete.",
    ]
    rows = []
    for pl in _PIPELINE_ORDER:
        r = data[pl]
        tok_in = _round_half_up(float(r["tokens_in"]))
        tok_out = _round_half_up(float(r["tokens_out"]))
        rows.append([
            pl,
            str(_round_half_up(float(r["words"]))),
            f"{tok_in:,}",
            f"{tok_out:,}",
            f"${float(r['cost_usd']):.2f}",
            f"{float(r['time_s']):.1f} s",
            f"{float(r['Judge_Faith']):.2f}",
            f"{float(r['Judge_Clarity']):.2f}",
            f"{float(r['Judge_Complete']):.2f}",
        ])
    return _padded_table(headers, rows)


def gen_faithfulness_en() -> str:
    """Ichmoukhamedov faithfulness metrics (3 rows: JSON to Text, Tool Use, Vision)."""
    data = _load_faithfulness()
    headers = ["Pipeline", "Rank Agr.", "Sign Agr.", "Value Agr."]
    rows = []
    for pl in _FAITH_ORDER:
        r = data[pl]
        rows.append([
            pl,
            f"{float(r['RA']):.3f}",
            f"{float(r['SA']):.3f}",
            f"{float(r['VA']):.3f}",
        ])
    return _padded_table(headers, rows)


def _global_feature_rows(fmt, fmt_rubric):
    """Shared row builder for the G2a per-feature table (EN/DE differ only in number format)."""
    rubric = _load_global_rubric()
    judge = _group_means(_load_judge_records("global_judge"), "form_pipeline")
    judge_o = _group_means(_load_judge_records("global_judge_openai"), "form_pipeline")
    rows = []
    for form in _GLOBAL_FORM_ORDER:
        j, o = judge[form], judge_o[form]
        rows.append([
            _GLOBAL_FORM_LABEL[form],
            fmt_rubric(rubric[form]),
            fmt(j["faithfulness"]),
            fmt(j["clarity"]),
            fmt(j["completeness"]),
            fmt(o["faithfulness"]),
        ])
    return rows


def gen_global_feature_en() -> str:
    """G2a: 4 handover forms x 2 XAI models x 9 features = 72 per-feature descriptions."""
    headers = ["Form", "Rubric total", "Judge Faith.", "Clarity", "Complete.",
               "Faith. (OpenAI)"]
    return _padded_table(headers, _global_feature_rows(
        lambda v: f"{v:.2f}", lambda v: f"{v:.3f}"))


def gen_global_formtype_en() -> str:
    """G2a judge faithfulness stratified by ground-truth shape type (the core finding)."""
    strata = _group_means(_load_judge_records("global_judge"), "form_type")
    headers = ["Shape type", "n", "Judge Faith.", "Clarity", "Complete."]
    rows = [[ft, str(strata[ft]["n"]), f"{strata[ft]['faithfulness']:.2f}",
             f"{strata[ft]['clarity']:.2f}", f"{strata[ft]['completeness']:.2f}"]
            for ft in _FORM_TYPE_ORDER if ft in strata]
    return _padded_table(headers, rows)


def _global_process_rows(fmt_int, fmt_1):
    proc = _load_global_process()
    rows = []
    for form in _GLOBAL_FORM_ORDER:
        m = proc[form]
        rows.append([
            _GLOBAL_FORM_LABEL[form],
            fmt_int(m["tok_in"]),
            fmt_int(m["tok_out"]),
            fmt_1(m["latency"]) + " s",
            "n/a" if form != "tooluse" else fmt_1(m["calls"]),
        ])
    return rows


def gen_global_process_en() -> str:
    """G2a process metrics per form: billed input/output tokens, latency, tool calls."""
    headers = ["Form", "Avg input tok.", "Avg output tok.", "Avg latency", "Avg tool calls"]
    return _padded_table(headers, _global_process_rows(
        lambda v: f"{_round_half_up(v):,}", lambda v: f"{v:.1f}"))


def _global_whole_rows(fmt):
    cov = _load_whole_coverage()
    judge = _group_means(_load_judge_records("global_whole_judge"), "form_pipeline")
    judge_o = _group_means(_load_judge_records("global_whole_judge_openai"), "form_pipeline")
    rows = []
    for cond in _WHOLE_ORDER:
        c = cov[cond]
        rows.append([
            cond,
            f"{c['ebm']}/9 \u00b7 {c['xgb']}/9",
            fmt(judge[cond]["faithfulness"]),
            fmt(judge_o[cond]["faithfulness"]),
            fmt(judge[cond]["completeness"]),
        ])
    return rows


def gen_global_whole_en() -> str:
    """G2b: 5 whole-model conditions x 2 XAI models, split into 90 per-feature records."""
    headers = ["Condition", "Coverage (ebm \u00b7 xgb)", "Judge Faith.", "Faith. (OpenAI)",
               "Complete."]
    return _padded_table(headers, _global_whole_rows(lambda v: f"{v:.2f}"))


# --- German table generators (for Readme_DE.md) ---

def gen_model_comparison_de() -> str:
    """Full model comparison in German locale (4 rows: Squared Error + Poisson-Log)."""
    sq = _load_metrics("squared_error")["metrics"]
    pl = _load_metrics("poisson_log")["metrics"]
    headers = ["Option", "Modell", "RMSE", "MAE", "R²", "Poisson-Dev.", "Neg. Vorhersagen"]
    rows = [
        [
            "Squared Error", "XGB",
            _de(sq["xgb"]["rmse"]), _de(sq["xgb"]["mae"]),
            _de(sq["xgb"]["r2"], 3), _de(sq["xgb"]["poisson_deviance"]),
            str(int(sq["xgb"]["n_negative_predictions"])),
        ],
        [
            "Squared Error", "EBM",
            _de(sq["ebm"]["rmse"]), _de(sq["ebm"]["mae"]),
            _de(sq["ebm"]["r2"], 3), _de(sq["ebm"]["poisson_deviance"]),
            str(int(sq["ebm"]["n_negative_predictions"])),
        ],
        [
            "**Poisson-Log**", "**XGB**",
            f"**{_de(pl['xgb']['rmse'])}**", f"**{_de(pl['xgb']['mae'])}**",
            f"**{_de(pl['xgb']['r2'], 3)}**", f"**{_de(pl['xgb']['poisson_deviance'])}**",
            f"**{int(pl['xgb']['n_negative_predictions'])}**",
        ],
        [
            "**Poisson-Log**", "**EBM**",
            f"**{_de(pl['ebm']['rmse'])}**", f"**{_de(pl['ebm']['mae'])}**",
            f"**{_de(pl['ebm']['r2'], 3)}**", f"**{_de(pl['ebm']['poisson_deviance'])}**",
            f"**{int(pl['ebm']['n_negative_predictions'])}**",
        ],
    ]
    return _minimal_table(headers, rows)


def gen_pipeline_quant_de() -> str:
    """Pipeline quantitative summary in German locale (cost/latency only, 4 rows)."""
    data = _load_eval_summary()
    headers = [
        "Pipeline", "Ø Wörter", "Ø Input-Tokens¹",
        "Ø Output-Tokens", "Gesamtkosten (20 Calls)", "Ø Latenz",
    ]
    rows = []
    for pl in _PIPELINE_ORDER:
        r = data[pl]
        tok_in = _round_half_up(float(r["tokens_in"]))
        tok_out = _round_half_up(float(r["tokens_out"]))
        cost = float(r["cost_usd"])
        rows.append([
            pl,
            str(_round_half_up(float(r["words"]))),
            _de_int(tok_in),
            _de_int(tok_out),
            f"{_de(cost)} USD",
            f"{_de(float(r['time_s']), 1)} s",
        ])
    return _minimal_table(headers, rows)


def gen_judge_scores_de() -> str:
    """LLM judge scores in German locale (4 rows)."""
    data = _load_eval_summary()
    headers = ["Pipeline", "Faithfulness", "Clarity", "Completeness"]
    rows = []
    for pl in _PIPELINE_ORDER:
        r = data[pl]
        rows.append([
            pl,
            _de(float(r["Judge_Faith"])),
            _de(float(r["Judge_Clarity"])),
            _de(float(r["Judge_Complete"])),
        ])
    return _minimal_table(headers, rows)


def gen_faithfulness_de() -> str:
    """Ichmoukhamedov faithfulness in German locale (3 rows)."""
    data = _load_faithfulness()
    headers = ["Pipeline", "RA (Rank)", "SA (Sign)", "VA (Value)"]
    rows = []
    for pl in _FAITH_ORDER:
        r = data[pl]
        rows.append([
            pl,
            _de(float(r["RA"]), 3),
            _de(float(r["SA"]), 3),
            _de(float(r["VA"]), 3),
        ])
    return _minimal_table(headers, rows)


def gen_global_feature_de() -> str:
    """G2a per-Feature-Tabelle, deutsches Zahlenformat."""
    headers = ["Form", "Rubric gesamt", "Judge Faithfulness", "Clarity", "Completeness",
               "Faithfulness (OpenAI)"]
    return _minimal_table(headers, _global_feature_rows(_de, lambda v: _de(v, 3)))


def gen_global_formtype_de() -> str:
    """G2a Judge-Faithfulness nach GT-Formtyp, deutsches Zahlenformat."""
    strata = _group_means(_load_judge_records("global_judge"), "form_type")
    headers = ["Formtyp", "n", "Judge Faithfulness", "Clarity", "Completeness"]
    rows = [[ft, str(strata[ft]["n"]), _de(strata[ft]["faithfulness"]),
             _de(strata[ft]["clarity"]), _de(strata[ft]["completeness"])]
            for ft in _FORM_TYPE_ORDER if ft in strata]
    return _minimal_table(headers, rows)


def gen_global_whole_de() -> str:
    """G2b Whole-Model-Tabelle, deutsches Zahlenformat."""
    headers = ["Bedingung", "Coverage (ebm \u00b7 xgb)", "Judge Faithfulness",
               "Faithfulness (OpenAI)", "Completeness"]
    return _minimal_table(headers, _global_whole_rows(_de))


def gen_global_process_de() -> str:
    """G2a Prozesskennzahlen je Form, deutsches Zahlenformat."""
    headers = ["Form", "Ø Input-Tokens", "Ø Output-Tokens", "Ø Latenz", "Ø Tool-Calls"]
    return _minimal_table(headers, _global_process_rows(
        _de_int, lambda v: _de(v, 1)))


# --- README updater ---

_SENTINEL_RE = re.compile(
    r"<!-- AUTO-TABLE:([^/\s>]+) -->\n(.*?)\n<!-- /AUTO-TABLE:\1 -->",
    re.DOTALL,
)


def extract_table(text: str, name: str) -> str:
    """Return the table content between sentinels for *name*."""
    pattern = re.compile(
        r"<!-- AUTO-TABLE:" + re.escape(name) + r" -->\n(.*?)\n<!-- /AUTO-TABLE:" + re.escape(name) + r" -->",
        re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        raise ValueError(f"Sentinel AUTO-TABLE:{name} not found")
    return m.group(1)


def replace_table(text: str, name: str, table: str) -> str:
    """Replace the sentinel block for *name* with *table*."""
    pattern = re.compile(
        r"<!-- AUTO-TABLE:" + re.escape(name) + r" -->.*?<!-- /AUTO-TABLE:" + re.escape(name) + r" -->",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise ValueError(f"Sentinel AUTO-TABLE:{name} not found in file")
    return pattern.sub(
        f"<!-- AUTO-TABLE:{name} -->\n{table}\n<!-- /AUTO-TABLE:{name} -->",
        text,
    )


# Table definitions: (readme_path, sentinel_name, generator_function)
_TABLES: list[tuple[Path, str, object]] = [
    (ROOT / "Readme.md",    "model-metrics",      gen_model_metrics_en),
    # global track (the main track since the 30.06. meeting)
    (ROOT / "Readme.md",    "global-feature",     gen_global_feature_en),
    (ROOT / "Readme.md",    "global-formtype",    gen_global_formtype_en),
    (ROOT / "Readme.md",    "global-whole",       gen_global_whole_en),
    (ROOT / "Readme.md",    "global-process",     gen_global_process_en),
    # local n=20 track (comparison basis)
    (ROOT / "Readme.md",    "pipeline-eval",      gen_pipeline_eval_en),
    (ROOT / "Readme.md",    "faithfulness",       gen_faithfulness_en),
    (ROOT / "Readme_DE.md", "model-comparison-de", gen_model_comparison_de),
    (ROOT / "Readme_DE.md", "global-feature-de",  gen_global_feature_de),
    (ROOT / "Readme_DE.md", "global-formtype-de", gen_global_formtype_de),
    (ROOT / "Readme_DE.md", "global-whole-de",    gen_global_whole_de),
    (ROOT / "Readme_DE.md", "global-process-de",  gen_global_process_de),
    (ROOT / "Readme_DE.md", "pipeline-quant-de",  gen_pipeline_quant_de),
    (ROOT / "Readme_DE.md", "judge-scores-de",    gen_judge_scores_de),
    (ROOT / "Readme_DE.md", "faithfulness-de",    gen_faithfulness_de),
]


def run(check: bool = False) -> int:
    stale: list[str] = []
    changed_files: set[Path] = set()
    file_contents: dict[Path, str] = {}

    for readme_path, name, gen_fn in _TABLES:
        if readme_path not in file_contents:
            file_contents[readme_path] = readme_path.read_text(encoding="utf-8")

        expected = gen_fn()
        current = extract_table(file_contents[readme_path], name)

        if current != expected:
            stale.append(f"{readme_path.name}:{name}")
            if not check:
                file_contents[readme_path] = replace_table(
                    file_contents[readme_path], name, expected
                )
                changed_files.add(readme_path)

    if check:
        if stale:
            print("STALE tables (README does not match results/):")
            for s in stale:
                print(f"  {s}")
            print("\nRun:  python utils/update_readme_tables.py")
            return 1
        print("OK - all README tables match results/")
        return 0

    for path in changed_files:
        path.write_text(file_contents[path], encoding="utf-8")
        print(f"Updated {path.name}")

    if not changed_files:
        print("Nothing to update - tables already up to date.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if any table is stale (do not write files).",
    )
    args = parser.parse_args()
    sys.exit(run(check=args.check))
