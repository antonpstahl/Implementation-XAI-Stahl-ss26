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
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

_PIPELINE_ORDER = ["Template", "JSON→Text", "Vision", "Tool-Use"]
_FAITH_ORDER = ["JSON→Text", "Tool-Use", "Vision"]


# ── markdown helpers ──────────────────────────────────────────────────────────

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


# ── number formatters ─────────────────────────────────────────────────────────

def _de(v: float, decimals: int = 2) -> str:
    """German locale float: comma decimal separator."""
    return f"{v:.{decimals}f}".replace(".", ",")


def _de_int(v: float) -> str:
    """German locale integer: narrow no-break space as thousands separator."""
    n = round(v)
    if n >= 1000:
        high, low = divmod(n, 1000)
        return f"{high} {low:03d}"
    return str(n)


# ── data loaders ──────────────────────────────────────────────────────────────

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
    p = RESULTS / "eval08_ichmoukhamedov" / "faithfulness_summary.csv"
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["pipeline_label"]] = row
    return rows


# ── English table generators ──────────────────────────────────────────────────

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
        tok_in = round(float(r["Tokens_in"]))
        tok_out = round(float(r["Tokens_out"]))
        rows.append([
            pl,
            str(round(float(r["Wörter"]))),
            f"{tok_in:,}",
            f"{tok_out:,}",
            f"${float(r['Kosten_USD']):.2f}",
            f"{float(r['Zeit_s']):.1f} s",
            f"{float(r['Judge_Faith']):.2f}",
            f"{float(r['Judge_Clarity']):.2f}",
            f"{float(r['Judge_Complete']):.2f}",
        ])
    return _padded_table(headers, rows)


def gen_faithfulness_en() -> str:
    """Ichmoukhamedov faithfulness metrics (3 rows: JSON→Text, Tool-Use, Vision)."""
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


# ── German table generators ───────────────────────────────────────────────────

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
        tok_in = round(float(r["Tokens_in"]))
        tok_out = round(float(r["Tokens_out"]))
        cost = float(r["Kosten_USD"])
        rows.append([
            pl,
            str(round(float(r["Wörter"]))),
            _de_int(tok_in),
            _de_int(tok_out),
            f"{_de(cost)} USD",
            f"{float(r['Zeit_s']):.1f} s",
        ])
    return _minimal_table(headers, rows)


def gen_judge_scores_de() -> str:
    """LLM-judge v1 scores in German locale (4 rows)."""
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


# ── README updater ────────────────────────────────────────────────────────────

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
    (ROOT / "Readme.md",    "pipeline-eval",      gen_pipeline_eval_en),
    (ROOT / "Readme.md",    "faithfulness",       gen_faithfulness_en),
    (ROOT / "Readme_DE.md", "model-comparison-de", gen_model_comparison_de),
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
        print("OK — all README tables match results/")
        return 0

    for path in changed_files:
        path.write_text(file_contents[path], encoding="utf-8")
        print(f"Updated {path.name}")

    if not changed_files:
        print("Nothing to update — tables already up to date.")
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
