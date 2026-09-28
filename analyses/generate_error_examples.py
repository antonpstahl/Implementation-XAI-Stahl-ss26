"""Qualitative error analysis of the global judge results, stratified by shape type.

Pulls the reference-based judge's ``faithfulness_reasoning`` for the low-scoring
explanations (failures) from ``results/global_judge/`` and emits a Markdown report
to ``analyses/error_examples_by_formtype.md``. The verbatim excerpts regenerate
from disk; the interpreted failure-mode taxonomy (INTERPRETATION below) is authored.

Run:  python3 analyses/generate_error_examples.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUDGE_DIR = ROOT / "results" / "global_judge"
OUT = Path(__file__).resolve().parent / "error_examples_by_formtype.md"

FORM_ORDER = ["near-flat", "categorical", "non-monotonic", "monotonic"]
FAIL_THRESHOLD = 3  # faithfulness <= this counts as a failure worth showing

# Authored interpretation per shape type (the excerpts below substantiate it).
INTERPRETATION = {
    "near-flat": (
        "The dominant failure. Two recurring modes: (1) **missing negligibility** — "
        "the feature is described as a meaningful driver instead of being flagged as "
        "negligible (e.g. 'a strong monotone decline reaching -0.7'); (2) **fabricated "
        "structure** — inventing a peak / non-monotonic shape the flat curve does not "
        "have. All four handover formats over-attribute here; the `template` baseline "
        "is not worse than the LLM formats on this stratum (n = 5 per format, so no "
        "ordering among them is read into it)."
    ),
    "categorical": (
        "Failures come from **mis-ranking the levels**: naming the wrong best/worst "
        "category (e.g. calling Sunday a top-positive weekday when it is the most "
        "negative), or retreating to a vague 'spread / variability' narrative instead "
        "of stating which levels are highest and lowest."
    ),
    "non-monotonic": (
        "Mostly handled well. The residual failure is **missing the downturn**: "
        "claiming a monotone rise and overlooking the inverted-U peak."
    ),
    "monotonic": (
        "At ceiling: every format, including the `template` baseline, captures the "
        "step change and its direction."
    ),
}


def load_by_formtype() -> dict[str, list[dict]]:
    by = defaultdict(list)
    for p in sorted(JUDGE_DIR.glob("*.json")):
        r = json.loads(p.read_text())
        by[r["form_type"]].append(r)
    return by


def build_report() -> str:
    by = load_by_formtype()
    lines: list[str] = []
    w = lines.append

    w("# Qualitative error analysis — global explanations by shape type\n")
    w("Reference-based judge (`claude-opus-4-8`) `faithfulness` reasoning for the "
      "**failure cases** (score ≤ {}), grouped by ground-truth shape type. "
      "Auto-generated from `results/global_judge/` by "
      "`analyses/generate_error_examples.py`.\n".format(FAIL_THRESHOLD))

    # summary table
    w("## Faithfulness by shape type\n")
    w("| shape type | mean | n | score distribution |")
    w("|---|---|---|---|")
    for ft in FORM_ORDER:
        rs = by.get(ft, [])
        if not rs:
            continue
        mean = sum(r["faithfulness"] for r in rs) / len(rs)
        dist = {s: sum(1 for r in rs if r["faithfulness"] == s)
                for s in sorted({r["faithfulness"] for r in rs})}
        dist_s = ", ".join(f"{k}:{v}" for k, v in dist.items())
        w(f"| {ft} | {mean:.2f} | {len(rs)} | {dist_s} |")
    w("")

    for ft in FORM_ORDER:
        rs = by.get(ft, [])
        if not rs:
            continue
        w(f"## {ft}\n")
        w(f"**Failure modes.** {INTERPRETATION.get(ft, '')}\n")
        fails = sorted((r for r in rs if r["faithfulness"] <= FAIL_THRESHOLD),
                       key=lambda r: r["faithfulness"])
        if not fails:
            w("_No failures at this threshold — all explanations scored above "
              f"{FAIL_THRESHOLD}._\n")
            continue
        w(f"**Failure excerpts ({len(fails)} of {len(rs)}):**\n")
        for r in fails:
            tag = f"{r['form_pipeline']} · {r['xai_model']} · {r['feature']}"
            w(f"- **[faith={r['faithfulness']}] {tag}** — "
              f"{r['faithfulness_reasoning']}")
        w("")

    return "\n".join(lines) + "\n"


def main() -> None:
    report = build_report()
    OUT.write_text(report)
    print(f"wrote {OUT.relative_to(ROOT)} ({report.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
