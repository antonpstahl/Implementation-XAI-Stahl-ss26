"""
Data guard: no persisted `explanation` field leaks a `<thinking>` or `<analysis>`
scratchpad block.

Background: run 1 shipped with `strip_scratchpad` matching only `<analysis>`. The
generator sometimes emits `<thinking>` instead, so 14/72 G2a records + 13 variance
draws + 4 G2b raw records carried the raw scratchpad into the persisted
`explanation` and into the judge input. The audit found it 2026-09-29; the fix
regenerates the affected XGB records (they see new plots anyway) and re-strips
the rest (analyses/restrip_scratchpad.py). This test locks the persisted state
so a future regression cannot silently ship leaked records again.

Exempt: `results/global_variance/json_xgb_weekday_gen1.json` is a documented
truncated draw with an unclosed `<analysis>`. It is excluded from analysis
elsewhere (utils.rubric / 04Gf cell 7) and the unclosed block is intentional
evidence of the truncation.

Regex is case-sensitive to match `utils.llm.strip_scratchpad`, a hypothetical
`<Thinking>` tag flagged here but not stripped there would be a bug, not two
consistent decisions.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Only the record folders that carry `explanation`. Judge dirs are excluded
# because their JSON has no `explanation` field, so parametrising over them
# would inflate the visible test count without checking anything real.
SCAN_DIRS = [
    RESULTS / "global",
    RESULTS / "global_variance",
    RESULTS / "global_whole",
    RESULTS / "global_whole_split",
]

EXEMPT = {  # relative paths under results/
    "global_variance/json_xgb_weekday_gen1.json",
}

LEAK_RE = re.compile(r"<(analysis|thinking)>")  # case-sensitive: matches strip_scratchpad


def _find_leaks() -> list[str]:
    """Return relative paths of records whose `explanation` still carries a
    scratchpad opening tag. Exempt paths are always excluded."""
    leaks: list[str] = []
    for d in SCAN_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.json")):
            rel = p.relative_to(RESULTS).as_posix()
            if rel in EXEMPT:
                continue
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            expl = rec.get("explanation")
            if isinstance(expl, str) and LEAK_RE.search(expl):
                leaks.append(rel)
    return leaks


def test_no_leaked_scratchpad_tag() -> None:
    """Every persisted `explanation` must be free of scratchpad opening tags."""
    leaks = _find_leaks()
    assert leaks == [], (
        f"{len(leaks)} record(s) still leak <analysis>/<thinking>. "
        "Re-run analyses/restrip_scratchpad.py --apply and re-judge, or "
        "regenerate the record. Leaked paths (first 10):\n  "
        + "\n  ".join(leaks[:10])
    )
