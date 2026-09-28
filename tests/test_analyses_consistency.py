"""
Consistency check: the auto-generated analysis reports must match results/ on disk.

Mirrors test_readme_consistency.py. Both analyses/*.md files are regenerated from
results/global_judge/ (and explanations/) by their generator scripts; if someone
re-runs the judge or the ground-truth derivation without re-running the generators,
the committed Markdown silently goes stale (caught once already: after the P0-1
judge re-run, error_examples_by_formtype.md still showed the pre-fix per-shape-type
faithfulness means).

Fix:  python analyses/generate_gt_verification.py
      python analyses/generate_error_examples.py
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analyses"))

import generate_gt_verification as gt_gen
import generate_error_examples as err_gen

_REPORTS = [
    (gt_gen.OUT, gt_gen.build_report, "analyses/generate_gt_verification.py"),
    (err_gen.OUT, err_gen.build_report, "analyses/generate_error_examples.py"),
]


@pytest.mark.parametrize("path,build_fn,generator", _REPORTS,
                         ids=[p.name for p, _, _ in _REPORTS])
def test_report_matches_results(path: Path, build_fn, generator: str) -> None:
    """Committed report must equal what its generator produces from results/ now."""
    on_disk = path.read_text(encoding="utf-8")
    expected = build_fn()
    assert on_disk == expected, (
        f"\n{path.relative_to(ROOT)} is out of sync with results/.\n"
        f"Run:  python {generator}\n"
    )
