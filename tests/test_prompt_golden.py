"""
Prompt regression golden test.

Test 1 - SHA-256 hash check:
  Freezes the exact byte contents of all prompt files.
  Fails on any change before the expensive full run starts.

  When a prompt is intentionally improved:
    1. Save the new file
    2. Recompute the hash: shasum -a 256 prompts/<file>.md
    3. Update GOLDEN_HASHES in this file
    4. Confirm pytest tests/test_prompt_golden.py is green

  judge_system.md is included in GOLDEN_HASHES because the judge prompt
  determines the measurement and changes should be confirmed explicitly.

Test 2 - key phrase assertion:
  Checks the semantically critical sentences of the prompt fix (yr sign rule +
  rank rule) directly as a text substring for a readable failure on regression.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

ROOT        = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"

GOLDEN_HASHES: dict[str, str] = {
    # English prompts with XML section tags (<analysis>/<prediction>/<drivers>/
    # <recommendation>), sign and rank rules, and the yr few shot calibration.
    "pipeline_04_json.md":   "89912121324f621f7a60fdf4fd58a0a42c7a363e3c5f05a9e4c36d26a934fe3c",
    "pipeline_05_vision.md": "c5c7a832dbf502088583cd02d7099de30031b328a8385e951f32e81e8011ce24",
    "pipeline_06_tooluse.md": "345503b794780b41102932ead0513a1e67f5085af8c2b37148b452c293491edf",
    "judge_system.md":       "01097af9368ba6e01c03aa382b4a6d76d9a0ca206615b5bcfa85fb64093329cc",
}


@pytest.mark.parametrize("filename,expected_hash", GOLDEN_HASHES.items())
def test_prompt_file_hash(filename: str, expected_hash: str) -> None:
    """A prompt file must not have changed since the frozen hash."""
    path = PROMPTS_DIR / filename
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected_hash, (
        f"\nPrompt '{filename}' has changed since the frozen hash.\n"
        f"  expected: {expected_hash}\n"
        f"  actual:   {actual}\n\n"
        "If the change is intentional (improved prompt):\n"
        "  update GOLDEN_HASHES in tests/test_prompt_golden.py.\n"
        "If not:\n"
        "  check git diff prompts/ and roll back."
    )


# ---------------------------------------------------------------------------
# Test 2 - key phrase assertion (prompt fix constraints)
# ---------------------------------------------------------------------------

# (filename, phrase, label) - each missing phrase is its own test case.
REQUIRED_PHRASES: list[tuple[str, str, str]] = [
    # yr sign rule (dominant error class C from the taxonomy)
    ("pipeline_04_json.md",   "yr=0 (2011) with a negative contribution", "yr-sign-fix"),
    ("pipeline_05_vision.md", "blue yr bar (yr=0, 2011)",                 "yr-sign-fix"),
    ("pipeline_06_tooluse.md", "yr=0 (2011) with a negative contribution","yr-sign-fix"),
    # rank rule
    ("pipeline_04_json.md",   "**Rank binding**",  "rank-rule"),
    ("pipeline_05_vision.md", "**Rank binding**",  "rank-rule"),
    ("pipeline_06_tooluse.md", "**Rank binding**", "rank-rule"),
]

_PHRASE_IDS = [f"{fn.replace('pipeline_', 'p').replace('.md', '')}/{label}"
               for fn, _, label in REQUIRED_PHRASES]


@pytest.mark.parametrize("filename,phrase,label", REQUIRED_PHRASES, ids=_PHRASE_IDS)
def test_prompt_contains_fix_phrase(filename: str, phrase: str, label: str) -> None:
    """Critical constraint sentences must appear verbatim in the prompt."""
    text = (PROMPTS_DIR / filename).read_text(encoding="utf-8")
    assert phrase in text, (
        f"\nConstraint '{label}' missing in '{filename}'.\n"
        f"  expected substring:\n    {phrase!r}\n\n"
        "Cause: the yr sign fix or the rank rule was removed or changed.\n"
        "Restore the prompt or adjust REQUIRED_PHRASES if changed on purpose."
    )


# ---------------------------------------------------------------------------
# Test 3 - strip_scratchpad
# ---------------------------------------------------------------------------

import sys
sys.path.insert(0, str(ROOT))
from utils.llm import strip_scratchpad  # noqa: E402


@pytest.mark.parametrize("raw,expected", [
    # Block is removed, the XML prose stays (<prediction> tags)
    (
        "<analysis>\nhr=8: positive, rank 1\nyr=0: negative, rank 2\n</analysis>\n\n<prediction>Text.</prediction>",
        "<prediction>Text.</prediction>",
    ),
    # No block, input unchanged
    (
        "<prediction>No scratchpad.</prediction>",
        "<prediction>No scratchpad.</prediction>",
    ),
    # Block with CRLF
    (
        "<analysis>\r\nhr=8: positive\r\n</analysis>\r\n<prediction>CRLF text.</prediction>",
        "<prediction>CRLF text.</prediction>",
    ),
    # Several blocks (robustness)
    (
        "<analysis>A</analysis>\n<analysis>B</analysis>\n<prediction>Double.</prediction>",
        "<prediction>Double.</prediction>",
    ),
    # Empty block
    (
        "<analysis></analysis>\n<prediction>Empty.</prediction>",
        "<prediction>Empty.</prediction>",
    ),
])
def test_strip_scratchpad(raw: str, expected: str) -> None:
    """strip_scratchpad removes <analysis> blocks and leaves the prose unchanged."""
    assert strip_scratchpad(raw) == expected
