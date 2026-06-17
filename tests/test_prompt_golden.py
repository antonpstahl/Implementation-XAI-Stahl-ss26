"""
Phase 3a — Prompt-Fix-Regression: Golden-Test.

Friert die SHA-256-Hashes der korrigierten Prompt-Dateien (Phase 3) ein.
Schlägt sofort fehl, wenn eine Prompt-Datei versehentlich zurückgerollt
oder verändert wird — bevor der teure Vollauf (Phase 3b) gestartet wird.

Wenn ein Prompt absichtlich verbessert wird:
  1. Neue Datei speichern
  2. Hash neu berechnen: shasum -a 256 prompts/<datei>.md
  3. GOLDEN_HASHES in dieser Datei aktualisieren
  4. pytest tests/test_prompt_golden.py grün bestätigen
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

ROOT        = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"

GOLDEN_HASHES: dict[str, str] = {
    "pipeline_04_json.md":  "3eca688e5e4a952f33065ef6cec86d4320ce9de803a9f5d917a630391e3fb90a",
    "pipeline_05_vision.md": "791d56d8f7257308c8adc9b4509473772e2e1b762b3c61d8010d35bd29fb86e7",
    "pipeline_06_tooluse.md": "6801db6db83b02621f84e733bb4e4a8d26d94b0ec6492c5f4878b9b75a692c30",
}


@pytest.mark.parametrize("filename,expected_hash", GOLDEN_HASHES.items())
def test_prompt_file_hash(filename: str, expected_hash: str) -> None:
    """Prompt-Datei darf sich seit dem Phase-3-Fix nicht geändert haben."""
    path = PROMPTS_DIR / filename
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected_hash, (
        f"\nPrompt '{filename}' hat sich seit dem Phase-3-Fix geändert.\n"
        f"  erwartet: {expected_hash}\n"
        f"  aktuell:  {actual}\n\n"
        "Wenn die Änderung beabsichtigt ist (verbesserter Prompt):\n"
        "  GOLDEN_HASHES in tests/test_prompt_golden.py aktualisieren.\n"
        "Wenn nicht:\n"
        "  git diff prompts/ prüfen und Rollback durchführen."
    )
