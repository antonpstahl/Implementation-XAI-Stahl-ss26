"""
Phase 3a — Prompt-Fix-Regression: Golden-Test.

Test 1 — SHA-256-Hash-Check:
  Friert die exakten Byte-Inhalte aller Prompt-Dateien ein.
  Schlägt bei jeder Änderung fehl, bevor der teure Vollauf (Phase 3b) startet.

  Wenn ein Prompt absichtlich verbessert wird:
    1. Neue Datei speichern
    2. Hash neu berechnen: shasum -a 256 prompts/<datei>.md
    3. GOLDEN_HASHES in dieser Datei aktualisieren
    4. pytest tests/test_prompt_golden.py grün bestätigen

  judge_system.md ist in GOLDEN_HASHES aufgenommen (Phase 3·2/A4), weil
  der Judge-Prompt die Messung bestimmt und Änderungen explizit bestätigt
  werden sollen.

Test 2 — Key-Phrase-Assertion:
  Prüft die semantisch kritischen Sätze des Phase-3-Fixes (yr-Vorzeichen +
  Rangregel) direkt als Textsubstring — lesbarer Fehler bei Regression.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

ROOT        = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"

GOLDEN_HASHES: dict[str, str] = {
    # Phase 3·2/B5: Few-shot-Kalibrierung ergänzt (je 1 BEISPIEL-Sektion in 04/05/06;
    # 3 Ankerbeispiele hoch/mittel/niedrig in judge_system.md).
    # Davor (Phase 3·2/A3): Prompts harmonisiert (gleiches Wortbudget 150-250,
    # identische Stil-/Schema-/Rollen-Bausteine; nur der modalspezifische Teil abweichend).
    "pipeline_04_json.md":   "d4ab8961740d7c01882086f17e99b11962aca3e6c567a7aeae8bf4b123232e6d",
    "pipeline_05_vision.md": "9bc1615b3dbe31891d3a9b75311eb13c7c372f59981010bc60d6c9c051804da8",
    "pipeline_06_tooluse.md": "c328eb0c67ee3fd0fcf27ac5dfb38425c1ab601509c713f8d420fa02afc3f7e7",
    # Judge-Prompt eingefroren (Phase 3·2/A4+B5): bestimmt die Messung;
    # Änderungen müssen explizit bestätigt werden.
    "judge_system.md":       "775af71edcc5d397d22fa7d279c16c597d8a035cae4b3716912ca56414007f32",
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


# ---------------------------------------------------------------------------
# Test 2 — Key-Phrase-Assertion (Phase-3-spezifische Constraints)
# ---------------------------------------------------------------------------

# (filename, phrase, label) — jede fehlende Phrase ist ein eigener Testfall.
REQUIRED_PHRASES: list[tuple[str, str, str]] = [
    # yr-Vorzeichenregel (dominante Fehlerklasse C aus Phase 3)
    ("pipeline_04_json.md",  "yr=0 (2011) mit negativem",                         "yr-sign-fix"),
    ("pipeline_05_vision.md", "ein blauer yr-Balken (yr=0, 2011) ist ein dämpfender Faktor", "yr-sign-fix"),
    ("pipeline_06_tooluse.md", "yr=0 (2011) mit negativem Beitrag",                "yr-sign-fix"),
    # Rangregel
    ("pipeline_04_json.md",   "**Rang bindend**",  "rank-rule"),
    ("pipeline_05_vision.md", "**Rang bindend**",  "rank-rule"),
    ("pipeline_06_tooluse.md", "**Rang bindend**", "rank-rule"),
]

_PHRASE_IDS = [f"{fn.replace('pipeline_', 'p').replace('.md', '')}/{label}"
               for fn, _, label in REQUIRED_PHRASES]


@pytest.mark.parametrize("filename,phrase,label", REQUIRED_PHRASES, ids=_PHRASE_IDS)
def test_prompt_contains_phase3_phrase(filename: str, phrase: str, label: str) -> None:
    """Kritische Phase-3-Constraint-Sätze müssen verbatim im Prompt enthalten sein."""
    text = (PROMPTS_DIR / filename).read_text(encoding="utf-8")
    assert phrase in text, (
        f"\nPhase-3-Constraint '{label}' fehlt in '{filename}'.\n"
        f"  Erwarteter Substring:\n    {phrase!r}\n\n"
        "Ursache: yr-Vorzeichenfehler-Fix oder Rangregel wurde entfernt/verändert.\n"
        "Prompt wiederherstellen oder REQUIRED_PHRASES anpassen, falls bewusst geändert."
    )
