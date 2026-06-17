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
    # Phase 3·2/B7+C: XML-Abschnitts-Tags (<vorhersage>/<treiber>/<empfehlung>),
    # positive Instruktionen (keine Verbote), Halluzinations-Notausgang in allen
    # drei Generierungs-Prompts. Davor (B6): Scratchpad; (B5): Few-shot; (A3): Harmonisierung.
    "pipeline_04_json.md":   "76e4efa276360c45cf6acf83c05aa260a7aab8d8dd438c95440bec4632f02155",
    "pipeline_05_vision.md": "c1ed676bbcd7037fc2067a91c7de71131dc8d4c9ca1e92f1359d78960b6108ed",
    "pipeline_06_tooluse.md": "0562c817c8e2b4de3ec5d6070b84580cc48606c11857091f8e4a8c5ea5fa1bd8",
    # Judge-Prompt (Phase 3·2/B7+C): AUSGABEFORMAT-Sektion mit XML-Schema ergänzt;
    # Ankerbeispiele auf XML-Format umgestellt; Änderungen explizit bestätigt.
    "judge_system.md":       "c19bafcc3ac584075ec5af640dfe8fce781fa360709a7ae05e82019edfa7e60f",
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


# ---------------------------------------------------------------------------
# Test 3 — strip_scratchpad (Phase 3·2/B6)
# ---------------------------------------------------------------------------

import sys
sys.path.insert(0, str(ROOT))
from utils.llm import strip_scratchpad  # noqa: E402


@pytest.mark.parametrize("raw,expected", [
    # Block wird entfernt, XML-Prosa bleibt (B7: <vorhersage>-Tags)
    (
        "<analyse>\nhr=8: positiv, Rang 1\nyr=0: negativ, Rang 2\n</analyse>\n\n<vorhersage>Text.</vorhersage>",
        "<vorhersage>Text.</vorhersage>",
    ),
    # Kein Block — Eingabe unverändert
    (
        "<vorhersage>Kein Scratchpad.</vorhersage>",
        "<vorhersage>Kein Scratchpad.</vorhersage>",
    ),
    # Block mit CRLF
    (
        "<analyse>\r\nhr=8: positiv\r\n</analyse>\r\n<vorhersage>CRLF-Text.</vorhersage>",
        "<vorhersage>CRLF-Text.</vorhersage>",
    ),
    # Mehrere Blöcke (robustness)
    (
        "<analyse>A</analyse>\n<analyse>B</analyse>\n<vorhersage>Doppelt.</vorhersage>",
        "<vorhersage>Doppelt.</vorhersage>",
    ),
    # Leerer Block
    (
        "<analyse></analyse>\n<vorhersage>Leer.</vorhersage>",
        "<vorhersage>Leer.</vorhersage>",
    ),
])
def test_strip_scratchpad(raw: str, expected: str) -> None:
    """strip_scratchpad entfernt <analyse>-Blöcke und lässt die Prosa unverändert."""
    assert strip_scratchpad(raw) == expected
