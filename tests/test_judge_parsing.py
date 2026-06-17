"""Tests für Judge-Parsing-Robustheit (Phase-3a-Gate).

Sichert den Phase-0-Fix (robustes JSON-Parsing + Retry) dauerhaft gegen Regression ab.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.judge import parse_judge_response, judge_with_retry
from tests.fixtures_judge import ALL_FIXTURES, FIXTURE_MARKDOWN_CODEBLOCK


# ── Hilfsfunktion: baut eine ask_fn-Mock-Response ────────────────────────────

def _make_response(text: str) -> dict:
    return {
        "content": [{"text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


VALID_RAW = FIXTURE_MARKDOWN_CODEBLOCK["raw"]
GARBAGE_RAW = "Ich kann diese Anfrage leider nicht beantworten."


# ── 1. Unit-Tests: parse_judge_response gegen alle Fixtures ──────────────────

@pytest.mark.parametrize("name,fixture", ALL_FIXTURES)
def test_parse_judge_response_expected_keys(name, fixture):
    result = parse_judge_response(fixture["raw"])
    for key, value in fixture["expected"].items():
        assert result.get(key) == value, (
            f"[{name}] Key '{key}': expected {value!r}, got {result.get(key)!r}"
        )


@pytest.mark.parametrize("name,fixture", ALL_FIXTURES)
def test_parse_judge_response_no_extra_score_keys(name, fixture):
    """Scores die nicht in expected sind, sollen nicht auftauchen (Garbage-Schutz)."""
    result = parse_judge_response(fixture["raw"])
    score_keys = {"faithfulness", "clarity", "completeness"}
    expected_scores = score_keys & set(fixture["expected"])
    result_scores = score_keys & set(result)
    assert result_scores == expected_scores, (
        f"[{name}] Unerwartete Score-Keys: {result_scores - expected_scores}"
    )


def test_parse_judge_response_garbage_returns_empty():
    result = parse_judge_response("lorem ipsum dolor sit amet")
    assert result == {}


def test_parse_judge_response_scores_are_int():
    result = parse_judge_response(VALID_RAW)
    for key in ("faithfulness", "clarity", "completeness"):
        assert isinstance(result[key], int), f"{key} sollte int sein, ist {type(result[key])}"


# ── 2. Mock-Test Retry-Logik: Garbage × 2, valides JSON beim 3. Call ─────────

def test_judge_with_retry_succeeds_on_third_attempt():
    ask_fn = MagicMock(side_effect=[
        _make_response(GARBAGE_RAW),   # Versuch 1 → kein Score
        _make_response(GARBAGE_RAW),   # Versuch 2 → kein Score
        _make_response(VALID_RAW),     # Versuch 3 → valide Scores
    ])

    result = judge_with_retry(ask_fn, "prompt", "system", "model", max_retries=3)

    assert ask_fn.call_count == 3, "ask_fn soll genau 3× aufgerufen werden"
    assert result["faithfulness"] == 5
    assert result["clarity"] == 4
    assert result["completeness"] == 4


def test_judge_with_retry_stops_early_on_success():
    """Wenn der erste Call schon valide ist, kein zweiter Call."""
    ask_fn = MagicMock(return_value=_make_response(VALID_RAW))

    result = judge_with_retry(ask_fn, "prompt", "system", "model", max_retries=3)

    assert ask_fn.call_count == 1
    assert result["faithfulness"] is not None


# ── 3. Erschöpfte Retries → None-Scores, kein Doppelzählen ──────────────────

def test_judge_with_retry_exhausted_returns_none_scores():
    """Alle Retries scheitern → Scores sind None, kein ValueError."""
    ask_fn = MagicMock(return_value=_make_response(GARBAGE_RAW))

    result = judge_with_retry(ask_fn, "prompt", "system", "model", max_retries=3)

    assert ask_fn.call_count == 3
    assert result["faithfulness"] is None
    assert result["clarity"] is None
    assert result["completeness"] is None


def test_judge_n_equals_n_with_partial_failures():
    """Simuliert n=5 Instanzen, 1 davon schlägt dauerhaft fehl.

    Judge_n (Einträge im Ergebnis) == n — kein Eintrag wird ausgelassen.
    None-Scores zählen als Eintrag, werden aber nicht als valider Score gezählt.
    """
    # Instanzen 1-4 gelingen beim 1. Versuch; Instanz 5 scheitert bei allen 3 Retries.
    responses = (
        [_make_response(VALID_RAW)] * 4
        + [_make_response(GARBAGE_RAW)] * 3
    )
    ask_fn = MagicMock(side_effect=responses)

    n = 5
    rows = []
    for _ in range(n):
        rows.append(judge_with_retry(ask_fn, "prompt", "system", "model", max_retries=3))

    # Judge_n == n: alle Instanzen haben einen Eintrag
    assert len(rows) == n

    # Valide Scores nur für die 4 erfolgreichen Instanzen
    valid = [r for r in rows if r["faithfulness"] is not None]
    failed = [r for r in rows if r["faithfulness"] is None]
    assert len(valid) == 4
    assert len(failed) == 1

    # Kein Doppelzählen: failed-Einträge erscheinen genau einmal
    assert len(rows) == len(valid) + len(failed)
