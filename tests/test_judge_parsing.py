"""Tests for judge parsing robustness.

Guards the robust JSON parsing + retry against regression.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.judge import parse_judge_response, judge_with_retry
from tests.fixtures_judge import ALL_FIXTURES, FIXTURE_MARKDOWN_CODEBLOCK


# --- Helper: build an ask_fn mock response ---

def _make_response(text: str) -> dict:
    return {
        "content": [{"text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


VALID_RAW = FIXTURE_MARKDOWN_CODEBLOCK["raw"]
GARBAGE_RAW = "Sorry, I cannot answer this request."


# --- 1. Unit tests: parse_judge_response against all fixtures ---

@pytest.mark.parametrize("name,fixture", ALL_FIXTURES)
def test_parse_judge_response_expected_keys(name, fixture):
    result = parse_judge_response(fixture["raw"])
    for key, value in fixture["expected"].items():
        assert result.get(key) == value, (
            f"[{name}] Key '{key}': expected {value!r}, got {result.get(key)!r}"
        )


@pytest.mark.parametrize("name,fixture", ALL_FIXTURES)
def test_parse_judge_response_no_extra_score_keys(name, fixture):
    """Scores not in expected must not appear (garbage protection)."""
    result = parse_judge_response(fixture["raw"])
    score_keys = {"faithfulness", "clarity", "completeness"}
    expected_scores = score_keys & set(fixture["expected"])
    result_scores = score_keys & set(result)
    assert result_scores == expected_scores, (
        f"[{name}] Unexpected score keys: {result_scores - expected_scores}"
    )


def test_parse_judge_response_garbage_returns_empty():
    result = parse_judge_response("lorem ipsum dolor sit amet")
    assert result == {}


def test_parse_judge_response_scores_are_int():
    result = parse_judge_response(VALID_RAW)
    for key in ("faithfulness", "clarity", "completeness"):
        assert isinstance(result[key], int), f"{key} should be int, is {type(result[key])}"


# --- 2. Mock test retry logic: garbage x 2, valid JSON on the 3rd call ---

def test_judge_with_retry_succeeds_on_third_attempt():
    ask_fn = MagicMock(side_effect=[
        _make_response(GARBAGE_RAW),   # attempt 1 -> no score
        _make_response(GARBAGE_RAW),   # attempt 2 -> no score
        _make_response(VALID_RAW),     # attempt 3 -> valid scores
    ])

    result = judge_with_retry(ask_fn, "prompt", "system", "model", max_retries=3)

    assert ask_fn.call_count == 3, "ask_fn should be called exactly 3 times"
    assert result["faithfulness"] == 5
    assert result["clarity"] == 4
    assert result["completeness"] == 4


def test_judge_with_retry_stops_early_on_success():
    """If the first call is already valid, no second call."""
    ask_fn = MagicMock(return_value=_make_response(VALID_RAW))

    result = judge_with_retry(ask_fn, "prompt", "system", "model", max_retries=3)

    assert ask_fn.call_count == 1
    assert result["faithfulness"] is not None


# --- 3. Exhausted retries -> None scores, no double counting ---

def test_judge_with_retry_exhausted_returns_none_scores():
    """All retries fail -> scores are None, no ValueError."""
    ask_fn = MagicMock(return_value=_make_response(GARBAGE_RAW))

    result = judge_with_retry(ask_fn, "prompt", "system", "model", max_retries=3)

    assert ask_fn.call_count == 3
    assert result["faithfulness"] is None
    assert result["clarity"] is None
    assert result["completeness"] is None


def test_judge_n_equals_n_with_partial_failures():
    """Simulates n=5 instances, 1 of which fails permanently.

    Judge_n (entries in the result) == n, no entry is dropped. None scores count as
    an entry but are not counted as a valid score.
    """
    # Instances 1 to 4 succeed on the first attempt; instance 5 fails all 3 retries.
    responses = (
        [_make_response(VALID_RAW)] * 4
        + [_make_response(GARBAGE_RAW)] * 3
    )
    ask_fn = MagicMock(side_effect=responses)

    n = 5
    rows = []
    for _ in range(n):
        rows.append(judge_with_retry(ask_fn, "prompt", "system", "model", max_retries=3))

    # Judge_n == n: every instance has an entry
    assert len(rows) == n

    # Valid scores only for the 4 successful instances
    valid = [r for r in rows if r["faithfulness"] is not None]
    failed = [r for r in rows if r["faithfulness"] is None]
    assert len(valid) == 4
    assert len(failed) == 1

    # No double counting: failed entries appear exactly once
    assert len(rows) == len(valid) + len(failed)
