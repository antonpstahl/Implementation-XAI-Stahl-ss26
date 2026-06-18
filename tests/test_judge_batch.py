"""
Phase 3a·B / A2 — Tests für den Batch-Judge mit Self-Consistency (utils.judge_batch_sc).

Deckt ab:
  * k Requests pro Eintrag werden als ein Batch eingereicht.
  * custom_ids haben das Schema {base_cid}-s0 … -s{k-1}.
  * Median-Aggregation client-seitig über k Samples (je Kriterium).
  * Fehlgeschlagene Samples werden aus dem Median ausgeschlossen.
  * Basis-CID ohne jedes erfolgreiche Sample → None-Scores.
  * Schema-Gleichheit mit judge_with_self_consistency (same keys).
  * base_cid-Längenvalidierung (> 61 Zeichen → ValueError).
  * temperature wird bei Opus-Modellen weggelassen (model_accepts_temperature).

Der Anthropic-Client wird durch den Fake aus test_batch ersetzt.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.judge import judge_batch_sc, SCORE_KEYS, parse_judge_response
from tests.test_batch import FakeBatches, make_client, NOSLEEP


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _xml(f: int, cl: int, co: int) -> str:
    return (
        f"<faithfulness_reasoning>r</faithfulness_reasoning>"
        f"<faithfulness>{f}</faithfulness>"
        f"<clarity_reasoning>r</clarity_reasoning>"
        f"<clarity>{cl}</clarity>"
        f"<completeness_reasoning>r</completeness_reasoning>"
        f"<completeness>{co}</completeness>"
    )


def _mk_succeeded(cid: str, f: int, cl: int, co: int) -> dict:
    text = _xml(f, cl, co)
    return {
        "custom_id": cid,
        "result": {
            "type": "succeeded",
            "message": {
                "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": 5, "output_tokens": 10},
            },
        },
    }


def _mk_errored(cid: str, etype: str = "invalid_request") -> dict:
    return {"custom_id": cid, "result": {"type": "errored", "error": {"type": etype}}}


ENTRIES = [
    ("jdg-v1-JSON_Text-xgb-224", "Erklärung 224"),
    ("jdg-v1-JSON_Text-xgb-580", "Erklärung 580"),
]

K = 3
MODEL = "claude-sonnet-4-6"


def _run(batches: FakeBatches, entries=ENTRIES, k=K, **kwargs) -> dict:
    return judge_batch_sc(
        entries,
        system="sys",
        model=MODEL,
        max_tokens=900,
        k=k,
        client=make_client(batches),
        sleep=NOSLEEP,
        **kwargs,
    )


# ── 1. k Requests pro Eintrag ────────────────────────────────────────────────

def test_submits_k_requests_per_entry():
    n = len(ENTRIES)
    results = [_mk_succeeded(f"{e[0]}-s{j}", 4, 3, 4) for e in ENTRIES for j in range(K)]
    batches = FakeBatches().queue(results)

    _run(batches)

    assert len(batches.created) == 1
    submitted_ids = batches.created[0][1]
    assert len(submitted_ids) == n * K
    for base_cid, _ in ENTRIES:
        for j in range(K):
            assert f"{base_cid}-s{j}" in submitted_ids


# ── 2. Median-Aggregation ────────────────────────────────────────────────────

def test_median_of_three_scores():
    # Sample scores: (5,4,3), (3,3,3), (4,4,4) → median: (4,4,3)
    entry = ENTRIES[0]
    score_sets = [(5, 4, 3), (3, 3, 3), (4, 4, 4)]
    results = [_mk_succeeded(f"{entry[0]}-s{j}", *s) for j, s in enumerate(score_sets)]
    # Second entry: all same scores
    results += [_mk_succeeded(f"{ENTRIES[1][0]}-s{j}", 2, 2, 2) for j in range(K)]
    batches = FakeBatches().queue(results)

    out = _run(batches)

    r = out[entry[0]]
    assert r["faithfulness"]  == 4   # median(5,3,4)
    assert r["clarity"]       == 4   # median(4,3,4)
    assert r["completeness"]  == 3   # median(3,3,4)


def test_median_two_samples_even():
    # k=2: (4,2) → median of [4,2] = 3.0 → int 3
    entry = ENTRIES[0]
    results = [
        _mk_succeeded(f"{entry[0]}-s0", 4, 4, 4),
        _mk_succeeded(f"{entry[0]}-s1", 2, 2, 2),
        _mk_succeeded(f"{ENTRIES[1][0]}-s0", 3, 3, 3),
        _mk_succeeded(f"{ENTRIES[1][0]}-s1", 3, 3, 3),
    ]
    batches = FakeBatches().queue(results)

    out = _run(batches, k=2)
    assert out[entry[0]]["faithfulness"] == 3


# ── 3. Fehlgeschlagene Samples ausgeschlossen ────────────────────────────────

def test_failed_sample_excluded_from_median():
    entry = ENTRIES[0]
    results = [
        _mk_succeeded(f"{entry[0]}-s0", 5, 5, 5),
        _mk_errored(f"{entry[0]}-s1", "invalid_request"),  # ausgeschlossen
        _mk_succeeded(f"{entry[0]}-s2", 3, 3, 3),
    ] + [_mk_succeeded(f"{ENTRIES[1][0]}-s{j}", 3, 3, 3) for j in range(K)]
    batches = FakeBatches().queue(results)

    out = _run(batches)
    # Median of [5, 3] = 4
    assert out[entry[0]]["faithfulness"] == 4


def test_all_samples_failed_gives_none_scores():
    entry = ENTRIES[0]
    results = [_mk_errored(f"{entry[0]}-s{j}", "invalid_request") for j in range(K)]
    results += [_mk_succeeded(f"{ENTRIES[1][0]}-s{j}", 3, 3, 3) for j in range(K)]
    batches = FakeBatches().queue(results)

    out = _run(batches)
    r = out[entry[0]]
    assert r["faithfulness"]  is None
    assert r["clarity"]       is None
    assert r["completeness"]  is None


# ── 4. Rückgabe-Schema identisch zu judge_with_self_consistency ───────────────

def test_return_schema_matches_self_consistency():
    results = [_mk_succeeded(f"{e[0]}-s{j}", 4, 3, 5) for e in ENTRIES for j in range(K)]
    batches = FakeBatches().queue(results)

    out = _run(batches)
    r = out[ENTRIES[0][0]]

    # Alle erwarteten Keys
    for key in SCORE_KEYS:
        assert key in r
    assert "faithfulness_reasoning"  in r
    assert "clarity_reasoning"       in r
    assert "completeness_reasoning"  in r
    assert "raw_responses"           in r
    assert isinstance(r["raw_responses"], list)
    assert "usage" in r
    assert "input_tokens"  in r["usage"]
    assert "output_tokens" in r["usage"]


# ── 5. Usage-Akkumulation ────────────────────────────────────────────────────

def test_usage_accumulated_over_k_samples():
    entry = ENTRIES[0]
    results = [_mk_succeeded(f"{entry[0]}-s{j}", 4, 4, 4) for j in range(K)]
    results += [_mk_succeeded(f"{ENTRIES[1][0]}-s{j}", 3, 3, 3) for j in range(K)]
    batches = FakeBatches().queue(results)

    out = _run(batches)
    # _mk_succeeded gibt in=5, out=10 → k=3 → 15, 30
    assert out[entry[0]]["usage"]["input_tokens"]  == 5 * K
    assert out[entry[0]]["usage"]["output_tokens"] == 10 * K


# ── 6. Rückgabe alle Einträge im dict ────────────────────────────────────────

def test_all_base_cids_in_result():
    results = [_mk_succeeded(f"{e[0]}-s{j}", 4, 3, 4) for e in ENTRIES for j in range(K)]
    batches = FakeBatches().queue(results)

    out = _run(batches)
    for base_cid, _ in ENTRIES:
        assert base_cid in out


# ── 7. base_cid-Längenvalidierung ───────────────────────────────────────────

def test_long_base_cid_raises():
    too_long = [("x" * 62, "prompt")]   # 62 + len("-s2") = 65 > 64
    with pytest.raises(ValueError, match="zu lang"):
        judge_batch_sc(too_long, system="s", model=MODEL, max_tokens=10, k=3,
                       client=make_client(FakeBatches()), sleep=NOSLEEP)


def test_base_cid_at_limit_ok():
    # 61 chars + "-s0" = 64 chars: should not raise
    ok_cid = "a" * 61
    results = [{"custom_id": f"{ok_cid}-s{j}", "result": {"type": "succeeded",
                "message": {"content": [{"type": "text", "text": _xml(4, 4, 4)}],
                             "usage": {"input_tokens": 1, "output_tokens": 1}}}}
               for j in range(3)]
    batches = FakeBatches().queue(results)
    out = judge_batch_sc([(ok_cid, "p")], system="s", model=MODEL, max_tokens=10, k=3,
                         client=make_client(batches), sleep=NOSLEEP)
    assert ok_cid in out


# ── 8. temperature-Gating (Opus → keine temperature) ────────────────────────

def test_opus_model_no_temperature_in_params():
    """Für Opus darf kein temperature-Feld in den batch-Requests auftauchen."""
    captured: list[dict] = []

    class CaptureBatches(FakeBatches):
        def create(self, requests):
            captured.extend(requests)
            # Alle succeeded zurückgeben
            result = []
            for r in requests:
                cid = r["custom_id"]
                result.append({"custom_id": cid, "result": {"type": "succeeded",
                    "message": {"content": [{"type": "text", "text": _xml(4, 4, 4)}],
                                "usage": {"input_tokens": 1, "output_tokens": 1}}}})
            self._store[f"batch_{self._n}"] = result
            bid = f"batch_{self._n}"; self._n += 1
            self.created.append((bid, [r["custom_id"] for r in requests]))
            return {"id": bid}

    judge_batch_sc(
        [("jdg-v3-test-xgb-1", "prompt")],
        system="sys",
        model="claude-opus-4-8",
        max_tokens=900,
        k=2,
        temperature=0.7,              # sollte weggelassen werden
        client=make_client(CaptureBatches()),
        sleep=NOSLEEP,
    )

    for req in captured:
        assert "temperature" not in req["params"], (
            f"temperature darf bei Opus nicht im Payload stehen: {req['params']}"
        )


def test_sonnet_model_temperature_in_params():
    """Für Sonnet wird temperature übergeben."""
    captured: list[dict] = []

    class CaptureBatches(FakeBatches):
        def create(self, requests):
            captured.extend(requests)
            result = []
            for r in requests:
                cid = r["custom_id"]
                result.append({"custom_id": cid, "result": {"type": "succeeded",
                    "message": {"content": [{"type": "text", "text": _xml(4, 4, 4)}],
                                "usage": {"input_tokens": 1, "output_tokens": 1}}}})
            self._store[f"batch_{self._n}"] = result
            bid = f"batch_{self._n}"; self._n += 1
            self.created.append((bid, [r["custom_id"] for r in requests]))
            return {"id": bid}

    judge_batch_sc(
        [("jdg-v1-test-xgb-1", "prompt")],
        system="sys",
        model="claude-sonnet-4-6",
        max_tokens=900,
        k=2,
        temperature=0.0,
        client=make_client(CaptureBatches()),
        sleep=NOSLEEP,
    )

    for req in captured:
        assert req["params"].get("temperature") == 0.0
