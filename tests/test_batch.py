"""
Phase 3a·B — Tests für den Anthropic-Batches-Helfer (utils/batch.py).

Deckt ab: custom_id-Roundtrip/Validierung, max_tokens=0-Verbot,
Ergebnis-Klassifikation (succeeded/errored-invalid/errored-server/expired/
canceled), Ergebnis-Mapping mit/ohne parse-Callback, Resubmit transienter
Fehler (server/expired), Logging-statt-Verwerfen bei invalid_request,
max_resubmits-Erschöpfung sowie batch_id-Persistenz + Poll-Resume.

Der Anthropic-Client wird durch einen Fake ersetzt; `sleep` ist injiziert,
sodass kein echter Netz-/Wartecall stattfindet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import batch
from utils.batch import (
    make_custom_id,
    message_request,
    classify_result,
    collect_results,
    submit_batch,
    wait_for_batch,
    run_batch,
    message_text,
    message_usage,
    STATUS_SUCCEEDED,
    STATUS_INVALID_REQUEST,
    STATUS_SERVER_ERROR,
    STATUS_EXPIRED,
    STATUS_CANCELED,
)
from utils.judge import parse_judge_response


# ── Fake-Client ──────────────────────────────────────────────────────────────

def _succeeded(cid: str, text: str = "ok", in_tok: int = 3, out_tok: int = 5) -> dict:
    return {
        "custom_id": cid,
        "result": {
            "type": "succeeded",
            "message": {
                "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
            },
        },
    }


def _errored(cid: str, etype: str) -> dict:
    return {"custom_id": cid, "result": {"type": "errored", "error": {"type": etype}}}


def _expired(cid: str) -> dict:
    return {"custom_id": cid, "result": {"type": "expired"}}


def _canceled(cid: str) -> dict:
    return {"custom_id": cid, "result": {"type": "canceled"}}


class FakeBatches:
    """Minimaler Stand-in für client.messages.batches."""

    def __init__(self):
        self.rounds: list[list[dict]] = []      # je create()-Aufruf eine Ergebnisliste
        self._store: dict[str, list[dict]] = {}  # batch_id -> Ergebnisliste
        self.created: list[tuple[str, list[str]]] = []
        self.retrieve_calls: list[str] = []
        self._n = 0

    def queue(self, results: list[dict]) -> "FakeBatches":
        self.rounds.append(results)
        return self

    def preset_batch(self, batch_id: str, results: list[dict]) -> None:
        self._store[batch_id] = results

    # --- SDK-Schnittstelle ---
    def create(self, requests):
        batch_id = f"batch_{self._n}"
        self._n += 1
        results = self.rounds.pop(0)
        self._store[batch_id] = results
        self.created.append((batch_id, [r["custom_id"] for r in requests]))
        return {"id": batch_id}

    def retrieve(self, batch_id):
        self.retrieve_calls.append(batch_id)
        return {"processing_status": "ended"}

    def results(self, batch_id):
        return iter(self._store[batch_id])


def make_client(batches: FakeBatches):
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


NOSLEEP = lambda _s: None  # noqa: E731


# ── 1. custom_id ─────────────────────────────────────────────────────────────

def test_make_custom_id_joins_parts():
    assert make_custom_id("gen", "json", "xgb", 1041, "g0") == "gen-json-xgb-1041-g0"


def test_make_custom_id_sanitizes_invalid_chars():
    assert make_custom_id("jdg", "v3", "pipeline 06") == "jdg-v3-pipeline_06"


def test_make_custom_id_rejects_too_long():
    with pytest.raises(ValueError):
        make_custom_id("x" * 65)


# ── 2. message_request / max_tokens=0 ────────────────────────────────────────

def test_message_request_shape():
    params = {"model": "claude-sonnet-4-6", "max_tokens": 900, "messages": []}
    req = message_request("gen-1", params)
    assert req == {"custom_id": "gen-1", "params": params}


def test_message_request_rejects_max_tokens_zero():
    with pytest.raises(ValueError, match="max_tokens=0"):
        message_request("gen-1", {"model": "m", "max_tokens": 0, "messages": []})


def test_message_request_rejects_bad_custom_id():
    with pytest.raises(ValueError):
        message_request("bad id!", {"max_tokens": 10})


# ── 3. classify_result (reine Funktion) ──────────────────────────────────────

def test_classify_succeeded_extracts_text_and_usage():
    c = classify_result(_succeeded("a", text="hallo", in_tok=7, out_tok=11))
    assert c["status"] == STATUS_SUCCEEDED
    assert c["text"] == "hallo"
    assert c["usage"] == {"input_tokens": 7, "output_tokens": 11}


def test_classify_errored_invalid_request():
    c = classify_result(_errored("a", "invalid_request"))
    assert c["status"] == STATUS_INVALID_REQUEST


def test_classify_errored_server_error():
    c = classify_result(_errored("a", "overloaded"))
    assert c["status"] == STATUS_SERVER_ERROR


def test_classify_expired_and_canceled():
    assert classify_result(_expired("a"))["status"] == STATUS_EXPIRED
    assert classify_result(_canceled("a"))["status"] == STATUS_CANCELED


# ── 4. Schema-Gleichheit batch ↔ real-time (Text/Usage) ──────────────────────

def test_message_text_and_usage_match_realtime_schema():
    msg = {"content": [{"type": "text", "text": " text "}],
           "usage": {"input_tokens": 2, "output_tokens": 4}}
    assert message_text(msg) == "text"                      # getrimmt wie ask_text
    assert message_usage(msg) == {"input_tokens": 2, "output_tokens": 4}


# ── 5. collect_results: Eimer + parse-Callback ───────────────────────────────

def test_collect_results_buckets():
    batches = FakeBatches()
    batches.preset_batch("b", [
        _succeeded("ok1"),
        _errored("srv", "overloaded"),
        _expired("exp"),
        _errored("inv", "invalid_request"),
        _canceled("can"),
    ])
    out = collect_results("b", client=make_client(batches))
    assert set(out["succeeded"]) == {"ok1"}
    assert set(out["resubmit"]) == {"srv", "exp"}
    assert set(out["invalid"]) == {"inv"}
    assert set(out["canceled"]) == {"can"}


def test_collect_results_applies_parse_to_judge_text():
    judge_xml = (
        "<faithfulness>5</faithfulness>"
        "<clarity>4</clarity>"
        "<completeness>3</completeness>"
    )
    batches = FakeBatches()
    batches.preset_batch("b", [_succeeded("jdg-1", text=judge_xml)])
    out = collect_results("b", client=make_client(batches), parse=parse_judge_response)
    assert out["succeeded"]["jdg-1"]["faithfulness"] == 5
    assert out["succeeded"]["jdg-1"]["clarity"] == 4
    assert out["succeeded"]["jdg-1"]["completeness"] == 3


# ── 6. submit / wait ─────────────────────────────────────────────────────────

def test_submit_batch_returns_id_and_persists(tmp_path):
    batches = FakeBatches().queue([_succeeded("a")])
    state = tmp_path / "state.json"
    reqs = [message_request("a", {"model": "m", "max_tokens": 10, "messages": []})]
    bid = submit_batch(reqs, client=make_client(batches), state_path=state)
    assert bid == "batch_0"
    assert json.loads(state.read_text())["batch_id"] == "batch_0"


def test_submit_batch_rejects_empty():
    with pytest.raises(ValueError):
        submit_batch([], client=make_client(FakeBatches()))


def test_wait_for_batch_returns_when_ended():
    batches = FakeBatches()
    b = wait_for_batch("b", client=make_client(batches), sleep=NOSLEEP)
    assert b["processing_status"] == "ended"
    assert batches.retrieve_calls == ["b"]


def test_wait_for_batch_times_out():
    class NeverEnds(FakeBatches):
        def retrieve(self, batch_id):
            return {"processing_status": "in_progress"}

    with pytest.raises(TimeoutError):
        wait_for_batch("b", client=make_client(NeverEnds()),
                       poll_interval_s=0, timeout_s=-1, sleep=NOSLEEP)


# ── 7. run_batch: Happy Path ─────────────────────────────────────────────────

def _req(cid: str) -> dict:
    return message_request(cid, {"model": "m", "max_tokens": 10, "messages": []})


def test_run_batch_all_succeed():
    batches = FakeBatches().queue([_succeeded("a"), _succeeded("b")])
    out = run_batch([_req("a"), _req("b")], client=make_client(batches), sleep=NOSLEEP)
    assert set(out["succeeded"]) == {"a", "b"}
    assert out["failed"] == {}
    assert len(batches.created) == 1


# ── 8. run_batch: Resubmit transienter Fehler ────────────────────────────────

def test_run_batch_resubmits_server_error_then_succeeds():
    batches = FakeBatches()
    batches.queue([_succeeded("a"), _errored("b", "overloaded")])  # Runde 1
    batches.queue([_succeeded("b")])                                # Runde 2 (resubmit)
    out = run_batch([_req("a"), _req("b")], client=make_client(batches), sleep=NOSLEEP)
    assert set(out["succeeded"]) == {"a", "b"}
    assert out["failed"] == {}
    # Zweiter Batch enthält nur den fehlgeschlagenen Request.
    assert len(batches.created) == 2
    assert batches.created[1][1] == ["b"]


def test_run_batch_resubmits_expired():
    batches = FakeBatches()
    batches.queue([_expired("a")])
    batches.queue([_succeeded("a")])
    out = run_batch([_req("a")], client=make_client(batches), sleep=NOSLEEP)
    assert set(out["succeeded"]) == {"a"}


# ── 9. run_batch: invalid_request wird geloggt, nicht resubmittet ────────────

def test_run_batch_invalid_request_is_failed_not_resubmitted():
    batches = FakeBatches().queue([_succeeded("a"), _errored("b", "invalid_request")])
    out = run_batch([_req("a"), _req("b")], client=make_client(batches), sleep=NOSLEEP)
    assert set(out["succeeded"]) == {"a"}
    assert set(out["failed"]) == {"b"}
    assert out["failed"]["b"]["status"] == STATUS_INVALID_REQUEST
    assert len(batches.created) == 1   # kein Resubmit


def test_run_batch_canceled_is_failed():
    batches = FakeBatches().queue([_canceled("a")])
    out = run_batch([_req("a")], client=make_client(batches), sleep=NOSLEEP)
    assert set(out["failed"]) == {"a"}


# ── 10. run_batch: max_resubmits-Erschöpfung ─────────────────────────────────

def test_run_batch_exhausts_max_resubmits():
    batches = FakeBatches()
    # Bei jedem Versuch erneut server_error → resubmit, bis Limit erreicht.
    for _ in range(5):
        batches.queue([_errored("a", "overloaded")])
    out = run_batch([_req("a")], client=make_client(batches),
                    max_resubmits=2, sleep=NOSLEEP)
    assert "a" in out["failed"]
    assert out["failed"]["a"]["status"] == "max_resubmits_exceeded"
    # 1 Erst-Submit + 2 Resubmits = 3 Batches.
    assert len(batches.created) == 3


# ── 11. batch_id-Persistenz + Poll-Resume ────────────────────────────────────

def test_run_batch_resumes_persisted_batch(tmp_path):
    """Bei vorhandener state-Datei wird der persistierte Batch gepollt, nicht neu eingereicht."""
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"batch_id": "resumed_batch", "n_requests": 1}))

    batches = FakeBatches()
    batches.preset_batch("resumed_batch", [_succeeded("a")])

    out = run_batch([_req("a")], client=make_client(batches),
                    state_path=state, sleep=NOSLEEP)

    assert set(out["succeeded"]) == {"a"}
    assert batches.created == []                       # kein neuer create()-Call
    assert batches.retrieve_calls == ["resumed_batch"]


def test_run_batch_persists_then_processes(tmp_path):
    state = tmp_path / "state.json"
    batches = FakeBatches().queue([_succeeded("a")])
    run_batch([_req("a")], client=make_client(batches),
              state_path=state, sleep=NOSLEEP)
    # Nach dem Erst-Submit wurde die batch_id persistiert.
    assert json.loads(state.read_text())["batch_id"] == "batch_0"
