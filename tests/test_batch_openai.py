"""Phase 3b — utils.batch_openai (OpenAI Batch API für den Cross-Vendor-Judge).

Spiegelt tests/test_batch.py: custom_id-Roundtrip, Ergebnis-Klassifikation,
Schema-Gleichheit zum Real-time-Pfad, Resubmit bei server_error, invalid_request-
Logging und Poll-Resume-Persistenz. Der OpenAI-Client wird gefaket (datei-basiert:
files.create/content, batches.create/retrieve).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import batch_openai as bo
from utils.batch import (
    STATUS_INVALID_REQUEST,
    STATUS_SERVER_ERROR,
    STATUS_SUCCEEDED,
    make_custom_id,
)


# ---------------------------------------------------------------------------
# Result-line builders (OpenAI batch output/error JSONL schema)
# ---------------------------------------------------------------------------

def _success_line(cid, content="<faithfulness>5</faithfulness>", pin=100, pout=50):
    return {
        "custom_id": cid,
        "response": {
            "status_code": 200,
            "body": {
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": pin, "completion_tokens": pout},
            },
        },
        "error": None,
    }


def _error_line(cid, status_code):
    return {
        "custom_id": cid,
        "response": {"status_code": status_code, "body": {"error": {"message": "x"}}},
        "error": None,
    }


# ---------------------------------------------------------------------------
# Fake OpenAI client — scripted rounds of {cid: line}
# ---------------------------------------------------------------------------

class _FakeContent:
    def __init__(self, text): self.text = text


class _FakeBatch(dict):
    """dict so _attr() reads it; mirrors OpenAI batch object fields."""


class FakeOpenAI:
    def __init__(self, rounds: list[dict], batch_status="completed"):
        """rounds: list per submit — {cid: success_line | error_line}."""
        self._rounds = rounds
        self._batch_status = batch_status
        self._files: dict[str, str] = {}
        self._batches: dict[str, _FakeBatch] = {}
        self.submit_count = 0
        self._fid = 0

        class _Files:
            def __init__(self, outer): self.o = outer
            def create(self, file, purpose):
                self.o._fid += 1
                return {"id": f"file-in-{self.o._fid}"}
            def content(self, file_id):
                return _FakeContent(self.o._files.get(file_id, ""))

        class _Batches:
            def __init__(self, outer): self.o = outer
            def create(self, input_file_id, endpoint, completion_window):
                idx = self.o.submit_count
                self.o.submit_count += 1
                lines = self.o._rounds[idx] if idx < len(self.o._rounds) else {}
                out = [l for l in lines.values() if l["response"]["status_code"] == 200]
                err = [l for l in lines.values() if l["response"]["status_code"] != 200]
                bid = f"batch-{idx}"
                ofid = f"file-out-{idx}" if out else None
                efid = f"file-err-{idx}" if err else None
                if ofid:
                    self.o._files[ofid] = "\n".join(json.dumps(x) for x in out)
                if efid:
                    self.o._files[efid] = "\n".join(json.dumps(x) for x in err)
                b = _FakeBatch(id=bid, status=self.o._batch_status,
                               output_file_id=ofid, error_file_id=efid)
                self.o._batches[bid] = b
                return b
            def retrieve(self, batch_id):
                return self.o._batches[batch_id]

        self.files = _Files(self)
        self.batches = _Batches(self)


def _no_sleep(_): pass


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------

def test_build_chat_body_includes_temperature_only_when_set():
    b = bo.build_chat_body("p", system="s", model="gpt-4o-mini", max_tokens=300)
    assert "temperature" not in b
    assert b["messages"][0]["role"] == "system"
    assert b["messages"][1]["content"] == "p"
    b2 = bo.build_chat_body("p", model="gpt-4o-mini", max_tokens=300, temperature=0.0)
    assert b2["temperature"] == 0.0
    assert b2["messages"][0]["role"] == "user"  # no system


def test_chat_request_rejects_zero_max_tokens():
    body = bo.build_chat_body("p", model="gpt-4o-mini", max_tokens=0)
    with pytest.raises(ValueError, match="max_tokens=0"):
        bo.chat_request("cid", body)


def test_chat_request_shape():
    body = bo.build_chat_body("p", model="gpt-4o-mini", max_tokens=300)
    req = bo.chat_request("jdg-x", body)
    assert req["custom_id"] == "jdg-x"
    assert req["method"] == "POST"
    assert req["url"] == "/v1/chat/completions"


# ---------------------------------------------------------------------------
# classify_result + text/usage extraction
# ---------------------------------------------------------------------------

def test_classify_success_extracts_text_and_usage():
    entry = bo.classify_result(_success_line("c1", content="hello", pin=12, pout=7))
    assert entry["status"] == STATUS_SUCCEEDED
    assert entry["text"] == "hello"
    assert entry["usage"] == {"input_tokens": 12, "output_tokens": 7}


@pytest.mark.parametrize("code,expected", [
    (429, STATUS_SERVER_ERROR),
    (500, STATUS_SERVER_ERROR),
    (503, STATUS_SERVER_ERROR),
    (400, STATUS_INVALID_REQUEST),
    (404, STATUS_INVALID_REQUEST),
])
def test_classify_status_codes(code, expected):
    assert bo.classify_result(_error_line("c", code))["status"] == expected


def test_classify_error_only_line_is_invalid():
    line = {"custom_id": "c", "response": None, "error": {"message": "bad"}}
    assert bo.classify_result(line)["status"] == STATUS_INVALID_REQUEST


# ---------------------------------------------------------------------------
# Roundtrip: submit → wait → collect
# ---------------------------------------------------------------------------

def test_run_batch_all_success_with_parse():
    cids = [make_custom_id("jdg", "oai", "JSON", "XGB", i) for i in (1, 2, 3)]
    reqs = [bo.chat_request(c, bo.build_chat_body("p", model="gpt-4o-mini",
                                                  max_tokens=300)) for c in cids]
    rounds = [{c: _success_line(c, content=f"r-{c}") for c in cids}]
    client = FakeOpenAI(rounds)

    out = bo.run_batch(reqs, client=client, parse=lambda t: {"text": t},
                       sleep=_no_sleep, poll_interval_s=0)
    assert client.submit_count == 1
    assert set(out["succeeded"]) == set(cids)
    assert out["succeeded"][cids[0]] == {"text": f"r-{cids[0]}"}
    assert out["failed"] == {}


def test_run_batch_resubmits_server_error():
    cids = ["a", "b"]
    reqs = [bo.chat_request(c, bo.build_chat_body("p", model="gpt-4o-mini",
                                                  max_tokens=300)) for c in cids]
    # Round 0: a ok, b 500. Round 1 (resubmit b): b ok.
    rounds = [
        {"a": _success_line("a"), "b": _error_line("b", 500)},
        {"b": _success_line("b")},
    ]
    client = FakeOpenAI(rounds)
    out = bo.run_batch(reqs, client=client, sleep=_no_sleep, poll_interval_s=0)
    assert client.submit_count == 2
    assert set(out["succeeded"]) == {"a", "b"}
    assert out["failed"] == {}


def test_run_batch_invalid_request_not_resubmitted():
    reqs = [bo.chat_request("a", bo.build_chat_body("p", model="gpt-4o-mini",
                                                    max_tokens=300))]
    rounds = [{"a": _error_line("a", 400)}]
    client = FakeOpenAI(rounds)
    out = bo.run_batch(reqs, client=client, sleep=_no_sleep, poll_interval_s=0)
    assert client.submit_count == 1            # no resubmit on 400
    assert "a" in out["failed"]
    assert out["succeeded"] == {}


def test_run_batch_max_resubmits_exhausted():
    reqs = [bo.chat_request("a", bo.build_chat_body("p", model="gpt-4o-mini",
                                                    max_tokens=300))]
    rounds = [{"a": _error_line("a", 500)} for _ in range(5)]
    client = FakeOpenAI(rounds)
    out = bo.run_batch(reqs, client=client, max_resubmits=2,
                       sleep=_no_sleep, poll_interval_s=0)
    # initial + 2 resubmits = 3 submits
    assert client.submit_count == 3
    assert out["failed"]["a"]["status"] == "max_resubmits_exceeded"


# ---------------------------------------------------------------------------
# Poll-resume persistence
# ---------------------------------------------------------------------------

def test_submit_persists_batch_id_and_resume(tmp_path):
    state = tmp_path / "_oai_state.json"
    reqs = [bo.chat_request("a", bo.build_chat_body("p", model="gpt-4o-mini",
                                                    max_tokens=300))]
    rounds = [{"a": _success_line("a")}]
    client = FakeOpenAI(rounds)
    bid = bo.submit_batch(reqs, client=client, state_path=state)
    assert json.loads(state.read_text())["batch_id"] == bid
    # resume: run_batch finds the persisted batch and polls it without re-creating
    out = bo.run_batch(reqs, client=client, state_path=state,
                       sleep=_no_sleep, poll_interval_s=0)
    assert client.submit_count == 1            # no second create
    assert "a" in out["succeeded"]
