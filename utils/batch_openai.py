"""utils/batch_openai.py - OpenAI batch helper for the cross vendor judge.

Like :mod:`utils.batch` (Anthropic), but for the OpenAI Batch API
(`/v1/chat/completions`, about 50 percent cheaper, asynchronous < 24 h, no RPM
limit). This makes the cross vendor judge (gpt-4o-mini) affordable even at a
larger n and runnable without the 20 s free tier delay.

The OpenAI flow is file based (unlike Anthropics inline requests):

    1. Build JSONL   - per line {"custom_id", "method":"POST",
                       "url":"/v1/chat/completions", "body": {...}}
    2. Upload        - client.files.create(purpose="batch")            -> file_id
    3. Start batch   - client.batches.create(endpoint=..., window="24h") -> batch_id
    4. Poll          - client.batches.retrieve(batch_id)               -> status
    5. Collect       - client.files.content(output_file_id | error_file_id)

Schema equality with the real time path (`utils.llm.ask_openai_text`):
`message_text`/`message_usage` return the same text and
`input_tokens`/`output_tokens`, so `parse_judge_response` works unchanged.

Error classes (like utils.batch): 200 -> succeeded; 429/5xx/expired -> resubmit;
other 4xx/batch `failed` -> invalid (logged, not silently dropped);
cancelled -> canceled. The client is injectable (tests use a fake).
"""

from __future__ import annotations

import json
import logging
import time as _time
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from utils.batch import (  # Wiederverwendung: identische custom_id-Constraints/Status
    STATUS_CANCELED,
    STATUS_EXPIRED,
    STATUS_INVALID_REQUEST,
    STATUS_SERVER_ERROR,
    STATUS_SUCCEEDED,
    _attr,
    make_custom_id,
)

logger = logging.getLogger(__name__)

_RESUBMITTABLE = frozenset({STATUS_SERVER_ERROR, STATUS_EXPIRED})
_MAX_BATCH_WAIT_S = 24 * 60 * 60
_ENDPOINT = "/v1/chat/completions"
_TERMINAL = frozenset({"completed", "failed", "expired", "cancelled", "canceled"})


# --- Request shape ---
def build_chat_body(
    prompt: str,
    *,
    system: str | None = None,
    model: str,
    max_tokens: int,
    temperature: float | None = None,
) -> dict:
    """Baut den `/v1/chat/completions`-Body (schema-gleich zu ask_openai_text)."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body: dict = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        body["temperature"] = temperature
    return body


def chat_request(custom_id: str, body: dict) -> dict:
    """Build a JSONL batch line ``{custom_id, method, url, body}``.

    Raises
    ------
    ValueError
        On ``max_tokens == 0`` (not allowed in batches).
    """
    if body.get("max_tokens", None) == 0:
        raise ValueError("max_tokens=0 is not allowed in the Batch API.")
    return {"custom_id": custom_id, "method": "POST", "url": _ENDPOINT, "body": body}


# --- Text/usage extraction (same schema as ask_openai_text) ---
def message_text(body: Any) -> str:
    """First choice message content from a chat completions body."""
    choices = _attr(body, "choices", []) or []
    if not choices:
        return ""
    message = _attr(choices[0], "message", {}) or {}
    return (_attr(message, "content", "") or "").strip()


def message_usage(body: Any) -> dict:
    """OpenAI usage -> {input_tokens, output_tokens} (like the real time path)."""
    usage = _attr(body, "usage", {}) or {}
    return {
        "input_tokens":  _attr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": _attr(usage, "completion_tokens", 0) or 0,
    }


def classify_result(line: Any) -> dict:
    """Classify a result line (pure function), out file or error file.

    Success: ``response.status_code == 200`` -> text/usage. Otherwise by status
    code: 429 / >= 500 -> server_error (resubmit); other 4xx -> invalid_request;
    a missing response with ``error`` -> invalid_request (logged). Maps to the same
    status constants as :mod:`utils.batch`.
    """
    custom_id = _attr(line, "custom_id")
    response = _attr(line, "response")
    error = _attr(line, "error")

    if response is not None:
        status_code = _attr(response, "status_code")
        body = _attr(response, "body")
        if status_code == 200 and body is not None:
            return {
                "custom_id": custom_id,
                "status": STATUS_SUCCEEDED,
                "text": message_text(body),
                "usage": message_usage(body),
                "body": body,
            }
        if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
            return {"custom_id": custom_id, "status": STATUS_SERVER_ERROR,
                    "error": _attr(response, "body", status_code)}
        return {"custom_id": custom_id, "status": STATUS_INVALID_REQUEST,
                "error": _attr(response, "body", status_code)}

    # No response -> error entry (error file). Default: invalid (do not resubmit).
    return {"custom_id": custom_id, "status": STATUS_INVALID_REQUEST, "error": error}


# --- Client access ---
def _get_client() -> Any:
    """Lazy: creates the OpenAI client only at the API call (tests inject)."""
    import os

    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise ImportError("Package 'openai' not installed (`pip install openai`).") from e
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)


def _read_file_content(client: Any, file_id: Optional[str]) -> str:
    """Load an OpenAI file as text (out/error file); empty if the id is missing."""
    if not file_id:
        return ""
    content = client.files.content(file_id)
    if hasattr(content, "text"):
        return content.text
    if hasattr(content, "read"):
        raw = content.read()
        return raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    return str(content)


def _iter_jsonl(text: str) -> Iterator[dict]:
    for raw in text.splitlines():
        raw = raw.strip()
        if raw:
            yield json.loads(raw)


# --- submit / wait / collect ---
def submit_batch(
    requests: list[dict],
    *,
    client: Any = None,
    state_path: Optional[Path | str] = None,
) -> str:
    """Upload `requests` as JSONL, start a batch, return the `batch_id`."""
    if not requests:
        raise ValueError("Empty request list, nothing to submit.")
    client = client or _get_client()

    jsonl = "\n".join(json.dumps(r, ensure_ascii=False) for r in requests)
    file_obj = client.files.create(
        file=("batch_requests.jsonl", jsonl.encode("utf-8")),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=_attr(file_obj, "id"),
        endpoint=_ENDPOINT,
        completion_window="24h",
    )
    batch_id = _attr(batch, "id")
    logger.info("OpenAI-Batch eingereicht: %s (%d Requests)", batch_id, len(requests))
    if state_path is not None:
        _persist_batch_id(state_path, batch_id, len(requests))
    return batch_id


def wait_for_batch(
    batch_id: str,
    *,
    client: Any = None,
    poll_interval_s: float = 30.0,
    timeout_s: float = _MAX_BATCH_WAIT_S,
    sleep: Callable[[float], None] = _time.sleep,
    on_poll: Optional[Callable[[Any], None]] = None,
) -> Any:
    """Poll `retrieve` until a terminal status is reached; return the batch."""
    client = client or _get_client()
    deadline = _time.monotonic() + timeout_s
    while True:
        batch = client.batches.retrieve(batch_id)
        status = _attr(batch, "status")
        if on_poll is not None:
            on_poll(batch)
        if status in _TERMINAL:
            return batch
        if _time.monotonic() >= deadline:
            raise TimeoutError(
                f"OpenAI batch {batch_id} not finished after {timeout_s:.0f}s "
                f"(status: {status})."
            )
        sleep(poll_interval_s)


def collect_results(
    batch: Any,
    *,
    client: Any = None,
    parse: Optional[Callable[[str], Any]] = None,
) -> dict:
    """Collect the out/error file of a finished batch into four buckets (like utils.batch).

    Batch status ``expired`` -> all (not delivered) as resubmit; ``failed`` /
    ``cancelled`` are mapped via the error file lines. `parse` (optional) is applied
    to the text of successful answers.
    """
    client = client or _get_client()
    status = _attr(batch, "status")

    succeeded: dict[str, Any] = {}
    resubmit: dict[str, dict] = {}
    invalid: dict[str, dict] = {}
    canceled: dict[str, dict] = {}

    out_text = _read_file_content(client, _attr(batch, "output_file_id"))
    err_text = _read_file_content(client, _attr(batch, "error_file_id"))

    for line in list(_iter_jsonl(out_text)) + list(_iter_jsonl(err_text)):
        entry = classify_result(line)
        cid, st = entry["custom_id"], entry["status"]
        if st == STATUS_SUCCEEDED:
            succeeded[cid] = parse(entry["text"]) if parse is not None else entry
        elif st in _RESUBMITTABLE:
            resubmit[cid] = entry
            logger.warning("Resubmit candidate %s (%s).", cid, st)
        elif st == STATUS_CANCELED:
            canceled[cid] = entry
        else:
            invalid[cid] = entry
            logger.error("invalid_request for %s, not resubmitted: %s",
                         cid, entry.get("error"))

    if status == "expired":
        logger.warning("OpenAI batch %s expired.", _attr(batch, "id"))

    return {"succeeded": succeeded, "resubmit": resubmit,
            "invalid": invalid, "canceled": canceled}


# --- Orchestration ---
def run_batch(
    requests: list[dict],
    *,
    client: Any = None,
    parse: Optional[Callable[[str], Any]] = None,
    state_path: Optional[Path | str] = None,
    max_resubmits: int = 2,
    poll_interval_s: float = 30.0,
    timeout_s: float = _MAX_BATCH_WAIT_S,
    sleep: Callable[[float], None] = _time.sleep,
) -> dict:
    """submit -> wait -> collect -> resubmit, end to end (cf. utils.batch.run_batch).

    Resubmits transient errors (429/5xx/expired) up to `max_resubmits`; each
    resubmit round also includes requests that did not come back in the batch at
    all (for example on `expired`). Poll resume via `state_path`.

    Returns::

        {"succeeded": {cid: ...}, "failed": {cid: classify-dict}, "batch_id": ...}
    """
    client = client or _get_client()
    by_id = {r["custom_id"]: r for r in requests}

    succeeded: dict[str, Any] = {}
    failed: dict[str, dict] = {}
    pending = list(requests)

    batch_id: Optional[str] = (
        _load_batch_id(state_path) if state_path is not None else None
    )
    if batch_id is not None:
        logger.info("Setze persistierten OpenAI-Batch fort: %s", batch_id)

    attempt = 0
    while pending:
        if batch_id is None:
            batch_id = submit_batch(pending, client=client, state_path=state_path)
        batch = wait_for_batch(
            batch_id, client=client, poll_interval_s=poll_interval_s,
            timeout_s=timeout_s, sleep=sleep,
        )
        collected = collect_results(batch, client=client, parse=parse)

        succeeded.update(collected["succeeded"])
        for cid, entry in collected["invalid"].items():
            failed[cid] = entry
        for cid, entry in collected["canceled"].items():
            failed[cid] = entry

        # Resubmit: explizite Resubmit-Kandidaten + im Batch nie gelieferte Requests.
        returned = (set(collected["succeeded"]) | set(collected["resubmit"])
                    | set(collected["invalid"]) | set(collected["canceled"]))
        resubmit_ids = list(collected["resubmit"])
        if _attr(batch, "status") in {"expired", "failed", "cancelled", "canceled"}:
            for cid in by_id:
                if cid not in returned and cid not in succeeded and cid not in failed:
                    resubmit_ids.append(cid)

        batch_id = None
        if not resubmit_ids:
            break

        attempt += 1
        if attempt > max_resubmits:
            logger.error("max_resubmits (%d) exhausted; %d requests open.",
                         max_resubmits, len(resubmit_ids))
            for cid in resubmit_ids:
                entry = dict(collected["resubmit"].get(cid, {"custom_id": cid}))
                entry["status"] = "max_resubmits_exceeded"
                failed[cid] = entry
            break

        logger.info("OpenAI resubmit round %d: %d requests.", attempt, len(resubmit_ids))
        pending = [by_id[cid] for cid in dict.fromkeys(resubmit_ids)]

    return {"succeeded": succeeded, "failed": failed, "batch_id": batch_id}


# --- batch_id persistence ---
def _persist_batch_id(state_path: Path | str, batch_id: str, n_requests: int) -> None:
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "batch_id": batch_id,
        "n_requests": n_requests,
        "submitted_at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))


def _load_batch_id(state_path: Optional[Path | str]) -> Optional[str]:
    if state_path is None:
        return None
    path = Path(state_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("batch_id")
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read batch_id from %s (%s); a new batch will be submitted.", path, exc)
        return None
