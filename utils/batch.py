"""
utils/batch.py - Anthropic Message Batches helper.

A pure cost lever: the Batches API processes Messages API
requests asynchronously at about 50 percent of the standard price. This helper
wraps submit -> wait -> collect over

    client.messages.batches.create | retrieve | results

and is deliberately cut so that the batch and the real time path produce schema
identical results (same text/usage per unit), so the eval (NB 05/06) stays
execution mode agnostic.

Key properties:
  * Unique `custom_id`s - `make_custom_id()` assembles them from parts and checks
    the Anthropic constraints (<= 64 chars, `[A-Za-z0-9_-]`).
  * `batch_id` persistence - `submit_batch(..., state_path=...)` writes the
    `batch_id` to disk so a crashed poll can be resumed (poll resume) instead of
    resubmitting the same batch.
  * Result mapping - `collect_results()` maps text + usage per `custom_id`; an
    optional `parse` callback (for example `parse_judge_response`) turns the text
    directly into scores.
  * Error classes - `errored`(server) / `expired` -> resubmit batch;
    `errored`(invalid_request) -> logged, not silently dropped; `canceled` ->
    recorded as a failure.
  * `max_tokens=0` is not allowed in batches - `message_request()` rejects it
    before the batch is submitted.

The SDK types (`Request`, `MessageCreateParamsNonStreaming`) are TypedDicts; at
runtime they are plain dicts. The helper therefore builds the request shape
`{"custom_id": ..., "params": ...}` directly and imports the `anthropic` package
only at the actual API call (`_get_client`), so the module import works without
the SDK installed (tests inject a fake client).
"""

from __future__ import annotations

import json
import logging
import re
import time as _time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

logger = logging.getLogger(__name__)

# --- Status constants ---
# Result types of the Batches API (result.result.type).
STATUS_SUCCEEDED       = "succeeded"
STATUS_INVALID_REQUEST = "invalid_request"   # errored, error.type == "invalid_request"
STATUS_SERVER_ERROR    = "server_error"      # errored, other error.type -> resubmit
STATUS_EXPIRED         = "expired"           # 24 h exceeded -> resubmit
STATUS_CANCELED        = "canceled"

# Error classes that justify a resubmit (transient).
_RESUBMITTABLE = frozenset({STATUS_SERVER_ERROR, STATUS_EXPIRED})

_CUSTOM_ID_RE  = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_BATCH_WAIT_S = 24 * 60 * 60  # Anthropic: a batch ends after 24 h at the latest.


# --- custom_id ---
def make_custom_id(*parts: Any) -> str:
    """Build a unique, API conformant `custom_id` from the parts.

    Parts are joined with ``-``; characters other than ``[A-Za-z0-9_-]`` are
    replaced by ``_``. Examples:
        make_custom_id("gen", pipeline, xai, iid, f"g{gen}")
        make_custom_id("jdg", ver, pipeline, xai, iid, f"s{k}")

    Raises
    ------
    ValueError
        If the result is empty or longer than 64 chars (Anthropic limit).
    """
    raw = "-".join(str(p) for p in parts)
    cid = re.sub(r"[^A-Za-z0-9_-]", "_", raw)
    if not _CUSTOM_ID_RE.match(cid):
        raise ValueError(
            f"Invalid custom_id {cid!r} (1 to 64 chars from [A-Za-z0-9_-] required)."
        )
    return cid


def message_request(custom_id: str, params: dict) -> dict:
    """Build a single batch request shape ``{"custom_id", "params"}``.

    `params` must match the Messages API parameters (model, max_tokens, messages,
    optional system/temperature/...), compatible with
    ``MessageCreateParamsNonStreaming``.

    Raises
    ------
    ValueError
        On an invalid `custom_id` or ``max_tokens == 0`` (not allowed in batches).
    """
    if not _CUSTOM_ID_RE.match(custom_id):
        raise ValueError(f"Invalid custom_id {custom_id!r}.")
    if params.get("max_tokens", None) == 0:
        raise ValueError(
            "max_tokens=0 is not allowed in the Batches API. "
            "Do not use batches for pre warming / empty requests."
        )
    return {"custom_id": custom_id, "params": params}


# --- Client / access helpers ---
def _get_client() -> Any:
    """Lazy: imports the Anthropic client only at the API call."""
    from utils.llm import _get_client as _llm_client  # uses the same key logic
    return _llm_client()


def _attr(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from an object attribute or a dict (SDK object vs test dict)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def message_text(message: Any) -> str:
    """Extract the first text block of a (batch) message, SDK object or dict."""
    content = _attr(message, "content", []) or []
    for block in content:
        if _attr(block, "type") == "text" or _attr(block, "text") is not None:
            return (_attr(block, "text", "") or "").strip()
    return ""


def message_usage(message: Any) -> dict:
    """Extract input_tokens/output_tokens, same schema as the real time path."""
    usage = _attr(message, "usage", {}) or {}
    return {
        "input_tokens":  _attr(usage, "input_tokens", 0) or 0,
        "output_tokens": _attr(usage, "output_tokens", 0) or 0,
    }


# --- submit / wait / collect ---
def submit_batch(
    requests: list[dict],
    *,
    client: Any = None,
    state_path: Optional[Path | str] = None,
) -> str:
    """Submit `requests` as one batch and return the `batch_id`.

    `requests` is a list of :func:`message_request` shapes. If `state_path` is set
    the `batch_id` is persisted (poll resume after a crash).
    """
    if not requests:
        raise ValueError("Empty request list, nothing to submit.")
    client = client or _get_client()

    batch = client.messages.batches.create(requests=requests)
    batch_id = _attr(batch, "id")
    logger.info("Batch submitted: %s (%d requests)", batch_id, len(requests))

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
    """Poll `retrieve` until ``processing_status == "ended"`` and return the batch.

    `sleep`/`on_poll` are injectable (tests, progress display).

    Raises
    ------
    TimeoutError
        If the batch is not finished after `timeout_s`.
    """
    client = client or _get_client()
    deadline = _time.monotonic() + timeout_s
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = _attr(batch, "processing_status")
        if on_poll is not None:
            on_poll(batch)
        if status == "ended":
            return batch
        if _time.monotonic() >= deadline:
            raise TimeoutError(
                f"Batch {batch_id} not finished after {timeout_s:.0f}s (status: {status})."
            )
        sleep(poll_interval_s)


def classify_result(result: Any) -> dict:
    """Classify a single batch result entry (pure function).

    Return dict per `custom_id`:
        status == STATUS_SUCCEEDED        -> + "text", "usage", "message"
        status == STATUS_INVALID_REQUEST  -> + "error"  (log, do not resubmit)
        status == STATUS_SERVER_ERROR     -> + "error"  (resubmit)
        status == STATUS_EXPIRED          -> (resubmit)
        status == STATUS_CANCELED         -> (failure)
    """
    custom_id = _attr(result, "custom_id")
    inner = _attr(result, "result")
    rtype = _attr(inner, "type")

    if rtype == "succeeded":
        message = _attr(inner, "message")
        return {
            "custom_id": custom_id,
            "status": STATUS_SUCCEEDED,
            "text": message_text(message),
            "usage": message_usage(message),
            "message": message,
        }
    if rtype == "errored":
        error = _attr(inner, "error")
        etype = _attr(error, "type")
        status = STATUS_INVALID_REQUEST if etype == "invalid_request" else STATUS_SERVER_ERROR
        return {"custom_id": custom_id, "status": status, "error": error}
    if rtype == "expired":
        return {"custom_id": custom_id, "status": STATUS_EXPIRED}
    if rtype == "canceled":
        return {"custom_id": custom_id, "status": STATUS_CANCELED}

    # Unknown type -> defensively treat as a server error (resubmit).
    return {"custom_id": custom_id, "status": STATUS_SERVER_ERROR, "error": rtype}


def iter_results(batch_id: str, *, client: Any = None) -> Iterator[dict]:
    """Iterate the results of a finished batch as classified dicts."""
    client = client or _get_client()
    for result in client.messages.batches.results(batch_id):
        yield classify_result(result)


def collect_results(
    batch_id: str,
    *,
    client: Any = None,
    parse: Optional[Callable[[str], Any]] = None,
) -> dict:
    """Collect the results of a finished batch into four buckets.

    `parse` (optional) is applied to the text of successful answers, for example
    ``utils.judge.parse_judge_response`` for judge batches. Without `parse`,
    ``succeeded[cid]`` holds the raw dict from :func:`classify_result`
    (text/usage/message), so the generation path can build the same record as the
    real time loop.

    Returns::

        {
            "succeeded":  {custom_id: parse(text) | classify-dict},
            "resubmit":   {custom_id: classify-dict},  # server_error / expired
            "invalid":    {custom_id: classify-dict},  # invalid_request (logged)
            "canceled":   {custom_id: classify-dict},
        }
    """
    succeeded: dict[str, Any] = {}
    resubmit: dict[str, dict] = {}
    invalid: dict[str, dict] = {}
    canceled: dict[str, dict] = {}

    for entry in iter_results(batch_id, client=client):
        cid, status = entry["custom_id"], entry["status"]
        if status == STATUS_SUCCEEDED:
            succeeded[cid] = parse(entry["text"]) if parse is not None else entry
        elif status in _RESUBMITTABLE:
            resubmit[cid] = entry
            logger.warning("Resubmit candidate %s (%s).", cid, status)
        elif status == STATUS_INVALID_REQUEST:
            invalid[cid] = entry
            logger.error("invalid_request for %s, not resubmitted: %s",
                         cid, entry.get("error"))
        else:  # canceled
            canceled[cid] = entry
            logger.error("Batch request %s canceled.", cid)

    return {
        "succeeded": succeeded,
        "resubmit":  resubmit,
        "invalid":   invalid,
        "canceled":  canceled,
    }


# --- Orchestration: submit -> wait -> collect -> resubmit ---
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
    """Run a batch end to end and resubmit transient errors.

    Flow per round: submit (or resume a persisted batch) -> wait -> collect.
    `errored`(server)/`expired` are resubmitted as a new batch up to
    `max_resubmits` times; `invalid_request` and `canceled` are logged and marked
    as finally failed.

    Poll resume: if a `batch_id` already exists under `state_path`, that batch is
    polled in the first round instead of submitting a new one (the request list
    must match the persisted batch).

    Returns::

        {"succeeded": {cid: ...}, "failed": {cid: classify-dict}, "batch_id": ...}
    """
    client = client or _get_client()
    by_id = {r["custom_id"]: r for r in requests}

    succeeded: dict[str, Any] = {}
    failed: dict[str, dict] = {}
    pending = list(requests)

    # Poll resume: resume the persisted batch_id in the first round.
    batch_id: Optional[str] = (
        _load_batch_id(state_path) if state_path is not None else None
    )
    if batch_id is not None:
        logger.info("Resuming persisted batch: %s", batch_id)

    attempt = 0
    while pending:
        if batch_id is None:
            batch_id = submit_batch(pending, client=client, state_path=state_path)
        wait_for_batch(
            batch_id, client=client, poll_interval_s=poll_interval_s,
            timeout_s=timeout_s, sleep=sleep,
        )
        collected = collect_results(batch_id, client=client, parse=parse)

        succeeded.update(collected["succeeded"])
        for cid, entry in collected["invalid"].items():
            failed[cid] = entry
        for cid, entry in collected["canceled"].items():
            failed[cid] = entry

        resubmit_ids = list(collected["resubmit"])
        batch_id = None  # the next round submits a fresh batch
        if not resubmit_ids:
            break

        attempt += 1
        if attempt > max_resubmits:
            logger.error("max_resubmits (%d) exhausted; %d requests remain open.",
                         max_resubmits, len(resubmit_ids))
            for cid in resubmit_ids:
                entry = dict(collected["resubmit"][cid])
                entry["status"] = "max_resubmits_exceeded"
                failed[cid] = entry
            break

        logger.info("Resubmit round %d: %d requests.", attempt, len(resubmit_ids))
        pending = [by_id[cid] for cid in resubmit_ids]

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
