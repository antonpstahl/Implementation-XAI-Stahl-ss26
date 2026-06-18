"""utils/batch_openai.py – OpenAI-Batch-Helfer für den Cross-Vendor-Judge (Phase 3b).

Analog zu :mod:`utils.batch` (Anthropic), aber für die **OpenAI Batch API**
(`/v1/chat/completions`, −50 %, asynchron < 24 h, kein RPM-Limit). Damit ist der
Cross-Vendor-Judge (gpt-4o-mini) auch bei n≈200 bezahlbar **und** ohne den
20-s-Free-Tier-Delay durchführbar.

Der OpenAI-Flow ist datei-basiert (≠ Anthropics Inline-Requests):

    1. JSONL bauen   – je Zeile {"custom_id", "method":"POST",
                       "url":"/v1/chat/completions", "body": {...}}
    2. Upload        – client.files.create(purpose="batch")            → file_id
    3. Batch starten – client.batches.create(endpoint=…, window="24h") → batch_id
    4. Pollen        – client.batches.retrieve(batch_id)               → status
    5. Einsammeln    – client.files.content(output_file_id | error_file_id)

Schema-Gleichheit zum Real-time-Pfad (`utils.llm.ask_openai_text`):
`message_text`/`message_usage` liefern denselben Text bzw.
`input_tokens`/`output_tokens`, sodass `parse_judge_response` unverändert greift.

Fehlerklassen (wie utils.batch): 200 → succeeded; 429/5xx/expired → resubmit;
sonstige 4xx/Batch-`failed` → invalid (geloggt, nicht still verworfen);
cancelled → canceled. Der Client ist injizierbar (Tests nutzen einen Fake).
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


# ── Request-Shape ────────────────────────────────────────────────────────────
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
    """Baut eine JSONL-Batch-Zeile ``{custom_id, method, url, body}``.

    Raises
    ------
    ValueError
        Bei ``max_tokens == 0`` (in Batches unzulässig).
    """
    if body.get("max_tokens", None) == 0:
        raise ValueError("max_tokens=0 ist in der Batch-API unzulässig.")
    return {"custom_id": custom_id, "method": "POST", "url": _ENDPOINT, "body": body}


# ── Text/Usage-Extraktion (schema-gleich zu ask_openai_text) ─────────────────
def message_text(body: Any) -> str:
    """Erster Choice-Message-Content aus einem Chat-Completions-Body."""
    choices = _attr(body, "choices", []) or []
    if not choices:
        return ""
    message = _attr(choices[0], "message", {}) or {}
    return (_attr(message, "content", "") or "").strip()


def message_usage(body: Any) -> dict:
    """OpenAI usage → {input_tokens, output_tokens} (wie der Real-time-Pfad)."""
    usage = _attr(body, "usage", {}) or {}
    return {
        "input_tokens":  _attr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": _attr(usage, "completion_tokens", 0) or 0,
    }


def classify_result(line: Any) -> dict:
    """Klassifiziert eine Ergebniszeile (reine Funktion) — Out- **oder** Error-File.

    Erfolg: ``response.status_code == 200`` → text/usage. Sonst nach Status-Code:
    429 / ≥ 500 → server_error (resubmit); übrige 4xx → invalid_request; fehlende
    Response mit ``error`` → invalid_request (geloggt). Mappt auf dieselben
    Status-Konstanten wie :mod:`utils.batch`.
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

    # Keine Response → Fehlereintrag (Error-File). Default: invalid (nicht resubmitten).
    return {"custom_id": custom_id, "status": STATUS_INVALID_REQUEST, "error": error}


# ── Client-Zugriff ───────────────────────────────────────────────────────────
def _get_client() -> Any:
    """Lazy: erstellt den OpenAI-Client erst beim API-Aufruf (Tests injizieren)."""
    import os

    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise ImportError("Paket 'openai' nicht installiert (`pip install openai`).") from e
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY nicht gesetzt.")
    return OpenAI(api_key=api_key)


def _read_file_content(client: Any, file_id: Optional[str]) -> str:
    """Lädt ein OpenAI-File als Text (out-/error-file); leer bei fehlendem id."""
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


# ── submit / wait / collect ──────────────────────────────────────────────────
def submit_batch(
    requests: list[dict],
    *,
    client: Any = None,
    state_path: Optional[Path | str] = None,
) -> str:
    """Lädt `requests` als JSONL hoch, startet einen Batch, gibt `batch_id` zurück."""
    if not requests:
        raise ValueError("Leere Request-Liste – nichts einzureichen.")
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
    """Pollt `retrieve`, bis ein terminaler Status erreicht ist; gibt Batch zurück."""
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
                f"OpenAI-Batch {batch_id} nach {timeout_s:.0f}s nicht beendet "
                f"(Status: {status})."
            )
        sleep(poll_interval_s)


def collect_results(
    batch: Any,
    *,
    client: Any = None,
    parse: Optional[Callable[[str], Any]] = None,
) -> dict:
    """Sammelt Out-/Error-File eines beendeten Batches in vier Eimer (wie utils.batch).

    Batch-Status ``expired`` → alle (nicht gelieferten) als resubmit; ``failed`` /
    ``cancelled`` werden über die Error-File-Zeilen abgebildet. `parse` (optional)
    wird auf den Text erfolgreicher Antworten angewandt.
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
            logger.warning("Resubmit-Kandidat %s (%s).", cid, st)
        elif st == STATUS_CANCELED:
            canceled[cid] = entry
        else:
            invalid[cid] = entry
            logger.error("invalid_request für %s – nicht resubmittet: %s",
                         cid, entry.get("error"))

    if status == "expired":
        logger.warning("OpenAI-Batch %s expired.", _attr(batch, "id"))

    return {"succeeded": succeeded, "resubmit": resubmit,
            "invalid": invalid, "canceled": canceled}


# ── Orchestrierung ───────────────────────────────────────────────────────────
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
    """submit → wait → collect → resubmit, end-to-end (vgl. utils.batch.run_batch).

    Resubmittet transiente Fehler (429/5xx/expired) bis `max_resubmits`; jede
    Resubmit-Runde umfasst auch Requests, die im Batch gar nicht zurückkamen
    (z. B. bei `expired`). Poll-Resume über `state_path`.

    Rückgabe::

        {"succeeded": {cid: …}, "failed": {cid: classify-dict}, "batch_id": …}
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
            logger.error("max_resubmits (%d) erschöpft; %d Requests offen.",
                         max_resubmits, len(resubmit_ids))
            for cid in resubmit_ids:
                entry = dict(collected["resubmit"].get(cid, {"custom_id": cid}))
                entry["status"] = "max_resubmits_exceeded"
                failed[cid] = entry
            break

        logger.info("OpenAI-Resubmit-Runde %d: %d Requests.", attempt, len(resubmit_ids))
        pending = [by_id[cid] for cid in dict.fromkeys(resubmit_ids)]

    return {"succeeded": succeeded, "failed": failed, "batch_id": batch_id}


# ── batch_id-Persistenz ──────────────────────────────────────────────────────
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
    except (json.JSONDecodeError, OSError):
        return None
