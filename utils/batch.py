"""
utils/batch.py – Anthropic-Message-Batches-Helfer (Phase 3a·B).

Reiner Kostenhebel für den teuren 3b-Lauf: die Batches API verarbeitet
Messages-API-Anfragen asynchron zu −50 % der Standardpreise. Dieser Helfer
kapselt **submit → wait → collect** über

    client.messages.batches.create | retrieve | results

und ist bewusst so geschnitten, dass der Batch- und der Real-time-Pfad
**schema-identische** Ergebnisse liefern (gleicher Text/Usage je Einheit),
damit die Eval (NB 07/08) ausführungsart-agnostisch bleibt.

Kerneigenschaften (DoD Phase 3a·B):
  * **Eindeutige `custom_id`s** – `make_custom_id()` setzt sie aus Teilen
    zusammen und prüft Anthropic-Constraints (≤ 64 Zeichen, `[A-Za-z0-9_-]`).
  * **`batch_id`-Persistenz** – `submit_batch(..., state_path=…)` schreibt die
    `batch_id` auf Platte, sodass ein abgestürzter Poll wieder aufgenommen
    werden kann (Poll-Resume), statt denselben Batch erneut einzureichen.
  * **Ergebnis-Mapping** – `collect_results()` ordnet je `custom_id` Text +
    Usage zu; ein optionaler `parse`-Callback (z. B. `parse_judge_response`)
    wandelt den Text direkt in Scores um.
  * **Fehlerklassen** – `errored`(server) / `expired` → Resubmit-Batch;
    `errored`(invalid_request) → geloggt, nicht still verworfen; `canceled`
    → als Fehler vermerkt.
  * **`max_tokens=0` ist in Batches unzulässig** – `message_request()` lehnt es
    ab, bevor der Batch eingereicht wird.

Die SDK-Typen (`Request`, `MessageCreateParamsNonStreaming`) sind TypedDicts;
zur Laufzeit sind es schlichte Dicts. Der Helfer baut daher die Request-Shape
`{"custom_id": …, "params": …}` direkt und importiert das `anthropic`-Paket
erst beim tatsächlichen API-Aufruf (`_get_client`), damit der Modulimport ohne
installiertes SDK gelingt (Tests injizieren einen Fake-Client).
"""

from __future__ import annotations

import json
import logging
import re
import time as _time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

logger = logging.getLogger(__name__)

# ── Status-Konstanten ────────────────────────────────────────────────────────
# Ergebnis-Typen der Batches API (result.result.type).
STATUS_SUCCEEDED       = "succeeded"
STATUS_INVALID_REQUEST = "invalid_request"   # errored, error.type == "invalid_request"
STATUS_SERVER_ERROR    = "server_error"      # errored, sonstiger error.type → resubmit
STATUS_EXPIRED         = "expired"           # 24 h überschritten → resubmit
STATUS_CANCELED        = "canceled"

# Fehlerklassen, die einen Resubmit rechtfertigen (transient).
_RESUBMITTABLE = frozenset({STATUS_SERVER_ERROR, STATUS_EXPIRED})

_CUSTOM_ID_RE  = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_BATCH_WAIT_S = 24 * 60 * 60  # Anthropic: Batch endet spätestens nach 24 h.


# ── custom_id ──────────────────────────────────────────────────────────────
def make_custom_id(*parts: Any) -> str:
    """Baut eine eindeutige, API-konforme `custom_id` aus den Teilen.

    Teile werden mit ``-`` verbunden, andere Zeichen als ``[A-Za-z0-9_-]`` durch
    ``_`` ersetzt. Beispiele (Phase 3a·B):
        make_custom_id("gen", pipeline, xai, iid, f"g{gen}")
        make_custom_id("jdg", ver, pipeline, xai, iid, f"s{k}")

    Raises
    ------
    ValueError
        Wenn das Ergebnis leer oder länger als 64 Zeichen ist (Anthropic-Limit).
    """
    raw = "-".join(str(p) for p in parts)
    cid = re.sub(r"[^A-Za-z0-9_-]", "_", raw)
    if not _CUSTOM_ID_RE.match(cid):
        raise ValueError(
            f"Ungültige custom_id {cid!r} (1–64 Zeichen aus [A-Za-z0-9_-] nötig)."
        )
    return cid


def message_request(custom_id: str, params: dict) -> dict:
    """Baut eine einzelne Batch-Request-Shape ``{"custom_id", "params"}``.

    `params` muss den Messages-API-Parametern entsprechen (model, max_tokens,
    messages, optional system/temperature/…) – kompatibel zu
    ``MessageCreateParamsNonStreaming``.

    Raises
    ------
    ValueError
        Bei ungültiger `custom_id` oder ``max_tokens == 0`` (in Batches unzulässig).
    """
    if not _CUSTOM_ID_RE.match(custom_id):
        raise ValueError(f"Ungültige custom_id {custom_id!r}.")
    if params.get("max_tokens", None) == 0:
        raise ValueError(
            "max_tokens=0 ist in der Batches API unzulässig. "
            "Für Pre-Warming/Leeranfragen nicht über Batches gehen."
        )
    return {"custom_id": custom_id, "params": params}


# ── Client / Zugriffshelfer ──────────────────────────────────────────────────
def _get_client() -> Any:
    """Lazy: importiert den Anthropic-Client erst beim API-Aufruf."""
    from utils.llm import _get_client as _llm_client  # nutzt dieselbe Key-Logik
    return _llm_client()


def _attr(obj: Any, key: str, default: Any = None) -> Any:
    """Liest `key` aus Objekt-Attribut **oder** Dict (SDK-Objekt ↔ Test-Dict)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def message_text(message: Any) -> str:
    """Extrahiert den ersten Text-Block einer (Batch-)Message – SDK-Obj oder Dict."""
    content = _attr(message, "content", []) or []
    for block in content:
        if _attr(block, "type") == "text" or _attr(block, "text") is not None:
            return (_attr(block, "text", "") or "").strip()
    return ""


def message_usage(message: Any) -> dict:
    """Extrahiert input_tokens/output_tokens – schema-gleich zum Real-time-Pfad."""
    usage = _attr(message, "usage", {}) or {}
    return {
        "input_tokens":  _attr(usage, "input_tokens", 0) or 0,
        "output_tokens": _attr(usage, "output_tokens", 0) or 0,
    }


# ── submit / wait / collect ──────────────────────────────────────────────────
def submit_batch(
    requests: list[dict],
    *,
    client: Any = None,
    state_path: Optional[Path | str] = None,
) -> str:
    """Reicht `requests` als einen Batch ein und gibt die `batch_id` zurück.

    `requests` ist eine Liste aus :func:`message_request`-Shapes. Bei gesetztem
    `state_path` wird die `batch_id` persistiert (Poll-Resume nach Absturz).
    """
    if not requests:
        raise ValueError("Leere Request-Liste – nichts einzureichen.")
    client = client or _get_client()

    batch = client.messages.batches.create(requests=requests)
    batch_id = _attr(batch, "id")
    logger.info("Batch eingereicht: %s (%d Requests)", batch_id, len(requests))

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
    """Pollt `retrieve`, bis ``processing_status == "ended"``; gibt den Batch zurück.

    `sleep`/`on_poll` sind injizierbar (Tests, Fortschrittsanzeige).

    Raises
    ------
    TimeoutError
        Wenn der Batch nach `timeout_s` nicht beendet ist.
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
                f"Batch {batch_id} nach {timeout_s:.0f}s nicht beendet (Status: {status})."
            )
        sleep(poll_interval_s)


def classify_result(result: Any) -> dict:
    """Klassifiziert einen einzelnen Batch-Result-Eintrag (reine Funktion).

    Rückgabe-Dict je `custom_id`:
        status == STATUS_SUCCEEDED        → + "text", "usage", "message"
        status == STATUS_INVALID_REQUEST  → + "error"  (loggen, nicht resubmitten)
        status == STATUS_SERVER_ERROR     → + "error"  (resubmit)
        status == STATUS_EXPIRED          → (resubmit)
        status == STATUS_CANCELED         → (Fehler)
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

    # Unbekannter Typ → defensiv als Server-Fehler behandeln (resubmit).
    return {"custom_id": custom_id, "status": STATUS_SERVER_ERROR, "error": rtype}


def iter_results(batch_id: str, *, client: Any = None) -> Iterator[dict]:
    """Iteriert die Ergebnisse eines beendeten Batches als klassifizierte Dicts."""
    client = client or _get_client()
    for result in client.messages.batches.results(batch_id):
        yield classify_result(result)


def collect_results(
    batch_id: str,
    *,
    client: Any = None,
    parse: Optional[Callable[[str], Any]] = None,
) -> dict:
    """Sammelt die Ergebnisse eines beendeten Batches in vier Eimer.

    `parse` (optional) wird auf den Text erfolgreicher Antworten angewandt –
    z. B. ``utils.judge.parse_judge_response`` für Judge-Batches. Ohne `parse`
    enthält ``succeeded[cid]`` das Roh-Dict aus :func:`classify_result`
    (Text/Usage/Message), sodass der Generierungs-Pfad denselben Record wie der
    Real-time-Loop bauen kann.

    Rückgabe::

        {
            "succeeded":  {custom_id: parse(text) | classify-dict},
            "resubmit":   {custom_id: classify-dict},  # server_error / expired
            "invalid":    {custom_id: classify-dict},  # invalid_request (geloggt)
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
            logger.warning("Resubmit-Kandidat %s (%s).", cid, status)
        elif status == STATUS_INVALID_REQUEST:
            invalid[cid] = entry
            logger.error("invalid_request für %s – wird nicht resubmittet: %s",
                         cid, entry.get("error"))
        else:  # canceled
            canceled[cid] = entry
            logger.error("Batch-Request %s canceled.", cid)

    return {
        "succeeded": succeeded,
        "resubmit":  resubmit,
        "invalid":   invalid,
        "canceled":  canceled,
    }


# ── Orchestrierung: submit → wait → collect → resubmit ───────────────────────
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
    """Führt einen Batch end-to-end aus und resubmittet transiente Fehler.

    Ablauf je Runde: einreichen (oder persistierten Batch wieder aufnehmen) →
    warten → einsammeln. `errored`(server)/`expired` werden bis zu
    `max_resubmits`-mal als neuer Batch erneut eingereicht; `invalid_request`
    und `canceled` werden geloggt und als endgültig fehlgeschlagen vermerkt.

    Poll-Resume: existiert unter `state_path` bereits eine `batch_id`, wird
    dieser Batch in der ersten Runde gepollt statt neu eingereicht (die
    Request-Liste muss dem persistierten Batch entsprechen).

    Rückgabe::

        {"succeeded": {cid: …}, "failed": {cid: classify-dict}, "batch_id": …}
    """
    client = client or _get_client()
    by_id = {r["custom_id"]: r for r in requests}

    succeeded: dict[str, Any] = {}
    failed: dict[str, dict] = {}
    pending = list(requests)

    # Poll-Resume: persistierte batch_id der ersten Runde wieder aufnehmen.
    batch_id: Optional[str] = (
        _load_batch_id(state_path) if state_path is not None else None
    )
    if batch_id is not None:
        logger.info("Setze persistierten Batch fort: %s", batch_id)

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
        batch_id = None  # nächste Runde reicht einen frischen Batch ein
        if not resubmit_ids:
            break

        attempt += 1
        if attempt > max_resubmits:
            logger.error("max_resubmits (%d) erschöpft; %d Requests bleiben offen.",
                         max_resubmits, len(resubmit_ids))
            for cid in resubmit_ids:
                entry = dict(collected["resubmit"][cid])
                entry["status"] = "max_resubmits_exceeded"
                failed[cid] = entry
            break

        logger.info("Resubmit-Runde %d: %d Requests.", attempt, len(resubmit_ids))
        pending = [by_id[cid] for cid in resubmit_ids]

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
