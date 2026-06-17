"""
utils/llm.py – Wrapper um den Anthropic-Client für die drei LLM-Pipelines.

Bündelt Konfiguration (Modell-ID, max_tokens) und bietet einfache Helfer
für Text-only- und multimodale (Vision) Anfragen. Wird von Notebooks
04, 05 und 06 verwendet.

Die konkrete Tool-Use-Schleife für Notebook 06 wird dort implementiert,
da sie modellspezifisch (Tool-Definitionen, Stop-Reason-Handling) ist.
"""

from __future__ import annotations

import base64
import os
import time as _time
from pathlib import Path
from typing import Any, Iterable

# Lädt .env automatisch, falls python-dotenv installiert ist.
# Fehlt das Paket, wird stattdessen die Shell-Umgebung verwendet.
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_file)
except ImportError:
    pass

# -----------------------------------------------------------------------------
# LLM-Konfiguration – zentrale Dokumentation aller Decoding-Parameter
#
# API:         Anthropic Messages API
# Abrufdatum:  2026-06-11
#
# Modell-IDs nach Rolle:
#   Erklärungsgenerierung  (NB 04 / 05 / 06)  → claude-sonnet-4-6
#   Faithfulness-Check     (NB 07)             → claude-sonnet-4-6
#   Judge v1 unkalibriert  (NB 07)             → claude-sonnet-4-6
#   Judge v2 kalibriert    (NB 07)             → claude-sonnet-4-6
#   Judge v3 unabhängig    (NB 07)             → claude-opus-4-8
#   Ichmoukhamedov-Metriken(NB 08)             → claude-sonnet-4-6
#
# max_tokens nach Kontext:
#   MAX_TOKENS_GENERATION      = 2048  (Pipelines 04 / 05 / 06)
#     Hinweis B6: Scratchpad (<analyse>…</analyse>) kommt vor der Prosa
#     (~50–100 Tokens) und wird vor dem Speichern via strip_scratchpad()
#     entfernt.  Notebooks 04/05 wurden von 600 auf diesen Wert angehoben.
#   MAX_TOKENS_FAITHFULNESS    = 300   (Faithfulness-Check NB 07)
#   MAX_TOKENS_JUDGE           = 900   (Judge-Calls NB 07, alle Versionen; +Reasoning)
#   MAX_TOKENS_ICHMOUKHAMEDOV  = 700   (LLM-Calls NB 08)
#
# ── Decoding-Temperaturen (Phase 3·2 / A2) ────────────────────────────────
#
#   JUDGE_TEMPERATURE = 0.0  (deterministisch)
#     Begründung: Der Judge ist ein Messinstrument, keine kreative Aufgabe.
#     Gleiche Eingabe → gleicher Score: maximale Reproduzierbarkeit und kein
#     Stochastik-Rauschen in den Messwerten. G-Eval (Liu et al. 2023) empfiehlt
#     temperature=0 für numerische Rubriken.
#     Wirkung auf n=20 Re-Run: Score-Std ≈ 0 (empirisch belegt, s. NB 07 Zelle v5).
#
#   GENERATION_TEMPERATURE = 1.0  (Anthropic-Default — bewusste Designentscheidung)
#     Begründung: Erklärungstexte sollen natürlich und nicht repetitiv wirken.
#     Phase 3b misst Varianz explizit (3 Generationen/Instanz), sodass die
#     Stochastik der Messung selbst Gegenstand der Untersuchung ist.
#     Limitation: reproduzierbare Erklärungen erfordern fixen Seed → wird im
#     Paper als Limitation benannt.
#
#   JUDGE_SC_K = 3  (Self-Consistency-Samples — nur wenn SC statt temp=0 genutzt)
#     SC kostet k× die Judge-Calls: bei n=200, 4 Pipelines, 2 XAI-Modellen
#     ergibt k=3 ca. 4 800 statt 1 600 Judge-Calls (→ Phase-3b-Kostenschätzung).
#     Bei JUDGE_TEMPERATURE=0 (deterministisch) ist SC wertlos — Implementierung
#     steht bereit (judge_with_self_consistency in utils/judge.py), ist aber
#     standardmäßig deaktiviert.
#
# Hinweis für das Paper: Modell-IDs und Parameterdefaults der Anthropic-API
# können sich nach dem Abrufdatum ändern. Für Reproduzierbarkeit sind exakte
# Versionspins und das Abrufdatum anzugeben.
# -----------------------------------------------------------------------------

DEFAULT_MODEL      = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 2048   # Erklärungsgenerierung (Pipelines 04 / 05 / 06)

MAX_TOKENS_GENERATION     = 2048
MAX_TOKENS_FAITHFULNESS   = 300
MAX_TOKENS_JUDGE          = 900   # +Reasoning (Reason-then-Score, Phase 3·2/A1)
MAX_TOKENS_ICHMOUKHAMEDOV = 700

# Decoding-Temperaturen (Phase 3·2 / A2)
JUDGE_TEMPERATURE      = 0.0   # deterministisch (s. Begründung oben)
GENERATION_TEMPERATURE = 1.0   # Anthropic-Default (bewusste Designentscheidung)
JUDGE_SC_K             = 3     # Self-Consistency k — nur wenn SC statt temp=0

import re as _re


def model_accepts_temperature(model: str) -> bool:
    """True, wenn das Modell den `temperature`-Parameter akzeptiert.

    Anthropic Claude Opus 4.7 / 4.8 sowie Fable lehnen `temperature` ab
    (HTTP 400 bei der Messages API) — sie steuern das Decoding ausschließlich
    über die Default-Stochastik. Für diese Modelle darf `temperature` nicht
    mitgeschickt werden. Alle übrigen Modelle (Sonnet, Haiku, OpenAI) akzeptieren
    den Parameter.

    Wirkung auf Self-Consistency: Bei abgelehntem `temperature` ziehen mehrere
    Calls ihre Diversität aus der Default-Stochastik (k Calls variieren trotzdem),
    statt aus einem explizit erhöhten `temperature`-Wert.
    """
    return _re.search(r"opus-4-[78]|fable", model) is None


def strip_scratchpad(text: str) -> str:
    """Removes the <analyse>…</analyse> scratchpad block from generated text.

    The block is written by the model before the prose (B6 — Think-before-write)
    and must be discarded before persisting the explanation.  Handles optional
    leading/trailing whitespace and CRLF line endings.
    """
    return _re.sub(r"<analyse>.*?</analyse>\s*", "", text, flags=_re.DOTALL).strip()


try:
    from anthropic import RateLimitError, APIConnectionError, InternalServerError
    _RETRYABLE_TYPES = (RateLimitError, APIConnectionError, InternalServerError)
except ImportError:
    _RETRYABLE_TYPES = (Exception,)


def _with_retry(fn: Any, *args: Any, max_retries: int = 2, **kwargs: Any) -> Any:
    """Wraps an API call with exponential backoff on transient errors."""
    delay = 5
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt < max_retries and isinstance(exc, _RETRYABLE_TYPES):
                print(f"[llm] {type(exc).__name__} – Retry {attempt + 1}/{max_retries} in {delay}s …")
                _time.sleep(delay)
                delay *= 2
            else:
                raise


def _get_client() -> Any:
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise ImportError(
            "Paket 'anthropic' nicht installiert. "
            "Bitte `pip install anthropic` ausführen."
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY nicht gesetzt.\n"
            "Entweder in .env eintragen (cp .env.example .env) "
            "oder als Umgebungsvariable exportieren:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return Anthropic(api_key=api_key)


# -----------------------------------------------------------------------------
# Request-Shape-Builder + Real-time-Runner (Phase 3a·B)
#
# Der Batch- und der Real-time-Pfad müssen **dieselbe** Request-Shape erzeugen,
# damit die On-Disk-Artefakte schema-identisch bleiben (NB 07/08 sind
# ausführungsart-agnostisch). Deshalb bauen `build_text_params` /
# `build_image_params` exakt die `messages.create`-Parameter; `run_params`
# führt sie real-time aus, `utils.batch.message_request` verpackt sie für den
# Batch. `ask_text` / `ask_with_images` bleiben die bequemen Real-time-Wrapper.
# -----------------------------------------------------------------------------
def _system_block(system: str | None, cache_system: bool) -> Any:
    """Baut den `system`-Parameter — als gecachten Block oder schlichten String."""
    if system and cache_system:
        return [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    return system or ""


def build_text_params(
    prompt: str,
    *,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cache_system: bool = False,
    temperature: float | None = None,
) -> dict:
    """Baut die `messages.create`-Parameter für eine Text-only-Anfrage.

    Gemeinsame Request-Shape für Real-time (`run_params`) und Batch
    (`utils.batch.message_request`). `temperature` wird bei Modellen, die sie
    ablehnen (Opus 4.7/4.8, Fable), weggelassen — siehe model_accepts_temperature().
    """
    params: dict = dict(
        model=model,
        max_tokens=max_tokens,
        system=_system_block(system, cache_system),
        messages=[{"role": "user", "content": prompt}],
    )
    if temperature is not None and model_accepts_temperature(model):
        params["temperature"] = temperature
    return params


def run_params(params: dict) -> dict:
    """Führt eine vorgebaute Request-Shape real-time aus (mit Retry)."""
    client = _get_client()
    resp = _with_retry(client.messages.create, **params)
    return resp.model_dump()


# -----------------------------------------------------------------------------
# Pipeline 04: JSON → Text  (mit Prompt-Caching für den System-Prompt)
# -----------------------------------------------------------------------------
def ask_text(
    prompt: str,
    *,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cache_system: bool = False,
    temperature: float | None = None,
) -> dict:
    """Text-only Anfrage an die Anthropic Messages API.

    Parameters
    ----------
    temperature : float | None
        None → Anthropic-Default (1.0); 0.0 → deterministisch (für Judge-Calls);
        0.2–0.4 → wenig stochastisch. Siehe JUDGE_TEMPERATURE / GENERATION_TEMPERATURE.
        Wird bei Modellen, die `temperature` ablehnen (Opus 4.7/4.8, Fable),
        automatisch verworfen — siehe model_accepts_temperature().
    """
    return run_params(
        build_text_params(
            prompt,
            system=system,
            model=model,
            max_tokens=max_tokens,
            cache_system=cache_system,
            temperature=temperature,
        )
    )


# -----------------------------------------------------------------------------
# Pipeline 05: Bilder + Text → Text
# -----------------------------------------------------------------------------
_MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB


def _encode_image(path: Path | str) -> dict:
    path = Path(path)
    suffix = path.suffix.lower().lstrip(".")
    media_type_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    if suffix not in media_type_map:
        raise ValueError(f"Bildformat .{suffix} nicht unterstützt.")

    size = path.stat().st_size
    if size > _MAX_IMAGE_BYTES:
        raise ValueError(
            f"{path.name} ist {size / 1024 / 1024:.1f} MB groß "
            f"(Limit: {_MAX_IMAGE_BYTES // 1024 // 1024} MB). "
            "Bild vorher komprimieren oder Auflösung reduzieren."
        )

    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type_map[suffix],
            "data": data,
        },
    }


# -----------------------------------------------------------------------------
# Cross-Vendor Judge: OpenAI-kompatibler Wrapper (Phase 2)
#
# API:         OpenAI Chat Completions API
# Abrufdatum:  2026-06-16
# Empfohlenes Modell (früher Test):  gpt-4o-mini  ($0.15/$0.60 per 1M in/out)
# Empfohlenes Modell (Finallauf):    gpt-4o       ($2.50/$10.00 per 1M in/out)
# Free-Tier Ratelimit: 3 RPM → request_delay_s=20 nötig (Tier 0).
#                      Ab Tier 1: request_delay_s=0 möglich.
# -----------------------------------------------------------------------------
OPENAI_JUDGE_MODEL_TEST  = "gpt-4o-mini"
OPENAI_JUDGE_MODEL_FINAL = "gpt-4o"
MAX_TOKENS_OPENAI_JUDGE  = 600


def _with_openai_retry(fn: Any, *args: Any, max_retries: int = 2, **kwargs: Any) -> Any:
    """Exponential backoff for OpenAI transient errors."""
    try:
        from openai import RateLimitError, APIConnectionError, InternalServerError
        _oai_retryable = (RateLimitError, APIConnectionError, InternalServerError)
    except ImportError:
        _oai_retryable = (Exception,)

    delay = 5
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt < max_retries and isinstance(exc, _oai_retryable):
                print(f"[openai] {type(exc).__name__} – Retry {attempt + 1}/{max_retries} in {delay}s …")
                _time.sleep(delay)
                delay *= 2
            else:
                raise


def ask_openai_text(
    prompt: str,
    *,
    system: str | None = None,
    model: str = OPENAI_JUDGE_MODEL_TEST,
    max_tokens: int = MAX_TOKENS_OPENAI_JUDGE,
    request_delay_s: float = 0.0,
    temperature: float | None = None,
) -> dict:
    """OpenAI Chat Completions call — gibt dasselbe Schema zurück wie ask_text().

    Parameters
    ----------
    prompt          : User-Nachricht
    system          : System-Prompt (wird als 'system' role übergeben)
    model           : Modell-ID, default gpt-4o-mini (günstig, Free-Tier-tauglich)
    max_tokens      : max. Output-Tokens
    request_delay_s : Pause vor dem Call (20s für Free-Tier-3-RPM-Limit empfohlen)
    temperature     : None → Modell-Default; 0.0 → deterministisch (für Judge-Calls)

    Returns
    -------
    dict mit 'content'[0]['text'], 'usage' (input_tokens/output_tokens), 'model'
    — kompatibel mit dem bestehenden _parse_judge_response()-Schema.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "Paket 'openai' nicht installiert. Bitte `pip install openai` ausführen."
        ) from e

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY nicht gesetzt.\n"
            "In .env eintragen oder exportieren:\n"
            "  export OPENAI_API_KEY=sk-proj-..."
        )

    client = OpenAI(api_key=api_key)

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if request_delay_s > 0:
        _time.sleep(request_delay_s)

    create_kwargs: dict = dict(model=model, messages=messages, max_tokens=max_tokens)
    if temperature is not None:
        create_kwargs["temperature"] = temperature

    resp = _with_openai_retry(client.chat.completions.create, **create_kwargs)

    return {
        "content": [{"text": resp.choices[0].message.content or ""}],
        "usage": {
            "input_tokens":  resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        },
        "model": resp.model,
    }


def build_image_params(
    prompt: str,
    image_paths: Iterable[Path | str],
    *,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cache_system: bool = True,
    temperature: float | None = None,
) -> dict:
    """Baut die `messages.create`-Parameter für eine multimodale Anfrage (NB 05).

    Bilder werden base64-kodiert in den User-Content gelegt. Gemeinsame
    Request-Shape für Real-time (`run_params`) und Batch
    (`utils.batch.message_request`).
    """
    content: list[dict] = [_encode_image(p) for p in image_paths]
    content.append({"type": "text", "text": prompt})

    params: dict = dict(
        model=model,
        max_tokens=max_tokens,
        system=_system_block(system, cache_system),
        messages=[{"role": "user", "content": content}],
    )
    if temperature is not None and model_accepts_temperature(model):
        params["temperature"] = temperature
    return params


def ask_with_images(
    prompt: str,
    image_paths: Iterable[Path | str],
    *,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cache_system: bool = True,
    temperature: float | None = None,
) -> dict:
    """Multimodale Anfrage mit einem oder mehreren Bildern (Notebook 05).

    Bilder werden base64-kodiert übergeben.
    temperature : siehe ask_text.
    """
    return run_params(
        build_image_params(
            prompt,
            image_paths,
            system=system,
            model=model,
            max_tokens=max_tokens,
            cache_system=cache_system,
            temperature=temperature,
        )
    )
