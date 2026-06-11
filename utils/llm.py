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
# Temperature: nicht gesetzt → Anthropic-Default = 1.0 (bei allen Calls)
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
#   MAX_TOKENS_FAITHFULNESS    = 300   (Faithfulness-Check NB 07)
#   MAX_TOKENS_JUDGE           = 600   (Judge-Calls NB 07, alle Versionen)
#   MAX_TOKENS_ICHMOUKHAMEDOV  = 700   (LLM-Calls NB 08)
#
# Hinweis für das Paper: Modell-IDs und Parameterdefaults der Anthropic-API
# können sich nach dem Abrufdatum ändern. Für Reproduzierbarkeit sind exakte
# Versionspins und das Abrufdatum anzugeben.
# -----------------------------------------------------------------------------

DEFAULT_MODEL      = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 2048   # Erklärungsgenerierung (Pipelines 04 / 05 / 06)

MAX_TOKENS_GENERATION     = 2048
MAX_TOKENS_FAITHFULNESS   = 300
MAX_TOKENS_JUDGE          = 600
MAX_TOKENS_ICHMOUKHAMEDOV = 700

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
# Pipeline 04: JSON → Text  (mit Prompt-Caching für den System-Prompt)
# -----------------------------------------------------------------------------
def ask_text(
    prompt: str,
    *,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cache_system: bool = False,
) -> dict:
    
    client = _get_client()

    if system and cache_system:
        system_block = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system_block = system or ""

    resp = _with_retry(
        client.messages.create,
        model=model,
        max_tokens=max_tokens,
        system=system_block,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.model_dump()


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


def ask_with_images(
    prompt: str,
    image_paths: Iterable[Path | str],
    *,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cache_system: bool = True,
) -> dict:
    """
    Multimodale Anfrage mit einem oder mehreren Bildern (Notebook 05).
    Bilder werden base64-kodiert übergeben.
    """
    client = _get_client()

    if system and cache_system:
        system_block = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system_block = system or ""

    content: list[dict] = [_encode_image(p) for p in image_paths]
    content.append({"type": "text", "text": prompt})

    resp = _with_retry(
        client.messages.create,
        model=model,
        max_tokens=max_tokens,
        system=system_block,
        messages=[{"role": "user", "content": content}],
    )
    return resp.model_dump()
