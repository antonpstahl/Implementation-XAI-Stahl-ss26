"""
utils/llm.py - wrapper around the Anthropic client for the three LLM pipelines.

Bundles configuration (model id, max_tokens) and offers simple helpers for text
only and multimodal (vision) requests. Used by notebooks 04, 05 and 06.

The concrete Tool Use loop for notebook 04Ld is implemented there, because it is
model specific (tool definitions, stop reason handling).
"""

from __future__ import annotations

import base64
import os
import time as _time
from pathlib import Path
from typing import Any, Iterable

# Loads .env automatically if python-dotenv is installed.
# If the package is missing, the shell environment is used instead.
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_file)
except ImportError:
    pass

# -----------------------------------------------------------------------------
# LLM configuration - central documentation of all decoding parameters
#
# API:           Anthropic Messages API
# Retrieved on:  2026-06-11
#
# Model ids by role:
#   Explanation generation  (NB 04Lb / 04Lc / 04Ld)  -> claude-sonnet-4-6
#   Judge, primary          (NB 05)               -> claude-opus-4-8
#   Judge, cross vendor     (NB 05)               -> gpt-4o-mini (OpenAI)
#   Ichmoukhamedov metrics  (NB 06)               -> claude-sonnet-4-6
#
# max_tokens by context:
#   MAX_TOKENS_GENERATION      = 2048  (pipelines 04 / 05 / 06)
#     Note: the scratchpad (<analysis> ... </analysis>) comes before the prose
#     (about 50 to 100 tokens) and is removed via strip_scratchpad() before saving.
#   MAX_TOKENS_JUDGE           = 900   (judge calls NB 05; + reasoning)
#   MAX_TOKENS_ICHMOUKHAMEDOV  = 700   (LLM calls NB 06)
#
# --- Decoding temperatures ---------------------------------------------------
#
#   JUDGE_TEMPERATURE = 0.0  (deterministic)
#     Reason: the judge is a measurement instrument, not a creative task. Same
#     input gives the same score: maximum reproducibility and no stochastic noise
#     in the measurements. G-Eval (Liu et al. 2023) recommends temperature=0 for
#     numeric rubrics. Note: Opus rejects temperature, so the Opus judge is not
#     fully deterministic; the OpenAI judge can use temperature=0.
#
#   GENERATION_TEMPERATURE = 1.0  (Anthropic default, a deliberate design choice)
#     Reason: explanation texts should read naturally and not be repetitive.
#     Limitation: reproducible explanations need a fixed seed, named as a
#     limitation in the paper.
#
#   JUDGE_SC_K = 3  (self consistency samples, only if SC is used instead of temp=0)
#     The self_consistency implementation in utils/judge.py is ready but disabled
#     by default (SC is worthless at JUDGE_TEMPERATURE=0).
#
# Note for the paper: Anthropic API model ids and parameter defaults can change
# after the retrieval date. For reproducibility report exact version pins and the
# retrieval date.
# -----------------------------------------------------------------------------

DEFAULT_MODEL      = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 2048   # explanation generation (pipelines 04 / 05 / 06)

MAX_TOKENS_GENERATION     = 2048
MAX_TOKENS_FAITHFULNESS   = 300
MAX_TOKENS_JUDGE          = 900   # + reasoning (reason then score)
MAX_TOKENS_ICHMOUKHAMEDOV = 700

# Decoding temperatures
JUDGE_TEMPERATURE      = 0.0   # deterministic (see reasoning above)
GENERATION_TEMPERATURE = 1.0   # Anthropic default (a deliberate design choice)
JUDGE_SC_K             = 3     # self consistency k, only if SC is used instead of temp=0

import re as _re


def model_accepts_temperature(model: str) -> bool:
    """True if the model accepts the `temperature` parameter.

    Anthropic Claude Opus 4.7 / 4.8 and Fable reject `temperature` (HTTP 400 on the
    Messages API), they steer decoding only via the default stochasticity. For
    those models `temperature` must not be sent. All other models (Sonnet, Haiku,
    OpenAI) accept the parameter.

    Effect on self consistency: when `temperature` is rejected, several calls draw
    their diversity from the default stochasticity (k calls still vary) instead of
    from an explicitly raised `temperature` value.
    """
    return _re.search(r"opus-4-[78]|fable", model) is None


def strip_scratchpad(text: str) -> str:
    """Removes the <analysis> ... </analysis> scratchpad block from generated text.

    The block is written by the model before the prose (think before write) and
    must be discarded before persisting the explanation. Handles optional leading
    or trailing whitespace and CRLF line endings.
    """
    return _re.sub(r"<analysis>.*?</analysis>\s*", "", text, flags=_re.DOTALL).strip()


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
                print(f"[llm] {type(exc).__name__} - retry {attempt + 1}/{max_retries} in {delay}s ...")
                _time.sleep(delay)
                delay *= 2
            else:
                raise


def _get_client() -> Any:
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise ImportError(
            "Package 'anthropic' not installed. "
            "Please run `pip install anthropic`."
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set.\n"
            "Either add it to .env (cp .env.example .env) "
            "or export it as an environment variable:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return Anthropic(api_key=api_key)


# -----------------------------------------------------------------------------
# Request shape builders + real time runner
#
# The batch and the real time path must produce the same request shape so the on
# disk artefacts stay schema identical (NB 05/06 are execution mode agnostic).
# Therefore `build_text_params` / `build_image_params` build exactly the
# `messages.create` parameters; `run_params` runs them real time,
# `utils.batch.message_request` wraps them for the batch. `ask_text` /
# `ask_with_images` remain the convenient real time wrappers.
# -----------------------------------------------------------------------------
def _system_block(system: str | None, cache_system: bool) -> Any:
    """Build the `system` parameter, as a cached block or a plain string."""
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
    """Build the `messages.create` parameters for a text only request.

    Shared request shape for real time (`run_params`) and batch
    (`utils.batch.message_request`). `temperature` is omitted for models that
    reject it (Opus 4.7/4.8, Fable), see model_accepts_temperature().
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
    """Run a prebuilt request shape real time (with retry)."""
    client = _get_client()
    resp = _with_retry(client.messages.create, **params)
    return resp.model_dump()


# -----------------------------------------------------------------------------
# Pipeline 04: JSON to Text  (with prompt caching for the system prompt)
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
    """Text only request to the Anthropic Messages API.

    Parameters
    ----------
    temperature : float | None
        None -> Anthropic default (1.0); 0.0 -> deterministic (for judge calls);
        0.2 to 0.4 -> slightly stochastic. See JUDGE_TEMPERATURE / GENERATION_TEMPERATURE.
        Dropped automatically for models that reject `temperature` (Opus 4.7/4.8,
        Fable), see model_accepts_temperature().
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
# Pipeline 05: images + text to text
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
        raise ValueError(f"Image format .{suffix} not supported.")

    size = path.stat().st_size
    if size > _MAX_IMAGE_BYTES:
        raise ValueError(
            f"{path.name} is {size / 1024 / 1024:.1f} MB "
            f"(limit: {_MAX_IMAGE_BYTES // 1024 // 1024} MB). "
            "Compress the image or reduce its resolution first."
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
# Cross vendor judge: OpenAI compatible wrapper
#
# API:           OpenAI Chat Completions API
# Retrieved on:  2026-06-16
# Recommended model (early test):  gpt-4o-mini  ($0.15/$0.60 per 1M in/out)
# Recommended model (final run):   gpt-4o       ($2.50/$10.00 per 1M in/out)
# Free tier rate limit: 3 RPM, so request_delay_s=20 is needed (tier 0).
#                       From tier 1: request_delay_s=0 is possible.
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
                print(f"[openai] {type(exc).__name__} - retry {attempt + 1}/{max_retries} in {delay}s ...")
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
    """OpenAI Chat Completions call, returns the same schema as ask_text().

    Parameters
    ----------
    prompt          : user message
    system          : system prompt (passed as the 'system' role)
    model           : model id, default gpt-4o-mini (cheap, free tier capable)
    max_tokens      : max output tokens
    request_delay_s : pause before the call (20s recommended for the free tier 3 RPM limit)
    temperature     : None -> model default; 0.0 -> deterministic (for judge calls)

    Returns
    -------
    dict with 'content'[0]['text'], 'usage' (input_tokens/output_tokens), 'model',
    compatible with the existing _parse_judge_response() schema.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "Package 'openai' not installed. Please run `pip install openai`."
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
    """Build the `messages.create` parameters for a multimodal request (NB 04Lc).

    Images are base64 encoded into the user content. Shared request shape for real
    time (`run_params`) and batch (`utils.batch.message_request`).
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
    """Multimodal request with one or more images (notebook 04Lc).

    Images are passed base64 encoded.
    temperature : see ask_text.
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
