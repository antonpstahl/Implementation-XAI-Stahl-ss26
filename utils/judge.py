"""Judge parsing: extract structured scores from LLM as judge answers."""

from __future__ import annotations

import json
import re


def parse_judge_response(raw: str) -> dict:
    """Extract judge scores robustly from XML tags, JSON or key:value plain text.

    Priority: XML tags -> JSON -> key:value regex (legacy fallback).
    Returns a dict with the keys faithfulness, clarity, completeness (int, 1 to 5)
    and optional *_reasoning keys. Missing scores are not set (no key in the dict),
    so the caller can detect None scores via .get().
    """
    # Remove markdown code block
    code_block = re.search(r'```(?:json)?\s*(.*?)(?:```|$)', raw, re.DOTALL)
    inner = code_block.group(1).strip() if code_block else raw

    scores: dict = {}

    # Primary: XML tags (robust output format)
    for key in ['faithfulness', 'clarity', 'completeness']:
        m = re.search(rf'<{key}>(\d)</{key}>', inner, re.IGNORECASE)
        if m:
            scores[key] = int(m.group(1))
        rm = re.search(rf'<{key}_reasoning>(.*?)</{key}_reasoning>', inner,
                       re.IGNORECASE | re.DOTALL)
        if rm:
            scores[f'{key}_reasoning'] = rm.group(1).strip()
    if all(scores.get(k) is not None for k in ('faithfulness', 'clarity', 'completeness')):
        return scores

    # Fallback 1: full JSON
    try:
        json_match = re.search(r'\{.*\}', inner, re.DOTALL)
        if json_match:
            d = json.loads(json_match.group())
            d_up = {k.upper(): v for k, v in d.items()}
            for key in ['FAITHFULNESS', 'CLARITY', 'COMPLETENESS']:
                if key in d_up and key.lower() not in scores:
                    scores[key.lower()] = int(d_up[key])
            for key in ['FAITHFULNESS_REASONING', 'CLARITY_REASONING', 'COMPLETENESS_REASONING']:
                base = key.replace('_REASONING', '').lower()
                rkey = f'{base}_reasoning'
                if key in d_up and rkey not in scores:
                    scores[rkey] = str(d_up[key])
            if all(scores.get(k) is not None for k in ('faithfulness', 'clarity', 'completeness')):
                return scores
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback 2: key:value regex (legacy, for older judge answers)
    for key in ['FAITHFULNESS', 'CLARITY', 'COMPLETENESS']:
        if key.lower() not in scores:
            m = re.search(rf'"?{key}"?\s*:\s*(\d)', inner, re.IGNORECASE)
            if m:
                scores[key.lower()] = int(m.group(1))
    for key in ['FAITHFULNESS_REASONING', 'CLARITY_REASONING', 'COMPLETENESS_REASONING']:
        base = key.replace('_REASONING', '').lower()
        rkey = f'{base}_reasoning'
        if rkey in scores:
            continue
        # Quoted value (JSON or partial JSON)
        m = re.search(rf'"?{key}"?\s*:\s*"([^"]+)', inner, re.IGNORECASE)
        if m:
            scores[rkey] = m.group(1)
            continue
        # Plain-text value (reason-then-score format; stop at next KEY: line or end)
        m = re.search(
            rf'^{key}\s*:\s*(.+?)(?=\n[A-Z_]{{3,}}\s*:|$)',
            inner, re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        if m:
            scores[rkey] = m.group(1).strip()

    return scores


SCORE_KEYS = ("faithfulness", "clarity", "completeness")

from typing import Any


def judge_batch_sc(
    entries: "list[tuple[str, str]]",
    *,
    system: str,
    model: str,
    max_tokens: int = 900,
    k: int = 3,
    temperature: "float | None" = None,
    client: Any = None,
    state_path: Any = None,
    **run_batch_kwargs: Any,
) -> "dict[str, dict]":
    """Batch based self consistency scoring.

    For each (base_cid, prompt) entry k batch requests are submitted
    (custom_ids: {base_cid}-s0 ... {base_cid}-s{k-1}). Results are aggregated per
    criterion by median on the client side.

    Return schema identical to judge_with_self_consistency:
        faithfulness, clarity, completeness  (int | None)
        faithfulness_reasoning, clarity_reasoning, completeness_reasoning (str)
        raw_responses  (list[str])
        usage  ({"input_tokens": int, "output_tokens": int})

    Temperature: dropped automatically for Opus 4.7/4.8 / Fable
    (model_accepts_temperature). Failed samples are excluded from the median;
    base entries with no success get None scores.

    Parameters
    ----------
    entries     : list of (base_custom_id, prompt). base_cid must be <= 61 chars
                  (3 chars reserved for "-s{j}").
    k           : number of samples per entry (diversity from default stochasticity
                  for Opus; for Sonnet/temperature > 0 from sampling).
    client      : Anthropic client (tests inject a fake).
    state_path  : path for batch_id persistence (poll resume after a crash).
    **run_batch_kwargs : passed through to utils.batch.run_batch
                  (sleep, poll_interval_s, max_resubmits, ...).

    Returns
    -------
    dict[base_cid -> aggregated_result]
    """
    import statistics as _statistics

    from utils.llm import build_text_params
    from utils.batch import message_request, run_batch

    max_suffix_len = len(f"-s{k - 1}")
    for base_cid, _ in entries:
        if len(base_cid) + max_suffix_len > 64:
            raise ValueError(
                f"base_cid {base_cid!r} is too long ({len(base_cid)} chars); "
                f"max {64 - max_suffix_len} allowed (reserved {max_suffix_len} for -s{{j}})."
            )

    requests: list[dict] = []
    for base_cid, prompt in entries:
        params = build_text_params(
            prompt,
            system=system,
            model=model,
            max_tokens=max_tokens,
            cache_system=True,
            temperature=temperature,
        )
        for j in range(k):
            requests.append(message_request(f"{base_cid}-s{j}", {**params}))

    outcome = run_batch(
        requests, client=client, state_path=state_path, **run_batch_kwargs
    )

    per_base: dict[str, list[dict]] = {b: [] for b, _ in entries}
    tok_in:  dict[str, int] = {b: 0 for b, _ in entries}
    tok_out: dict[str, int] = {b: 0 for b, _ in entries}

    for sample_cid, entry in outcome["succeeded"].items():
        base_cid = sample_cid.rsplit("-", 1)[0]
        if base_cid not in per_base:
            continue
        usage = entry.get("usage", {})
        tok_in[base_cid]  += usage.get("input_tokens", 0)
        tok_out[base_cid] += usage.get("output_tokens", 0)
        parsed = parse_judge_response(entry.get("text", ""))
        per_base[base_cid].append({**parsed, "raw_response": entry.get("text", "")})

    aggregated: dict[str, dict] = {}
    for base_cid, samples in per_base.items():
        agg: dict = {}
        for key in SCORE_KEYS:
            vals = [s[key] for s in samples if s.get(key) is not None]
            agg[key] = int(_statistics.median(vals)) if vals else None
        for rkey in ("faithfulness_reasoning", "clarity_reasoning", "completeness_reasoning"):
            agg[rkey] = samples[0].get(rkey, "") if samples else ""
        agg["raw_responses"] = [s["raw_response"] for s in samples]
        agg["usage"] = {"input_tokens": tok_in[base_cid], "output_tokens": tok_out[base_cid]}
        aggregated[base_cid] = agg

    return aggregated


def judge_with_retry(ask_fn, prompt: str, system: str, model: str,
                     max_tokens: int = 900, max_retries: int = 3,
                     temperature: float | None = None) -> dict:
    """Call ask_fn and retry up to max_retries times on incomplete parsing.

    ask_fn must have the same interface as utils.llm.ask_text:
        ask_fn(prompt, system=..., model=..., max_tokens=..., cache_system=...,
               temperature=...) -> response

    temperature : None -> model default (1.0); 0.0 -> deterministic (JUDGE_TEMPERATURE).
                  For reproducibility and minimal score variance, models that accept
                  `temperature` (Sonnet, OpenAI) should get JUDGE_TEMPERATURE=0.0.
                  For Opus 4.7/4.8 `temperature` is rejected by the API and dropped
                  automatically in utils.llm.ask_text, so the judge cannot be fixed
                  deterministically there (default stochasticity).

    Returns: dict with scores (missing scores as None) + raw_response + usage.
    """
    scores: dict = {}
    raw = ""
    in_tok = out_tok = 0

    for _ in range(max_retries):
        response = ask_fn(prompt, system=system, model=model,
                          max_tokens=max_tokens, cache_system=True,
                          temperature=temperature)
        usage = response.get("usage", {})
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        raw = response["content"][0]["text"].strip()
        scores = parse_judge_response(raw)
        if all(scores.get(k) is not None for k in SCORE_KEYS):
            break

    return {
        "faithfulness":  scores.get("faithfulness"),
        "clarity":       scores.get("clarity"),
        "completeness":  scores.get("completeness"),
        "faithfulness_reasoning":  scores.get("faithfulness_reasoning", ""),
        "clarity_reasoning":       scores.get("clarity_reasoning", ""),
        "completeness_reasoning":  scores.get("completeness_reasoning", ""),
        "raw_response": raw,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


def judge_with_self_consistency(
    ask_fn, prompt: str, system: str, model: str,
    max_tokens: int = 900, k: int = 3, temperature: float = 0.7,
) -> dict:
    """Self consistency judge: k samples at a given temperature, median per score.

    Costs k times the judge calls of judge_with_retry.

    Model dependence of `temperature`:
      * Sonnet/OpenAI accept `temperature`. At JUDGE_TEMPERATURE=0 (deterministic)
        SC adds no extra information, so judge_with_retry with temperature=0 should
        be preferred. SC is only useful at temperature > 0.
      * Opus 4.7/4.8 (and Fable) reject `temperature`; the parameter is dropped
        automatically in utils.llm.ask_text. The k calls still vary via the default
        stochasticity, so SC draws its diversity from that, the passed temperature
        value has no effect.

    Cost effect at scale: k=3, n=200, 4 pipelines, 2 XAI models
        -> 200 x 4 x 2 x k = 4800 judge calls instead of 1600 (factor k=3).
        This must be included in the cost estimate.

    Returns: dict with aggregated scores (median), raw answers and cumulative usage.
    """
    import statistics

    per_run: list[dict] = []
    total_in = total_out = 0

    for _ in range(k):
        result = judge_with_retry(
            ask_fn, prompt, system, model,
            max_tokens=max_tokens, max_retries=3, temperature=temperature,
        )
        per_run.append(result)
        total_in  += result["usage"]["input_tokens"]
        total_out += result["usage"]["output_tokens"]

    aggregated: dict = {}
    for key in SCORE_KEYS:
        vals = [r[key] for r in per_run if r[key] is not None]
        aggregated[key] = int(statistics.median(vals)) if vals else None

    for rkey in ("faithfulness_reasoning", "clarity_reasoning", "completeness_reasoning"):
        aggregated[rkey] = per_run[0].get(rkey, "")

    aggregated["raw_responses"] = [r["raw_response"] for r in per_run]
    aggregated["usage"] = {"input_tokens": total_in, "output_tokens": total_out}
    return aggregated
