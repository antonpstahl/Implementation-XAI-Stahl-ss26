"""Judge-Parsing: extrahiert strukturierte Scores aus LLM-as-Judge-Antworten."""

from __future__ import annotations

import json
import re


def parse_judge_response(raw: str) -> dict:
    """Extrahiert Judge-Scores robust aus XML-Tags, JSON oder Key:Value-Plaintext.

    Priorität: XML-Tags (B7) → JSON → Key:Value-Regex (Legacy-Fallback).
    Gibt ein dict mit den Keys faithfulness, clarity, completeness (int, 1-5)
    und optionalen *_reasoning-Keys zurück. Fehlende Scores werden nicht gesetzt
    (kein Key im dict), sodass der Aufrufer None-Scores per .get() erkennen kann.
    """
    # Markdown-Codeblock entfernen
    code_block = re.search(r'```(?:json)?\s*(.*?)(?:```|$)', raw, re.DOTALL)
    inner = code_block.group(1).strip() if code_block else raw

    scores: dict = {}

    # Primär: XML-Tags (B7 — robustes Ausgabeformat)
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

    # Fallback 1: vollständiges JSON
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

    # Fallback 2: Key:Value-Regex (Legacy — für ältere Judge-Antworten)
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


def judge_with_retry(ask_fn, prompt: str, system: str, model: str,
                     max_tokens: int = 900, max_retries: int = 3,
                     temperature: float | None = None) -> dict:
    """Ruft ask_fn auf und wiederholt bis zu max_retries mal bei unvollständigem Parsing.

    ask_fn muss dasselbe Interface wie utils.llm.ask_text haben:
        ask_fn(prompt, system=..., model=..., max_tokens=..., cache_system=...,
               temperature=...) -> response

    temperature : None → Modell-Default (1.0); 0.0 → deterministisch (JUDGE_TEMPERATURE).
                  Für Reproduzierbarkeit und minimale Score-Varianz sollte bei
                  Modellen, die `temperature` akzeptieren (Sonnet, OpenAI),
                  JUDGE_TEMPERATURE=0.0 übergeben werden (Phase 3·2/A2). Bei Opus
                  4.7/4.8 wird `temperature` von der API abgelehnt und in
                  utils.llm.ask_text automatisch verworfen — dort ist der Judge
                  nicht deterministisch fixierbar (Default-Stochastik).

    Rückgabe: dict mit scores (fehlende Scores als None) + raw_response + usage.
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
    """Self-Consistency-Judge: k Samples bei gegebener temperature, Median je Score.

    Kostet k× die Judge-Calls von judge_with_retry.

    Modellabhängigkeit von `temperature`:
      • Sonnet/OpenAI akzeptieren `temperature`. Bei JUDGE_TEMPERATURE=0
        (deterministisch) liefert SC keine zusätzliche Information — in diesem
        Fall sollte judge_with_retry mit temperature=0 bevorzugt werden. SC ist
        nur bei temperature>0 sinnvoll.
      • Opus 4.7/4.8 (und Fable) lehnen `temperature` ab; der Parameter wird in
        utils.llm.ask_text automatisch verworfen. Die k Calls variieren dort
        dennoch über die Default-Stochastik, sodass SC die Diversität daraus
        bezieht — der übergebene temperature-Wert bleibt wirkungslos.

    Kostenwirkung Phase 3b: k=3, n=200, 4 Pipelines, 2 XAI-Modelle
        → 200 × 4 × 2 × k = 4 800 Judge-Calls statt 1 600 (Faktor k=3).
        Dies ist in die Kostenschätzung (Phase 3b) einzurechnen.

    Rückgabe: dict mit aggregierten Scores (Median), Roh-Antworten und
    kumuliertem usage.
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
