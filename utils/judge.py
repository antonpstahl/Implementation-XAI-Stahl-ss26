"""Judge-Parsing: extrahiert strukturierte Scores aus LLM-as-Judge-Antworten."""

from __future__ import annotations

import json
import re


def parse_judge_response(raw: str) -> dict:
    """Extrahiert Judge-Scores robust aus Plaintext, JSON oder Markdown-Codeblock.

    Gibt ein dict mit den Keys faithfulness, clarity, completeness (int, 1-5)
    und optionalen *_reasoning-Keys zurück. Fehlende Scores werden nicht gesetzt
    (kein Key im dict), sodass der Aufrufer None-Scores per .get() erkennen kann.
    """
    # Markdown-Codeblock entfernen
    code_block = re.search(r'```(?:json)?\s*(.*?)(?:```|$)', raw, re.DOTALL)
    inner = code_block.group(1).strip() if code_block else raw

    scores: dict = {}

    # Vollständiges JSON versuchen
    try:
        json_match = re.search(r'\{.*\}', inner, re.DOTALL)
        if json_match:
            d = json.loads(json_match.group())
            d_up = {k.upper(): v for k, v in d.items()}
            for key in ['FAITHFULNESS', 'CLARITY', 'COMPLETENESS']:
                if key in d_up:
                    scores[key.lower()] = int(d_up[key])
            for key in ['FAITHFULNESS_REASONING', 'CLARITY_REASONING', 'COMPLETENESS_REASONING']:
                base = key.replace('_REASONING', '').lower()
                if key in d_up:
                    scores[f'{base}_reasoning'] = str(d_up[key])
            if len(scores) >= 3:
                return scores
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: Regex auf JSON-Keys (auch bei abgeschnittenem JSON)
    for key in ['FAITHFULNESS', 'CLARITY', 'COMPLETENESS']:
        m = re.search(rf'"?{key}"?\s*:\s*(\d)', inner, re.IGNORECASE)
        if m:
            scores[key.lower()] = int(m.group(1))
    for key in ['FAITHFULNESS_REASONING', 'CLARITY_REASONING', 'COMPLETENESS_REASONING']:
        base = key.replace('_REASONING', '').lower()
        m = re.search(rf'"?{key}"?\s*:\s*"([^"]+)', inner, re.IGNORECASE)
        if m:
            scores[f'{base}_reasoning'] = m.group(1)

    return scores
