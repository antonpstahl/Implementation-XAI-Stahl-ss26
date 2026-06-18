"""utils/eval.py – Skalierungs-Evaluation (Phase 3b).

Gen-aware Loading der Generierungs-Artefakte und Aufbau des Judge-Prompts für
den n≈200-Vollauf. Bewusst getrennt vom n=20-**Validitäts**-Notebook (NB 07):

  * NB 07 bleibt unangetastet (alle v1/v2/v3/v4/v5-Caches + Inter-Judge-
    Agreement gelten weiter, n = 20).
  * `07b_Scaling_Evaluation` nutzt diese Helfer und fährt **nur** den finalen
    Opus-Judge + Cross-Vendor auf den 200 Instanzen × N Generationen.

`build_judge_prompt` ist ein **treuer Port** des Judge-Prompt-Aufbaus aus NB 07
(Zelle 9) — identisches Format (XML-Reason-then-Score, dieselben
menschenlesbaren Feature-Werte), damit der Skalierungs-Judge exakt nach der in
Phase 3·2 validierten Rubrik bewertet. Der einzige Unterschied: das Tool-Use-
Transkript (Pipeline 06) wird **explizit** übergeben (`tool_trace`), statt aus
einem festen Pfad gelesen zu werden — denn bei N Generationen ist die
Trace-Datei generationsspezifisch (`…_gen{g}.json`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from utils import EXPLANATIONS_DIR, RESULTS_DIR
from utils.generation import generation_filename

LOSS_KEY_DEFAULT = "poisson_log"

# Pipeline-Kürzel → Anzeigename (identisch zu NB 07).
PIPELINE_LABELS = {
    "00": "Template",
    "04": "JSON→Text",
    "05": "Vision",
    "06": "Tool-Use",
}

# Deterministische Pipelines: erzeugen pro Instanz identischen Text → 1 Generation
# genügt (keine Stochastik zu messen). Template (00) ist der Textbaustein-Generator.
DETERMINISTIC_PIPELINES = frozenset({"00"})

# Tool-Use-Pipeline (client-seitiger Tool-Loop; Trace wird dem Judge beigelegt).
TOOLUSE_PIPELINES = frozenset({"06"})

# Kosten pro 1M Token (claude-sonnet-4-6) — nur fürs Reporting, identisch zu NB 07.
COST_INPUT_PER_M      = 3.00
COST_CACHE_READ_PER_M = 0.30
COST_OUTPUT_PER_M     = 15.00

# Menschenlesbare Feature-Werte für den Judge (treuer Port aus NB 07 Zelle 9).
WEEKDAYS_JUDGE = {0: "Sonntag", 1: "Montag", 2: "Dienstag", 3: "Mittwoch",
                  4: "Donnerstag", 5: "Freitag", 6: "Samstag"}
MONTHS_JUDGE   = {1: "Januar", 2: "Februar", 3: "März", 4: "April", 5: "Mai",
                  6: "Juni", 7: "Juli", 8: "August", 9: "September",
                  10: "Oktober", 11: "November", 12: "Dezember"}
WEATHER_JUDGE  = {1: "klar/wenige Wolken", 2: "Nebel/bewölkt",
                  3: "leichter Regen/Schnee", 4: "Starkregen/Gewitter"}


def n_generations_for(pipeline: str, n_generations_scale: int) -> int:
    """Generationen pro Einheit je Pipeline: deterministische → 1, sonst Scale."""
    return 1 if pipeline in DETERMINISTIC_PIPELINES else n_generations_scale


def load_scale_records(
    pipelines: list[str],
    xai_models: list[str],
    instance_ids: list[int],
    n_generations_scale: int,
    *,
    results_dir: Path = RESULTS_DIR,
    loss_key: str = LOSS_KEY_DEFAULT,
    require_complete: bool = False,
) -> pd.DataFrame:
    """Lädt die Generierungs-Artefakte des Skalierungslaufs gen-aware in einen df.

    Liest ``pipeline{p}/{xai}_inst{iid}[_gen{g}].json`` über alle Pipelines ×
    XAI-Modelle × Instanzen × Generationen. Das Dateinamensschema folgt
    :func:`utils.generation.generation_filename`: deterministische Pipelines
    (Template) ohne ``_gen``-Suffix (1 Generation), LLM-Pipelines mit Suffix.

    Jede Zeile trägt zusätzlich zur NB-07-Spaltenmenge eine ``generation``-Spalte
    (0-basiert) und — für Tool-Use — die volle ``tool_calls``-Liste (für den
    Judge-Trace). Fehlende Dateien werden gemeldet; mit ``require_complete=True``
    lösen sie einen ``FileNotFoundError`` aus (Schutz vor stillen Lücken vor der
    Auswertung).
    """
    records: list[dict] = []
    missing: list[str] = []

    for pipeline in pipelines:
        p_dir = results_dir / f"pipeline{pipeline}"
        n_gen = n_generations_for(pipeline, n_generations_scale)
        for xai in xai_models:
            for iid in instance_ids:
                for gen_idx in range(n_gen):
                    fname = generation_filename(xai, iid, gen_idx, n_gen)
                    f = p_dir / fname
                    if not f.exists():
                        missing.append(str(f))
                        continue
                    d = json.loads(f.read_text())
                    usage   = d.get("usage", {})
                    in_tok  = usage.get("input_tokens", 0)
                    out_tok = usage.get("output_tokens", 0)
                    cache_r = usage.get("cache_read_input_tokens", 0)
                    regular_in = max(in_tok - cache_r, 0)
                    cost = (
                        regular_in * COST_INPUT_PER_M
                        + cache_r  * COST_CACHE_READ_PER_M
                        + out_tok  * COST_OUTPUT_PER_M
                    ) / 1_000_000
                    records.append({
                        "pipeline":       pipeline,
                        "pipeline_label": PIPELINE_LABELS[pipeline],
                        "xai_model":      xai.upper(),
                        "instance_id":    iid,
                        "generation":     gen_idx,
                        "explanation":    d.get("explanation", ""),
                        "word_count":     len(d.get("explanation", "").split()),
                        "tok_input":      in_tok,
                        "tok_output":     out_tok,
                        "tok_cache":      cache_r,
                        "tok_total":      in_tok + out_tok,
                        "cost_usd":       round(cost, 5),
                        "elapsed_s":      d.get("elapsed_s", 0),
                        "n_tool_calls":   d.get("n_tool_calls", 0),
                        "tool_calls":     d.get("tool_calls", []),
                        "y_true":         d.get("y_true", None),
                        "prediction":     d.get("prediction", None),
                    })

    if missing:
        msg = f"{len(missing)} fehlende Generierungs-Datei(en) (erste 5): {missing[:5]}"
        if require_complete:
            raise FileNotFoundError(msg)
        print(f"⚠️  {msg}")

    return pd.DataFrame(records)


def _tool_trace_block(tool_calls: list[dict]) -> list[dict]:
    """Baut das Judge-Trace-Format aus einer ``tool_calls``-Liste (NB-07-Schema)."""
    return [
        {
            "round":     i + 1,
            "tool":      c.get("tool"),
            "arguments": c.get("arguments"),
            "result":    c.get("result_preview"),
        }
        for i, c in enumerate(tool_calls or [])
    ]


def build_judge_prompt(
    row: dict,
    xai_model: str,
    instance_id: int,
    *,
    loss_key: str = LOSS_KEY_DEFAULT,
    explanations_dir: Path = EXPLANATIONS_DIR,
    tool_trace: Optional[list[dict]] = None,
) -> str:
    """Baut den Judge-User-Prompt (JSON) für eine Erklärung — Port aus NB 07.

    Identisch zum Validitäts-Notebook: menschenlesbare Feature-Werte, Top-3-
    Treiber, Reason-then-Score-XML-Ausgabeanweisung. Für Tool-Use wird das
    Transkript über `tool_trace` (Liste von ``tool_calls``-Dicts) beigelegt, statt
    es aus einem festen Pfad zu lesen (generationsspezifisch bei N > 1).
    """
    local_path = explanations_dir / f"local_{xai_model.lower()}_{loss_key}_inst{instance_id}.json"
    l = json.loads(local_path.read_text())
    fv = l["feature_values"]

    top3 = [{"feature": c["feature"], "contribution": c["contribution"],
             "value": c["value"]}
            for c in l["contributions"][:3]]

    fv_readable = {
        "uhrzeit":             f"{int(fv['hr']):02d}:00 Uhr",
        "wochentag":           WEEKDAYS_JUDGE.get(int(fv["weekday"]), str(fv["weekday"])),
        "monat":               MONTHS_JUDGE.get(int(fv["mnth"]), str(fv["mnth"])),
        "jahr":                "2011" if int(fv["yr"]) == 0 else "2012",
        "wetter":              WEATHER_JUDGE.get(int(fv["weathersit"]), str(fv["weathersit"])),
        "temperatur_celsius":  f"~{float(fv['temp']) * 41:.1f} °C",
        "luftfeuchtigkeit":    f"{float(fv['hum']) * 100:.0f} %",
        "windgeschwindigkeit": f"{float(fv['windspeed']) * 67:.1f} km/h",
        "feiertag":            "ja" if int(fv["holiday"]) == 1 else "nein",
    }

    ground_truth = {
        "model":                   xai_model,
        "prediction":              l["prediction"],
        "y_true":                  l["y_true"],
        "top3_drivers":            top3,
        "feature_values_readable": fv_readable,
    }

    pipeline = row.get("pipeline", "")
    if (pipeline in TOOLUSE_PIPELINES or pipeline == "06_tooluse") and tool_trace:
        ground_truth["tool_call_trace"] = _tool_trace_block(tool_trace)
        ground_truth["tool_trace_note"] = (
            "Die abgerufenen Werte (Beiträge, Percentile, Counterfactuals) "
            "sind korrekt und dürfen als Belege für Faithfulness gewertet werden."
        )

    output_instruction = (
        "Antworte ausschließlich im XML-Format aus dem System-Prompt (B7).\n"
        "Je Kriterium: erst Begründung (1–2 Sätze), dann Score als XML-Tag.\n"
        "\n"
        "<faithfulness_reasoning>Ankerpunkt wählen, Abzüge prüfen, Endpunktzahl berechnen</faithfulness_reasoning>\n"
        "<faithfulness>N</faithfulness>\n"
        "<clarity_reasoning>Ankerpunkt wählen, Abzüge prüfen, Endpunktzahl berechnen</clarity_reasoning>\n"
        "<clarity>N</clarity>\n"
        "<completeness_reasoning>Ankerpunkt wählen, Abzüge prüfen, Endpunktzahl berechnen</completeness_reasoning>\n"
        "<completeness>N</completeness>"
    )
    return json.dumps({
        "task": (
            "Bewerte die folgende Erklärung nach der definierten Rubrik. "
            "Vergib für jedes Kriterium einen Score (1–5) und begründe kurz."
        ),
        "ground_truth": ground_truth,
        "explanation": row["explanation"],
        "output_format": output_instruction,
    }, ensure_ascii=False, indent=2)
