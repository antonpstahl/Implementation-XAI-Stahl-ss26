"""utils/eval.py - scaling evaluation.

Generation aware loading of the generation artefacts and building of the judge
prompt for a larger run. Deliberately separate from the n=20 validity notebook
(NB 05):

  * NB 05 stays untouched (its judge caches and inter judge agreement hold, n = 20).
  * `07b_Scaling_Evaluation` uses these helpers and runs only the final Opus judge
    plus cross vendor on the 200 instances x N generations.

`build_judge_prompt` is a faithful port of the judge prompt build from NB 05
(cell 9), identical format (XML reason then score, the same human readable feature
values), so the scaling judge scores exactly by the validated rubric. The only
difference: the Tool Use transcript (pipeline 06) is passed explicitly
(`tool_trace`) instead of read from a fixed path, because at N generations the
trace file is generation specific (`..._gen{g}.json`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from utils import EXPLANATIONS_DIR, RESULTS_DIR
from utils.generation import generation_filename

LOSS_KEY_DEFAULT = "poisson_log"

# Pipeline code to display name (identical to NB 05).
PIPELINE_LABELS = {
    "00": "Template",
    "04": "JSON to Text",
    "05": "Vision",
    "06": "Tool Use",
}

# Deterministic pipelines: produce identical text per instance, so 1 generation is
# enough (no stochasticity to measure). Template (00) is the template generator.
DETERMINISTIC_PIPELINES = frozenset({"00"})

# Tool Use pipeline (client side tool loop; the trace is attached for the judge).
TOOLUSE_PIPELINES = frozenset({"06"})

# Cost per 1M tokens (claude-sonnet-4-6), reporting only, identical to NB 05.
COST_INPUT_PER_M      = 3.00
COST_CACHE_READ_PER_M = 0.30
COST_OUTPUT_PER_M     = 15.00

# Human readable feature values for the judge (faithful port from NB 05 cell 9).
WEEKDAYS_JUDGE = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
                  4: "Thursday", 5: "Friday", 6: "Saturday"}
MONTHS_JUDGE   = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May",
                  6: "June", 7: "July", 8: "August", 9: "September",
                  10: "October", 11: "November", 12: "December"}
WEATHER_JUDGE  = {1: "clear/few clouds", 2: "mist/cloudy",
                  3: "light rain/snow", 4: "heavy rain/thunderstorm"}


def n_generations_for(pipeline: str, n_generations_scale: int) -> int:
    """Generations per unit per pipeline: deterministic -> 1, otherwise scale."""
    return 1 if pipeline in DETERMINISTIC_PIPELINES else n_generations_scale


def load_scale_records(
    pipelines: list[str],
    xai_models: list[str],
    instance_ids: list[int],
    n_generations_scale: int,
    *,
    results_dir: Path = RESULTS_DIR,
    loss_key: str = LOSS_KEY_DEFAULT,
    scale_subdir: str = "scale",
    require_complete: bool = False,
) -> pd.DataFrame:
    """Load the scaling run generation artefacts generation aware into a df.

    Reads ``pipeline{p}/{scale_subdir}/{xai}_inst{iid}[_gen{g}].json`` over all
    pipelines x XAI models x instances x generations. The file name scheme follows
    :func:`utils.generation.generation_filename`: deterministic pipelines (Template)
    without a ``_gen`` suffix (1 generation), LLM pipelines with a suffix.

    The subfolder `scale_subdir` (default ``"scale"``) separates the larger run
    physically from the n=20 validity run (which sits directly under
    ``pipeline{p}/``). ``scale_subdir=""`` reads directly from ``pipeline{p}/``
    (for example for tests).

    Beyond the NB 05 column set each row carries a ``generation`` column (0 based)
    and, for Tool Use, the full ``tool_calls`` list (for the judge trace). Missing
    files are reported; with ``require_complete=True`` they raise a
    ``FileNotFoundError`` (guard against silent gaps before evaluation).
    """
    records: list[dict] = []
    missing: list[str] = []

    for pipeline in pipelines:
        p_dir = results_dir / f"pipeline{pipeline}"
        if scale_subdir:
            p_dir = p_dir / scale_subdir
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
        msg = f"{len(missing)} missing generation file(s) (first 5): {missing[:5]}"
        if require_complete:
            raise FileNotFoundError(msg)
        print(f"WARNING: {msg}")

    return pd.DataFrame(records)


def _tool_trace_block(tool_calls: list[dict]) -> list[dict]:
    """Build the judge trace format from a ``tool_calls`` list (NB 05 schema)."""
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
    """Build the judge user prompt (JSON) for one explanation, port from NB 05.

    Identical to the validity notebook: human readable feature values, top 3
    drivers, reason then score XML output instruction. For Tool Use the transcript
    is attached via `tool_trace` (a list of ``tool_calls`` dicts) instead of read
    from a fixed path (generation specific at N > 1).
    """
    local_path = explanations_dir / f"local_{xai_model.lower()}_{loss_key}_inst{instance_id}.json"
    l = json.loads(local_path.read_text())
    fv = l["feature_values"]

    top3 = [{"feature": c["feature"], "contribution": c["contribution"],
             "value": c["value"]}
            for c in l["contributions"][:3]]

    fv_readable = {
        "time":                f"{int(fv['hr']):02d}:00",
        "weekday":             WEEKDAYS_JUDGE.get(int(fv["weekday"]), str(fv["weekday"])),
        "month":               MONTHS_JUDGE.get(int(fv["mnth"]), str(fv["mnth"])),
        "year":                "2011" if int(fv["yr"]) == 0 else "2012",
        "weather":             WEATHER_JUDGE.get(int(fv["weathersit"]), str(fv["weathersit"])),
        "temperature_celsius": f"~{float(fv['temp']) * 41:.1f} C",
        "humidity":            f"{float(fv['hum']) * 100:.0f} %",
        "wind_speed":          f"{float(fv['windspeed']) * 67:.1f} km/h",
        "holiday":             "yes" if int(fv["holiday"]) == 1 else "no",
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
            "The retrieved values (contributions, percentiles, counterfactuals) "
            "are correct and may be counted as evidence for faithfulness."
        )

    output_instruction = (
        "Answer only in the XML format from the system prompt.\n"
        "Per criterion: first the reasoning (1 to 2 sentences), then the score as an XML tag.\n"
        "\n"
        "<faithfulness_reasoning>Choose anchor point, check deductions, compute final score</faithfulness_reasoning>\n"
        "<faithfulness>N</faithfulness>\n"
        "<clarity_reasoning>Choose anchor point, check deductions, compute final score</clarity_reasoning>\n"
        "<clarity>N</clarity>\n"
        "<completeness_reasoning>Choose anchor point, check deductions, compute final score</completeness_reasoning>\n"
        "<completeness>N</completeness>"
    )
    return json.dumps({
        "task": (
            "Score the following explanation using the defined rubric. "
            "Give a score (1 to 5) for each criterion and justify briefly."
        ),
        "ground_truth": ground_truth,
        "explanation": row["explanation"],
        "output_format": output_instruction,
    }, ensure_ascii=False, indent=2)
