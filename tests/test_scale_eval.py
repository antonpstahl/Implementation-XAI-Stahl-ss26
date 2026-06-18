"""Phase 3b — utils.eval (gen-aware Loading + Judge-Prompt).

Sichert den Skalierungs-Eval-Pfad ab:
  * Template (00) wird mit 1 Generation geladen (deterministisch, kein _gen-Suffix),
    die LLM-Pipelines mit N Generationen (_gen{idx}-Suffix).
  * load_scale_records baut die erwarteten Spalten inkl. generation/tool_calls und
    meldet bzw. erzwingt Vollständigkeit.
  * build_judge_prompt erzeugt gültiges JSON mit dem NB-07-Schema und bettet das
    Tool-Transkript nur für Pipeline 06 (und nur bei vorhandenem trace) ein.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.eval import (
    build_judge_prompt,
    load_scale_records,
    n_generations_for,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal generation + explanation artifacts on disk
# ---------------------------------------------------------------------------

def _write_generation(results_dir: Path, pipeline: str, xai: str, iid: int,
                      gen_idx: int, n_gen: int, *, tool_calls=None):
    from utils.generation import generation_filename
    d = results_dir / f"pipeline{pipeline}"
    d.mkdir(parents=True, exist_ok=True)
    rec = {
        "pipeline":    f"{pipeline}",
        "xai_model":   xai,
        "instance_id": iid,
        "explanation": f"Erklärung {pipeline}/{xai}/{iid}/g{gen_idx}",
        "usage":       {"input_tokens": 100, "output_tokens": 50,
                        "cache_read_input_tokens": 80},
        "prediction":  390.0,
        "y_true":      387,
    }
    if tool_calls is not None:
        rec["tool_calls"] = tool_calls
        rec["n_tool_calls"] = len(tool_calls)
    (d / generation_filename(xai, iid, gen_idx, n_gen)).write_text(
        json.dumps(rec, ensure_ascii=False)
    )


def _write_local_explanation(expl_dir: Path, xai: str, iid: int,
                             loss_key: str = "poisson_log"):
    expl_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "model":       xai,
        "instance_id": iid,
        "prediction":  390.0,
        "y_true":      387,
        "feature_values": {
            "hr": 8, "weekday": 4, "mnth": 6, "yr": 0, "weathersit": 1,
            "temp": 0.50, "hum": 0.88, "windspeed": 0.10, "holiday": 0,
        },
        "contributions": [
            {"feature": "hr",  "contribution": 1.23, "value": 8},
            {"feature": "yr",  "contribution": -0.226, "value": 0},
            {"feature": "hum", "contribution": 0.05, "value": 0.88},
            {"feature": "temp", "contribution": 0.02, "value": 0.50},
        ],
    }
    (expl_dir / f"local_{xai.lower()}_{loss_key}_inst{iid}.json").write_text(
        json.dumps(data, ensure_ascii=False)
    )


# ---------------------------------------------------------------------------
# n_generations_for
# ---------------------------------------------------------------------------

def test_template_gets_one_generation():
    assert n_generations_for("00", 3) == 1


@pytest.mark.parametrize("pipeline", ["04", "05", "06"])
def test_llm_pipelines_get_scale_generations(pipeline):
    assert n_generations_for(pipeline, 3) == 3


# ---------------------------------------------------------------------------
# load_scale_records — gen-aware filenames + columns
# ---------------------------------------------------------------------------

def test_load_scale_records_gen_aware(tmp_path):
    iids = [101, 202]
    xai_models = ["xgb", "ebm"]
    # Template: 1 generation (no suffix); 04: 3 generations (_gen suffix)
    for xai in xai_models:
        for iid in iids:
            _write_generation(tmp_path, "00", xai, iid, 0, 1)
            for g in range(3):
                _write_generation(tmp_path, "04", xai, iid, g, 3)

    df = load_scale_records(["00", "04"], xai_models, iids, 3,
                            results_dir=tmp_path)

    # Template: 2 models × 2 instances × 1 gen = 4 rows
    assert len(df[df.pipeline == "00"]) == 4
    # 04: 2 × 2 × 3 = 12 rows
    assert len(df[df.pipeline == "04"]) == 12
    # generation column present and correct range
    assert set(df[df.pipeline == "00"]["generation"]) == {0}
    assert set(df[df.pipeline == "04"]["generation"]) == {0, 1, 2}
    # xai_model is upper-cased like NB 07
    assert set(df["xai_model"]) == {"XGB", "EBM"}


def test_load_scale_records_reports_missing(tmp_path, capsys):
    # Only write one of the expected files → the rest are missing
    _write_generation(tmp_path, "04", "xgb", 101, 0, 3)
    df = load_scale_records(["04"], ["xgb"], [101], 3, results_dir=tmp_path)
    out = capsys.readouterr().out
    assert "fehlende" in out
    assert len(df) == 1  # only the one written


def test_load_scale_records_require_complete_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scale_records(["04"], ["xgb"], [101], 3,
                           results_dir=tmp_path, require_complete=True)


def test_load_scale_records_carries_tool_calls(tmp_path):
    tc = [{"tool": "get_pd", "arguments": {"f": "hr"}, "result_preview": "..."}]
    _write_generation(tmp_path, "06", "xgb", 101, 0, 3, tool_calls=tc)
    _write_generation(tmp_path, "06", "xgb", 101, 1, 3, tool_calls=tc)
    _write_generation(tmp_path, "06", "xgb", 101, 2, 3, tool_calls=tc)
    df = load_scale_records(["06"], ["xgb"], [101], 3, results_dir=tmp_path)
    assert all(len(r) == 1 for r in df["tool_calls"])
    assert df.iloc[0]["n_tool_calls"] == 1


# ---------------------------------------------------------------------------
# build_judge_prompt
# ---------------------------------------------------------------------------

def test_build_judge_prompt_valid_json_schema(tmp_path):
    _write_local_explanation(tmp_path, "xgb", 101)
    row = {"pipeline": "04", "explanation": "Die Uhrzeit treibt die Nachfrage."}
    prompt = build_judge_prompt(row, "xgb", 101, explanations_dir=tmp_path)
    parsed = json.loads(prompt)
    assert set(parsed) == {"task", "ground_truth", "explanation", "output_format"}
    gt = parsed["ground_truth"]
    assert gt["prediction"] == 390.0 and gt["y_true"] == 387
    assert len(gt["top3_drivers"]) == 3
    assert gt["feature_values_readable"]["jahr"] == "2011"      # yr=0
    assert gt["feature_values_readable"]["uhrzeit"] == "08:00 Uhr"
    assert "tool_call_trace" not in gt                          # not pipeline 06


def test_build_judge_prompt_embeds_trace_for_tooluse(tmp_path):
    _write_local_explanation(tmp_path, "xgb", 101)
    row = {"pipeline": "06", "explanation": "Tool-Use Erklärung."}
    trace = [{"tool": "get_pd", "arguments": {"feature": "hr"},
              "result_preview": "peak at 8h"}]
    prompt = build_judge_prompt(row, "xgb", 101, explanations_dir=tmp_path,
                                tool_trace=trace)
    gt = json.loads(prompt)["ground_truth"]
    assert "tool_call_trace" in gt
    assert gt["tool_call_trace"][0]["round"] == 1
    assert gt["tool_call_trace"][0]["tool"] == "get_pd"
    assert gt["tool_call_trace"][0]["result"] == "peak at 8h"


def test_build_judge_prompt_no_trace_when_absent(tmp_path):
    """Pipeline 06 but no trace provided → no tool_call_trace key (defensive)."""
    _write_local_explanation(tmp_path, "xgb", 101)
    row = {"pipeline": "06", "explanation": "x"}
    gt = json.loads(build_judge_prompt(row, "xgb", 101,
                                       explanations_dir=tmp_path))["ground_truth"]
    assert "tool_call_trace" not in gt
