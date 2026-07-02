"""
Generation loop and persistence tests.

A test with a mocked LLM shows that generations per instance are persisted
correctly, resumable after an abort and idempotent (no double counting).

Tests `utils.run_resumable_generation`, into which the skip if exists loop
previously inlined three times (NB 04Lb/04Lc/04Ld) was extracted. The LLM call is
replaced by a counting `generate` callback.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.generation import (
    run_resumable_generation,
    generation_filename,
    load_local_explanation,
    load_global_explanation,
    build_generation_record,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODELS = ["xgb", "ebm"]
INSTANCES = [101, 202, 303]


def make_generate(calls: list[tuple]):
    """Returns a fake `generate` that records its invocations and builds a record."""

    def generate(model_name: str, instance_id: int, gen_idx: int):
        calls.append((model_name, instance_id, gen_idx))
        return {
            "xai_model": model_name,
            "instance_id": instance_id,
            "generation_idx": gen_idx,
            "explanation": f"text-{model_name}-{instance_id}-{gen_idx}",
        }

    return generate


# ---------------------------------------------------------------------------
# Filename schema
# ---------------------------------------------------------------------------

def test_filename_single_generation_has_no_suffix():
    """n_generations == 1 keeps the legacy name so existing artefacts resume."""
    assert generation_filename("xgb", 101, 0, 1) == "xgb_inst101.json"


def test_filename_multi_generation_has_gen_suffix():
    assert generation_filename("ebm", 202, 2, 3) == "ebm_inst202_gen2.json"


# ---------------------------------------------------------------------------
# Persistence (loss-free)
# ---------------------------------------------------------------------------

def test_persists_one_file_per_unit(tmp_path):
    calls: list[tuple] = []
    results = run_resumable_generation(
        model_names=MODELS,
        instance_ids=INSTANCES,
        out_dir=tmp_path,
        generate=make_generate(calls),
    )
    expected = len(MODELS) * len(INSTANCES)
    assert len(calls) == expected
    assert len(results) == expected
    written = sorted(p.name for p in tmp_path.glob("*.json"))
    assert len(written) == expected
    # Each file is valid JSON carrying the expected record.
    for p in tmp_path.glob("*.json"):
        rec = json.loads(p.read_text())
        assert rec["explanation"].startswith("text-")


def test_record_content_matches_generate_output(tmp_path):
    run_resumable_generation(
        model_names=["xgb"],
        instance_ids=[101],
        out_dir=tmp_path,
        generate=make_generate([]),
    )
    rec = json.loads((tmp_path / "xgb_inst101.json").read_text())
    assert rec == {
        "xai_model": "xgb",
        "instance_id": 101,
        "generation_idx": 0,
        "explanation": "text-xgb-101-0",
    }


# ---------------------------------------------------------------------------
# Idempotency (no double counting)
# ---------------------------------------------------------------------------

def test_rerun_is_idempotent(tmp_path):
    """A second full run must not call generate again nor duplicate records."""
    first_calls: list[tuple] = []
    first = run_resumable_generation(
        model_names=MODELS, instance_ids=INSTANCES, out_dir=tmp_path,
        generate=make_generate(first_calls),
    )

    second_calls: list[tuple] = []
    second = run_resumable_generation(
        model_names=MODELS, instance_ids=INSTANCES, out_dir=tmp_path,
        generate=make_generate(second_calls),
    )

    assert second_calls == []                       # nothing regenerated
    assert len(second) == len(first)                # no double counting
    assert second == first                          # identical records


# ---------------------------------------------------------------------------
# Resume after interruption
# ---------------------------------------------------------------------------

def test_resume_only_generates_missing(tmp_path):
    """After a partial run, only the missing units are generated."""
    # Simulate a crash after the first 2 of 6 units by pre-writing those files.
    done = run_resumable_generation(
        model_names=["xgb"], instance_ids=INSTANCES, out_dir=tmp_path,
        generate=make_generate([]),
    )
    assert len(done) == len(INSTANCES)              # 3 files now on disk

    resume_calls: list[tuple] = []
    results = run_resumable_generation(
        model_names=MODELS, instance_ids=INSTANCES, out_dir=tmp_path,
        generate=make_generate(resume_calls),
    )
    # Only the ebm units are new; the 3 xgb units are resumed from disk.
    assert sorted(resume_calls) == sorted(
        ("ebm", iid, 0) for iid in INSTANCES
    )
    assert len(results) == len(MODELS) * len(INSTANCES)


def test_resumed_records_come_from_disk(tmp_path):
    """Resumed records are read back verbatim, even if generate would differ."""
    run_resumable_generation(
        model_names=["xgb"], instance_ids=[101], out_dir=tmp_path,
        generate=make_generate([]),
    )

    def different_generate(model_name, instance_id, gen_idx):
        return {"explanation": "SHOULD-NOT-APPEAR"}

    results = run_resumable_generation(
        model_names=["xgb"], instance_ids=[101], out_dir=tmp_path,
        generate=different_generate,
    )
    assert results[0]["explanation"] == "text-xgb-101-0"


# ---------------------------------------------------------------------------
# Error skip (NB 04Ld tool-use loop may fail on an instance)
# ---------------------------------------------------------------------------

def test_none_record_is_not_persisted_and_retried_next_run(tmp_path):
    """generate returning None leaves the unit open (no file, retried later)."""
    state = {"fail": True}

    def flaky_generate(model_name, instance_id, gen_idx):
        if state["fail"]:
            return None                              # simulate failure
        return {"explanation": "recovered"}

    first = run_resumable_generation(
        model_names=["xgb"], instance_ids=[101], out_dir=tmp_path,
        generate=flaky_generate,
    )
    assert first == []                               # nothing persisted
    assert list(tmp_path.glob("*.json")) == []

    state["fail"] = False
    second = run_resumable_generation(
        model_names=["xgb"], instance_ids=[101], out_dir=tmp_path,
        generate=flaky_generate,
    )
    assert len(second) == 1
    assert second[0]["explanation"] == "recovered"


# ---------------------------------------------------------------------------
# Multiple generations per instance (Phase 3b: 3 generations)
# ---------------------------------------------------------------------------

def test_three_generations_per_instance_persisted(tmp_path):
    calls: list[tuple] = []
    results = run_resumable_generation(
        model_names=MODELS, instance_ids=INSTANCES, out_dir=tmp_path,
        generate=make_generate(calls), n_generations=3,
    )
    expected = len(MODELS) * len(INSTANCES) * 3
    assert len(calls) == expected
    assert len(results) == expected
    # Distinct files per generation index.
    assert (tmp_path / "xgb_inst101_gen0.json").exists()
    assert (tmp_path / "xgb_inst101_gen2.json").exists()
    assert len(list(tmp_path.glob("*.json"))) == expected


def test_multi_generation_resume_is_idempotent(tmp_path):
    run_resumable_generation(
        model_names=MODELS, instance_ids=INSTANCES, out_dir=tmp_path,
        generate=make_generate([]), n_generations=3,
    )
    rerun_calls: list[tuple] = []
    run_resumable_generation(
        model_names=MODELS, instance_ids=INSTANCES, out_dir=tmp_path,
        generate=make_generate(rerun_calls), n_generations=3,
    )
    assert rerun_calls == []


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def test_hooks_fire_for_result_then_skip(tmp_path):
    results_seen: list[tuple] = []
    skips_seen: list[tuple] = []

    run_resumable_generation(
        model_names=["xgb"], instance_ids=[101], out_dir=tmp_path,
        generate=make_generate([]),
        on_result=lambda rec, m, i, g: results_seen.append((m, i, g)),
        on_skip=lambda rec, m, i, g: skips_seen.append((m, i, g)),
    )
    assert results_seen == [("xgb", 101, 0)]
    assert skips_seen == []

    run_resumable_generation(
        model_names=["xgb"], instance_ids=[101], out_dir=tmp_path,
        generate=make_generate([]),
        on_result=lambda rec, m, i, g: results_seen.append((m, i, g)),
        on_skip=lambda rec, m, i, g: skips_seen.append((m, i, g)),
    )
    assert skips_seen == [("xgb", 101, 0)]           # second run resumes


def test_out_dir_created_if_missing(tmp_path):
    nested = tmp_path / "does" / "not" / "exist"
    run_resumable_generation(
        model_names=["xgb"], instance_ids=[101], out_dir=nested,
        generate=make_generate([]),
    )
    assert (nested / "xgb_inst101.json").exists()


# ---------------------------------------------------------------------------
# build_generation_record - golden schemas of the three pipelines (NB 04Lb/04Lc/04Ld)
#
# Freezes the records previously built inline: same keys, same order, same values.
# Breaks if the central builder diverges from the notebook schema (silent
# divergence = the expensive bug).
# ---------------------------------------------------------------------------

USAGE_FULL = {"input_tokens": 600, "output_tokens": 510, "cache_read_input_tokens": 1024}
USAGE_TOOL = {"input_tokens": 3489, "output_tokens": 1225}


def test_record_schema_04Lb_json():
    expected = {
        "pipeline":    "04Lb",
        "llm_model":   "claude-sonnet-4-6",
        "loss_key":    "poisson_log",
        "xai_model":   "xgb",
        "instance_id": 224,
        "explanation": "Explanation text",
        "elapsed_s":   11.7,
        "usage":       {"input_tokens": 600, "output_tokens": 510,
                        "cache_read_input_tokens": 1024},
        "prediction":  270.4,
        "y_true":      270,
    }
    rec = build_generation_record(
        pipeline="04Lb", model_name="xgb", instance_id=224,
        explanation="Explanation text", usage=USAGE_FULL,
        llm_model="claude-sonnet-4-6", loss_key="poisson_log",
        prediction=270.4, y_true=270, elapsed_s=11.7,
    )
    assert rec == expected
    assert list(rec) == list(expected)  # identical key order


def test_record_schema_04Lc_vision():
    expected = {
        "pipeline":    "04Lc",
        "llm_model":   "claude-sonnet-4-6",
        "loss_key":    "poisson_log",
        "xai_model":   "ebm",
        "instance_id": 580,
        "explanation": "Image explanation",
        "plot_file":   "waterfall_ebm_poisson_log_inst580.png",
        "elapsed_s":   12.3,
        "usage":       {"input_tokens": 600, "output_tokens": 510,
                        "cache_read_input_tokens": 1024},
        "prediction":  5.1,
        "y_true":      5,
    }
    rec = build_generation_record(
        pipeline="04Lc", model_name="ebm", instance_id=580,
        explanation="Image explanation", usage=USAGE_FULL,
        llm_model="claude-sonnet-4-6", loss_key="poisson_log",
        prediction=5.1, y_true=5, elapsed_s=12.3,
        extra={"plot_file": "waterfall_ebm_poisson_log_inst580.png"},
    )
    assert rec == expected
    assert list(rec) == list(expected)


def test_record_schema_04Ld_tooluse():
    call_log = [{"tool": "get_shap_values", "arguments": {"instance_id": 224}}]
    expected = {
        "pipeline":     "04Ld",
        "llm_model":    "claude-sonnet-4-6",
        "loss_key":     "poisson_log",
        "xai_model":    "xgb",
        "instance_id":  224,
        "explanation":  "Tool explanation",
        "stop_reason":  "end_turn",
        "tool_calls":   call_log,
        "n_tool_calls": 1,
        "elapsed_s":    28.8,
        "usage":        {"input_tokens": 3489, "output_tokens": 1225},
        "y_true":       270.0,
    }
    rec = build_generation_record(
        pipeline="04Ld", model_name="xgb", instance_id=224,
        explanation="Tool explanation", usage=USAGE_TOOL,
        llm_model="claude-sonnet-4-6", loss_key="poisson_log",
        y_true=270.0, elapsed_s=28.8, include_cache=False,
        extra={"stop_reason": "end_turn", "tool_calls": call_log,
               "n_tool_calls": 1},
    )
    assert rec == expected
    assert list(rec) == list(expected)        # order: extra before elapsed_s
    assert "prediction" not in rec            # Tool Use carries no prediction
    assert "cache_read_input_tokens" not in rec["usage"]


# ---------------------------------------------------------------------------
# load_local_explanation / load_global_explanation
# ---------------------------------------------------------------------------

def test_load_local_and_global_explanation(tmp_path):
    (tmp_path / "local_xgb_poisson_log_inst224.json").write_text(
        json.dumps({"prediction": 270.4, "y_true": 270}), encoding="utf-8")
    (tmp_path / "global_xgb_poisson_log.json").write_text(
        json.dumps({"global_importance": [{"rank": 0, "feature": "hr"}]}),
        encoding="utf-8")

    loc = load_local_explanation("xgb", 224, explanations_dir=tmp_path)
    glo = load_global_explanation("xgb", explanations_dir=tmp_path)
    assert loc["prediction"] == 270.4 and loc["y_true"] == 270
    assert glo["global_importance"][0]["feature"] == "hr"


def test_load_local_explanation_custom_loss_key(tmp_path):
    (tmp_path / "local_ebm_squared_error_inst5.json").write_text(
        json.dumps({"prediction": 1.0, "y_true": 2}), encoding="utf-8")
    loc = load_local_explanation("ebm", 5, loss_key="squared_error",
                                 explanations_dir=tmp_path)
    assert loc["y_true"] == 2
