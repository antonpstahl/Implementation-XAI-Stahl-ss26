"""Tests for utils.global_feature + utils.global_tools (global per-feature pipeline, G2a).

No API calls: artifact loaders, JSON payload, filename/record schema, the resumable
feature loop (skip/idempotent/lossless/error-skip), and the GlobalToolBox dispatch
(data tools, image tools, error handling). Reads the real G0/G1 artifacts under
explanations/.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import EXPLANATIONS_DIR
from utils.global_feature import (
    build_feature_json_payload,
    build_global_record,
    describe_curve,
    feature_importance_map,
    global_generation_filename,
    list_global_features,
    load_global_curve,
    readable_feature_value,
    run_resumable_global_generation,
    shape_plot_path,
    beeswarm_plot_path,
)
from utils.global_tools import (
    GLOBAL_TOOL_DEFINITIONS,
    GlobalToolBox,
    ToolImage,
    _tool_result_content,
)

PLOTS = EXPLANATIONS_DIR / "plots" / "global"


# -----------------------------------------------------------------------------
# Artifact loaders / payload
# -----------------------------------------------------------------------------

def test_features_in_rank_order():
    feats = list_global_features("ebm", explanations_dir=EXPLANATIONS_DIR)
    assert len(feats) == 9
    ranks = [feature_importance_map("ebm", explanations_dir=EXPLANATIONS_DIR)[f]["rank"] for f in feats]
    assert ranks == sorted(ranks)  # rank 1 first


def test_json_payload_matches_curve_and_importance():
    payload = build_feature_json_payload("ebm", "hr", explanations_dir=EXPLANATIONS_DIR)
    curve = load_global_curve("ebm", "hr", explanations_dir=EXPLANATIONS_DIR)
    imp = feature_importance_map("ebm", explanations_dir=EXPLANATIONS_DIR)["hr"]
    assert payload["feature"] == "hr"
    assert payload["kind"] == curve["kind"]
    assert payload["curve"]["x"] == curve["x"]
    assert payload["curve"]["y"] == curve["y"]
    assert len(payload["curve"]["x"]) == len(payload["curve"]["y"])
    assert payload["importance"] == imp["importance"]
    assert payload["rank"] == imp["rank"]
    assert payload["n_features"] == 9
    assert payload["target"]["name"] == "cnt"


def test_plot_paths_resolve():
    assert shape_plot_path("ebm", "hr", plots_dir=PLOTS).name == "ebm_shape_hr.png"
    assert shape_plot_path("xgb", "hr", plots_dir=PLOTS).name == "xgb_dependence_hr.png"
    assert shape_plot_path("ebm", "hr", plots_dir=PLOTS).exists()
    assert shape_plot_path("xgb", "temp", plots_dir=PLOTS).exists()
    assert beeswarm_plot_path("ebm", plots_dir=PLOTS).exists()


def test_bad_model_name_rejected():
    with pytest.raises(ValueError):
        shape_plot_path("lgbm", "hr", plots_dir=PLOTS)


# -----------------------------------------------------------------------------
# Curve derivation (shared baseline + G3 ground-truth source)
# -----------------------------------------------------------------------------

def test_readable_feature_value():
    assert readable_feature_value("hr", 7) == "07:00"
    assert readable_feature_value("temp", 0.68).startswith("~27.9")
    assert readable_feature_value("yr", 0) == "2011"
    assert readable_feature_value("yr", 1) == "2012"
    assert readable_feature_value("weekday", 3) == "Wednesday"
    assert readable_feature_value("holiday", 1) == "holiday"


def test_describe_curve_fields_and_domains():
    d = describe_curve("ebm", "hr", explanations_dir=EXPLANATIONS_DIR)
    assert set(d) == {"feature", "kind", "direction", "monotonicity", "shape",
                      "peak_x", "peak_value", "peak_label"}
    assert d["direction"] in {"rising", "falling", "mixed", "flat"}
    assert d["monotonicity"] in {"monotonic", "non_monotonic", "flat"}
    assert d["shape"] in {"monotonic", "non_monotonic", "categorical", "near_flat"}


def test_describe_curve_categorical_and_monotonic():
    # hr is a categorical curve
    assert describe_curve("ebm", "hr", explanations_dir=EXPLANATIONS_DIR)["shape"] == "categorical"
    # yr is a monotonic step for the EBM (2011 -> 2012 rising)
    yr = describe_curve("ebm", "yr", explanations_dir=EXPLANATIONS_DIR)
    assert yr["monotonicity"] == "monotonic" and yr["direction"] == "rising"
    assert yr["peak_label"] == "2012"


def test_describe_curve_matches_manual_peak():
    # peak_label must be the readable value at the argmax of y
    curve = load_global_curve("ebm", "temp", explanations_dir=EXPLANATIONS_DIR)
    y = [float(v) for v in curve["y"]]
    x = [float(v) for v in curve["x"]]
    peak_raw = x[max(range(len(y)), key=lambda i: y[i])]
    d = describe_curve("ebm", "temp", explanations_dir=EXPLANATIONS_DIR)
    assert d["peak_label"] == readable_feature_value("temp", peak_raw)


# -----------------------------------------------------------------------------
# Filename + record schema
# -----------------------------------------------------------------------------

def test_filename_scheme():
    assert global_generation_filename("json", "ebm", "hr") == "json_ebm_hr.json"
    assert global_generation_filename("vision", "XGB", "temp") == "vision_xgb_temp.json"
    # suffix only from the 2nd generation on
    assert global_generation_filename("tooluse", "ebm", "yr", 0, 3) == "tooluse_ebm_yr_gen0.json"


def test_record_schema():
    rec = build_global_record(
        form="json", model_name="ebm", feature="hr", explanation="text",
        usage={"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 3},
        llm_model="claude-x", elapsed_s=1.2, extra={"plot_file": "p.png"},
    )
    assert rec["scope"] == "global"
    assert rec["form"] == "json"
    assert rec["xai_model"] == "ebm"
    assert rec["feature"] == "hr"
    assert rec["plot_file"] == "p.png"           # extra inserted after explanation
    assert "instance_id" not in rec and "y_true" not in rec
    assert rec["usage"]["cache_read_input_tokens"] == 3


def test_record_no_cache_for_tooluse():
    rec = build_global_record(
        form="tooluse", model_name="xgb", feature="temp", explanation="t",
        usage={"input_tokens": 1, "output_tokens": 1}, llm_model="m", include_cache=False,
    )
    assert "cache_read_input_tokens" not in rec["usage"]


# -----------------------------------------------------------------------------
# Resumable loop contract (mirror of the local generation loop)
# -----------------------------------------------------------------------------

def _fake_gen_factory(calls):
    def gen(model, feature, gi):
        calls.append((model, feature, gi))
        return build_global_record(
            form="json", model_name=model, feature=feature,
            explanation=f"{model}:{feature}:{gi}",
            usage={"input_tokens": 1, "output_tokens": 1}, llm_model="test",
        )
    return gen


def test_loop_produces_and_persists(tmp_path):
    calls = []
    recs = run_resumable_global_generation(
        form="json", model_names=["ebm"], features=["hr", "temp"],
        out_dir=tmp_path, generate=_fake_gen_factory(calls),
    )
    assert len(recs) == 2
    assert {p.name for p in tmp_path.glob("*.json")} == {"json_ebm_hr.json", "json_ebm_temp.json"}
    assert len(calls) == 2


def test_loop_idempotent_resume(tmp_path):
    calls = []
    gen = _fake_gen_factory(calls)
    r1 = run_resumable_global_generation(form="json", model_names=["ebm"], features=["hr"], out_dir=tmp_path, generate=gen)
    r2 = run_resumable_global_generation(form="json", model_names=["ebm"], features=["hr"], out_dir=tmp_path, generate=gen)
    assert len(r1) == len(r2) == 1
    assert len(calls) == 1            # second run skips (file exists)
    assert r1[0] == r2[0]


def test_loop_error_skip_leaves_unit_open(tmp_path):
    def gen(model, feature, gi):
        return None                   # simulate an error
    recs = run_resumable_global_generation(
        form="json", model_names=["ebm"], features=["hr"], out_dir=tmp_path, generate=gen,
    )
    assert recs == []
    assert list(tmp_path.glob("*.json")) == []


# -----------------------------------------------------------------------------
# GlobalToolBox
# -----------------------------------------------------------------------------

@pytest.fixture()
def box():
    return GlobalToolBox("ebm", explanations_dir=EXPLANATIONS_DIR, plots_dir=PLOTS)


def test_tool_definitions_names():
    assert [t["name"] for t in GLOBAL_TOOL_DEFINITIONS] == [
        "get_target_overview", "get_feature_importances",
        "get_feature_curve", "get_feature_plot", "get_beeswarm_plot",
    ]


def test_toolbox_data_tools(box):
    ov = box.dispatch("get_target_overview", {})
    assert ov["target"]["name"] == "cnt" and len(ov["features"]) == 9
    imps = box.dispatch("get_feature_importances", {})
    assert imps[0]["rank"] == 1
    cur = box.dispatch("get_feature_curve", {"feature": "temp"})
    assert cur["feature"] == "temp" and "x" in cur and cur["rank"] == 2


def test_toolbox_image_tools_and_content(box):
    img = box.dispatch("get_feature_plot", {"feature": "hr"})
    assert isinstance(img, ToolImage) and img.path.name == "ebm_shape_hr.png"
    content = _tool_result_content(img)
    assert [b["type"] for b in content] == ["text", "image"]
    bee = box.dispatch("get_beeswarm_plot", {})
    assert isinstance(bee, ToolImage) and bee.path.name == "ebm_beeswarm.png"


def test_toolbox_errors_are_results_not_exceptions(box):
    assert "error" in box.dispatch("get_feature_plot", {"feature": "nope"})
    assert "error" in box.dispatch("unknown_tool", {})
    # every dispatch is logged, including the errors
    assert len(box.call_log) == 2
