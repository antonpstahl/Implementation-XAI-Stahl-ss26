"""Tests for utils.global_whole (whole-model pipeline, Phase G2b).

No API calls: condition registry, whole-model payloads (full "all" vs the info-matched
numeric beeswarm control), vision plot paths, prompt assembly (byte-identical core +
per-condition handover), the forced per-feature schema splitter (shared recommendation,
dropped features), that a split record round-trips through the G3 rubric, and the
output-token truncation guards (P0-1). Reads the real G0/G1 artifacts under
explanations/.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import EXPLANATIONS_DIR, PROMPTS_DIR
from utils import rubric
from utils.global_feature import list_global_features
from utils.global_whole import (
    WHOLE_CONDITIONS,
    WHOLE_CONDITIONS_BY_NAME,
    TruncatedGenerationError,
    WholeCondition,
    assert_not_truncated,
    find_truncated_records,
    invalidate_whole_condition,
    is_truncated,
    assemble_whole_system_prompt,
    beeswarm_colour_direction,
    build_whole_json_all_payload,
    build_whole_json_beeswarm_payload,
    build_whole_record,
    run_resumable_whole_generation,
    split_whole_model_record,
    whole_generation_filename,
    whole_plot_paths,
    write_split_records,
)

PLOTS = EXPLANATIONS_DIR / "plots" / "global"
FEATURES = list_global_features("ebm", explanations_dir=EXPLANATIONS_DIR)


# -----------------------------------------------------------------------------
# Condition registry
# -----------------------------------------------------------------------------

def test_five_conditions_cover_both_axes():
    names = {c.name for c in WHOLE_CONDITIONS}
    assert names == {"json_all", "json_beeswarm", "vision_all",
                     "vision_beeswarm", "tooluse_all"}
    # Axis 2: exactly one pull condition, the rest push.
    assert [c.name for c in WHOLE_CONDITIONS if c.mechanism == "pull"] == ["tooluse_all"]
    # every "beeswarm" condition is a representation control, every "all" is full info.
    for c in WHOLE_CONDITIONS:
        assert c.representation in {"all", "beeswarm"}
        assert WHOLE_CONDITIONS_BY_NAME[c.name] is c


# -----------------------------------------------------------------------------
# Payloads: full "all" vs info-matched beeswarm
# -----------------------------------------------------------------------------

def test_json_all_payload_has_every_feature_curve():
    p = build_whole_json_all_payload("ebm", explanations_dir=EXPLANATIONS_DIR)
    assert p["n_features"] == len(FEATURES)
    got = {e["feature"] for e in p["features"]}
    assert got == set(FEATURES)
    for e in p["features"]:
        assert e["curve"]["x"] and e["curve"]["y"]
        assert len(e["curve"]["x"]) == len(e["curve"]["y"])
        assert "rank" in e and "importance" in e


def test_json_all_aggregates_xgb_scatter_to_grid():
    # XGB curve JSON is raw per-instance SHAP scatter (~12k points/feature); the whole-
    # model payload must aggregate it or the 9-feature prompt exceeds the context window.
    p = build_whole_json_all_payload("xgb", explanations_dir=EXPLANATIONS_DIR)
    total_points = sum(len(e["curve"]["x"]) for e in p["features"])
    assert total_points < 500, f"xgb json_all not aggregated ({total_points} points)"
    for e in p["features"]:
        assert len(e["curve"]["x"]) == len(e["curve"]["y"])


def test_json_beeswarm_is_info_matched_no_curve():
    p = build_whole_json_beeswarm_payload("ebm", explanations_dir=EXPLANATIONS_DIR)
    assert p["n_features"] == len(FEATURES)
    for e in p["features"]:
        # only rank + colour direction + coarse spread, deliberately no curve/peak/shape
        assert set(e.keys()) == {"feature", "rank", "colour_direction", "spread"}
        assert e["spread"] in {"narrow", "moderate", "wide"}
    # ranks form the full 1..9 set (feature ordering, what the swarm shows)
    assert sorted(e["rank"] for e in p["features"]) == list(range(1, len(FEATURES) + 1))


def test_beeswarm_direction_matches_curve_shape():
    # hr is categorical -> no ordinal colour trend
    assert "category" in beeswarm_colour_direction("ebm", "hr", explanations_dir=EXPLANATIONS_DIR)
    # temp is non-monotonic (rise then saturate) -> no clean colour separation
    assert "mixed" in beeswarm_colour_direction("ebm", "temp", explanations_dir=EXPLANATIONS_DIR)
    # yr is monotone rising -> higher raises demand; holiday is monotone falling
    assert beeswarm_colour_direction("ebm", "yr", explanations_dir=EXPLANATIONS_DIR) == \
        "higher values raise demand"
    assert beeswarm_colour_direction("ebm", "holiday", explanations_dir=EXPLANATIONS_DIR) == \
        "higher values lower demand"


# -----------------------------------------------------------------------------
# Vision plot paths
# -----------------------------------------------------------------------------

def test_plot_paths_all_vs_beeswarm():
    all_paths = whole_plot_paths("xgb", "all", plots_dir=PLOTS,
                                 explanations_dir=EXPLANATIONS_DIR)
    assert len(all_paths) == len(FEATURES)
    assert all(p.exists() for p in all_paths)
    swarm = whole_plot_paths("xgb", "beeswarm", plots_dir=PLOTS,
                             explanations_dir=EXPLANATIONS_DIR)
    assert len(swarm) == 1 and swarm[0].exists() and "beeswarm" in swarm[0].name


def test_plot_paths_reject_unknown_representation():
    with pytest.raises(ValueError):
        whole_plot_paths("xgb", "nope", plots_dir=PLOTS, explanations_dir=EXPLANATIONS_DIR)


# -----------------------------------------------------------------------------
# Prompt assembly
# -----------------------------------------------------------------------------

def test_prompt_core_identical_across_conditions_only_handover_differs():
    def core_of(name):
        s = assemble_whole_system_prompt(name, "ebm", prompts_dir=PROMPTS_DIR,
                                         features=FEATURES)
        # everything up to the handover section is the shared core
        return s.split("## HOW YOU RECEIVE THE INFORMATION")[0]
    cores = {core_of(c.name) for c in WHOLE_CONDITIONS}
    assert len(cores) == 1  # byte-identical core -> comparison measures the condition


def test_prompt_fills_placeholders_and_feature_list():
    s = assemble_whole_system_prompt("vision_all", "xgb", prompts_dir=PROMPTS_DIR,
                                     features=FEATURES)
    assert "{{" not in s and "}}" not in s
    assert "XGBoost" in s and "EBM" not in s.split("FEATURE SCHEMA")[0]
    for f in FEATURES:
        assert f in s  # the forced schema names every feature


def test_prompt_rejects_unknown_condition():
    with pytest.raises(ValueError):
        assemble_whole_system_prompt("bogus", "ebm", prompts_dir=PROMPTS_DIR,
                                     features=FEATURES)


# -----------------------------------------------------------------------------
# The forced per-feature schema splitter
# -----------------------------------------------------------------------------

def _whole_record(explanation, condition="vision_all"):
    c = WHOLE_CONDITIONS_BY_NAME[condition]
    return build_whole_record(
        condition=c, model_name="ebm", explanation=explanation,
        usage={"input_tokens": 1, "output_tokens": 1}, llm_model="stub")


def test_split_parses_blocks_and_shares_recommendation():
    text = (
        "<analysis>x</analysis>\n"
        "[FEATURE: hr]\n[EFFECT] Commuter peaks morning and evening.\n[IMPORTANCE] Rank 1 of 9.\n"
        "[FEATURE: temp]\n[EFFECT] Rises then saturates.\n[IMPORTANCE] Rank 2 of 9.\n"
        "[RECOMMENDATION] Staff up at rush hour on warm days."
    )
    rec = _whole_record(text)
    parts = split_whole_model_record(rec, features=["hr", "temp"])
    assert [p["feature"] for p in parts] == ["hr", "temp"]
    for p in parts:
        assert not p["dropped"]
        assert "[EFFECT]" in p["explanation"] and "[IMPORTANCE]" in p["explanation"]
        # the single whole-model recommendation is shared into each feature record
        assert "Staff up at rush hour" in p["explanation"]
        assert p["form"] == "vision_all" and p["xai_model"] == "ebm"


@pytest.mark.parametrize("header", [
    "[FEATURE: hr]",                 # strict schema
    "**[FEATURE: hr]**",             # markdown emphasis around the bracket
    "[FEATURE hr]",                  # missing colon
    "[FEATURE: hr (hour of day)]",   # parenthetical after the name
    "[ FEATURE : hr ]",              # extra spacing
    "### hr",                        # heading style instead of the bracket
    "## hr - hour of day",           # heading with a trailing description
    "### Feature: hr",               # heading carrying the FEATURE marker word
])
def test_split_tolerates_header_deviations(header):
    # P3: cosmetic header deviations must NOT be mis-scored as a dropped feature
    # (the coverage metric would otherwise be biased downward on the real run).
    text = (f"{header}\n[EFFECT] Peaks at rush hour.\n[IMPORTANCE] rank 1 of 9.\n"
            "[RECOMMENDATION] plan around commute.")
    parts = split_whole_model_record(_whole_record(text), features=["hr"])
    assert parts[0]["dropped"] is False, f"header {header!r} wrongly dropped"
    assert "[EFFECT]" in parts[0]["explanation"]


def test_split_heading_only_answer_covers_all():
    # an answer that uses headings throughout (no [FEATURE] brackets at all)
    text = ("### hr\n[EFFECT] commuter peaks.\n[IMPORTANCE] rank 1.\n"
            "### temp\n[EFFECT] rises then saturates.\n[IMPORTANCE] rank 2.\n"
            "## Recommendation\nStaff up at rush hour on warm days.")
    parts = split_whole_model_record(_whole_record(text), features=["hr", "temp"])
    assert [p["feature"] for p in parts] == ["hr", "temp"]
    assert all(not p["dropped"] for p in parts)
    # the heading-style recommendation is still shared into each feature block
    assert all("Staff up at rush hour" in p["explanation"] for p in parts)


def test_split_still_detects_genuinely_missing_feature():
    # tolerance must not mask a real omission: hum is never named -> dropped
    text = ("[FEATURE: hr (hour)]\n[EFFECT] peaks at commute.\n[IMPORTANCE] rank 1.\n"
            "[RECOMMENDATION] plan.")
    parts = split_whole_model_record(_whole_record(text), features=["hr", "hum"])
    by_f = {p["feature"]: p for p in parts}
    assert by_f["hr"]["dropped"] is False
    assert by_f["hum"]["dropped"] is True


def test_split_marks_dropped_features():
    text = ("[FEATURE: hr]\n[EFFECT] Peaks at rush hour.\n[IMPORTANCE] Rank 1.\n"
            "[RECOMMENDATION] Plan around commute.")
    rec = _whole_record(text)
    parts = split_whole_model_record(rec, features=["hr", "temp", "hum"])
    by_f = {p["feature"]: p for p in parts}
    assert by_f["hr"]["dropped"] is False and by_f["hr"]["explanation"]
    # temp/hum were never mentioned -> dropped, empty explanation (scored as a miss)
    assert by_f["temp"]["dropped"] is True and by_f["temp"]["explanation"] == ""
    assert by_f["hum"]["dropped"] is True


def test_split_record_roundtrips_through_rubric():
    # a well-formed hr block should score well on the deterministic rubric (rank 1 hit,
    # both commuter peaks captured), proving the split output is G3-scoreable unchanged.
    text = (
        "[FEATURE: hr]\n[EFFECT] Demand is highest at the morning commute around 8 and "
        "the evening peak around 17-18, and lowest overnight.\n[IMPORTANCE] The most "
        "important feature, rank 1 of 9.\n[RECOMMENDATION] Concentrate bikes at commute times."
    )
    rec = _whole_record(text, condition="vision_all")
    part = next(p for p in split_whole_model_record(rec, features=["hr"]))
    gt = rubric.load_ground_truth("ebm", "hr")
    score = rubric.rubric_score(part["explanation"], gt)
    assert score["rank"] == 1.0          # "rank 1 of 9" parsed and correct
    assert score["structure"] == 1.0     # both commuter peaks -> full structure
    assert score["total"] > 0.9


# -----------------------------------------------------------------------------
# write_split_records + resumable loop
# -----------------------------------------------------------------------------

def test_write_split_records_filenames(tmp_path):
    rec = _whole_record(
        "[FEATURE: hr]\n[EFFECT] e\n[IMPORTANCE] Rank 1.\n[RECOMMENDATION] r",
        condition="json_beeswarm")
    paths = write_split_records([rec], features=["hr", "temp"], split_dir=tmp_path)
    names = sorted(p.name for p in paths)
    assert names == ["json_beeswarm_ebm_hr.json", "json_beeswarm_ebm_temp.json"]
    loaded = json.loads((tmp_path / "json_beeswarm_ebm_hr.json").read_text())
    assert loaded["feature"] == "hr" and loaded["condition"] == "json_beeswarm"


def test_resumable_whole_loop_skips_existing(tmp_path):
    calls = []

    def gen(model, cond):
        calls.append((model, cond.name))
        return _whole_record("[FEATURE: hr]\n[EFFECT] e\n[IMPORTANCE] i\n[RECOMMENDATION] r",
                             condition=cond.name)

    conds = [WHOLE_CONDITIONS_BY_NAME["json_all"]]
    run_resumable_whole_generation(model_names=["ebm"], conditions=conds,
                                   out_dir=tmp_path, generate=gen)
    assert (tmp_path / whole_generation_filename("json_all", "ebm")).exists()
    # second run: file exists -> no new generate call (resume)
    run_resumable_whole_generation(model_names=["ebm"], conditions=conds,
                                   out_dir=tmp_path, generate=gen)
    assert len(calls) == 1


def test_generate_returning_none_leaves_unit_open(tmp_path):
    conds = [WHOLE_CONDITIONS_BY_NAME["json_all"]]
    run_resumable_whole_generation(model_names=["ebm"], conditions=conds,
                                   out_dir=tmp_path, generate=lambda m, c: None)
    assert not (tmp_path / whole_generation_filename("json_all", "ebm")).exists()


# -----------------------------------------------------------------------------
# Truncation guards (P0-1): a token-ceiling hit must never pass as a modality effect
# -----------------------------------------------------------------------------

def _rec(*, stop_reason=None, max_tokens=None, out_tokens=100, condition="vision_all"):
    return build_whole_record(
        condition=WHOLE_CONDITIONS_BY_NAME[condition], model_name="xgb",
        explanation="[FEATURE: hr]\n[EFFECT] e\n[IMPORTANCE] i\n[RECOMMENDATION] r",
        usage={"input_tokens": 10, "output_tokens": out_tokens}, llm_model="stub",
        stop_reason=stop_reason, max_tokens=max_tokens)


def test_record_carries_stop_reason_and_max_tokens_on_every_path():
    # The pre-fix bug: only the tool-use path persisted stop_reason, so a truncated
    # json/vision answer was indistinguishable from a short one.
    rec = _rec(stop_reason="end_turn", max_tokens=16384)
    assert rec["stop_reason"] == "end_turn" and rec["max_tokens"] == 16384
    # both keys exist even when unknown, so the schema is uniform across modalities
    bare = _rec()
    assert bare["stop_reason"] is None and bare["max_tokens"] is None


def test_extra_stop_reason_still_wins_for_tooluse_call_sites():
    rec = build_whole_record(
        condition=WHOLE_CONDITIONS_BY_NAME["tooluse_all"], model_name="xgb",
        explanation="x", usage={"input_tokens": 1, "output_tokens": 1},
        llm_model="stub", extra={"stop_reason": "end_turn", "n_tool_calls": 3})
    assert rec["stop_reason"] == "end_turn" and rec["n_tool_calls"] == 3


@pytest.mark.parametrize("kwargs", [
    {"stop_reason": "max_tokens", "max_tokens": 16384, "out_tokens": 16384},
    {"stop_reason": None, "max_tokens": 4096, "out_tokens": 4096},   # arithmetic fallback
])
def test_is_truncated_detects_ceiling_hits(kwargs):
    assert is_truncated(_rec(**kwargs))
    with pytest.raises(TruncatedGenerationError):
        assert_not_truncated(_rec(**kwargs))


def test_complete_answer_passes_the_gate():
    rec = _rec(stop_reason="end_turn", max_tokens=16384, out_tokens=2578)
    assert not is_truncated(rec)
    assert assert_not_truncated(rec) is rec


def test_legacy_record_needs_an_assumed_ceiling():
    # A pre-P0-1 record carries neither field; without the caller supplying the old
    # ceiling it cannot self-report, which is exactly how the artefact stayed invisible.
    legacy = {"condition": "vision_beeswarm", "xai_model": "xgb",
              "usage": {"input_tokens": 1, "output_tokens": 4096}}
    assert not is_truncated(legacy)
    assert is_truncated(legacy, assume_max_tokens=4096)


def test_find_truncated_records_scans_a_directory(tmp_path):
    (tmp_path / "ok.json").write_text(json.dumps(
        _rec(stop_reason="end_turn", max_tokens=16384, out_tokens=500)))
    (tmp_path / "cut.json").write_text(json.dumps(
        _rec(stop_reason="max_tokens", max_tokens=16384, out_tokens=16384)))
    (tmp_path / "legacy.json").write_text(json.dumps(
        {"condition": "vision_all", "xai_model": "xgb",
         "usage": {"input_tokens": 1, "output_tokens": 4096}}))
    assert [p.name for p in find_truncated_records(tmp_path)] == ["cut.json"]
    assert [p.name for p in find_truncated_records(tmp_path, assume_max_tokens=4096)] == \
        ["cut.json", "legacy.json"]


def test_truncated_record_is_never_persisted_by_the_resumable_loop(tmp_path):
    # The notebook gate: generate_api raises, the loop stores nothing for that condition.
    def gen(model, cond):
        return assert_not_truncated(
            _rec(stop_reason="max_tokens", max_tokens=16384, out_tokens=16384,
                 condition=cond.name))

    conds = [WHOLE_CONDITIONS_BY_NAME["vision_all"]]
    with pytest.raises(TruncatedGenerationError):
        run_resumable_whole_generation(model_names=["xgb"], conditions=conds,
                                       out_dir=tmp_path, generate=gen)
    assert not (tmp_path / whole_generation_filename("vision_all", "xgb")).exists()


def test_invalidate_clears_the_whole_downstream_chain(tmp_path):
    # run_global_judge is idempotent, so a stale judge verdict for a regenerated answer
    # would silently survive. Invalidation must reach every downstream directory.
    layout = {
        "global_whole": ["vision_all_xgb.json", "json_all_xgb.json"],
        "global_whole_split": ["vision_all_xgb_hr.json", "vision_all_xgb_temp.json",
                               "json_all_xgb_hr.json", "vision_all_ebm_hr.json"],
        "global_whole_judge": ["vision_all_xgb_hr.json", "json_all_xgb_hr.json"],
        "global_whole_judge_openai": ["vision_all_xgb_hr.json"],
    }
    for sub, names in layout.items():
        (tmp_path / sub).mkdir()
        for n in names:
            (tmp_path / sub / n).write_text("{}")

    deleted = invalidate_whole_condition("vision_all", "xgb", results_dir=tmp_path)
    assert {p.name for p in deleted} == {"vision_all_xgb.json", "vision_all_xgb_hr.json",
                                         "vision_all_xgb_temp.json"}
    assert len(deleted) == 5   # 1 raw + 2 split + 1 judge + 1 judge_openai
    # a different condition and a different xai_model are untouched
    assert (tmp_path / "global_whole" / "json_all_xgb.json").exists()
    assert (tmp_path / "global_whole_split" / "json_all_xgb_hr.json").exists()
    assert (tmp_path / "global_whole_split" / "vision_all_ebm_hr.json").exists()


def test_invalidate_is_safe_on_a_condition_that_was_never_run(tmp_path):
    for sub in ("global_whole", "global_whole_split", "global_whole_judge",
                "global_whole_judge_openai"):
        (tmp_path / sub).mkdir()
    assert invalidate_whole_condition("tooluse_all", "ebm", results_dir=tmp_path) == []
