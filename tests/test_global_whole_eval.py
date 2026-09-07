"""Tests for the whole-model (G2b) eval helpers in utils.global_eval.

No API calls: builds whole-model records, writes split records with the real writer,
and checks load_whole_rubric (axis attachment, dropped -> total miss, beeswarm
fair_total) plus the coverage / axis-1 / axis-2 / beeswarm-readability tables, including
the disaggregated axis-2 view that keeps a pooled push mean from hiding a weak condition.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import global_eval
from utils.global_whole import (
    WHOLE_CONDITIONS_BY_NAME,
    build_whole_record,
    write_split_records,
)

FEATS = ["hr", "yr", "hum"]


def _rec(condition, explanation):
    return build_whole_record(
        condition=WHOLE_CONDITIONS_BY_NAME[condition], model_name="ebm",
        explanation=explanation, usage={}, llm_model="stub")


def _make_split_dir(tmp_path):
    r_all = _rec("json_all",
        "[FEATURE: hr]\n[EFFECT] Demand is highest at the morning and evening commute.\n"
        "[IMPORTANCE] rank 1 of 9.\n"
        "[FEATURE: yr]\n[EFFECT] higher values raise demand.\n[IMPORTANCE] rank 3 of 9.\n"
        "[FEATURE: hum]\n[EFFECT] higher values lower demand.\n[IMPORTANCE] rank 5 of 9.\n"
        "[RECOMMENDATION] plan around commute.")
    r_bee = _rec("vision_beeswarm",   # hum omitted -> dropped
        "[FEATURE: hr]\n[EFFECT] Demand is highest at the morning and evening commute.\n"
        "[IMPORTANCE] rank 1 of 9.\n"
        "[FEATURE: yr]\n[EFFECT] higher values raise demand.\n[IMPORTANCE] rank 3 of 9.\n"
        "[RECOMMENDATION] plan.")
    write_split_records([r_all, r_bee], features=FEATS,
                        split_dir=tmp_path / "global_whole_split")


def test_load_whole_rubric_attaches_axes_and_zeroes_drops(tmp_path):
    _make_split_dir(tmp_path)
    df = global_eval.load_whole_rubric(results_dir=tmp_path)
    assert set(df["condition"]) == {"json_all", "vision_beeswarm"}

    # dropped feature -> honest total miss (all sub-scores 0)
    drop = df[(df.condition == "vision_beeswarm") & (df.feature == "hum")].iloc[0]
    assert drop["dropped"] and drop["total"] == 0.0 and drop["direction"] == 0.0

    # beeswarm fair_total scores only direction+rank; 'all' fair_total == full total
    bee_hr = df[(df.condition == "vision_beeswarm") & (df.feature == "hr")].iloc[0]
    assert bee_hr["fair_total"] == round((bee_hr["direction"] + bee_hr["rank"]) / 2, 4)
    all_hr = df[(df.condition == "json_all") & (df.feature == "hr")].iloc[0]
    assert all_hr["fair_total"] == all_hr["total"]


def test_whole_coverage_counts(tmp_path):
    _make_split_dir(tmp_path)
    df = global_eval.load_whole_rubric(results_dir=tmp_path)
    cov = global_eval.whole_coverage(df)
    assert cov.loc["json_all", "ebm"] == 3          # all 3 features described
    assert cov.loc["vision_beeswarm", "ebm"] == 2   # hum dropped


def test_axis_tables_run(tmp_path):
    _make_split_dir(tmp_path)
    df = global_eval.load_whole_rubric(results_dir=tmp_path)
    a1 = global_eval.axis1_representation(df)
    assert {"direction", "rank", "structure", "fair_total"}.issubset(a1.columns)
    # axis-2 restricts to 'all' conditions only (constant full information)
    a2 = global_eval.axis2_mechanism(df)
    assert "ebm" in a2.columns
    bee = global_eval.beeswarm_readability(df)
    assert "vision_beeswarm" in bee.index


def test_load_whole_rubric_none_when_absent(tmp_path):
    assert global_eval.load_whole_rubric(results_dir=tmp_path) is None


def test_axis2_pairwise_exposes_a_push_condition_that_beats_pull():
    """The pooled push mean can hide that one push condition actually wins.

    Synthetic: pull sits between a strong and a weak push condition, so the pooled mean
    says "pull > push" while the strong push condition in fact beats it.
    """
    import pandas as pd

    df = pd.DataFrame([
        {"condition": "tooluse_all", "representation": "all", "mechanism": "pull",
         "xai_model": "ebm", "fair_total": 0.90},
        {"condition": "json_all", "representation": "all", "mechanism": "push",
         "xai_model": "ebm", "fair_total": 0.96},
        {"condition": "vision_all", "representation": "all", "mechanism": "push",
         "xai_model": "ebm", "fair_total": 0.60},
        # a beeswarm row must be ignored: it belongs to axis 1, not here
        {"condition": "json_beeswarm", "representation": "beeswarm", "mechanism": "push",
         "xai_model": "ebm", "fair_total": 0.10},
    ])

    pooled = global_eval.axis2_mechanism(df, value="fair_total")
    assert pooled.loc["pull", "ebm"] > pooled.loc["push", "ebm"]   # the misleading view

    pw = global_eval.axis2_mechanism_pairwise(df, value="fair_total")
    assert set(pw["push_condition"]) == {"json_all", "vision_all"}   # beeswarm excluded
    json_row = pw[pw.push_condition == "json_all"].iloc[0]
    vision_row = pw[pw.push_condition == "vision_all"].iloc[0]
    assert json_row["delta_pull_minus_push"] < 0      # json_all actually beat pull
    assert vision_row["delta_pull_minus_push"] > 0


def test_axis2_pairwise_on_the_real_records():
    df = global_eval.load_whole_rubric()
    if df is None:
        import pytest
        pytest.skip("04Ge not run with RUN_API")
    pw = global_eval.axis2_mechanism_pairwise(df)
    assert list(pw.columns) == ["push_condition", "xai_model", "pull", "push",
                                "delta_pull_minus_push"]
    assert len(pw) == 4          # 2 push conditions x 2 xai models
    assert "tooluse_all" not in set(pw["push_condition"])
