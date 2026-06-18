"""Phase 3b — utils.faithfulness (gen-aware RA/SA/VA + Extraktionsvalidität).

Sichert den Skalierungs-Faithfulness-Pfad ab:
  * `compute_faithfulness` ist ein **treuer Port** aus NB 08 (ϕ-Ausschluss,
    Out-of-Top-K-Skip, Vorzeichen, Wert-Match mit Denormalisierung, None bei n=0).
  * `parse_extraction` ist robust gegen Fließtext-Umrandung und kaputtes JSON.
  * `extraction_coverage` liefert die Validitäts-Proxys (parse_empty, Top-K-Recall,
    Out-of-Top-K, Rang-0-Treffer) — der Extraktor ist fehleranfällig (NB 09).
  * `build_faithfulness_df` ist gen-aware (custom_id-Lookup) und behandelt fehlende
    Extraktionen als parse_empty.
  * `extraction_validity_summary` aggregiert je Pipeline korrekt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.faithfulness import (
    build_faithfulness_df,
    compute_faithfulness,
    extraction_base_cid,
    extraction_coverage,
    extraction_validity_summary,
    is_value_match,
    parse_extraction,
)

# Top-4 Ground-Truth (Beiträge im Log-Raum): hr(+), yr(−), hum(+), temp(+).
GT = [
    {"feature": "hr",   "contribution": 1.23,  "value": 8},
    {"feature": "yr",   "contribution": -0.226, "value": 0},
    {"feature": "hum",  "contribution": 0.05,  "value": 0.88},
    {"feature": "temp", "contribution": 0.02,  "value": 0.50},
]


# ---------------------------------------------------------------------------
# parse_extraction
# ---------------------------------------------------------------------------

def test_parse_extraction_plain_json():
    assert parse_extraction('{"hr": {"rank": 0}}') == {"hr": {"rank": 0}}


def test_parse_extraction_embedded_in_text():
    raw = 'Hier ist das Ergebnis:\n{"hr": {"rank": 0}}\nFertig.'
    assert parse_extraction(raw) == {"hr": {"rank": 0}}


def test_parse_extraction_invalid_returns_empty():
    assert parse_extraction("kein json hier") == {}
    assert parse_extraction('{"hr": kaputt}') == {}


# ---------------------------------------------------------------------------
# is_value_match
# ---------------------------------------------------------------------------

def test_value_match_exact_and_tolerance():
    assert is_value_match("hr", 8.0, 8.0)
    assert is_value_match("hr", 8.5, 8.0, tol=1.0)
    assert not is_value_match("hr", 12.0, 8.0, tol=1.0)


def test_value_match_denormalized_temperature():
    # GT temp normalisiert 0.50 → ~20.5 °C; LLM zitiert oft die °C-Form.
    assert is_value_match("temp", 20.5, 0.50)
    # normalisierte Form passt ebenfalls
    assert is_value_match("temp", 0.50, 0.50)


# ---------------------------------------------------------------------------
# compute_faithfulness — treuer Port
# ---------------------------------------------------------------------------

def test_compute_faithfulness_all_correct():
    ext = {
        "hr":   {"rank": 0, "sign":  1, "value": 8},
        "yr":   {"rank": 1, "sign": -1, "value": None},
        "hum":  {"rank": 2, "sign":  1, "value": 0.88},
    }
    m = compute_faithfulness(ext, GT)
    assert m["RA"] == 1.0 and m["RA_n"] == 3
    assert m["SA"] == 1.0 and m["SA_n"] == 3
    # value: hr=8 ✓, hum=0.88 ✓ (yr value None → nicht gewertet)
    assert m["VA"] == 1.0 and m["VA_n"] == 2
    assert m["n_extracted"] == 3


def test_compute_faithfulness_out_of_topk_skipped():
    # 'windspeed' ist nicht unter Top-4 → komplett übersprungen (Nenner unberührt).
    ext = {
        "hr":        {"rank": 0, "sign": 1, "value": None},
        "windspeed": {"rank": 1, "sign": 1, "value": 5},
    }
    m = compute_faithfulness(ext, GT)
    assert m["RA_n"] == 1 and m["RA"] == 1.0   # nur hr gewertet
    assert m["VA_n"] == 0 and m["VA"] is None  # hr value None, windspeed übersprungen


def test_compute_faithfulness_sign_and_rank_errors():
    ext = {
        "hr": {"rank": 1, "sign": -1, "value": None},   # falscher Rang + falsches Vorzeichen
        "yr": {"rank": 1, "sign": -1, "value": None},   # korrekt
    }
    m = compute_faithfulness(ext, GT)
    assert m["RA_hits"] == 1 and m["RA_n"] == 2 and m["RA"] == 0.5
    assert m["SA_hits"] == 1 and m["SA_n"] == 2 and m["SA"] == 0.5


def test_compute_faithfulness_empty_extraction_gives_none():
    m = compute_faithfulness({}, GT)
    assert m["RA"] is None and m["SA"] is None and m["VA"] is None
    assert m["n_extracted"] == 0


# ---------------------------------------------------------------------------
# extraction_coverage — Validitäts-Proxys
# ---------------------------------------------------------------------------

def test_coverage_recall_and_out_of_topk():
    ext = {
        "hr":        {"rank": 0, "sign": 1, "value": 8},
        "yr":        {"rank": 1, "sign": -1, "value": None},
        "windspeed": {"rank": 2, "sign": 1, "value": None},  # nicht Top-K → Rauschen
    }
    cov = extraction_coverage(ext, GT)
    assert cov["parse_empty"] is False
    assert cov["n_in_topk"] == 2 and cov["n_out_of_topk"] == 1
    assert cov["topk_total"] == 4 and cov["topk_covered"] == 2
    assert cov["topk_recall"] == 0.5


def test_coverage_r0_match_and_mismatch():
    hit = extraction_coverage({"hr": {"rank": 0}}, GT)
    assert hit["gt_r0_feature"] == "hr" and hit["ext_r0_feature"] == "hr"
    assert hit["r0_match"] == 1
    miss = extraction_coverage({"yr": {"rank": 0}}, GT)
    assert miss["ext_r0_feature"] == "yr" and miss["r0_match"] == 0
    none = extraction_coverage({"hr": {"rank": 2}}, GT)  # kein Rang-0 extrahiert
    assert none["ext_r0_feature"] is None and none["r0_match"] is None


def test_coverage_empty_extraction():
    cov = extraction_coverage({}, GT)
    assert cov["parse_empty"] is True
    assert cov["topk_covered"] == 0 and cov["topk_recall"] == 0.0
    assert cov["r0_match"] is None


# ---------------------------------------------------------------------------
# build_faithfulness_df — gen-aware custom_id lookup
# ---------------------------------------------------------------------------

def _write_local_explanation(expl_dir: Path, xai: str, iid: int,
                             loss_key: str = "poisson_log"):
    expl_dir.mkdir(parents=True, exist_ok=True)
    data = {"model": xai, "instance_id": iid, "prediction": 390.0, "y_true": 387,
            "feature_values": {"hr": 8, "yr": 0, "hum": 0.88, "temp": 0.50},
            "contributions": GT}
    (expl_dir / f"local_{xai.lower()}_{loss_key}_inst{iid}.json").write_text(
        json.dumps(data, ensure_ascii=False)
    )


def _scale_df():
    rows = []
    for xai in ["XGB", "EBM"]:
        for gen in range(3):
            rows.append({"pipeline": "04", "pipeline_label": "JSON→Text",
                         "xai_model": xai, "instance_id": 101, "generation": gen,
                         "explanation": "x"})
    return pd.DataFrame(rows)


def test_build_faithfulness_df_gen_aware(tmp_path):
    _write_local_explanation(tmp_path, "xgb", 101)
    _write_local_explanation(tmp_path, "ebm", 101)
    df = _scale_df()

    # Extraktionen nur für XGB-gen0 vorhanden → restliche cids fehlen (parse_empty).
    cid = extraction_base_cid("ext", "04", "XGB", 101, 0)
    by_cid = {cid: {"hr": {"rank": 0, "sign": 1, "value": 8}}}

    out = build_faithfulness_df(df, by_cid, explanations_dir=tmp_path)
    assert len(out) == 6
    assert set(out["generation"]) == {0, 1, 2}

    hit = out[(out.xai_model == "XGB") & (out.generation == 0)].iloc[0]
    assert hit["RA"] == 1.0 and not hit["parse_empty"] and hit["r0_match"] == 1

    # alle anderen cids fehlen → leere Extraktion
    assert out["parse_empty"].sum() == 5


# ---------------------------------------------------------------------------
# extraction_validity_summary
# ---------------------------------------------------------------------------

def test_validity_summary_aggregates_per_pipeline(tmp_path):
    _write_local_explanation(tmp_path, "xgb", 101)
    _write_local_explanation(tmp_path, "ebm", 101)
    df = _scale_df()
    cid = extraction_base_cid("ext", "04", "XGB", 101, 0)
    by_cid = {cid: {"hr": {"rank": 0, "sign": 1, "value": 8},
                    "windspeed": {"rank": 1, "sign": 1, "value": None}}}
    faith = build_faithfulness_df(df, by_cid, explanations_dir=tmp_path)
    summ = extraction_validity_summary(faith)

    assert "JSON→Text" in summ.index
    row = summ.loc["JSON→Text"]
    assert row["n_narratives"] == 6
    # 5 von 6 Narrativen leer → Parse-Ausfallrate 5/6 (auf 4 Stellen gerundet)
    assert abs(row["parse_empty_rate"] - 5 / 6) < 1e-3
    # Out-of-Top-K-Rate: 1 von (1 in-topK + 1 out) = 0.5
    assert abs(row["out_of_topk_rate"] - 0.5) < 1e-6
    # nur das eine erfolgreiche Narrativ hat r0_match=1
    assert row["r0_match_rate"] == 1.0
