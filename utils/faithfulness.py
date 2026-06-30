"""utils/faithfulness.py - scaling faithfulness following Ichmoukhamedov et al. (2024).

Ground truth anchored faithfulness metrics (RA/SA/VA, Eq. 1 in the paper) for a
larger run. Faithful port of the logic developed on n=20 in
`06_Evaluation_Ichmoukhamedov`: identical extraction prompt, identical parser and a
byte for byte identical `compute_faithfulness` (reproducing the n=20 numbers),
extended with:

  * generation aware processing (N generations per instance, own `custom_id`s),
  * a batch extraction path (extraction is a single step call, so fully batchable,
    about 50 percent cheaper, like the judge), and
  * extraction validity (`extraction_coverage` / `extraction_validity_summary`):
    automatically computable proxies for how reliable the measurement instrument
    (the extractor) is, because the error taxonomy (NB 07) showed 19/30 extractor
    artefacts and a rank 0 extractor accuracy of only 55 percent. RA/SA/VA measure
    precision, not recall (NB 06 section 4.1); the validity figures quantify exactly
    that limitation.

Deliberately separate from the n=20 notebook (NB 06 stays untouched); `08b_Scaling_
Faithfulness` uses these helpers on the scaling artefacts under
``results/pipeline0X/scale/``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from utils import EXPLANATIONS_DIR, RESULTS_DIR
from utils.batch import make_custom_id
from utils.explanations import FEATURE_SCHEMA, HUM_FACTOR, TEMP_FACTOR, WIND_FACTOR

LOSS_KEY_DEFAULT = "poisson_log"
TOP_K_DEFAULT = 4  # paper: top 4 features by absolute contribution

# Extraction prompt + system (faithful port from NB 06 cell 5).
EXTRACTION_SYSTEM = (
    "You are an extraction model for XAI narratives of a bike rental model.\n"
    "Extract the requested information only from the narrative.\n"
    "Answer only with a valid JSON object, no text, no markdown blocks."
)

# Denormalisation: normalised feature values to human readable units.
# Factors come from utils.explanations (single source), so the C/%/km-h
# conversion does not diverge from the generation or judge prompt.
_DENORM = {
    "temp":      lambda v: v * TEMP_FACTOR,
    "hum":       lambda v: v * HUM_FACTOR,
    "windspeed": lambda v: v * WIND_FACTOR,
}


def build_extraction_prompt(explanation: str, xai_model: str) -> str:
    """Build the extraction user prompt (JSON), faithful port from NB 06."""
    feat_descs = {f: FEATURE_SCHEMA[f]["description"] for f in FEATURE_SCHEMA}
    payload = {
        "task": (
            f"Extract structured information from the following English narrative "
            f"about a {xai_model} regression model for a bike rental company. "
            f"The model predicts hourly bike rentals. "
            f"Positive contributions raise the prediction, negative ones lower it."
        ),
        "narrative": explanation,
        "all_features": list(FEATURE_SCHEMA.keys()),
        "feature_descriptions": feat_descs,
        "extraction_instruction": (
            "For each feature mentioned as important in the narrative, return an object:\n"
            "  rank: 0 based importance rank according to the narrative (0 = most important feature)\n"
            "  sign: +1 if the feature raises the prediction, -1 if it lowers it\n"
            "  value: numeric feature value if explicitly named in the narrative, else null\n"
            "  assumption: a single sentence of background knowledge why the feature has this effect; "
            '"None" if no background knowledge was added'
        ),
        "output_format_example": {
            "hr":   {"rank": 0, "sign":  1, "value": 13,   "assumption": "Midday is typically high demand."},
            "temp": {"rank": 1, "sign": -1, "value": None, "assumption": "Cold deters cyclists."},
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_extraction(raw: str) -> dict:
    """Extract the JSON object from the extractor answer (faithful port, NB 06)."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {}


def is_value_match(feat: str, extracted: float, gt: float, tol: float = 1.0) -> bool:
    """Value comparison with tolerance; also checks denormalised units (faithful port)."""
    if abs(extracted - gt) <= tol:
        return True
    if feat in _DENORM:
        dv = _DENORM[feat](gt)
        if abs(extracted - dv) <= tol:
            return True
    return False


def compute_faithfulness(extraction: dict, gt_contributions: list) -> dict:
    """RA, SA, VA following Eq. 1 from Ichmoukhamedov et al. (2024), faithful port from NB 06.

    phi = null/not extracted is removed from the denominator. Iterates over the
    extracted features (precision, not recall, see NB 06 section 4.1; recall and
    coverage are provided by :func:`extraction_coverage`).
    """
    gt_rank  = {c["feature"]: i for i, c in enumerate(gt_contributions)}
    gt_sign  = {c["feature"]: (1 if c["contribution"] >= 0 else -1)
                for c in gt_contributions}
    gt_value = {c["feature"]: c["value"] for c in gt_contributions}

    ra_hits, ra_n = 0, 0
    sa_hits, sa_n = 0, 0
    va_hits, va_n = 0, 0

    for feat, info in extraction.items():
        feat_key = feat.lower()
        if feat_key not in gt_rank:
            continue  # feature not in the top K, skip (as in the paper)

        r = info.get("rank")
        if r is not None:
            ra_n += 1
            try:
                if int(float(r)) == gt_rank[feat_key]:
                    ra_hits += 1
            except (ValueError, TypeError):
                pass

        s = info.get("sign")
        if s is not None:
            sa_n += 1
            try:
                if int(float(s)) == gt_sign[feat_key]:
                    sa_hits += 1
            except (ValueError, TypeError):
                pass

        v = info.get("value")
        if v is not None and str(v).lower() not in ("null", "none", ""):
            try:
                v_float = float(v)
                gt_v    = float(gt_value.get(feat_key, 0))
                va_n += 1
                if is_value_match(feat_key, v_float, gt_v):
                    va_hits += 1
            except (ValueError, TypeError):
                pass

    return {
        "RA": round(ra_hits / ra_n, 4) if ra_n > 0 else None,
        "SA": round(sa_hits / sa_n, 4) if sa_n > 0 else None,
        "VA": round(va_hits / va_n, 4) if va_n > 0 else None,
        "RA_hits": ra_hits, "RA_n": ra_n,
        "SA_hits": sa_hits, "SA_n": sa_n,
        "VA_hits": va_hits, "VA_n": va_n,
        "n_extracted": len(extraction),
    }


def extraction_coverage(extraction: dict, gt_contributions: list) -> dict:
    """Extraction validity per narrative, proxies for the reliability of the extractor.

    RA/SA/VA only score the mentioned features (precision). These figures make the
    recall and noise share visible, important because the extractor itself is error
    prone (NB 07: 19/30 extractor artefacts, rank 0 accuracy 55 percent):

      * ``parse_empty``    - the extractor produced no valid JSON (measurement failure).
      * ``topk_recall``    - share of the top K ground truth features the extractor
                             captured at all (counterpart to the precision of RA/SA/VA).
      * ``n_out_of_topk``  - extracted features that are not in the top K (silently
                             skipped in `compute_faithfulness`, a noise proxy).
      * ``r0_match``       - does the feature the extractor marked as rank 0 match the
                             SHAP rank 0? (1/0/None), directly related to the NB 07
                             figure (55 percent rank 0 accuracy).
    """
    gt_features = [c["feature"].lower() for c in gt_contributions]
    gt_set = set(gt_features)
    ext_keys = [str(k).lower() for k in extraction.keys()]

    in_topk = [k for k in ext_keys if k in gt_set]
    out_of_topk = [k for k in ext_keys if k not in gt_set]
    covered = gt_set & set(ext_keys)

    gt_r0 = gt_features[0] if gt_features else None
    ext_r0 = None
    for k, info in extraction.items():
        r = info.get("rank") if isinstance(info, dict) else None
        if r is None:
            continue
        try:
            if int(float(r)) == 0:
                ext_r0 = str(k).lower()
                break
        except (ValueError, TypeError):
            pass

    r0_match = None
    if ext_r0 is not None and gt_r0 is not None:
        r0_match = 1 if ext_r0 == gt_r0 else 0

    return {
        "parse_empty":   len(extraction) == 0,
        "n_in_topk":     len(in_topk),
        "n_out_of_topk": len(out_of_topk),
        "topk_total":    len(gt_set),
        "topk_covered":  len(covered),
        "topk_recall":   round(len(covered) / len(gt_set), 4) if gt_set else None,
        "ext_r0_feature": ext_r0,
        "gt_r0_feature":  gt_r0,
        "r0_match":       r0_match,
    }


def load_gt_contributions(
    xai_model: str,
    instance_id: int,
    *,
    top_k: int = TOP_K_DEFAULT,
    loss_key: str = LOSS_KEY_DEFAULT,
    explanations_dir: Path = EXPLANATIONS_DIR,
) -> list:
    """Top K SHAP ground truth contributions from ``local_{xai}_{loss}_inst{iid}.json``."""
    p = explanations_dir / f"local_{xai_model.lower()}_{loss_key}_inst{instance_id}.json"
    gt = json.loads(p.read_text())
    return gt["contributions"][:top_k]


def extraction_base_cid(prefix: str, pipeline: str, xai_model: str,
                        instance_id: int, generation: int) -> str:
    """Generation aware `custom_id` for an extraction (like the judge cids in NB 07b)."""
    return make_custom_id(prefix, pipeline, xai_model, instance_id, f"g{generation}")


def build_faithfulness_df(
    df: pd.DataFrame,
    extraction_by_cid: dict,
    *,
    prefix: str = "ext",
    top_k: int = TOP_K_DEFAULT,
    loss_key: str = LOSS_KEY_DEFAULT,
    explanations_dir: Path = EXPLANATIONS_DIR,
) -> pd.DataFrame:
    """Build the per narrative faithfulness table (RA/SA/VA + validity) generation aware.

    `df` are the generation aware generation records (from
    :func:`utils.eval.load_scale_records`); `extraction_by_cid` maps the extraction
    `custom_id` (see :func:`extraction_base_cid`) to the parsed extraction dict
    (`run_batch(...)['succeeded']`). Per row the ground truth (cached per (xai,
    instance)), `compute_faithfulness` and `extraction_coverage` are merged. Missing
    `custom_id`s mean an empty extraction (counts as `parse_empty`).
    """
    gt_cache: dict = {}
    rows = []
    for _, row in df.iterrows():
        xai = row["xai_model"]
        iid = int(row["instance_id"])
        gen = int(row.get("generation", 0))
        gt_key = (xai.lower(), iid)
        if gt_key not in gt_cache:
            gt_cache[gt_key] = load_gt_contributions(
                xai, iid, top_k=top_k, loss_key=loss_key,
                explanations_dir=explanations_dir,
            )
        gt = gt_cache[gt_key]

        cid = extraction_base_cid(prefix, row["pipeline"], xai, iid, gen)
        ext = extraction_by_cid.get(cid, {}) or {}

        rows.append({
            "pipeline":       row["pipeline"],
            "pipeline_label": row["pipeline_label"],
            "xai_model":      xai,
            "instance_id":    iid,
            "generation":     gen,
            **compute_faithfulness(ext, gt),
            **extraction_coverage(ext, gt),
        })
    return pd.DataFrame(rows)


def extraction_validity_summary(faith_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregated extraction validity per pipeline (for the validity disclaimer).

    Columns: number of narratives, parse failure rate, mean extracted features, mean
    top K recall, out of top K rate (noise share of the extraction) and rank 0 hit
    rate of the extractor (related to the NB 07 figure of 55 percent). RA/SA/VA are
    only interpretable in the light of this coverage (precision, not recall, NB 06
    section 4.1).
    """
    g = faith_df.groupby("pipeline_label")
    out = g.agg(
        n_narratives=("n_extracted", "size"),
        parse_empty_rate=("parse_empty", "mean"),
        mean_n_extracted=("n_extracted", "mean"),
        mean_topk_recall=("topk_recall", "mean"),
        in_topk_total=("n_in_topk", "sum"),
        out_of_topk_total=("n_out_of_topk", "sum"),
    )
    denom = out["in_topk_total"] + out["out_of_topk_total"]
    out["out_of_topk_rate"] = (out["out_of_topk_total"] / denom).where(denom > 0)
    out["r0_match_rate"] = g["r0_match"].apply(lambda s: s.dropna().mean())
    return out.round(4)


# --- Error analysis / extraction validity (NB 07 section 6) ------------------

def rank0_correctness(extraction: dict, gt_contributions: list) -> dict:
    """Rank 0 fidelity of the extractor including the sign (extension of ``extraction_coverage``).

    ``extraction_coverage`` checks only the feature identity (``r0_match``). For the
    extraction validity (NB 07 section 6) the sign also counts: a rank 0 is only
    correct if feature and sign both match the SHAP rank 0.

    ``gt_contributions`` are the (top K) contributions, sorted descending by absolute
    contribution. Returns: gt/ext rank 0 feature and sign, ``feat_match``,
    ``sign_match`` and ``r0_correct`` (= both at once).
    """
    gt_r0 = gt_contributions[0] if gt_contributions else None
    gt_r0_feat = gt_r0["feature"].lower() if gt_r0 else None
    gt_r0_sign = (1 if gt_r0["contribution"] > 0 else -1) if gt_r0 else None

    ext_r0_feat = None
    ext_r0_sign = None
    for k, info in extraction.items():
        if not isinstance(info, dict):
            continue
        try:
            if int(float(info.get("rank"))) == 0:
                ext_r0_feat = str(k).lower()
                raw_sign = info.get("sign")
                ext_r0_sign = int(raw_sign) if raw_sign is not None else None
                break
        except (ValueError, TypeError):
            pass

    feat_match = gt_r0_feat is not None and gt_r0_feat == ext_r0_feat
    sign_match = ext_r0_feat is not None and gt_r0_sign == ext_r0_sign
    return {
        "gt_r0_feat":  gt_r0_feat,  "gt_r0_sign":  gt_r0_sign,
        "ext_r0_feat": ext_r0_feat, "ext_r0_sign": ext_r0_sign,
        "feat_match":  feat_match,  "sign_match":  sign_match,
        "r0_correct":  bool(feat_match and sign_match),
    }


def correct_metric(obs_val: float, n_extracted: int) -> float:
    """Upper bound of an RA/SA metric if the rank 0 error was a measurement artefact.

    NB 07 section 6.4: if the extractor provably captured the rank 0 feature wrong
    (measurement error, not explanation error), a correct extraction would have added
    a rank 0 hit. The corrected metric adds one hit out of ``n``: ``(obs*n + 1)/n``,
    capped at 1.0. For ``n_extracted == 0`` the value stays unchanged.
    """
    if n_extracted <= 0:
        return obs_val
    return min((obs_val * n_extracted + 1) / n_extracted, 1.0)


def load_explanation_text(
    pipeline_prefix: str,
    xai_model: str,
    instance_id: int,
    *,
    results_dir: Path = RESULTS_DIR,
) -> str:
    """Explanation text from ``results/pipeline{prefix}/{xai}_inst{iid}.json`` (n=20 run).

    Canonical loader for the error analysis (NB 07), replaces three inline loaders
    duplicated there. Returns ``''`` if the file is missing (for example before the
    generation run).
    """
    p = results_dir / f"pipeline{pipeline_prefix}" / f"{xai_model.lower()}_inst{instance_id}.json"
    if not p.exists():
        return ""
    return json.loads(p.read_text()).get("explanation", "")
