"""utils/faithfulness.py – Skalierungs-Faithfulness nach Ichmoukhamedov et al. (2024).

Ground-truth-verankerte Faithfulness-Metriken (RA/SA/VA, Gl. 1 im Paper) für den
n≈200-Vollauf (Phase 3b). **Treuer Port** der in `08_Evaluation_Ichmoukhamedov`
auf n=20 entwickelten Logik — identischer Extraktions-Prompt, identischer Parser
und eine **byte-für-byte identische** `compute_faithfulness` (die n=20-Zahlen
reproduzieren), erweitert um:

  * **gen-aware** Verarbeitung (N Generationen pro Instanz, eigene `custom_id`s),
  * einen **Batch-Extraktionspfad** (Extraktion ist ein Einzelschritt-Call → voll
    batchbar, −50 % Kosten, wie der Judge), und
  * **Extraktionsvalidität** (`extraction_coverage` / `extraction_validity_summary`):
    automatisch berechenbare Proxys dafür, wie verlässlich das *Messinstrument*
    (der Extraktor) ist — denn die Fehlertaxonomie (NB 09) zeigte 19/30
    Extraktor-Artefakte und eine Rang-0-Extraktor-Genauigkeit von nur 55 %.
    RA/SA/VA messen **Präzision, nicht Recall** (NB 08 §4.1); die Validitäts-
    Kennzahlen machen genau diese Einschränkung quantitativ.

Bewusst getrennt vom n=20-Notebook (NB 08 bleibt unangetastet); `08b_Scaling_
Faithfulness` nutzt diese Helfer auf den Skalierungs-Artefakten unter
``results/pipeline0X/scale/``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from utils import EXPLANATIONS_DIR
from utils.batch import make_custom_id
from utils.explanations import FEATURE_SCHEMA, HUM_FACTOR, TEMP_FACTOR, WIND_FACTOR

LOSS_KEY_DEFAULT = "poisson_log"
TOP_K_DEFAULT = 4  # Paper: top-4 Features nach |Beitrag|

# Extraktions-Prompt + System (treuer Port aus NB 08 Zelle 5).
EXTRACTION_SYSTEM = (
    "Du bist ein Extraktionsmodell für XAI-Narrative eines Fahrradverleih-Modells.\n"
    "Extrahiere die angeforderten Informationen ausschließlich aus dem Narrativ.\n"
    "Antworte ausschließlich mit einem validen JSON-Objekt — kein Text, keine Markdown-Blöcke."
)

# Denormalisierung: normalisierte Featurewerte → menschlich lesbare Einheiten.
# Quelle der Faktoren: utils.explanations (DRY-konsolidiert, Phase 3·2), damit
# °C/%/km-h-Umrechnung nicht gegen Generierungs-/Judge-Prompt divergiert.
_DENORM = {
    "temp":      lambda v: v * TEMP_FACTOR,
    "hum":       lambda v: v * HUM_FACTOR,
    "windspeed": lambda v: v * WIND_FACTOR,
}


def build_extraction_prompt(explanation: str, xai_model: str) -> str:
    """Baut den Extraktions-User-Prompt (JSON) — treuer Port aus NB 08."""
    feat_descs = {f: FEATURE_SCHEMA[f]["description"] for f in FEATURE_SCHEMA}
    payload = {
        "aufgabe": (
            f"Extrahiere strukturierte Informationen aus dem folgenden deutschen Narrativ "
            f"zu einem {xai_model}-Regressionsmodell für einen Fahrradverleih. "
            f"Das Modell sagt stündliche Fahrrad-Ausleihen voraus. "
            f"Positive Beiträge erhöhen die Vorhersage, negative senken sie."
        ),
        "narrativ": explanation,
        "alle_features": list(FEATURE_SCHEMA.keys()),
        "feature_beschreibungen": feat_descs,
        "extraktionsanweisung": (
            "Gib für jedes Feature, das im Narrativ als wichtig erwähnt wird, ein Objekt zurück:\n"
            "  rank: 0-basierter Wichtigkeitsrang laut Narrativ (0 = wichtigstes Feature)\n"
            "  sign: +1 wenn das Feature die Vorhersage erhöht, -1 wenn es sie senkt\n"
            "  value: numerischer Featurewert, falls explizit im Narrativ genannt, sonst null\n"
            "  assumption: einziger Satz mit Hintergrundwissen warum das Feature diesen Einfluss hat; "
            '"None" falls kein Hintergrundwissen hinzugefügt wurde'
        ),
        "ausgabeformat_beispiel": {
            "hr":   {"rank": 0, "sign":  1, "value": 13,   "assumption": "Mittagszeit ist typischerweise nachfragereich."},
            "temp": {"rank": 1, "sign": -1, "value": None, "assumption": "Kälte schreckt Radfahrer ab."},
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_extraction(raw: str) -> dict:
    """Extrahiert das JSON-Objekt aus der Extraktor-Antwort (treuer Port, NB 08)."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {}


def is_value_match(feat: str, extracted: float, gt: float, tol: float = 1.0) -> bool:
    """Wertvergleich mit Toleranz; prüft auch denormalisierte Einheiten (treuer Port)."""
    if abs(extracted - gt) <= tol:
        return True
    if feat in _DENORM:
        dv = _DENORM[feat](gt)
        if abs(extracted - dv) <= tol:
            return True
    return False


def compute_faithfulness(extraction: dict, gt_contributions: list) -> dict:
    """RA, SA, VA nach Gl. 1 aus Ichmoukhamedov et al. (2024) — treuer Port aus NB 08.

    ϕ = null/nicht extrahiert → wird aus dem Nenner herausgerechnet. Iteriert über
    die **extrahierten** Features (Präzision, nicht Recall — siehe NB 08 §4.1;
    Recall/Coverage liefert :func:`extraction_coverage`).
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
            continue  # Feature nicht unter Top-K → überspringen (wie im Paper)

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
    """Extraktionsvalidität je Narrativ — Proxys für die Verlässlichkeit des Extraktors.

    RA/SA/VA werten nur die *erwähnten* Features (Präzision). Diese Kennzahlen machen
    den Recall- und Rausch-Anteil sichtbar — wichtig, weil der Extraktor selbst
    fehleranfällig ist (NB 09: 19/30 Extraktor-Artefakte, Rang-0-Genauigkeit 55 %):

      * ``parse_empty``    — der Extraktor lieferte kein gültiges JSON (Messausfall).
      * ``topk_recall``    — Anteil der Top-K-Ground-Truth-Features, die der Extraktor
                             überhaupt erfasst hat (Gegenstück zur Präzision von RA/SA/VA).
      * ``n_out_of_topk``  — extrahierte Features, die **nicht** unter Top-K liegen
                             (werden in `compute_faithfulness` still übersprungen → Rausch-Proxy).
      * ``r0_match``       — stimmt das vom Extraktor als Rang 0 markierte Feature mit
                             dem SHAP-Rang-0 überein? (1/0/None) — direkter Bezug zur
                             NB-09-Kennzahl (55 % Rang-0-Genauigkeit).
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
    """Top-K SHAP-Ground-Truth-Beiträge aus ``local_{xai}_{loss}_inst{iid}.json``."""
    p = explanations_dir / f"local_{xai_model.lower()}_{loss_key}_inst{instance_id}.json"
    gt = json.loads(p.read_text())
    return gt["contributions"][:top_k]


def extraction_base_cid(prefix: str, pipeline: str, xai_model: str,
                        instance_id: int, generation: int) -> str:
    """Gen-aware `custom_id` für eine Extraktion (analog zu den Judge-cids in NB 07b)."""
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
    """Baut die per-Narrativ-Faithfulness-Tabelle (RA/SA/VA + Validität) gen-aware.

    `df` sind die gen-aware Generierungs-Records (aus
    :func:`utils.eval.load_scale_records`); `extraction_by_cid` mappt die
    Extraktions-`custom_id` (siehe :func:`extraction_base_cid`) auf das geparste
    Extraktions-Dict (`run_batch(...)['succeeded']`). Pro Zeile werden Ground-Truth
    (je (xai, instance) gecacht), `compute_faithfulness` und `extraction_coverage`
    zusammengeführt. Fehlende `custom_id`s ⇒ leere Extraktion (zählt als `parse_empty`).
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
    """Aggregierte Extraktionsvalidität je Pipeline (für den Validitäts-Disclaimer).

    Spalten: Narrativ-Anzahl, Parse-Ausfallrate, Ø extrahierte Features, Ø Top-K-Recall,
    Out-of-Top-K-Rate (Rausch-Anteil der Extraktion) und Rang-0-Trefferquote des
    Extraktors (Bezug zur NB-09-Kennzahl 55 %). RA/SA/VA sind nur im Licht dieser
    Coverage interpretierbar (Präzision, nicht Recall — NB 08 §4.1).
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
