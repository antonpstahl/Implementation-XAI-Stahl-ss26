"""Deterministic, LLM-free rubric score for global feature explanations (G3, step a).

Compares a generated per-feature explanation (the ``[EFFECT]/[IMPORTANCE]/
[RECOMMENDATION]`` sections produced by the 04G* pipelines) against the structured
ground truth from :mod:`utils.groundtruth`. Returns three sub-scores in ``[0, 1]``
plus their mean, so the number is fully auditable and reproducible:

  * ``direction``  -- overall direction / monotonicity stated correctly?
  * ``rank``       -- global-importance rank plausible?
  * ``structure``  -- key shape captured? (peak for non-monotone, both commute
                      peaks for ``hr``, correct top category, or "negligible"
                      framing for near-flat features)

This is the non-LLM half of the G3 evaluation; the reference-based LLM judge
(step b) handles the semantic nuance the keyword rules deliberately skip.
Everything here is intentionally transparent regex/keyword matching — no model
calls — so scores are deterministic across runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

EXPL_DIR = Path(__file__).resolve().parent.parent / "explanations"
GT_DIR = EXPL_DIR / "global_groundtruth"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "global"

_MONTHS = {1: "january", 2: "february", 3: "march", 4: "april", 5: "may",
           6: "june", 7: "july", 8: "august", 9: "september", 10: "october",
           11: "november", 12: "december"}
_WEEKDAYS = {0: "sunday", 1: "monday", 2: "tuesday", 3: "wednesday",
             4: "thursday", 5: "friday", 6: "saturday"}
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
             "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9}

# direction vocabulary
_UP = r"increase|increasing|rise|rising|rises|higher|grow|growth|more|greater|positive|boost|up\b"
_DOWN = r"decrease|decreasing|fall|falls|falling|lower|less|fewer|reduce|reduces|reducing|drop|drops|negative|dampen|damping|down\b"
_NONMONO = (r"non-?monoton|inverted-?u|u-?shaped?|peak|optimum|optimal|"
            r"highest around|maximum around|then (?:decreas|fall|declin|drop|reduc)|"
            r"rises? .*? (?:falls?|declin|drops?)|saturat|plateau|tapers?")
_NEGLIGIBLE = (r"negligible|minor|weak|small(?:est)?|little|marginal|minimal|"
               r"least important|not (?:very )?important|limited (?:effect|impact|influence)|"
               r"barely|insignificant|low importance")


# ---------------------------------------------------------------------------
# section parsing
# ---------------------------------------------------------------------------

def split_sections(explanation: str) -> dict[str, str]:
    """Return {effect, importance, recommendation} lower-cased section bodies."""
    out = {"effect": "", "importance": "", "recommendation": ""}
    markers = {"[EFFECT]": "effect", "[IMPORTANCE]": "importance",
               "[RECOMMENDATION]": "recommendation"}
    # strip any <thinking>...</thinking> block first
    text = re.sub(r"<thinking>.*?</thinking>", "", explanation, flags=re.S | re.I)
    positions = [(m.start(), markers[m.group().upper()])
                 for m in re.finditer(r"\[(?:EFFECT|IMPORTANCE|RECOMMENDATION)\]",
                                      text, flags=re.I)]
    positions.sort()
    for i, (start, key) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        body = text[start:end]
        body = re.sub(r"^\[[A-Za-z]+\]", "", body).strip()
        body = re.sub(r"[*`]", "", body)  # drop markdown emphasis / code ticks
        out[key] = body.lower()
    return out


# ---------------------------------------------------------------------------
# sub-scores
# ---------------------------------------------------------------------------

def _score_direction(effect: str, gt: dict) -> float:
    up = bool(re.search(_UP, effect))
    down = bool(re.search(_DOWN, effect))
    nonmono = bool(re.search(_NONMONO, effect))
    form = gt["form"]

    if form == "categorical":
        # direction is "which categories are high/low"; scored under structure.
        # Here we only require that the text engages with variation across levels.
        return 1.0 if (up or down or "categor" in effect or "level" in effect
                       or "highest" in effect or "lowest" in effect) else 0.5

    if form in ("monotonic", "near-flat"):
        want_up = gt["direction"] == "increasing"
        if want_up and up and not (down and not up):
            return 1.0 if not nonmono else 0.5  # spurious peak claim => partial
        if not want_up and down:
            return 1.0 if not nonmono else 0.5
        # mentioned only the wrong direction
        if (want_up and down and not up) or (not want_up and up and not down):
            return 0.0
        return 0.5  # direction unclear / both present

    # non-monotonic
    if nonmono:
        return 1.0
    if up and down:
        return 0.75  # rise-and-fall implied without explicit peak word
    return 0.25 if (up or down) else 0.0


def _extract_rank(importance: str) -> int | None:
    # normalise markdown emphasis / code ticks so "**7th**" parses like "7th"
    importance = re.sub(r"[*_`]", "", importance)
    # "rank 2", "ranks 7th", "ranked #2"
    m = re.search(r"(?:rank(?:s|ed)?|#)\s*#?\s*(\d+)(?:st|nd|rd|th)?", importance)
    if m:
        return int(m.group(1))
    # "7th out of 9", "2 of 9", "8 / 9"
    m = re.search(r"\b(\d+)(?:st|nd|rd|th)?\s*(?:of|out of|/)\s*9\b", importance)
    if m:
        return int(m.group(1))
    _most = r"most\s+(?:important|influential|impactful|significant)"
    m = re.search(rf"\b(\d+)(?:st|nd|rd|th)\s+{_most}", importance)
    if m:
        return int(m.group(1))
    for word, num in _ORDINALS.items():
        if re.search(rf"\b{word}\s+{_most}", importance):
            return num
    if re.search(rf"\b(?:single|the)\s+{_most}", importance):
        return 1
    if re.search(rf"\b{_most}\b", importance) and "second" not in importance:
        return 1
    if re.search(r"\bleast important\b", importance):
        return 9
    return None


def _score_rank(importance: str, gt: dict) -> float:
    claimed = _extract_rank(importance)
    true = gt["importance_rank"]
    if claimed is None:
        return 0.0
    diff = abs(claimed - true)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.5
    # same coarse tier (top-3 / middle / bottom-3) => partial credit
    tier = lambda r: 0 if r <= 3 else (1 if r <= 6 else 2)
    return 0.25 if tier(claimed) == tier(true) else 0.0


def _category_aliases(feature: str, cat, schema: dict) -> list[str]:
    """Search strings for a categorical level (numeric + semantic label)."""
    s = str(cat).split(".")[0]
    aliases = [s]
    try:
        n = int(float(cat))
    except (TypeError, ValueError):
        n = None
    if feature == "mnth" and n in _MONTHS:
        aliases += [_MONTHS[n], _MONTHS[n][:3]]
    elif feature == "weekday" and n in _WEEKDAYS:
        aliases.append(_WEEKDAYS[n])
    elif feature == "hr" and n is not None:
        aliases.append(f"{n}:00")
    # semantic label from the model schema (weathersit / yr / holiday)
    cats = schema.get(feature, {}).get("categories", {})
    if s in cats:
        label = cats[s].lower()
        aliases += re.split(r"[/\s]+", label)
    return [a for a in aliases if a and len(a) >= 3 or a.isdigit()]


def _score_structure(effect: str, gt: dict, schema: dict) -> float:
    form = gt["form"]

    if form == "near-flat":
        return 1.0 if re.search(_NEGLIGIBLE, effect) else 0.0

    if form == "non-monotonic":
        has_turn = bool(re.search(_NONMONO, effect))
        both_dirs = bool(re.search(_UP, effect)) and bool(re.search(_DOWN, effect))
        return 1.0 if has_turn else (0.5 if both_dirs else 0.0)

    if form == "monotonic":
        want_up = gt["direction"] == "increasing"
        hit = re.search(_UP if want_up else _DOWN, effect)
        false_peak = re.search(r"non-?monoton|inverted-?u|u-?shaped?", effect)
        return 1.0 if (hit and not false_peak) else (0.5 if hit else 0.0)

    # categorical
    if gt["feature"] == "hr":
        morning = bool(re.search(r"\bmorning\b|7\s*[-–]?\s*9|\b[7-9](?::00)?\s*a", effect)) \
            or bool(re.search(r"\b8\b", effect))
        evening = bool(re.search(r"\bevening\b|after ?noon|16\s*[-–]|17|18|19|"
                                  r"\b[4-7]\s*p", effect))
        return 0.5 * morning + 0.5 * evening

    top = gt.get("top_categories", [])
    if not top:
        return 0.0
    top1_aliases = _category_aliases(gt["feature"], top[0]["cat"], schema)
    if any(re.search(rf"\b{re.escape(a)}\b", effect) for a in top1_aliases):
        return 1.0
    for c in top[1:]:
        al = _category_aliases(gt["feature"], c["cat"], schema)
        if any(re.search(rf"\b{re.escape(a)}\b", effect) for a in al):
            return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

_SCHEMA_CACHE: dict[str, dict] = {}


def _schema(model: str) -> dict:
    if model not in _SCHEMA_CACHE:
        p = EXPL_DIR / f"global_{model}_poisson_log.json"
        _SCHEMA_CACHE[model] = json.loads(p.read_text()).get("feature_schema", {})
    return _SCHEMA_CACHE[model]


def load_ground_truth(model: str, feature: str) -> dict:
    return json.loads((GT_DIR / f"{model}_{feature}.json").read_text())


def rubric_score(explanation: str, gt: dict) -> dict:
    """Score one generated explanation against its ground truth.

    Returns a dict with the three sub-scores, their mean ``total`` (all in
    ``[0, 1]``), and the parsed rank for auditing.
    """
    schema = _schema(gt["model"])
    sec = split_sections(explanation)
    direction = _score_direction(sec["effect"], gt)
    rank = _score_rank(sec["importance"], gt)
    structure = _score_structure(sec["effect"], gt, schema)
    total = round((direction + rank + structure) / 3, 4)
    return {
        "direction": direction,
        "rank": rank,
        "structure": structure,
        "total": total,
        "parsed_rank": _extract_rank(sec["importance"]),
        "form": gt["form"],
    }


def score_result_file(path: Path) -> dict:
    rec = json.loads(path.read_text())
    gt = load_ground_truth(rec["xai_model"], rec["feature"])
    sec = split_sections(rec["explanation"])
    missing = [k for k in ("effect", "importance", "recommendation") if not sec[k]]
    s = rubric_score(rec["explanation"], gt)
    return {
        "form_pipeline": rec["form"],
        "xai_model": rec["xai_model"],
        "feature": rec["feature"],
        "form_type": gt["form"],
        "incomplete": ";".join(missing),
        **s,
    }


# ---------------------------------------------------------------------------
# CLI: score every results/global/* file, write CSV, print stratified summary
# ---------------------------------------------------------------------------

def _main() -> None:
    import csv
    from collections import defaultdict

    rows = [score_result_file(p) for p in sorted(RESULTS_DIR.glob("*.json"))]

    out_csv = RESULTS_DIR.parent / "global_rubric.csv"
    cols = ["form_pipeline", "xai_model", "feature", "form_type",
            "direction", "rank", "structure", "total", "parsed_rank", "incomplete"]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # stratified summary: mean total by (pipeline modality x form_type)
    agg: dict = defaultdict(list)
    for r in rows:
        agg[(r["form_pipeline"], r["form_type"])].append(r["total"])
    modalities = sorted({r["form_pipeline"] for r in rows})
    form_types = ["monotonic", "non-monotonic", "categorical", "near-flat"]

    incomplete = [r for r in rows if r["incomplete"]]
    unparsed = [r for r in rows if r["parsed_rank"] in (None, "")]

    print(f"\nScored {len(rows)} explanations -> {out_csv}")
    if incomplete:
        print(f"\n! {len(incomplete)} incomplete generation(s) (missing sections):")
        for r in incomplete:
            print(f"    {r['form_pipeline']:9s} {r['xai_model']} {r['feature']:10s} "
                  f"missing: {r['incomplete']}")
    if unparsed:
        print(f"! {len(unparsed)} explanation(s) with no parsable rank.")
    print()
    hdr = f"{'modality':10s} " + " ".join(f"{ft:14s}" for ft in form_types) + " mean"
    print(hdr)
    print("-" * len(hdr))
    for mod in modalities:
        cells = []
        allv = []
        for ft in form_types:
            v = agg.get((mod, ft), [])
            allv += v
            cells.append(f"{sum(v)/len(v):.2f} (n={len(v):>2d})" if v else "   -     ")
        mean = f"{sum(allv)/len(allv):.2f}" if allv else "-"
        print(f"{mod:10s} " + " ".join(f"{c:14s}" for c in cells) + f" {mean}")


if __name__ == "__main__":
    _main()
