"""Deterministic ground-truth derivation from G0 global curves (Phase G3, step 1).

Reads ``explanations/global_curve_{model}_{feature}.json`` and the model-level
importance file ``explanations/global_{model}_poisson_log.json`` and emits one
structured reference per feature to
``explanations/global_groundtruth/{model}_{feature}.json``.

Two curve formats are handled transparently:
  * EBM  -> clean shape-function grid (x sorted-unique, y = contribution).
  * XGB  -> per-instance SHAP scatter (x = feature value, y = SHAP; duplicate x).
            Aggregated to a grid by grouping on x and averaging y before any
            monotonicity / peak logic runs.

The derived fields are the GT contract for the G3 rubric and the reference-based
judge: ``form``, ``direction``, ``monotonicity``, ``peak``, ``importance_rank`` plus
supporting quantities. Output is semi-automatic and meant to be domain-verified
afterwards (edit the JSON by hand where the curve is ambiguous).

Classification defaults (documented, overridable via ``Thresholds``):
  * flat_threshold = 0.025 -> importance below this absolute value => "near-flat".
  * mono_tol       = 0.15  -> a continuous curve counts as monotone when the
                             movement against the dominant direction is <= 15 %
                             of the total absolute variation.
  * top_k_cats     = 3     -> how many best/worst categories to record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

# The form/direction/monotonicity classifier is shared with the deterministic baseline
# (utils.global_feature.describe_curve), so GT and baseline can never disagree on the
# form type (planning/korrekturen17_06.md P2). groundtruth stays the authority for the
# thresholds; describe_curve is aligned to it.
from .global_feature import aggregate_curve, classify_shape

EXPL_DIR = Path(__file__).resolve().parent.parent / "explanations"
GT_DIR = EXPL_DIR / "global_groundtruth"
MODELS = ("ebm", "xgb")

# Manual domain verification, applied after the automatic derivation (and
# preserved across regeneration). Keyed by feature. Presence marks the reference
# as human-reviewed (verified=True) and attaches a limitation note; the derived
# curve values are left untouched (documented, not altered).
DOMAIN_NOTES = {
    "weathersit": ("Category 4 (heavy rain/thunderstorm) has very few samples, so "
                   "its curve contribution is noisy and its high rank is not "
                   "domain-reliable. True weather severity is monotone 1>2>3>4 "
                   "(clear > mist > light rain > heavy rain)."),
}

# (model, feature) pairs whose derived reference was VERIFIED on 17.07 (P1): the label
# was mechanically re-derived against the curve (form / direction / monotonicity /
# importance_rank / peak / top-categories all reproduce) AND scanned for domain
# reliability via per-level training-sample counts. All 18 pass the mechanical check
# (0 corrections needed). See analyses/gt_verification.md for the per-feature report and
# the threshold sensitivity (15/18 form labels are stable across a 0.015-0.035 ×
# 0.10-0.25 grid; ebm/temp, ebm/weekday and xgb/windspeed sit near a boundary, flagged
# there for a final visual confirmation, not a mechanical error).
VERIFIED: set[tuple[str, str]] = {
    (m, f) for m in MODELS for f in (
        "hr", "temp", "yr", "mnth", "hum", "weathersit", "weekday", "windspeed", "holiday",
    )
}

# Manual field corrections applied after derivation, keyed by (model, feature): a dict of
# {field: value} overriding a mechanically-wrong derived value (documented, regeneration-
# stable). Empty because the 17.07 review found 0 mechanical errors; kept as the wiring
# for any future hand-correction.
OVERRIDES: dict[tuple[str, str], dict] = {}


@dataclass
class Thresholds:
    flat_threshold: float = 0.025  # absolute importance below => near-flat
    mono_tol: float = 0.15         # reversal share tolerated as still-monotone
    top_k_cats: int = 3            # best/worst categories to record


# ---------------------------------------------------------------------------
# loading helpers
# ---------------------------------------------------------------------------

def _load_curve(model: str, feature: str) -> dict:
    p = EXPL_DIR / f"global_curve_{model}_{feature}.json"
    return json.loads(p.read_text())


def _load_ranks(model: str) -> dict[str, int]:
    p = EXPL_DIR / f"global_{model}_poisson_log.json"
    gi = json.loads(p.read_text())["global_importance"]
    return {r["feature"]: r["rank"] for r in gi}


def _feature_schema(model: str) -> dict:
    p = EXPL_DIR / f"global_{model}_poisson_log.json"
    return json.loads(p.read_text()).get("feature_schema", {})


def _aggregate(x: list, y: list) -> tuple[list, list]:
    """Group duplicate x, average y, return sorted-by-x grid.

    Delegates to the shared :func:`utils.global_feature.aggregate_curve` so the GT and
    the baseline aggregate identically (single source of truth, P2). Works for both EBM
    (already a grid) and XGB scatter; categorical string x kept as-is.
    """
    return aggregate_curve(x, y)


# ---------------------------------------------------------------------------
# derivation of structured fields
# ---------------------------------------------------------------------------

def _total_variation(y: list[float]) -> float:
    return sum(abs(y[i + 1] - y[i]) for i in range(len(y) - 1))


def _signed_movement(y: list[float]) -> tuple[float, float]:
    """Return (up_sum, down_sum) of absolute step magnitudes."""
    up = down = 0.0
    for i in range(len(y) - 1):
        d = y[i + 1] - y[i]
        if d >= 0:
            up += d
        else:
            down += -d
    return up, down


def _continuous_fields(xs: list[float], ys: list[float], th: Thresholds) -> dict:
    """Monotonicity, direction, peak/trough for an aggregated continuous grid."""
    tv = _total_variation(ys)
    up, down = _signed_movement(ys)
    net = ys[-1] - ys[0]
    y_min, y_max = min(ys), max(ys)
    i_max, i_min = ys.index(y_max), ys.index(y_min)

    # reversal share: movement against the dominant direction
    reversal = min(up, down) / tv if tv > 0 else 0.0
    is_monotone = reversal <= th.mono_tol

    if is_monotone:
        direction = "increasing" if net >= 0 else "decreasing"
        monotonicity = f"monotonic_{direction}"
        peak = None
    else:
        direction = "mixed"
        monotonicity = "non_monotonic"
        # inverted-U (rise then fall) -> report max; U-shape -> report min
        if i_max not in (0, len(ys) - 1):
            peak = {"type": "max", "x": xs[i_max], "y": round(y_max, 5)}
        elif i_min not in (0, len(ys) - 1):
            peak = {"type": "min", "x": xs[i_min], "y": round(y_min, 5)}
        else:
            peak = {"type": "max", "x": xs[i_max], "y": round(y_max, 5)}

    # sign-change crossing (where contribution flips baseline), if any
    crossing = None
    for i in range(len(ys) - 1):
        if (ys[i] < 0) != (ys[i + 1] < 0) and ys[i] != ys[i + 1]:
            # linear interpolation of the x at y=0
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
            crossing = round(x0 + (0 - y0) * (x1 - x0) / (y1 - y0), 4)
            break

    return {
        "direction": direction,
        "monotonicity": monotonicity,
        "peak": peak,
        "effect_range": round(y_max - y_min, 5),
        "y_at_min_x": round(ys[0], 5),
        "y_at_max_x": round(ys[-1], 5),
        "sign_change_x": crossing,
    }


def _categorical_fields(xs: list, ys: list[float], th: Thresholds) -> dict:
    order = sorted(range(len(ys)), key=lambda i: ys[i], reverse=True)
    k = min(th.top_k_cats, len(order))
    top = [{"cat": xs[i], "y": round(ys[i], 5)} for i in order[:k]]
    bottom = [{"cat": xs[i], "y": round(ys[i], 5)} for i in order[-k:][::-1]]
    return {
        "direction": "categorical",
        "monotonicity": "n/a",
        "peak": None,
        "effect_range": round(max(ys) - min(ys), 5),
        "top_categories": top,
        "bottom_categories": bottom,
    }


def derive_feature(model: str, feature: str, ranks: dict[str, int],
                   th: Thresholds) -> dict:
    curve = _load_curve(model, feature)
    kind = curve["kind"]
    importance = float(curve.get("importance"))
    xs, ys = _aggregate(curve["x"], curve["y"])

    n_unique = len(xs)
    is_binary = n_unique == 2

    if kind == "categorical":
        fields = _categorical_fields(xs, ys, th)
    else:
        fields = _continuous_fields([float(x) for x in xs], ys, th)

    # form / direction / monotonicity from the shared classifier (single source of truth
    # with the baseline, P2); _continuous_fields still supplies peak/sign_change/ranges.
    cls = classify_shape(ys, kind, importance,
                         flat_importance=th.flat_threshold, mono_tol=th.mono_tol)
    fields["direction"] = cls["direction"]
    fields["monotonicity"] = cls["monotonicity"]
    form = cls["form"]

    gt = {
        "model": model,
        "feature": feature,
        "kind": kind,
        "is_binary": is_binary,
        "importance": round(importance, 5),
        "importance_rank": ranks[feature],
        "form": form,
        "n_grid": n_unique,
        "verified": False,  # flipped to True by VERIFIED / DOMAIN_NOTES (P1 review)
        **fields,
    }
    # P1: a reference counts as verified once it is in VERIFIED (mechanically re-derived +
    # domain-scanned) or carries a domain note. A note is attached where present.
    if (model, feature) in VERIFIED or feature in DOMAIN_NOTES:
        gt["verified"] = True
    if feature in DOMAIN_NOTES:
        gt["note"] = DOMAIN_NOTES[feature]
    # apply any documented manual field correction (regeneration-stable)
    gt.update(OVERRIDES.get((model, feature), {}))
    return gt


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def build_all(th: Thresholds | None = None) -> list[dict]:
    th = th or Thresholds()
    GT_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for model in MODELS:
        ranks = _load_ranks(model)
        for feature in ranks:
            gt = derive_feature(model, feature, ranks, th)
            (GT_DIR / f"{model}_{feature}.json").write_text(
                json.dumps(gt, indent=2) + "\n"
            )
            out.append(gt)
    return out


def print_table(records: list[dict]) -> None:
    hdr = f"{'model':5s} {'feature':11s} {'rank':4s} {'imp':8s} {'form':13s} " \
          f"{'direction':10s} {'peak/struct'}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(records, key=lambda d: (d["model"], d["importance_rank"])):
        if r["peak"]:
            struct = f"{r['peak']['type']}@x={r['peak']['x']}"
        elif "top_categories" in r:
            struct = "top=" + ",".join(str(c["cat"]) for c in r["top_categories"])
        else:
            struct = f"net {r.get('y_at_min_x')}->{r.get('y_at_max_x')}"
        print(f"{r['model']:5s} {r['feature']:11s} {r['importance_rank']:<4d} "
              f"{r['importance']:<8.4f} {r['form']:13s} {r['direction']:10s} {struct}")


if __name__ == "__main__":
    recs = build_all()
    print_table(recs)
    print(f"\nWrote {len(recs)} files to {GT_DIR}")
