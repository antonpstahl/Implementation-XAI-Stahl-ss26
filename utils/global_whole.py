"""utils/global_whole.py - Phase G2b: whole-model global explanation (additive, exploratory).

Where G2a (``utils.global_feature``) asks about **one feature per call**, G2b asks the
LLM to describe the **whole model** (all features) in a single call. It is the additive,
exploratory extension of the revision plan (see
``planning/Revision_30062026_Umsetzungsplan.md`` -> Phase G2b): not meeting-mandated
(the meeting said "one feature at a time"), but it opens the two comparisons the
per-feature track cannot make:

  * **Axis 1 - representation / information content:** all 9 per-feature plots/curves
    (*_all) vs the single beeswarm (*_beeswarm). These carry *different* information, so
    they are scored **per GT field**, not as a head-to-head winner; the beeswarm is only
    scored on what it can carry (direction + rank).
  * **Axis 2 - mechanism (push vs pull):** full-push (vision_all / json_all) vs pull
    (tooluse_all, the model retrieves everything itself), at constant full information.

Plus the beeswarm-readability control: ``vision_beeswarm`` vs ``json_beeswarm``. Both
carry the *same* information (rank + colour direction + coarse spread); the JSON variant
is **deliberately info-matched to what the swarm image conveys** - ranking, colour
direction and horizontal spread, **no per-value curve** (plan G2b Auflage). So the
difference between the two isolates the pure modality effect ("can the LLM *read* the
beeswarm?"), not an information-content gap.

Scoring reuse (the point of the forced schema): a whole-model answer must emit a rigid
``[FEATURE: x] [EFFECT] ... [IMPORTANCE] ...`` block per feature (plus one whole-model
``[RECOMMENDATION]``). :func:`split_whole_model_record` cuts that into per-feature
records byte-compatible with the G2a records, so the existing G3 rubric
(:mod:`utils.rubric`) and reference-based judge (``utils.eval.run_global_judge``) score
each feature unchanged - directly measuring the meeting's "what if only 5 of 9 features
are right?" concern (a feature the whole-model answer drops scores as a miss).

Everything reads the pre-computed G0/G1 artifacts only (no model at description time),
exactly like ``utils.global_feature``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .global_feature import (
    _global_json,
    aggregate_curve,
    beeswarm_plot_path,
    describe_curve,
    feature_importance_map,
    list_global_features,
    load_global_curve,
    shape_plot_path,
)

# Whole-model outputs live in their own directories, leaving the approved G2a core
# (results/global, results/global_judge) frozen and untouched.
WHOLE_RESULTS_SUBDIR = "global_whole"        # raw whole-model records: {condition}_{model}.json
WHOLE_SPLIT_SUBDIR = "global_whole_split"    # per-feature split records: {condition}_{model}_{feature}.json

_VALUE_SPACE = "contribution to the target in log space (exp -> multiplicative on cnt)"


# -----------------------------------------------------------------------------
# Condition registry (the five whole-model conditions x 2 models = 10 calls)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class WholeCondition:
    """One whole-model condition.

    ``modality``       : json | vision | tooluse  (how the payload is delivered).
    ``representation`` : all | beeswarm  (all 9 curves/plots vs the single beeswarm).
    ``mechanism``      : push | pull  (information pushed in vs pulled via tools).
    """

    name: str
    modality: str
    representation: str
    mechanism: str


WHOLE_CONDITIONS: list[WholeCondition] = [
    WholeCondition("json_all",        "json",    "all",      "push"),
    WholeCondition("json_beeswarm",   "json",    "beeswarm", "push"),
    WholeCondition("vision_all",      "vision",  "all",      "push"),
    WholeCondition("vision_beeswarm", "vision",  "beeswarm", "push"),
    WholeCondition("tooluse_all",     "tooluse", "all",      "pull"),
]
WHOLE_CONDITIONS_BY_NAME: dict[str, WholeCondition] = {c.name: c for c in WHOLE_CONDITIONS}


# -----------------------------------------------------------------------------
# Payloads: full-information ("all") - all 9 per-feature curves in one JSON
# -----------------------------------------------------------------------------

def build_whole_json_all_payload(
    model_name: str,
    *,
    explanations_dir: Path | str,
    loss_key: str = "poisson_log",
) -> dict:
    """Full-push JSON payload: every feature's curve in one object (Axis-2 push side).

    Information-matched to the ``vision_all`` condition (all 9 plots): each feature's
    global curve plus rank/importance context. Curves are **aggregated to a grid**
    (``aggregate_curve``: mean contribution per unique feature value), a no-op for the
    EBM shape functions, but essential for XGB, whose curve JSON is the raw per-instance
    SHAP scatter (~12k points/feature). Dumping all 9 raw would be ~1M+ tokens and
    exceed the context window; the aggregated grid is the mean dependence trend the
    plot shows, and is what the model can actually read. (The vision_all plots still
    carry the per-point scatter density, a small, documented representation difference.)
    """
    g = _global_json(model_name, explanations_dir=explanations_dir, loss_key=loss_key)
    features = list_global_features(
        model_name, explanations_dir=explanations_dir, loss_key=loss_key
    )
    imp = feature_importance_map(
        model_name, explanations_dir=explanations_dir, loss_key=loss_key
    )
    entries = []
    for f in features:
        curve = load_global_curve(model_name, f, explanations_dir=explanations_dir)
        xs, ys = aggregate_curve(curve["x"], curve["y"])
        entries.append({
            "feature": f,
            "kind": curve["kind"],
            "importance": imp[f]["importance"],
            "rank": imp[f]["rank"],
            "curve": {"x": xs, "y": [round(v, 5) for v in ys]},
        })
    return {
        "target": g["task"],
        "value_space": _VALUE_SPACE + "; curve = mean contribution per feature value",
        "model": model_name.lower(),
        "n_features": len(features),
        "features": entries,
    }


# -----------------------------------------------------------------------------
# Payloads: info-matched numeric beeswarm (the readability control)
# -----------------------------------------------------------------------------
_SPREAD_LABELS = ("narrow", "moderate", "wide")


def _curve_amplitude(model_name: str, feature: str, *, explanations_dir: Path | str) -> float:
    curve = load_global_curve(model_name, feature, explanations_dir=explanations_dir)
    y = [float(v) for v in curve["y"]]
    return (max(y) - min(y)) if y else 0.0


def _spread_label(amplitude: float, thresholds: tuple[float, float]) -> str:
    """Coarse horizontal-spread label (narrow|moderate|wide) from tertile thresholds."""
    lo, hi = thresholds
    if amplitude <= lo:
        return _SPREAD_LABELS[0]
    if amplitude <= hi:
        return _SPREAD_LABELS[1]
    return _SPREAD_LABELS[2]


def beeswarm_colour_direction(
    model_name: str, feature: str, *, explanations_dir: Path | str
) -> str:
    """What the beeswarm COLOUR conveys for a feature: do high values push up or down?

    Derived from the G0 curve direction (same source as everything else), phrased as the
    colour-gradient reading of a swarm row. Non-monotonic / flat curves have no clean
    colour separation on the swarm, and categorical features have no ordinal colour
    trend - both are reported as such (this honestly *is* what the beeswarm shows, and it
    is deliberately less than the full curve: no shape, no peak).
    """
    d = describe_curve(model_name, feature, explanations_dir=explanations_dir)
    if d["kind"] == "categorical":
        return "varies by category (no ordinal colour trend)"
    direction = d["direction"]
    if direction == "rising":
        return "higher values raise demand"
    if direction == "falling":
        return "higher values lower demand"
    return "mixed / no clear colour separation"  # mixed (non-monotonic) or flat


def build_whole_json_beeswarm_payload(
    model_name: str,
    *,
    explanations_dir: Path | str,
    loss_key: str = "poisson_log",
) -> dict:
    """Info-matched numeric beeswarm: per feature ONLY rank + colour direction + spread.

    The JSON counterpart to the beeswarm image, matched to what a viewer reads off the
    swarm (feature ordering, colour direction, horizontal spread). It deliberately omits
    the per-value curve, shape and peak so the vision-vs-json beeswarm comparison
    measures the *modality* (visual reading vs equivalent numbers), not an information
    gap (plan G2b Auflage). Reported as a control, not a new information source.
    """
    features = list_global_features(
        model_name, explanations_dir=explanations_dir, loss_key=loss_key
    )
    imp = feature_importance_map(
        model_name, explanations_dir=explanations_dir, loss_key=loss_key
    )
    g = _global_json(model_name, explanations_dir=explanations_dir, loss_key=loss_key)

    amps = {f: _curve_amplitude(model_name, f, explanations_dir=explanations_dir)
            for f in features}
    ordered = sorted(amps.values())
    n = len(ordered)
    # tertile thresholds over the 9 amplitudes -> narrow / moderate / wide
    thresholds = (ordered[n // 3], ordered[2 * n // 3]) if n >= 3 else (0.0, 0.0)

    entries = [{
        "feature": f,
        "rank": imp[f]["rank"],
        "colour_direction": beeswarm_colour_direction(
            model_name, f, explanations_dir=explanations_dir),
        "spread": _spread_label(amps[f], thresholds),
    } for f in features]

    return {
        "target": g["task"],
        "model": model_name.lower(),
        "n_features": len(features),
        "note": ("beeswarm-equivalent summary: feature ranking + colour direction + "
                 "horizontal spread only, no per-value curve - matched to what the "
                 "beeswarm image conveys (a readability control, not extra information)."),
        "features": entries,
    }


# -----------------------------------------------------------------------------
# Vision plot paths ("all" = every feature plot; "beeswarm" = the single swarm)
# -----------------------------------------------------------------------------

def whole_plot_paths(
    model_name: str,
    representation: str,
    *,
    plots_dir: Path | str,
    explanations_dir: Path | str,
    loss_key: str = "poisson_log",
) -> list[Path]:
    """Image paths for a vision whole-model condition.

    ``representation="all"``       -> all 9 shape/dependence PNGs (rank order).
    ``representation="beeswarm"``  -> the single beeswarm PNG.
    """
    if representation == "beeswarm":
        return [beeswarm_plot_path(model_name, plots_dir=plots_dir)]
    if representation == "all":
        features = list_global_features(
            model_name, explanations_dir=explanations_dir, loss_key=loss_key
        )
        return [shape_plot_path(model_name, f, plots_dir=plots_dir) for f in features]
    raise ValueError(f"Unknown representation '{representation}' (expected 'all'|'beeswarm').")


# -----------------------------------------------------------------------------
# Prompt assembly (shared by all whole-model conditions; own template)
# -----------------------------------------------------------------------------
MODEL_LABELS = {"ebm": "EBM", "xgb": "XGBoost"}


def assemble_whole_system_prompt(
    condition_name: str,
    model_name: str,
    *,
    prompts_dir: Path | str,
    features: Iterable[str],
    prompt_file: str = "global_whole.md",
) -> str:
    """Build the whole-model system prompt for one (condition, model).

    Parses ``prompts/global_whole.md`` (byte-identical core + per-condition handover
    block), fills ``{{MODEL}}`` and ``{{FEATURE_LIST}}`` (the exact feature names the
    forced per-feature schema must cover). Raises on any leftover ``{{placeholder}}``.
    """
    md = (Path(prompts_dir) / prompt_file).read_text()
    try:
        core = md.split("# SYSTEM PROMPT CORE", 1)[1].split("# HANDOVER FORMAT variants", 1)[0]
        variants = md.split("# HANDOVER FORMAT variants", 1)[1]
    except IndexError as exc:  # pragma: no cover - template shape guard
        raise ValueError("global_whole.md is missing an expected section header.") from exc

    core = core[core.index("\n") + 1:]                  # drop the rest of the header line
    core = re.sub(r"\n-{3,}\s*$", "", core.strip())     # drop trailing '---' separator

    model_key = model_name.lower()
    if model_key not in MODEL_LABELS:
        raise ValueError(f"Unknown model_name '{model_name}' (expected 'ebm' or 'xgb').")
    if condition_name not in WHOLE_CONDITIONS_BY_NAME:
        raise ValueError(f"Unknown condition '{condition_name}'.")

    handover = _whole_handover_block(variants, condition_name)
    feature_list = ", ".join(features)
    sys_prompt = (
        core.replace("{{HANDOVER_FORMAT}}", handover)
        .replace("{{MODEL}}", MODEL_LABELS[model_key])
        .replace("{{FEATURE_LIST}}", feature_list)
    )
    leftover = re.findall(r"\{\{[^}]+\}\}", sys_prompt)
    if leftover:
        raise ValueError(f"Unresolved placeholders in assembled system prompt: {leftover}")
    return sys_prompt.strip()


def _whole_handover_block(variants_section: str, condition_name: str) -> str:
    m = re.search(
        r"^##\s+" + re.escape(condition_name) + r"\b[^\n]*\n```\n(.*?)\n```",
        variants_section, re.DOTALL | re.MULTILINE,
    )
    if m is None:
        raise ValueError(f"No handover block for condition '{condition_name}' in the template.")
    return m.group(1).strip()


# -----------------------------------------------------------------------------
# Output filename + record schema
# -----------------------------------------------------------------------------

def whole_generation_filename(condition_name: str, model_name: str) -> str:
    """File name of one whole-model generation: ``{condition}_{model}.json``."""
    return f"{condition_name}_{model_name.lower()}.json"


class TruncatedGenerationError(RuntimeError):
    """A whole-model answer hit the output-token ceiling and is therefore incomplete.

    Raised by :func:`assert_not_truncated`. A truncated answer silently loses whole
    ``[FEATURE: ...]`` blocks, so its coverage score measures the token limit rather
    than the modality - the exact failure that invalidated the first ``04Ge`` vision
    run (see ``planning/korrekturen17_06.md`` / Schreibplan P0-1).
    """


def is_truncated(record: dict, *, assume_max_tokens: Optional[int] = None) -> bool:
    """True if *record* shows an output-token ceiling hit.

    Two independent signals, because neither alone is sufficient:

    * ``stop_reason == "max_tokens"`` - the API's own verdict, authoritative when present.
    * ``output_tokens >= max_tokens`` - the arithmetic fallback.

    Records written **before** the P0-1 fix carry neither field (the JSON/vision path
    persisted no ``stop_reason``, which is precisely why the truncation was invisible).
    Such a record cannot self-report, so the caller must supply the ceiling the old run
    used via ``assume_max_tokens`` (4096 for the first ``04Ge`` run); it is only applied
    when the record has no ``max_tokens`` of its own.
    """
    if record.get("stop_reason") == "max_tokens":
        return True
    cap = record.get("max_tokens") or assume_max_tokens
    out_tok = (record.get("usage") or {}).get("output_tokens")
    return bool(cap and out_tok and out_tok >= cap)


def assert_not_truncated(record: dict, *, assume_max_tokens: Optional[int] = None) -> dict:
    """Return *record*, or raise :class:`TruncatedGenerationError` if it was cut off.

    The hard gate for the billed ``04Ge`` run: a truncated answer must never reach
    ``results/global_whole/``, because downstream coverage/Achse-1/Achse-2 numbers
    would then report a token-limit artefact as a modality effect.
    """
    if is_truncated(record, assume_max_tokens=assume_max_tokens):
        usage = record.get("usage") or {}
        raise TruncatedGenerationError(
            f"{record.get('condition')}/{record.get('xai_model')} was truncated: "
            f"stop_reason={record.get('stop_reason')!r}, "
            f"output_tokens={usage.get('output_tokens')}, "
            f"max_tokens={record.get('max_tokens') or assume_max_tokens}. Raise "
            f"MAX_TOKENS and re-run this condition; do not score this record."
        )
    return record


def find_truncated_records(
    out_dir: Path | str, *, assume_max_tokens: Optional[int] = None
) -> list[Path]:
    """All persisted whole-model records in *out_dir* that hit the token ceiling.

    Audit helper for the invalidation step before a re-run: delete what this returns,
    then let :func:`run_resumable_whole_generation` recompute exactly those conditions.
    ``assume_max_tokens`` is forwarded to :func:`is_truncated` for pre-P0-1 records.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    hits = []
    for path in sorted(out_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if is_truncated(record, assume_max_tokens=assume_max_tokens):
            hits.append(path)
    return hits


# Every downstream directory a whole-model record feeds into. Invalidating a condition
# has to clear all of them: run_global_judge is idempotent, so a stale judge record for a
# regenerated answer would silently survive and be scored as if it were current.
WHOLE_JUDGE_SUBDIRS = ("global_whole_judge", "global_whole_judge_openai")


def invalidate_whole_condition(
    condition: str,
    model_name: str,
    *,
    results_dir: Path | str,
    judge_subdirs: Iterable[str] = WHOLE_JUDGE_SUBDIRS,
) -> list[Path]:
    """Delete the raw record, split records and judge scores for one (condition, model).

    Returns the deleted paths. Use before a re-run: the resumable loop then recomputes
    exactly this condition, ``write_split_records`` overwrites its split records, and
    ``run_global_judge`` re-scores them instead of loading the superseded verdicts.
    """
    results_dir = Path(results_dir)
    stem = f"{condition}_{model_name.lower()}"
    targets = [results_dir / WHOLE_RESULTS_SUBDIR / f"{stem}.json"]
    targets += sorted((results_dir / WHOLE_SPLIT_SUBDIR).glob(f"{stem}_*.json"))
    for sub in judge_subdirs:
        targets += sorted((results_dir / sub).glob(f"{stem}_*.json"))

    deleted = []
    for path in targets:
        if path.exists():
            path.unlink()
            deleted.append(path)
    return deleted


def build_whole_record(
    *,
    condition: WholeCondition,
    model_name: str,
    explanation: str,
    usage: dict,
    llm_model: str,
    loss_key: str = "poisson_log",
    elapsed_s: Optional[float] = None,
    include_cache: bool = True,
    stop_reason: Optional[str] = None,
    max_tokens: Optional[int] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Persisted record for one whole-model explanation.

    Carries the condition axes (modality/representation/mechanism) so the eval can slice
    by axis without re-deriving them. ``extra`` holds modality-specific fields (vision:
    ``plot_files``; tool-use: ``tool_calls`` / ``n_tool_calls``).

    ``stop_reason`` and ``max_tokens`` are written for **every** modality, not just
    tool-use. Without them a truncated answer is indistinguishable from a short one, and
    the resulting coverage gap reads as a modality effect (Schreibplan P0-1); with them
    :func:`is_truncated` can decide it from the record alone. ``stop_reason`` may also
    arrive inside ``extra`` (the tool-use loop's return value) - that value wins, so the
    existing tool-use call sites keep working unchanged.
    """
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    usage_d = {"input_tokens": in_tok, "output_tokens": out_tok}
    if include_cache:
        usage_d["cache_read_input_tokens"] = usage.get("cache_read_input_tokens", 0)

    record: dict[str, Any] = {
        "condition": condition.name,
        "scope": "global_whole",
        "modality": condition.modality,
        "representation": condition.representation,
        "mechanism": condition.mechanism,
        "llm_model": llm_model,
        "loss_key": loss_key,
        "xai_model": model_name.lower(),
        "explanation": explanation,
    }
    record["stop_reason"] = stop_reason
    record["max_tokens"] = max_tokens
    if extra:
        record.update(extra)
    record["elapsed_s"] = elapsed_s
    record["usage"] = usage_d
    return record


# -----------------------------------------------------------------------------
# The forced per-feature schema splitter (makes whole-model scoreable by G3)
# -----------------------------------------------------------------------------
# A block header is either a "[FEATURE ...]" marker or a markdown heading line. The
# feature name is resolved against the KNOWN feature list (word-boundary match), so a
# real LLM's cosmetic deviations - a parenthetical "[FEATURE: hr (hour of day)]",
# markdown emphasis "**[FEATURE: hr]**", a missing colon, or a bare heading "### hr" -
# are NOT mis-scored as a dropped feature. The coverage metric must reflect what the
# model described, not how it formatted the header (see planning/korrekturen17_06.md P3).
_BRACKET_HEADER = re.compile(r"\[\s*FEATURE\b[^\]]*\]", re.I)
_HEADING_LINE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*$", re.M)
_RECOMMENDATION = re.compile(r"\[\s*RECOMMENDATION\b[^\]]*\]", re.I)
_RECOMMENDATION_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+recommendation\b.*$", re.I | re.M)


def _resolve_feature(fragment: str, features: list[str]) -> Optional[str]:
    """Return the known feature named in a header fragment (word-boundary), or None.

    If several known names appear (unusual for a header), the earliest-occurring one
    wins, so the resolution is deterministic.
    """
    low = fragment.lower()
    hits = [f for f in features if re.search(rf"\b{re.escape(f)}\b", low)]
    if not hits:
        return None
    return min(hits, key=low.index)


def _find_feature_headers(body: str, features: list[str]) -> list[tuple[int, int, str]]:
    """Locate per-feature block headers as (start, header_end, feature), in order.

    Primary: "[FEATURE ...]" brackets (tolerant of colons/parentheticals/emphasis).
    Fallback: markdown heading lines whose text names a feature NOT already found via a
    bracket - covers answers that use "### hr" instead of the bracket schema. Guarding
    the fallback to still-uncovered features avoids double-counting and false positives
    from headings that merely mention an already-parsed feature.
    """
    headers: list[tuple[int, int, str]] = []
    covered: set[str] = set()
    for m in _BRACKET_HEADER.finditer(body):
        feat = _resolve_feature(m.group(0), features)
        if feat is not None:
            headers.append((m.start(), m.end(), feat))
            covered.add(feat)
    for m in _HEADING_LINE.finditer(body):
        if _BRACKET_HEADER.search(m.group(0)):
            continue  # heading already holds a bracket header (counted above)
        feat = _resolve_feature(m.group(1), features)
        if feat is not None and feat not in covered:
            headers.append((m.start(), m.end(), feat))
            covered.add(feat)
    headers.sort(key=lambda h: h[0])
    return headers


def split_whole_model_record(record: dict, *, features: Iterable[str]) -> list[dict]:
    """Split a whole-model record into per-feature records the G3 scorers accept.

    Each ``[FEATURE: x]`` block (containing ``[EFFECT]``/``[IMPORTANCE]``) becomes one
    record with ``form``/``xai_model``/``feature``/``explanation`` - the exact schema
    the G2a records use, so :func:`utils.rubric.score_result_file` and
    ``utils.eval.run_global_judge`` consume it unchanged. The single whole-model
    ``[RECOMMENDATION]`` is appended to every feature block (it is genuinely
    whole-model, so sharing keeps the completeness criterion comparable across
    conditions).

    A feature with **no** block in the answer yields an **empty** explanation: the
    honest "the whole-model answer dropped this feature" outcome - the rubric flags it
    incomplete and the judge scores it as a miss (the meeting's "only 5 of 9 right"
    problem, now measurable).
    """
    features = list(features)
    text = record["explanation"]

    rec_m = _RECOMMENDATION.search(text) or _RECOMMENDATION_HEADING.search(text)
    if rec_m is not None:
        shared_rec = text[rec_m.end():].strip()
        body = text[:rec_m.start()]
    else:
        shared_rec = ""
        body = text

    headers = _find_feature_headers(body, features)
    blocks: dict[str, str] = {}
    for i, (_h_start, h_end, feat) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(body)
        if feat not in blocks:                       # first occurrence of a feature wins
            blocks[feat] = body[h_end:end].strip()

    out: list[dict] = []
    for feature in features:
        block = blocks.get(feature, "")
        if block:
            explanation = block
            if not _RECOMMENDATION.search(block) and shared_rec:
                explanation = f"{block}\n[RECOMMENDATION] {shared_rec}"
        else:
            explanation = ""  # dropped feature -> scored as a miss
        out.append({
            "form": record["condition"],          # -> form_pipeline in the eval
            "scope": "global_whole",
            "llm_model": record["llm_model"],
            "loss_key": record.get("loss_key", "poisson_log"),
            "xai_model": record["xai_model"],
            "feature": feature,
            "explanation": explanation,
            "dropped": block == "",
            "condition": record["condition"],
            "modality": record.get("modality"),
            "representation": record.get("representation"),
            "mechanism": record.get("mechanism"),
        })
    return out


def write_split_records(
    records: Iterable[dict],
    *,
    features: Iterable[str],
    split_dir: Path | str,
) -> list[Path]:
    """Materialise per-feature split records for a batch of whole-model records.

    Writes ``{condition}_{model}_{feature}.json`` under ``split_dir`` (mirrors the
    G2a filename convention) so the existing rubric/judge glob picks them up. Returns
    the written paths. Idempotent by content (overwrites - splitting is deterministic).
    """
    features = list(features)
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for record in records:
        for pf in split_whole_model_record(record, features=features):
            path = split_dir / f"{pf['form']}_{pf['xai_model']}_{pf['feature']}.json"
            path.write_text(json.dumps(pf, indent=2, ensure_ascii=False))
            written.append(path)
    return written


# -----------------------------------------------------------------------------
# Resumable generation loop (unit = whole model, i.e. model x condition)
# -----------------------------------------------------------------------------
# (model, condition) -> record dict, or None to skip (error).
WholeGenerateFn = Callable[[str, WholeCondition], Optional[dict]]
WholeHookFn = Callable[[dict, str, WholeCondition], None]


def run_resumable_whole_generation(
    *,
    model_names: Iterable[str],
    conditions: Iterable[WholeCondition],
    out_dir: Path | str,
    generate: WholeGenerateFn,
    on_skip: Optional[WholeHookFn] = None,
    on_result: Optional[WholeHookFn] = None,
) -> list[dict]:
    """Run whole-model generation over all (model x condition) and persist.

    Same resume/idempotent/lossless/error-skip contract as
    :func:`utils.global_feature.run_resumable_global_generation`, only the iterated unit
    is (model, condition) and the file is ``{condition}_{model}.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for model_name in model_names:
        for condition in conditions:
            out_file = out_dir / whole_generation_filename(condition.name, model_name)
            if out_file.exists():
                record = json.loads(out_file.read_text())
                results.append(record)
                if on_skip is not None:
                    on_skip(record, model_name, condition)
                continue

            record = generate(model_name, condition)
            if record is None:
                continue

            out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False))
            results.append(record)
            if on_result is not None:
                on_result(record, model_name, condition)

    return results
