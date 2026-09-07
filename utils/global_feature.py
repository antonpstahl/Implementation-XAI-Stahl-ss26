"""utils/global_feature.py - infrastructure for the *global, per-feature* LLM pipelines.

Phase G2a (revision 30.06): the description unit shifts from a single instance to a
single **feature** - the LLM describes the effect of one feature on the target
(hourly ``cnt``) at a time. Three delivery forms share this module:

  * **JSON**  (04Gb): the feature's shape/dependence curve values as a compact payload.
  * **Vision**(04Gc): the feature's shape/dependence PNG (path helpers here; the image
    request itself is ``utils.llm.ask_with_images``).
  * **Tool-Use** (04Gd): see :mod:`utils.global_tools` (GlobalToolBox + loop) which
    reuses the artifact loaders below.

Everything reads the pre-computed G0/G1 artifacts (no model needed at description time):
  * ``explanations/global_curve_{model}_{feature}.json`` -> ``{feature, kind, x, y, importance}``
  * ``explanations/global_{model}_{loss}.json``          -> ``global_importance`` (+ rank), task, metrics
  * ``explanations/plots/global/{ebm_shape|xgb_dependence}_{feature}.png`` and ``{model}_beeswarm.png``

The generation loop mirrors ``utils.generation.run_resumable_generation`` exactly (skip /
idempotent / lossless / error-skip), only the iterated unit is the feature.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# (model, feature, generation_idx) -> record dict, or None to skip (error).
GlobalGenerateFn = Callable[[str, str, int], Optional[dict]]
GlobalHookFn = Callable[[dict, str, str, int], None]

# Results of the global track live under results/global/ (plan G2), separate from the
# local n=20 track (results/04L*). Filenames are {form}_{model}_{feature}.json.
GLOBAL_RESULTS_SUBDIR = "global"


# -----------------------------------------------------------------------------
# Artifact paths
# -----------------------------------------------------------------------------

def shape_plot_filename(model_name: str, feature: str) -> str:
    """PNG file name of a feature's global plot.

    EBM ships a shape plot, XGB a SHAP dependence plot - both live in
    ``explanations/plots/global/`` and are the Vision-pipeline input for the feature.
    """
    model_name = model_name.lower()
    if model_name == "ebm":
        return f"ebm_shape_{feature}.png"
    if model_name == "xgb":
        return f"xgb_dependence_{feature}.png"
    raise ValueError(f"Unknown model_name '{model_name}' (expected 'ebm' or 'xgb').")


def shape_plot_path(model_name: str, feature: str, *, plots_dir: Path | str) -> Path:
    """Absolute path of a feature's global shape/dependence PNG."""
    return Path(plots_dir) / shape_plot_filename(model_name, feature)


def beeswarm_plot_path(model_name: str, *, plots_dir: Path | str) -> Path:
    """Absolute path of a model's global beeswarm PNG."""
    return Path(plots_dir) / f"{model_name.lower()}_beeswarm.png"


# -----------------------------------------------------------------------------
# Artifact loaders
# -----------------------------------------------------------------------------

def load_global_curve(
    model_name: str, feature: str, *, explanations_dir: Path | str
) -> dict:
    """Load one feature's global curve JSON (``{feature, kind, x, y, importance}``)."""
    p = Path(explanations_dir) / f"global_curve_{model_name.lower()}_{feature}.json"
    return json.loads(p.read_text())


def _global_json(model_name: str, *, explanations_dir: Path | str, loss_key: str) -> dict:
    p = Path(explanations_dir) / f"global_{model_name.lower()}_{loss_key}.json"
    return json.loads(p.read_text())


def feature_importance_map(
    model_name: str, *, explanations_dir: Path | str, loss_key: str = "poisson_log"
) -> dict[str, dict]:
    """``{feature: {"importance": float, "rank": int}}`` from the model's global JSON."""
    g = _global_json(model_name, explanations_dir=explanations_dir, loss_key=loss_key)
    return {
        item["feature"]: {"importance": item["importance"], "rank": item["rank"]}
        for item in g["global_importance"]
    }


def list_global_features(
    model_name: str, *, explanations_dir: Path | str, loss_key: str = "poisson_log"
) -> list[str]:
    """All features in global-importance order (rank 1 first) - the loop unit list."""
    g = _global_json(model_name, explanations_dir=explanations_dir, loss_key=loss_key)
    return [item["feature"] for item in g["global_importance"]]


# -----------------------------------------------------------------------------
# Curve derivation (shared source of truth: deterministic baseline 04Ga + G3 ground truth)
# -----------------------------------------------------------------------------
# Discrete-coded features: readable_feature_value truncates via int(), so continuous
# shape-curve x-grid points (bin midpoints like 0.75) are rounded to hit the right label.
_DISCRETE_FEATURES = {"hr", "yr", "mnth", "weekday", "weathersit", "holiday"}
_WEEKDAYS = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
             4: "Thursday", 5: "Friday", 6: "Saturday"}
_MONTHS = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
           7: "July", 8: "August", 9: "September", 10: "October", 11: "November",
           12: "December"}
_WEATHER = {1: "clear/few clouds", 2: "mist/cloudy", 3: "light rain/snow",
            4: "heavy rain/thunderstorm"}


def readable_feature_value(feature: str, raw: float) -> str:
    """Denormalise a raw feature value into a readable string (e.g. hr 7 -> ``07:00``).

    Single source of the denormalisation used by the global template baseline and the
    G3 ground truth, so both verbalise feature values identically.
    """
    if feature == "hr":
        return f"{int(raw):02d}:00"
    if feature == "temp":
        return f"~{raw * 41:.1f} C"
    if feature == "hum":
        return f"{raw * 100:.0f} %"
    if feature == "windspeed":
        return f"{raw * 67:.1f} km/h"
    if feature == "yr":
        return "2011" if int(raw) == 0 else "2012"
    if feature == "mnth":
        return _MONTHS.get(int(raw), str(int(raw)))
    if feature == "weekday":
        return _WEEKDAYS.get(int(raw), str(int(raw)))
    if feature == "weathersit":
        return _WEATHER.get(int(raw), str(int(raw)))
    if feature == "holiday":
        return "holiday" if int(raw) == 1 else "no holiday"
    return str(raw)


# Canonical classification thresholds: the SINGLE source of truth shared by the
# deterministic baseline (describe_curve) and the G3 ground truth (utils.groundtruth),
# so the two can never disagree on the form type (planning/korrekturen17_06.md P2).
# Values are the GT authority's (utils.groundtruth.Thresholds): near-flat by *global
# importance*, monotonic by *reversal share*, NOT by curve amplitude.
_FLAT_IMPORTANCE = 0.025   # global importance below this => near-flat
_MONO_TOL = 0.15           # reversal share (movement against the net) tolerated as monotone


def aggregate_curve(x: list, y: list) -> tuple[list, list]:
    """Group duplicate x, average y, return a sorted-by-x grid.

    EBM shape functions are already a unique grid (no-op ordering); XGB SHAP scatter has
    duplicate x (many instances per value) and is collapsed to a grid so the monotonicity
    / peak logic sees the mean trend, not the per-instance noise. Categorical string x is
    kept as-is (sorted numerically where possible).
    """
    buckets: dict = {}
    for xi, yi in zip(x, y):
        buckets.setdefault(xi, []).append(float(yi))

    def _key(k):
        try:
            return (0, float(k))
        except (TypeError, ValueError):
            return (1, str(k))

    keys = sorted(buckets, key=_key)
    return keys, [sum(v) / len(v) for v in (buckets[k] for k in keys)]


def _total_variation(y: list[float]) -> float:
    return sum(abs(y[i + 1] - y[i]) for i in range(len(y) - 1))


def _signed_movement(y: list[float]) -> tuple[float, float]:
    up = down = 0.0
    for i in range(len(y) - 1):
        d = y[i + 1] - y[i]
        if d >= 0:
            up += d
        else:
            down += -d
    return up, down


def classify_shape(
    ys: list[float],
    kind: str,
    importance: float,
    *,
    flat_importance: float = _FLAT_IMPORTANCE,
    mono_tol: float = _MONO_TOL,
) -> dict:
    """Canonical form / direction / monotonicity of an *aggregated* curve.

    The single classifier shared by :func:`describe_curve` (deterministic baseline 04Ga)
    and :mod:`utils.groundtruth` (G3 ground truth), so both label a feature's form
    identically (planning/korrekturen17_06.md P2). Criteria follow the GT authority:
    near-flat by **global importance** (``< flat_importance``), monotonic by **reversal
    share** (``<= mono_tol``). Direction / monotonicity are computed independently of the
    near-flat override, so a near-flat feature still carries its underlying sign.

    Returns ``{form, direction, monotonicity}`` (canonical vocab):
      * ``form``         : ``near-flat`` | ``monotonic`` | ``non-monotonic`` | ``categorical``.
      * ``direction``    : ``increasing`` | ``decreasing`` | ``mixed`` | ``categorical``.
      * ``monotonicity`` : ``monotonic_increasing`` | ``monotonic_decreasing`` |
                           ``non_monotonic`` | ``n/a``.
    """
    if kind == "categorical":
        direction, monotonicity = "categorical", "n/a"
    else:
        tv = _total_variation(ys)
        up, down = _signed_movement(ys)
        net = ys[-1] - ys[0] if len(ys) >= 2 else 0.0
        reversal = min(up, down) / tv if tv > 0 else 0.0
        if reversal <= mono_tol:
            direction = "increasing" if net >= 0 else "decreasing"
            monotonicity = f"monotonic_{direction}"
        else:
            direction, monotonicity = "mixed", "non_monotonic"

    if importance < flat_importance:
        form = "near-flat"
    elif kind == "categorical":
        form = "categorical"
    else:
        form = "monotonic" if monotonicity.startswith("monotonic") else "non-monotonic"
    return {"form": form, "direction": direction, "monotonicity": monotonicity}


# Canonical -> describe_curve's compact vocab (kept for the 04Ga template + tests).
_SHAPE_VOCAB = {"near-flat": "near_flat", "non-monotonic": "non_monotonic",
                "monotonic": "monotonic", "categorical": "categorical"}
_DIRECTION_VOCAB = {"increasing": "rising", "decreasing": "falling",
                    "mixed": "mixed", "categorical": "mixed"}


def describe_curve(
    model_name: str,
    feature: str,
    *,
    explanations_dir: Path | str,
    flat_importance: float = _FLAT_IMPORTANCE,
    mono_tol: float = _MONO_TOL,
) -> dict:
    """Derive the structural facts of a feature's global curve (deterministic baseline).

    Uses the shared :func:`classify_shape` (importance-based near-flat, reversal-share
    monotonicity, on the aggregated curve), so the form type is **identical** to the G3
    ground truth (:mod:`utils.groundtruth`), the baseline is scored against the same
    classification it was written from (planning/korrekturen17_06.md P2). A near-flat
    feature keeps its underlying ``direction`` but reports ``shape="near_flat"``.

    Returns ``{feature, kind, direction, monotonicity, shape, peak_x, peak_value,
    peak_label}``:
      * ``direction``    : ``rising`` | ``falling`` | ``mixed`` (| ``flat``).
      * ``monotonicity`` : ``monotonic`` | ``non_monotonic`` (| ``flat``).
      * ``shape``        : ``monotonic`` | ``non_monotonic`` | ``categorical`` |
                           ``near_flat``, the coarse form type for the G3 stratification.
      * ``peak_*``       : the x (raw + readable) and y where the contribution is highest.
    """
    curve = load_global_curve(model_name, feature, explanations_dir=explanations_dir)
    kind = curve["kind"]
    importance = float(curve.get("importance", 0.0))
    xs, ys = aggregate_curve([float(v) for v in curve["x"]],
                             [float(v) for v in curve["y"]])
    n = len(ys)

    cls = classify_shape(ys, kind, importance,
                         flat_importance=flat_importance, mono_tol=mono_tol)
    shape = _SHAPE_VOCAB[cls["form"]]
    direction = _DIRECTION_VOCAB[cls["direction"]]
    monotonicity = "monotonic" if cls["monotonicity"].startswith("monotonic") else "non_monotonic"

    peak_i = max(range(n), key=lambda i: ys[i]) if n else 0
    peak_raw = xs[peak_i] if n else 0.0
    if feature in _DISCRETE_FEATURES:
        peak_raw = round(peak_raw)

    return {
        "feature": feature,
        "kind": kind,
        "direction": direction,
        "monotonicity": monotonicity,
        "shape": shape,
        "peak_x": peak_raw,
        "peak_value": ys[peak_i] if n else None,
        "peak_label": readable_feature_value(feature, peak_raw),
    }


# -----------------------------------------------------------------------------
# JSON pipeline payload (04Gb)
# -----------------------------------------------------------------------------

def build_feature_json_payload(
    model_name: str,
    feature: str,
    *,
    explanations_dir: Path | str,
    loss_key: str = "poisson_log",
) -> dict:
    """Compact per-feature payload for the JSON pipeline (04Gb).

    Faithful to the G0 curve (the ground-truth source): the raw ``x`` grid and the
    ``y`` contributions (log space), plus importance/rank context and the target
    description so the model knows *what* the effect acts on. No denormalisation -
    the values match the plot the Vision pipeline sees, keeping the two forms
    information-matched.
    """
    g = _global_json(model_name, explanations_dir=explanations_dir, loss_key=loss_key)
    curve = load_global_curve(model_name, feature, explanations_dir=explanations_dir)
    imp = feature_importance_map(
        model_name, explanations_dir=explanations_dir, loss_key=loss_key
    )[feature]
    return {
        "target": g["task"],                       # name / description / type of cnt
        "value_space": "contribution to the target in log space (exp -> multiplicative on cnt)",
        "model": model_name.lower(),
        "feature": feature,
        "kind": curve["kind"],                      # continuous | categorical
        "importance": imp["importance"],
        "rank": imp["rank"],
        "n_features": len(g["global_importance"]),
        "curve": {"x": curve["x"], "y": curve["y"]},
    }


# -----------------------------------------------------------------------------
# Prompt assembly (shared by 04Gb / 04Gc / 04Gd)
# -----------------------------------------------------------------------------
# The single prompt template prompts/global_feature.md has a byte-identical core plus
# per-modality handover blocks. Parse it once here so the three notebooks don't each
# re-implement the markdown extraction (and can't drift apart).
MODEL_LABELS = {"ebm": "EBM", "xgb": "XGBoost"}
ARTIFACT_LABELS = {"ebm": "EBM shape plot", "xgb": "XGBoost SHAP dependence plot"}


def _handover_block(variants_section: str, form: str) -> str:
    m = re.search(
        r"^##\s+" + re.escape(form) + r"\b[^\n]*\n```\n(.*?)\n```",
        variants_section, re.DOTALL | re.MULTILINE,
    )
    if m is None:
        raise ValueError(f"No handover block for form '{form}' in the prompt template.")
    return m.group(1).strip()


def assemble_global_system_prompt(
    form: str,
    model_name: str,
    *,
    prompts_dir: Path | str,
    prompt_file: str = "global_feature.md",
) -> str:
    """Build the system prompt for one (form, model) from ``global_feature.md``.

    Splices the modality's handover block into the shared core and fills {{MODEL}}
    (and, for vision, {{ARTIFACT}}). ``form`` in {"json", "vision", "tooluse"}.
    Raises if any ``{{placeholder}}`` is left unresolved (catches template drift).
    The per-feature {{FEATURE}}/{{HANDOVER}} placeholders live in the USER MESSAGE, not
    here; the notebook builds that at call time.
    """
    md = (Path(prompts_dir) / prompt_file).read_text()
    try:
        core = md.split("# SYSTEM PROMPT CORE", 1)[1].split("# HANDOVER FORMAT variants", 1)[0]
        variants = md.split("# HANDOVER FORMAT variants", 1)[1].split("# USER MESSAGE pattern", 1)[0]
    except IndexError as exc:  # pragma: no cover - template shape guard
        raise ValueError("global_feature.md is missing an expected section header.") from exc

    core = core[core.index("\n") + 1:]                  # drop the rest of the header line
    core = re.sub(r"\n-{3,}\s*$", "", core.strip())     # drop trailing '---' separator

    model_key = model_name.lower()
    if model_key not in MODEL_LABELS:
        raise ValueError(f"Unknown model_name '{model_name}' (expected 'ebm' or 'xgb').")

    sys_prompt = (
        core.replace("{{HANDOVER_FORMAT}}", _handover_block(variants, form))
        .replace("{{MODEL}}", MODEL_LABELS[model_key])
    )
    if "{{ARTIFACT}}" in sys_prompt:
        sys_prompt = sys_prompt.replace("{{ARTIFACT}}", ARTIFACT_LABELS[model_key])

    leftover = re.findall(r"\{\{[^}]+\}\}", sys_prompt)
    if leftover:
        raise ValueError(f"Unresolved placeholders in assembled system prompt: {leftover}")
    return sys_prompt.strip()


# -----------------------------------------------------------------------------
# Output filename + record schema
# -----------------------------------------------------------------------------

def global_generation_filename(
    form: str,
    model_name: str,
    feature: str,
    generation_idx: int = 0,
    n_generations: int = 1,
) -> str:
    """File name of one global generation: ``{form}_{model}_{feature}[_gen{idx}].json``.

    ``form`` is the delivery modality: ``"json"`` | ``"vision"`` | ``"tooluse"``.
    Mirrors :func:`utils.generation.generation_filename`: no ``_gen`` suffix for a
    single generation (backward-friendly), suffixed from 2 on.
    """
    base = f"{form}_{model_name.lower()}_{feature}"
    if n_generations == 1:
        return f"{base}.json"
    return f"{base}_gen{generation_idx}.json"


def seed_generation_zero(
    *,
    form: str,
    model_names: Iterable[str],
    features: Iterable[str],
    src_dir: Path | str,
    out_dir: Path | str,
    n_generations: int,
) -> list[Path]:
    """Copy each frozen single-generation record in as generation 0 of a variance run.

    A multi-draw variance study (P1-2) does not need to pay for the first draw: the
    existing ``{form}_{model}_{feature}.json`` was produced by the same prompt and code
    path, so it is a legitimate draw. Copying it to ``..._gen0.json`` cuts the billed
    calls from ``k`` to ``k-1`` per cell and anchors the spread to the exact records the
    reported results came from.

    Idempotent: an existing generation-0 file is left untouched, so a re-run never
    replaces a draw that later draws were already compared against. Missing sources are
    skipped and returned separately by the caller's own check, not raised - a partial
    subset is a legitimate state while earlier notebooks are still running.

    Returns the paths newly written.
    """
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model_name in model_names:
        for feature in features:
            src = src_dir / global_generation_filename(form, model_name, feature)
            dst = out_dir / global_generation_filename(
                form, model_name, feature, 0, n_generations
            )
            if not src.exists() or dst.exists():
                continue
            dst.write_text(src.read_text())
            written.append(dst)
    return written


def build_global_record(
    *,
    form: str,
    model_name: str,
    feature: str,
    explanation: str,
    usage: dict,
    llm_model: str,
    loss_key: str = "poisson_log",
    elapsed_s: Optional[float] = None,
    include_cache: bool = True,
    extra: Optional[dict] = None,
) -> dict:
    """Persisted record for one global per-feature explanation.

    Global analogue of :func:`utils.generation.build_generation_record`: no
    ``instance_id`` / ``y_true`` / ``prediction`` (there is no single instance), but a
    ``scope="global"`` marker, the ``form`` and the ``feature``. ``extra`` carries
    modality-specific fields (Vision: ``plot_file``; Tool-Use: ``stop_reason`` /
    ``tool_calls`` / ``n_tool_calls``), inserted right after ``explanation``.
    """
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    usage_d = {"input_tokens": in_tok, "output_tokens": out_tok}
    if include_cache:
        usage_d["cache_read_input_tokens"] = usage.get("cache_read_input_tokens", 0)

    record: dict[str, Any] = {
        "form": form,
        "scope": "global",
        "llm_model": llm_model,
        "loss_key": loss_key,
        "xai_model": model_name.lower(),
        "feature": feature,
        "explanation": explanation,
    }
    if extra:
        record.update(extra)
    record["elapsed_s"] = elapsed_s
    record["usage"] = usage_d
    return record


# -----------------------------------------------------------------------------
# Resumable generation loop (unit = feature)
# -----------------------------------------------------------------------------

def run_resumable_global_generation(
    *,
    form: str,
    model_names: Iterable[str],
    features: Iterable[str],
    out_dir: Path | str,
    generate: GlobalGenerateFn,
    n_generations: int = 1,
    on_skip: Optional[GlobalHookFn] = None,
    on_result: Optional[GlobalHookFn] = None,
) -> list[dict]:
    """Run generation over all (model x feature x generation) and persist.

    Same contract as :func:`utils.generation.run_resumable_generation`, only the unit
    is the feature:
      * **Resume:** existing target file is loaded and appended, no ``generate`` call.
      * **Idempotency:** a second full run produces no duplicates.
      * **Lossless:** each record is written immediately before the next unit.
      * **Error skip:** ``generate`` returning None writes nothing (unit stays open).

    Parameters
    ----------
    form           : delivery modality, e.g. "json" | "vision" | "tooluse".
    model_names    : XAI model keys, for example ["xgb", "ebm"].
    features       : feature names (see :func:`list_global_features`).
    out_dir        : target directory (typically ``RESULTS_DIR / "global"``); created.
    generate       : callback returning the record for (model, feature, gen_idx), or
                     None to skip.
    n_generations  : generations per feature. Default 1.
    on_skip        : optional hook (record, model, feature, gen_idx) on a resume skip.
    on_result      : optional hook (record, model, feature, gen_idx) after persistence.

    Returns
    -------
    list[dict] : all records in iteration order (loaded + newly produced).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for model_name in model_names:
        for feature in features:
            for gen_idx in range(n_generations):
                out_file = out_dir / global_generation_filename(
                    form, model_name, feature, gen_idx, n_generations
                )
                if out_file.exists():
                    record = json.loads(out_file.read_text())
                    results.append(record)
                    if on_skip is not None:
                        on_skip(record, model_name, feature, gen_idx)
                    continue

                record = generate(model_name, feature, gen_idx)
                if record is None:
                    continue

                out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False))
                results.append(record)
                if on_result is not None:
                    on_result(record, model_name, feature, gen_idx)

    return results
