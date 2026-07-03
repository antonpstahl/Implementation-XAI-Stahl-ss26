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
