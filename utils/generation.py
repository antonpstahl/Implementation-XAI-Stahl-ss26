"""
utils/generation.py - shared generation building blocks for pipelines 04/05/06.

The three LLM pipelines previously inlined the same persistence/resume loop, the
explanation loading and the record building three times. That triplication is the
most expensive place for silent divergence, so it is centralised here:

  * `run_resumable_generation` - skip if exists loop that makes the n=20 run
    resumable after an API abort, lossless and idempotent (no double counting on
    re-run). The modality specific work (build prompt, call LLM) stays in the
    `generate` callback of each pipeline.
  * `load_local_explanation` / `load_global_explanation` - central explanation IO.
  * `build_generation_record` - one record schema for all three pipelines.

`n_generations == 1` keeps the file name scheme (`{model}_inst{iid}.json`); higher
values append `_gen{idx}` (prepared for possible repeated sampling runs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# generate(model_name, instance_id, generation_idx) -> record dict | None
#   Returns None to skip this generation (for example after an error in the
#   Tool Use loop, NB 04d), then nothing is persisted.
GenerateFn = Callable[[str, int, int], Optional[dict]]
HookFn = Callable[[dict, str, int, int], None]


def load_local_explanation(
    model_name: str,
    instance_id: int,
    *,
    loss_key: str = "poisson_log",
    explanations_dir: Path | str,
) -> dict:
    """Local SHAP/EBM explanation of a test instance (`local_{model}_{loss}_inst{id}.json`).

    Previously inlined three times in NB 04b/04c; centralised so the path scheme and
    loss key live in one place.
    """
    p = Path(explanations_dir) / f"local_{model_name}_{loss_key}_inst{instance_id}.json"
    return json.loads(p.read_text())


def load_global_explanation(
    model_name: str,
    *,
    loss_key: str = "poisson_log",
    explanations_dir: Path | str,
) -> dict:
    """Global feature importance of a model (`global_{model}_{loss}.json`)."""
    p = Path(explanations_dir) / f"global_{model_name}_{loss_key}.json"
    return json.loads(p.read_text())


_UNSET = object()


def build_generation_record(
    *,
    pipeline: str,
    model_name: str,
    instance_id: int,
    explanation: str,
    usage: dict,
    llm_model: str,
    loss_key: str,
    y_true: float,
    prediction: Any = _UNSET,
    elapsed_s: Optional[float] = None,
    include_cache: bool = True,
    extra: Optional[dict] = None,
) -> dict:
    """Build the persisted explanation record - one schema for all three pipelines.

    Reproduces the records previously inlined three times in NB 04b/04c/06 exactly
    (including key order), parametrised over the few real differences:

    * ``extra``         - modality specific fields, inserted directly after
                          ``explanation`` (NB 04c: ``plot_file``; NB 04d:
                          ``stop_reason`` / ``tool_calls`` / ``n_tool_calls``).
    * ``prediction``    - omitted when not passed (NB 04d carries no prediction in
                          the record).
    * ``include_cache`` - ``cache_read_input_tokens`` in ``usage`` (NB 04b/04c: yes;
                          NB 04d Tool Use: no).
    """
    in_tok  = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    usage_d = {"input_tokens": in_tok, "output_tokens": out_tok}
    if include_cache:
        usage_d["cache_read_input_tokens"] = usage.get("cache_read_input_tokens", 0)

    record = {
        "pipeline":    pipeline,
        "llm_model":   llm_model,
        "loss_key":    loss_key,
        "xai_model":   model_name,
        "instance_id": instance_id,
        "explanation": explanation,
    }
    if extra:
        record.update(extra)
    record["elapsed_s"] = elapsed_s
    record["usage"]     = usage_d
    if prediction is not _UNSET:
        record["prediction"] = prediction
    record["y_true"] = y_true
    return record


def generation_filename(
    model_name: str,
    instance_id: int,
    generation_idx: int = 0,
    n_generations: int = 1,
) -> str:
    """File name of a single generation.

    For `n_generations == 1` without a generation suffix (backward compatible with
    the already committed artefacts); from 2 on with `_gen{idx}`.
    """
    if n_generations == 1:
        return f"{model_name}_inst{instance_id}.json"
    return f"{model_name}_inst{instance_id}_gen{generation_idx}.json"


def run_resumable_generation(
    *,
    model_names: Iterable[str],
    instance_ids: Iterable[int],
    out_dir: Path | str,
    generate: GenerateFn,
    n_generations: int = 1,
    on_skip: Optional[HookFn] = None,
    on_result: Optional[HookFn] = None,
) -> list[dict]:
    """Run generation over all (model x instance x generation) and persist.

    Contract:
      * **Resume:** if the target file already exists it is loaded and the record
        appended to the result, no further `generate` call.
      * **Idempotency:** a second full run does not call `generate` again and
        produces no duplicates (same length, same records).
      * **Lossless:** each produced record is written as JSON immediately before
        moving to the next unit.
      * **Error skip:** if `generate` returns None, nothing is written and nothing
        appended (the unit stays open and is retried on the next run).

    Parameters
    ----------
    model_names    : XAI model keys, for example ["xgb", "ebm"].
    instance_ids   : test instance IDs (utils.INSTANCE_IDS).
    out_dir        : target directory; created if needed.
    generate       : callback that returns the record for (model, iid, gen_idx)
                     or None to skip.
    n_generations  : generations per instance. Default 1.
    on_skip        : optional hook (record, model, iid, gen_idx) on a resume skip.
    on_result      : optional hook (record, model, iid, gen_idx) after persistence.

    Returns
    -------
    list[dict] : all records in iteration order (loaded + newly produced).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for model_name in model_names:
        for iid in instance_ids:
            for gen_idx in range(n_generations):
                out_file = out_dir / generation_filename(
                    model_name, iid, gen_idx, n_generations
                )
                if out_file.exists():
                    record = json.loads(out_file.read_text())
                    results.append(record)
                    if on_skip is not None:
                        on_skip(record, model_name, iid, gen_idx)
                    continue

                record = generate(model_name, iid, gen_idx)
                if record is None:
                    continue

                out_file.write_text(
                    json.dumps(record, indent=2, ensure_ascii=False)
                )
                results.append(record)
                if on_result is not None:
                    on_result(record, model_name, iid, gen_idx)

    return results

