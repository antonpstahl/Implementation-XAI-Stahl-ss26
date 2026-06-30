"""
utils/generation.py – Gemeinsame Generierungs-Bausteine der Pipelines 04/05/06.

Die drei LLM-Pipelines teilten bisher denselben Persistenz-/Resume-Loop, das
Laden der Erklärungen und den Record-Aufbau dreifach inline. Diese Triplikation
ist die teuerste Stelle für stille Divergenz; hier zentralisiert:

  * `run_resumable_generation` – skip-if-exists-Loop, der den n=20-Lauf nach einem
    API-Abbruch wiederaufnehmbar, verlustfrei und idempotent (kein Doppelzählen
    beim Re-Run) macht. Die modalitätsspezifische Arbeit (Prompt bauen, LLM rufen)
    bleibt im `generate`-Callback der jeweiligen Pipeline.
  * `load_local_explanation` / `load_global_explanation` – zentrale Erklärungs-IO.
  * `build_generation_record` – ein Record-Schema für alle drei Pipelines.

`n_generations == 1` hält das Dateinamensschema (`{model}_inst{iid}.json`); höhere
Werte hängen `_gen{idx}` an (für etwaige Repeated-Sampling-Läufe vorbereitet).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# generate(model_name, instance_id, generation_idx) -> record dict | None
#   Gibt None zurück, um diese Generation zu überspringen (z. B. nach einem
#   Fehler in der Tool-Use-Schleife, NB 04d) — dann wird nichts persistiert.
GenerateFn = Callable[[str, int, int], Optional[dict]]
HookFn = Callable[[dict, str, int, int], None]


def load_local_explanation(
    model_name: str,
    instance_id: int,
    *,
    loss_key: str = "poisson_log",
    explanations_dir: Path | str,
) -> dict:
    """Lokale SHAP-/EBM-Erklärung einer Test-Instanz (`local_{model}_{loss}_inst{id}.json`).

    Bisher in NB 04b/04c dreifach inline geladen; zentralisiert, damit Pfadschema und
    Loss-Schlüssel an einer Stelle leben.
    """
    p = Path(explanations_dir) / f"local_{model_name}_{loss_key}_inst{instance_id}.json"
    return json.loads(p.read_text())


def load_global_explanation(
    model_name: str,
    *,
    loss_key: str = "poisson_log",
    explanations_dir: Path | str,
) -> dict:
    """Globale Feature-Importance eines Modells (`global_{model}_{loss}.json`)."""
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
    """Baut den persistierten Erklärungs-Record — ein Schema für alle drei Pipelines.

    Reproduziert die zuvor in NB 04b/04c/06 dreifach inline gebauten Records **exakt**
    (inkl. Key-Reihenfolge), parametrisiert über die wenigen echten Unterschiede:

    * ``extra``         — modalitätsspezifische Felder, direkt nach ``explanation``
                          eingefügt (NB 04c: ``plot_file``; NB 04d: ``stop_reason`` /
                          ``tool_calls`` / ``n_tool_calls``).
    * ``prediction``    — weggelassen, wenn nicht übergeben (NB 04d führt keine
                          Vorhersage im Record).
    * ``include_cache`` — ``cache_read_input_tokens`` in ``usage`` (NB 04b/04c: ja;
                          NB 04d Tool-Use: nein).
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
    """Dateiname einer einzelnen Generation.

    Bei `n_generations == 1` ohne Generations-Suffix (rückwärtskompatibel zu den
    bereits committeten Artefakten); ab 2 mit `_gen{idx}`.
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
    """Führt die Generierung über alle (Modell × Instanz × Generation) aus und persistiert.

    Kontrakt:
      * **Resume:** Existiert die Zieldatei bereits, wird sie geladen und der Record
        an das Ergebnis angehängt — kein erneuter `generate`-Aufruf.
      * **Idempotenz:** Ein zweiter vollständiger Lauf ruft `generate` kein weiteres
        Mal auf und erzeugt keine Duplikate (gleiche Länge, gleiche Records).
      * **Verlustfrei:** Jeder erzeugte Record wird sofort als JSON geschrieben,
        bevor zur nächsten Einheit gegangen wird.
      * **Fehler-Skip:** Gibt `generate` None zurück, wird nichts geschrieben und
        nichts angehängt (die Einheit bleibt offen und wird beim nächsten Lauf
        erneut versucht).

    Parameters
    ----------
    model_names    : XAI-Modell-Schlüssel, z. B. ["xgb", "ebm"].
    instance_ids   : Test-Instanz-IDs (utils.INSTANCE_IDS).
    out_dir        : Zielverzeichnis; wird bei Bedarf angelegt.
    generate       : Callback, das den Record für (model, iid, gen_idx) liefert
                     oder None zum Überspringen.
    n_generations  : Generationen pro Instanz (Phase 3b: 3). Default 1.
    on_skip        : optionaler Hook (record, model, iid, gen_idx) bei Resume-Skip.
    on_result      : optionaler Hook (record, model, iid, gen_idx) nach Persistenz.

    Returns
    -------
    list[dict] : alle Records in Iterationsreihenfolge (geladen + neu erzeugt).
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

