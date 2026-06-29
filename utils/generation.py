"""
utils/generation.py – Wiederaufnehmbare Generierungs-Schleife für die Pipelines 04/05/06.

Die drei LLM-Pipelines teilten bisher denselben Persistenz-/Resume-Loop dreifach
inline (skip-if-exists → laden, sonst generieren → schreiben). Diese Triplikation
ist die teuerste Stelle für stille Divergenz: der mehrstündige Phase-3b-Lauf muss
wiederaufnehmbar (resume nach Abbruch), verlustfrei und idempotent (kein
Doppelzählen beim Re-Run) sein.

`run_resumable_generation` kapselt genau diesen Kontrakt; die modalitätsspezifische
Arbeit (Prompt bauen, LLM rufen, Record bauen) bleibt im `generate`-Callback der
jeweiligen Pipeline. `n_generations` ist für Phase 3b vorbereitet (3 Generationen
pro Instanz); bei `n_generations == 1` bleibt das Dateinamensschema unverändert
(`{model}_inst{iid}.json`), sodass bestehende Artefakte weiter wiederaufgenommen
werden.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

# generate(model_name, instance_id, generation_idx) -> record dict | None
#   Gibt None zurück, um diese Generation zu überspringen (z. B. nach einem
#   Fehler in der Tool-Use-Schleife, NB 06) — dann wird nichts persistiert.
GenerateFn = Callable[[str, int, int], Optional[dict]]
HookFn = Callable[[dict, str, int, int], None]

# build_request(model_name, instance_id, generation_idx) -> messages.create-Params | None
#   Liefert die Request-Shape (utils.llm.build_text_params / build_image_params)
#   oder None, um diese Einheit zu überspringen (nicht batchen).
BuildRequestFn = Callable[[str, int, int], Optional[dict]]
# build_record(model_name, instance_id, generation_idx, text, usage) -> record dict
#   Baut denselben Record wie der Real-time-Pfad (Schema-Identität batch↔real-time).
BuildRecordFn = Callable[[str, int, int, str, dict], dict]


def load_local_explanation(
    model_name: str,
    instance_id: int,
    *,
    loss_key: str = "poisson_log",
    explanations_dir: Path | str,
) -> dict:
    """Lokale SHAP-/EBM-Erklärung einer Test-Instanz (`local_{model}_{loss}_inst{id}.json`).

    Bisher in NB 04/05 dreifach inline geladen; zentralisiert, damit Pfadschema und
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

    Reproduziert die zuvor in NB 04/05/06 dreifach inline gebauten Records **exakt**
    (inkl. Key-Reihenfolge), parametrisiert über die wenigen echten Unterschiede:

    * ``extra``         — modalitätsspezifische Felder, direkt nach ``explanation``
                          eingefügt (NB 05: ``plot_file``; NB 06: ``stop_reason`` /
                          ``tool_calls`` / ``n_tool_calls``).
    * ``prediction``    — weggelassen, wenn nicht übergeben (NB 06 führt keine
                          Vorhersage im Record).
    * ``include_cache`` — ``cache_read_input_tokens`` in ``usage`` (NB 04/05: ja;
                          NB 06 Tool-Use: nein).
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


def run_batch_generation(
    *,
    model_names: Iterable[str],
    instance_ids: Iterable[int],
    out_dir: Path | str,
    build_request: BuildRequestFn,
    build_record: BuildRecordFn,
    pipeline_tag: str,
    n_generations: int = 1,
    client: Any = None,
    state_path: Optional[Path | str] = None,
    on_skip: Optional[HookFn] = None,
    on_result: Optional[HookFn] = None,
    **run_batch_kwargs: Any,
) -> list[dict]:
    """Batch-Variante von :func:`run_resumable_generation` (Phase 3a·B).

    Statt jede Einheit einzeln real-time zu rufen, werden **alle fehlenden**
    Einheiten (dieselbe skip-if-exists-Logik wie im Real-time-Loop) als **ein**
    Anthropic-Batch eingereicht (−50 % Kosten) und die Ergebnisse in **dieselben
    Dateien** geschrieben. Damit die Eval (NB 07/08) ausführungsart-agnostisch
    bleibt, baut **derselbe** `build_record`-Callback den Record wie der
    Real-time-Pfad → schema-identische Artefakte (Golden-Vergleich batch↔real-time).

    Kontrakt (wie der Real-time-Loop):
      * **Resume:** Existierende Zieldateien werden geladen, nicht erneut
        angefragt; sind alle Einheiten vorhanden, wird **kein** Batch eingereicht.
      * **Verlustfrei:** Jeder erfolgreiche Batch-Eintrag wird als JSON
        geschrieben, bevor die Ergebnisliste gebaut wird.
      * **Fehler-Skip:** Einheiten mit `build_request → None` oder endgültig
        fehlgeschlagene Batch-Requests (invalid_request/canceled/erschöpfte
        Resubmits) bleiben ohne Datei und werden beim nächsten Lauf erneut
        versucht.

    Parameters
    ----------
    build_request   : (model, iid, gen) → messages.create-Params (utils.llm
                      build_text_params / build_image_params) oder None zum Skip.
    build_record    : (model, iid, gen, text, usage) → Record-Dict, **identisch**
                      zum Real-time-Pfad.
    pipeline_tag    : Präfix der `custom_id` (z. B. "gen04", "gen05").
    client          : Anthropic-Client (Tests injizieren einen Fake).
    state_path      : optionale `batch_id`-Persistenz (Poll-Resume nach Absturz).
    run_batch_kwargs: an `utils.batch.run_batch` durchgereicht (poll_interval_s,
                      max_resubmits, sleep, …).

    Returns
    -------
    list[dict] : alle vorhandenen Records in Iterationsreihenfolge (geladen + neu).
    """
    from utils.batch import make_custom_id, message_request, run_batch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Iterationsreihenfolge identisch zu run_resumable_generation.
    units: list[tuple[str, int, int, Path, bool]] = []
    requests: list[dict] = []
    cid_map: dict[str, tuple[str, int, int, Path]] = {}

    for model_name in model_names:
        for iid in instance_ids:
            for gen_idx in range(n_generations):
                out_file = out_dir / generation_filename(
                    model_name, iid, gen_idx, n_generations
                )
                existed = out_file.exists()
                units.append((model_name, iid, gen_idx, out_file, existed))
                if existed:
                    continue
                params = build_request(model_name, iid, gen_idx)
                if params is None:
                    continue
                cid = make_custom_id(pipeline_tag, model_name, iid, f"g{gen_idx}")
                if cid in cid_map:
                    raise ValueError(f"custom_id-Kollision: {cid!r}")
                cid_map[cid] = (model_name, iid, gen_idx, out_file)
                requests.append(message_request(cid, params))

    succeeded_cids: set[str] = set()
    if requests:
        outcome = run_batch(
            requests, client=client, state_path=state_path, **run_batch_kwargs
        )
        for cid, entry in outcome["succeeded"].items():
            model_name, iid, gen_idx, out_file = cid_map[cid]
            record = build_record(
                model_name, iid, gen_idx, entry["text"], entry["usage"]
            )
            out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False))
            succeeded_cids.add(cid)
        if outcome["failed"]:
            logger.error(
                "%d Batch-Request(s) endgültig fehlgeschlagen; Einheiten bleiben "
                "offen und werden beim nächsten Lauf erneut versucht: %s",
                len(outcome["failed"]), sorted(outcome["failed"]),
            )

    # Ergebnisse in Iterationsreihenfolge zusammenstellen (geladen + neu).
    results: list[dict] = []
    for model_name, iid, gen_idx, out_file, existed in units:
        if existed:
            record = json.loads(out_file.read_text())
            results.append(record)
            if on_skip is not None:
                on_skip(record, model_name, iid, gen_idx)
            continue
        cid = make_custom_id(pipeline_tag, model_name, iid, f"g{gen_idx}")
        if cid in succeeded_cids:
            record = json.loads(out_file.read_text())
            results.append(record)
            if on_result is not None:
                on_result(record, model_name, iid, gen_idx)

    return results
