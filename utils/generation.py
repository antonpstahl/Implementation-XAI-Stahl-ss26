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
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# generate(model_name, instance_id, generation_idx) -> record dict | None
#   Gibt None zurück, um diese Generation zu überspringen (z. B. nach einem
#   Fehler in der Tool-Use-Schleife, NB 06) — dann wird nichts persistiert.
GenerateFn = Callable[[str, int, int], Optional[dict]]
HookFn = Callable[[dict, str, int, int], None]


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
