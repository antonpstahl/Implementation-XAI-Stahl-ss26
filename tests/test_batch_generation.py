"""
Phase 3a·B — Tests für den Batch-Generierungs-Pfad (utils.run_batch_generation).

Deckt ab:
  * **Skip-Logik:** nur fehlende Einheiten werden als **ein** Batch eingereicht;
    existierende Dateien werden geladen, nicht erneut angefragt.
  * **Voller Resume:** sind alle Einheiten vorhanden, wird kein Batch eingereicht.
  * **Verlustfreie Persistenz:** erfolgreiche Einträge landen unter denselben
    Dateinamen wie der Real-time-Loop (`generation_filename`).
  * **Fehler-Skip:** endgültig fehlgeschlagene Requests (invalid_request) und
    `build_request → None` lassen die Einheit offen → nächster Lauf wiederholt.
  * **Golden-Vergleich batch ↔ real-time:** Bei identischem (text, usage) erzeugen
    `run_batch_generation` und `run_resumable_generation` **byte-identische**
    On-Disk-Artefakte (DoD: „batch == real-time" auf 2 Instanzen).

Der Anthropic-Client wird durch den Fake aus test_batch ersetzt; `sleep` ist
injiziert, sodass kein echter Netz-/Wartecall stattfindet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.generation import (
    run_batch_generation,
    run_resumable_generation,
    generation_filename,
)
from utils.batch import make_custom_id
from utils.llm import strip_scratchpad

# Fake-Client + Ergebnis-Fabriken aus dem Batch-Helfer-Test wiederverwenden.
from tests.test_batch import (
    FakeBatches,
    make_client,
    _succeeded,
    _errored,
    NOSLEEP,
)


MODELS = ["xgb", "ebm"]
INSTANCES = [224, 580]
TAG = "gen04"
USAGE = {"input_tokens": 11, "output_tokens": 22}


# ── gemeinsamer build_request / build_record (wie in NB 04/05) ────────────────

def _text_of(model_name: str, iid: int) -> str:
    # Scratchpad-Präfix bewusst dabei, damit strip_scratchpad mitgetestet wird.
    return f"<analyse>rang</analyse>Erklärung-{model_name}-{iid}"


def build_request(model_name, iid, gen_idx):
    # Inhalt egal — der Fake-Client ignoriert die Params; nur die Shape zählt.
    return {"model": "claude-sonnet-4-6", "max_tokens": 64, "messages": []}


def build_record(model_name, iid, gen_idx, text, usage, elapsed_s=None):
    return {
        "pipeline":    "04_json",
        "xai_model":   model_name,
        "instance_id": iid,
        "explanation": strip_scratchpad(text),
        "elapsed_s":   elapsed_s,
        "usage": {
            "input_tokens":            usage.get("input_tokens", 0),
            "output_tokens":           usage.get("output_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        },
    }


def _cid(model_name, iid, gen_idx=0):
    return make_custom_id(TAG, model_name, iid, f"g{gen_idx}")


def _succeeded_for(units):
    """Baut die Batch-Ergebnisliste für die angegebenen (model, iid)-Einheiten."""
    return [
        _succeeded(_cid(m, i), text=_text_of(m, i),
                   in_tok=USAGE["input_tokens"], out_tok=USAGE["output_tokens"])
        for (m, i) in units
    ]


def _run_batch_gen(out_dir, batches, **kwargs):
    return run_batch_generation(
        model_names=MODELS,
        instance_ids=INSTANCES,
        out_dir=out_dir,
        build_request=build_request,
        build_record=build_record,
        pipeline_tag=TAG,
        client=make_client(batches),
        sleep=NOSLEEP,
        **kwargs,
    )


# ── 1. Skip-Logik: nur fehlende Einheiten in einem Batch ─────────────────────

def test_submits_all_missing_as_one_batch(tmp_path):
    all_units = [(m, i) for m in MODELS for i in INSTANCES]
    batches = FakeBatches().queue(_succeeded_for(all_units))

    results = _run_batch_gen(tmp_path, batches)

    assert len(batches.created) == 1                       # genau ein Batch
    assert set(batches.created[0][1]) == {_cid(m, i) for (m, i) in all_units}
    assert len(results) == len(all_units)
    # Eine Datei je Einheit unter dem kanonischen Namen.
    for m, i in all_units:
        assert (tmp_path / generation_filename(m, i)).exists()


def test_resume_only_submits_missing(tmp_path):
    # xgb-Einheiten vorab schreiben (simulierter Teilabbruch).
    for i in INSTANCES:
        (tmp_path / generation_filename("xgb", i)).write_text(
            json.dumps(build_record("xgb", i, 0, _text_of("xgb", i), USAGE))
        )

    missing = [("ebm", i) for i in INSTANCES]
    batches = FakeBatches().queue(_succeeded_for(missing))

    results = _run_batch_gen(tmp_path, batches)

    # Nur die fehlenden ebm-Einheiten werden eingereicht.
    assert set(batches.created[0][1]) == {_cid("ebm", i) for i in INSTANCES}
    assert len(results) == len(MODELS) * len(INSTANCES)


def test_full_resume_submits_no_batch(tmp_path):
    for m in MODELS:
        for i in INSTANCES:
            (tmp_path / generation_filename(m, i)).write_text(
                json.dumps(build_record(m, i, 0, _text_of(m, i), USAGE))
            )

    batches = FakeBatches()                                # keine Runden gequeued
    results = _run_batch_gen(tmp_path, batches)

    assert batches.created == []                           # kein create()-Call
    assert len(results) == len(MODELS) * len(INSTANCES)


# ── 2. Reihenfolge der Ergebnisse identisch zum Real-time-Loop ───────────────

def test_results_in_iteration_order(tmp_path):
    all_units = [(m, i) for m in MODELS for i in INSTANCES]
    batches = FakeBatches().queue(_succeeded_for(all_units))
    results = _run_batch_gen(tmp_path, batches)
    order = [(r["xai_model"], r["instance_id"]) for r in results]
    assert order == all_units


# ── 3. Fehler-Skip: invalid_request lässt die Einheit offen ──────────────────

def test_failed_unit_is_not_persisted_and_retried(tmp_path):
    all_units = [(m, i) for m in MODELS for i in INSTANCES]
    bad = ("ebm", INSTANCES[-1])
    round1 = [
        _succeeded(_cid(m, i), text=_text_of(m, i)) if (m, i) != bad
        else _errored(_cid(*bad), "invalid_request")
        for (m, i) in all_units
    ]
    batches = FakeBatches().queue(round1)

    results = _run_batch_gen(tmp_path, batches)

    assert not (tmp_path / generation_filename(*bad)).exists()
    assert len(results) == len(all_units) - 1
    # Zweiter Lauf reicht nur die offene Einheit erneut ein.
    batches.queue(_succeeded_for([bad]))
    results2 = _run_batch_gen(tmp_path, batches)
    assert set(batches.created[-1][1]) == {_cid(*bad)}
    assert (tmp_path / generation_filename(*bad)).exists()
    assert len(results2) == len(all_units)


def test_build_request_none_skips_unit(tmp_path):
    skip = ("ebm", INSTANCES[0])

    def build_request_skip(model_name, iid, gen_idx):
        if (model_name, iid) == skip:
            return None
        return build_request(model_name, iid, gen_idx)

    submitted = [(m, i) for m in MODELS for i in INSTANCES if (m, i) != skip]
    batches = FakeBatches().queue(_succeeded_for(submitted))

    results = run_batch_generation(
        model_names=MODELS, instance_ids=INSTANCES, out_dir=tmp_path,
        build_request=build_request_skip, build_record=build_record,
        pipeline_tag=TAG, client=make_client(batches), sleep=NOSLEEP,
    )

    assert skip not in {(r["xai_model"], r["instance_id"]) for r in results}
    assert not (tmp_path / generation_filename(*skip)).exists()
    assert set(batches.created[0][1]) == {_cid(m, i) for (m, i) in submitted}


# ── 4. Hooks ─────────────────────────────────────────────────────────────────

def test_on_result_then_on_skip(tmp_path):
    all_units = [(m, i) for m in MODELS for i in INSTANCES]
    batches = FakeBatches().queue(_succeeded_for(all_units))

    seen_result: list[tuple] = []
    _run_batch_gen(tmp_path, batches,
                   on_result=lambda r, m, i, g: seen_result.append((m, i)))
    assert sorted(seen_result) == sorted(all_units)

    seen_skip: list[tuple] = []
    _run_batch_gen(tmp_path, FakeBatches(),
                   on_skip=lambda r, m, i, g: seen_skip.append((m, i)))
    assert sorted(seen_skip) == sorted(all_units)


# ── 5. Golden-Vergleich: batch == real-time (DoD) ────────────────────────────

def test_batch_artifacts_identical_to_realtime(tmp_path):
    """Bei identischem (text, usage) sind die On-Disk-Dateien byte-identisch."""
    rt_dir = tmp_path / "realtime"
    bt_dir = tmp_path / "batch"

    # Real-time-Pfad: generate liefert den Record über denselben build_record
    # (elapsed_s=None, wie der Batch-Pfad — die Zeitmessung ist der einzige Punkt,
    # an dem sich die Pfade systematisch unterscheiden würden).
    def generate(model_name, iid, gen_idx):
        return build_record(model_name, iid, gen_idx, _text_of(model_name, iid), USAGE)

    run_resumable_generation(
        model_names=MODELS, instance_ids=INSTANCES, out_dir=rt_dir,
        generate=generate,
    )

    all_units = [(m, i) for m in MODELS for i in INSTANCES]
    batches = FakeBatches().queue(_succeeded_for(all_units))
    run_batch_generation(
        model_names=MODELS, instance_ids=INSTANCES, out_dir=bt_dir,
        build_request=build_request, build_record=build_record,
        pipeline_tag=TAG, client=make_client(batches), sleep=NOSLEEP,
    )

    rt_files = sorted(p.name for p in rt_dir.glob("*.json"))
    bt_files = sorted(p.name for p in bt_dir.glob("*.json"))
    assert rt_files == bt_files and rt_files                # gleiche Dateinamen
    for name in rt_files:
        assert (rt_dir / name).read_text() == (bt_dir / name).read_text()
