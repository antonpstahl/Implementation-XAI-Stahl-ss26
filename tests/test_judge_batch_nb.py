"""
Phase 3a·B — NB07 Batch-Judge Roundtrip-Tests.

Prüft den custom_id-Roundtrip und die Schema-Gleichheit zwischen dem Batch-
und dem Real-time-Pfad des Judge-Laufs in NB07 (Zellen 10/18/21/34 — v1–v4).

Abgedeckte Szenarien:
  * custom_id-Format: make_custom_id("jdg", version, pipeline_label, xai, iid)
    ist gültig und eindeutig für alle NB07-Pipeline-Labels (inkl. Sonderzeichen
    wie "→" in "JSON→Text" und "-" in "Tool-Use").
  * Ergebnis-Mapping: zip(entries, df_rows) ordnet base_cid korrekt der Zeile
    zu — Reihenfolge bleibt erhalten.
  * Schema-Gleichheit: Batch-Zeilen haben dieselben Keys wie der Real-time-Loop.
  * Judge_n == n wenn alle SC-Samples erfolgreich sind.
  * Judge_n < n wenn eine Basis-CID keine erfolgreichen Samples hat (None-Scores
    werden von pandas .count() ausgeschlossen).
  * Partial-Failure: verbleibende Einträge bleiben korrekt im DataFrame.
  * Opus-Modell: temperature wird weggelassen (Gating wie in den NB07-Zellen).
  * k×len(entries) Requests in einem Batch (k=JUDGE_SC_K).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.batch import make_custom_id
from utils.judge import judge_batch_sc, SCORE_KEYS
from tests.test_batch import FakeBatches, make_client, NOSLEEP
from tests.test_judge_batch import _xml, _mk_succeeded, _mk_errored

# ── NB07-Konstanten (Spiegel der Notebook-Konfiguration) ─────────────────────

PIPELINE_LABELS = {
    '00': 'Template',
    '04': 'JSON→Text',
    '05': 'Vision',
    '06': 'Tool-Use',
}
PIPELINES   = ['00', '04', '05', '06']
XAI_MODELS  = ['xgb', 'ebm']
INSTANCE_IDS = [42, 100]   # kleines Sample für schnelle Tests
K = 3
SONNET = 'claude-sonnet-4-6'
OPUS   = 'claude-opus-4-8'


# ── Hilfsfunktionen (NB07-Zellen-Logik) ─────────────────────────────────────

def _build_rows(pipelines=PIPELINES, xai_models=XAI_MODELS, instance_ids=INSTANCE_IDS):
    """Simuliert df.iterrows() aus NB07 (ohne echten DataFrame)."""
    return [
        {
            'pipeline_label': PIPELINE_LABELS[p],
            'xai_model':      xai,
            'instance_id':    iid,
        }
        for p in pipelines
        for xai in xai_models
        for iid in instance_ids
    ]


def _entries(rows, version='v1'):
    """Baut die entries-Liste wie in NB07-Zellen 10/18/21/34."""
    return [
        (
            make_custom_id("jdg", version,
                           row['pipeline_label'], row['xai_model'], row['instance_id']),
            f"prompt_{row['instance_id']}",
        )
        for row in rows
    ]


def _judge_rows_from_sc(sc: dict, entries, rows) -> list[dict]:
    """Rekonstruiert die _rows-Liste wie in NB07-Zellen 10/18/21/34."""
    result = []
    for (base_cid, _), row in zip(entries, rows):
        r = sc.get(base_cid, {})
        result.append({
            'pipeline_label': row['pipeline_label'],
            'xai_model':      row['xai_model'],
            'instance_id':    row['instance_id'],
            'faithfulness':   r.get('faithfulness'),
            'clarity':        r.get('clarity'),
            'completeness':   r.get('completeness'),
            'reasoning': {
                'faithfulness': r.get('faithfulness_reasoning', ''),
                'clarity':      r.get('clarity_reasoning', ''),
                'completeness': r.get('completeness_reasoning', ''),
            },
            'raw_response': (
                r.get('raw_responses', [''])[0] if r.get('raw_responses') else ''
            ),
        })
    return result


def _run_sc(batches, entries, model=SONNET, k=K, temperature=0.0):
    return judge_batch_sc(
        entries,
        system='sys',
        model=model,
        max_tokens=900,
        k=k,
        temperature=temperature,
        client=make_client(batches),
        sleep=NOSLEEP,
    )


# ── 1. custom_id-Format: gültig + eindeutig für alle Pipeline-Labels ─────────

def test_nb07_custom_ids_valid_and_unique_all_versions():
    """make_custom_id('jdg', version, pipeline_label, xai, iid) → kein Fehler, alle eindeutig."""
    import re
    _RE = re.compile(r'^[A-Za-z0-9_-]{1,64}$')
    rows = _build_rows()
    for version in ('v1', 'v2', 'v3', 'v4'):
        ents = _entries(rows, version)
        cids = [cid for cid, _ in ents]
        for cid in cids:
            assert _RE.match(cid), f"Ungültige custom_id: {cid!r}"
            assert len(cid) <= 61, f"custom_id zu lang für k=3: {cid!r} ({len(cid)} Z.)"
        assert len(set(cids)) == len(cids), \
            f"Doppelte custom_ids in Version {version}"


def test_nb07_special_pipeline_labels_in_custom_id():
    """Sonderzeichen in Pipeline-Labels ('→', '-') werden korrekt ersetzt/beibehalten."""
    cid_json   = make_custom_id("jdg", "v1", "JSON→Text", "xgb", 42)
    cid_tool   = make_custom_id("jdg", "v1", "Tool-Use",  "xgb", 42)
    cid_tmpl   = make_custom_id("jdg", "v1", "Template",  "xgb", 42)
    cid_vision = make_custom_id("jdg", "v1", "Vision",    "xgb", 42)

    assert "JSON_Text" in cid_json   # '→' → '_'
    assert "Tool-Use"  in cid_tool   # '-' bleibt
    assert "Template"  in cid_tmpl
    assert "Vision"    in cid_vision

    # Alle vier eindeutig
    assert len({cid_json, cid_tool, cid_tmpl, cid_vision}) == 4


# ── 2. Ergebnis-Mapping: Reihenfolge bleibt erhalten ─────────────────────────

def test_nb07_result_mapping_order_preserved():
    """zip(entries, rows) ordnet Ergebnis-CIDs korrekt den Original-Zeilen zu."""
    rows = _build_rows(pipelines=['04', '05'], instance_ids=[42, 100])
    ents = _entries(rows, 'v1')
    results = [_mk_succeeded(f"{cid}-s{j}", 4, 3, 5)
               for cid, _ in ents for j in range(K)]
    batches = FakeBatches().queue(results)

    sc = _run_sc(batches, ents)
    judge_rows = _judge_rows_from_sc(sc, ents, rows)

    assert len(judge_rows) == len(rows)
    for jr, orig in zip(judge_rows, rows):
        assert jr['pipeline_label'] == orig['pipeline_label']
        assert jr['xai_model']      == orig['xai_model']
        assert jr['instance_id']    == orig['instance_id']


# ── 3. Schema-Gleichheit batch vs. real-time ─────────────────────────────────

def test_nb07_batch_rows_have_realtime_schema():
    """Jede Batch-Zeile hat dieselben Keys wie der Real-time-Loop in NB07."""
    rows = _build_rows()
    ents = _entries(rows, 'v1')
    results = [_mk_succeeded(f"{cid}-s{j}", 4, 3, 5)
               for cid, _ in ents for j in range(K)]
    batches = FakeBatches().queue(results)

    sc = _run_sc(batches, ents)
    judge_rows = _judge_rows_from_sc(sc, ents, rows)

    EXPECTED_KEYS = {
        'pipeline_label', 'xai_model', 'instance_id',
        'faithfulness', 'clarity', 'completeness',
        'reasoning', 'raw_response',
    }
    for row in judge_rows:
        assert set(row.keys()) == EXPECTED_KEYS
        assert isinstance(row['reasoning'], dict)
        assert set(row['reasoning'].keys()) == {'faithfulness', 'clarity', 'completeness'}


# ── 4. Judge_n == n wenn alle Samples erfolgreich ────────────────────────────

def test_nb07_judge_n_equals_n_all_succeed():
    """Alle Scores nicht-None → Judge_n == n (pandas .count() zählt alle Zeilen)."""
    rows = _build_rows()
    ents = _entries(rows, 'v3')
    results = [_mk_succeeded(f"{cid}-s{j}", 4, 3, 5)
               for cid, _ in ents for j in range(K)]
    batches = FakeBatches().queue(results)

    sc = _run_sc(batches, ents, model=OPUS, temperature=None)
    judge_rows = _judge_rows_from_sc(sc, ents, rows)

    df = pd.DataFrame(judge_rows)
    judge_n = df.groupby('pipeline_label')[['faithfulness']].count()

    expected_per_pipeline = len(XAI_MODELS) * len(INSTANCE_IDS)
    for label in PIPELINE_LABELS.values():
        assert judge_n.loc[label, 'faithfulness'] == expected_per_pipeline, \
            f"Judge_n falsch für {label}: {judge_n.loc[label, 'faithfulness']}"


# ── 5. Judge_n < n wenn ein Eintrag komplett fehlschlägt ─────────────────────

def test_nb07_judge_n_excludes_none_scores():
    """Scheitern aller SC-Samples → None-Scores → nicht in Judge_n gezählt."""
    rows = _build_rows(pipelines=['04'], xai_models=['xgb'], instance_ids=[42, 100])
    ents = _entries(rows, 'v1')

    # Erster Eintrag schlägt komplett fehl; zweiter gelingt.
    failed_cid  = ents[0][0]
    success_cid = ents[1][0]
    results = (
        [_mk_errored(f"{failed_cid}-s{j}",  "invalid_request") for j in range(K)]
        + [_mk_succeeded(f"{success_cid}-s{j}", 4, 3, 5) for j in range(K)]
    )
    batches = FakeBatches().queue(results)

    sc = _run_sc(batches, ents)
    judge_rows = _judge_rows_from_sc(sc, ents, rows)

    df = pd.DataFrame(judge_rows)
    judge_n = df.groupby('pipeline_label')[['faithfulness']].count()

    # Nur 1 von 2 Einträgen hat gültige Scores.
    assert judge_n.loc['JSON→Text', 'faithfulness'] == 1


# ── 6. Partial failure: restliche Einträge korrekt im DataFrame ──────────────

def test_nb07_partial_failure_other_entries_intact():
    """Scheitert eine Basis-CID, bleiben alle anderen Zeilen korrekt befüllt."""
    # Minimal-Setup: genau 2 Einträge (single pipeline, 2 XAI-Modelle, 1 Instanz)
    rows = _build_rows(pipelines=['04'], xai_models=['xgb', 'ebm'], instance_ids=[42])
    ents = _entries(rows, 'v2')
    assert len(ents) == 2

    bad_cid = ents[1][0]   # zweiter Eintrag schlägt fehl
    results = [_mk_succeeded(f"{ents[0][0]}-s{j}", 5, 5, 5) for j in range(K)]
    results += [_mk_errored(f"{bad_cid}-s{j}", "invalid_request") for j in range(K)]
    batches = FakeBatches().queue(results)

    sc = _run_sc(batches, ents)
    judge_rows = _judge_rows_from_sc(sc, ents, rows)

    good = [r for r in judge_rows if r['faithfulness'] is not None]
    bad  = [r for r in judge_rows if r['faithfulness'] is None]

    assert len(good) == 1
    assert good[0]['faithfulness'] == 5
    assert len(bad) == 1
    # Zeile mit None hat trotzdem alle Keys (Schema vollständig)
    assert set(bad[0].keys()) == {
        'pipeline_label', 'xai_model', 'instance_id',
        'faithfulness', 'clarity', 'completeness',
        'reasoning', 'raw_response',
    }


# ── 7. Opus: temperature wird weggelassen (NB07-Gating-Logik) ────────────────

def test_nb07_opus_temperature_gating():
    """temperature=JUDGE_TEMPERATURE if model != 'claude-opus-4-8' else None → kein 400."""
    captured: list[dict] = []

    class CapBatches(FakeBatches):
        def create(self, requests):
            captured.extend(requests)
            res = []
            for r in requests:
                cid = r['custom_id']
                res.append({'custom_id': cid, 'result': {'type': 'succeeded',
                    'message': {'content': [{'type': 'text', 'text': _xml(4, 4, 4)}],
                                'usage': {'input_tokens': 1, 'output_tokens': 1}}}})
            bid = f'cap_{self._n}'
            self._store[bid] = res
            self._n += 1
            self.created.append((bid, [r['custom_id'] for r in requests]))
            return {'id': bid}

    rows = _build_rows(pipelines=['04'], xai_models=['xgb'], instance_ids=[42])
    ents = _entries(rows, 'v3')

    # NB07-Gating: Opus → temperature=None
    judge_batch_sc(
        ents,
        system='sys',
        model=OPUS,
        max_tokens=900,
        k=2,
        temperature=0.0,          # wird durch Gating in build_text_params verworfen
        client=make_client(CapBatches()),
        sleep=NOSLEEP,
    )

    for req in captured:
        assert 'temperature' not in req['params'], \
            f"temperature darf bei Opus nicht im Payload stehen: {req['params']}"


# ── 8. k × len(entries) Requests in einem Batch ─────────────────────────────

def test_nb07_k_times_n_requests_submitted():
    """judge_batch_sc reicht genau k × len(entries) Requests als einen Batch ein."""
    rows = _build_rows()   # 4 Pipelines × 2 XAI × 2 Instanzen = 16 Zeilen
    ents = _entries(rows, 'v1')
    results = [_mk_succeeded(f"{cid}-s{j}", 4, 3, 5)
               for cid, _ in ents for j in range(K)]
    batches = FakeBatches().queue(results)

    _run_sc(batches, ents)

    assert len(batches.created) == 1
    assert len(batches.created[0][1]) == len(ents) * K
