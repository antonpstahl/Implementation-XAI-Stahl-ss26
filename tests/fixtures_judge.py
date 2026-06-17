"""
Fixtures für Judge-Parsing-Tests.

Jede Fixture ist ein dict mit:
  - raw:      die rohe LLM-Antwort (str)
  - expected: erwartetes Ergebnis von parse_judge_response (dict)
              Felder die nicht gesetzt sein sollen → Key fehlt im expected-dict
"""

# ── 1. Normalfall: reiner Markdown-Codeblock (wie echte Modell-Antworten) ────
FIXTURE_MARKDOWN_CODEBLOCK = {
    "raw": """\
```json
{
  "FAITHFULNESS": 5,
  "CLARITY": 4,
  "COMPLETENESS": 4,
  "FAITHFULNESS_REASONING": "Alle drei Top-3-Treiber korrekt benannt.",
  "CLARITY_REASONING": "Alltagssprache, kein Fachjargon.",
  "COMPLETENESS_REASONING": "Alle drei Pflichtabschnitte vorhanden."
}
```""",
    "expected": {
        "faithfulness": 5,
        "clarity": 4,
        "completeness": 4,
        "faithfulness_reasoning": "Alle drei Top-3-Treiber korrekt benannt.",
        "clarity_reasoning": "Alltagssprache, kein Fachjargon.",
        "completeness_reasoning": "Alle drei Pflichtabschnitte vorhanden.",
    },
}

# ── 2. JSON eingebettet in Fließtext ─────────────────────────────────────────
FIXTURE_JSON_IN_FLIESSTEXT = {
    "raw": """\
Hier ist meine Bewertung der Erklärung:

Die Erklärung ist insgesamt solide. Meine Scores:

{"FAITHFULNESS": 3, "CLARITY": 5, "COMPLETENESS": 2,
 "FAITHFULNESS_REASONING": "Treiber teilweise falsch.",
 "CLARITY_REASONING": "Sehr verständlich.",
 "COMPLETENESS_REASONING": "Empfehlung fehlt."}

Ich hoffe das hilft.""",
    "expected": {
        "faithfulness": 3,
        "clarity": 5,
        "completeness": 2,
        "faithfulness_reasoning": "Treiber teilweise falsch.",
        "clarity_reasoning": "Sehr verständlich.",
        "completeness_reasoning": "Empfehlung fehlt.",
    },
}

# ── 3. Klares JSON-Objekt ohne Codeblock, kein Fließtext ─────────────────────
FIXTURE_PLAIN_JSON = {
    "raw": """\
{
  "faithfulness": 4,
  "clarity": 3,
  "completeness": 5,
  "faithfulness_reasoning": "Wichtige Features erwähnt.",
  "clarity_reasoning": "Etwas technisch.",
  "completeness_reasoning": "Vollständig."
}""",
    "expected": {
        "faithfulness": 4,
        "clarity": 3,
        "completeness": 5,
        "faithfulness_reasoning": "Wichtige Features erwähnt.",
        "clarity_reasoning": "Etwas technisch.",
        "completeness_reasoning": "Vollständig.",
    },
}

# ── 4a. Fehlende Felder: nur zwei von drei Scores vorhanden ──────────────────
FIXTURE_MISSING_FIELDS = {
    "raw": """\
```json
{
  "FAITHFULNESS": 2,
  "CLARITY": 4
}
```""",
    "expected": {
        "faithfulness": 2,
        "clarity": 4,
        # completeness fehlt absichtlich
    },
}

# ── 4b. Abgeschnittenes JSON (Regex-Fallback nötig) ───────────────────────────
FIXTURE_TRUNCATED_JSON = {
    "raw": """\
{
  "FAITHFULNESS": 1,
  "CLARITY": 2,
  "COMPLETENESS": 3,
  "FAITHFULNESS_REASONING": "Falsch.""",  # kein schließendes }
    "expected": {
        "faithfulness": 1,
        "clarity": 2,
        "completeness": 3,
    },
}

# ── 5. Vollständiger Garbage: kein JSON, keine Scores extrahierbar ────────────
FIXTURE_GARBAGE = {
    "raw": "Ich kann diese Anfrage leider nicht beantworten. Bitte versuchen Sie es erneut.",
    "expected": {},  # leeres dict — kein Score extrahierbar
}

# ── 6. Reason-then-Score Plain-Text (A1-Format: Begründung vor Score) ─────────
FIXTURE_REASON_THEN_SCORE_PLAINTEXT = {
    "raw": """\
FAITHFULNESS_REASONING: Ankerpunkt 5: alle drei Top-3-Treiber korrekt benannt, Vorhersagezahl korrekt. Kein Abzug. Endpunktzahl = max(1, 5+0) = 5.
FAITHFULNESS: 5
CLARITY_REASONING: Ankerpunkt 4: ein leichter Fachbegriff vorhanden. Ein Abzug. Endpunktzahl = max(1, 4-1) = 3.
CLARITY: 3
COMPLETENESS_REASONING: Ankerpunkt 5: alle drei Pflichtabschnitte substanziell vorhanden. Kein Abzug. Endpunktzahl = max(1, 5+0) = 5.
COMPLETENESS: 5""",
    "expected": {
        "faithfulness": 5,
        "clarity": 3,
        "completeness": 5,
        "faithfulness_reasoning": "Ankerpunkt 5: alle drei Top-3-Treiber korrekt benannt, Vorhersagezahl korrekt. Kein Abzug. Endpunktzahl = max(1, 5+0) = 5.",
        "clarity_reasoning": "Ankerpunkt 4: ein leichter Fachbegriff vorhanden. Ein Abzug. Endpunktzahl = max(1, 4-1) = 3.",
        "completeness_reasoning": "Ankerpunkt 5: alle drei Pflichtabschnitte substanziell vorhanden. Kein Abzug. Endpunktzahl = max(1, 5+0) = 5.",
    },
}

# ── Alle Fixtures als Liste für parametrisierten Einsatz ─────────────────────
ALL_FIXTURES = [
    ("markdown_codeblock",           FIXTURE_MARKDOWN_CODEBLOCK),
    ("json_in_fliesstext",           FIXTURE_JSON_IN_FLIESSTEXT),
    ("plain_json",                   FIXTURE_PLAIN_JSON),
    ("missing_fields",               FIXTURE_MISSING_FIELDS),
    ("truncated_json",               FIXTURE_TRUNCATED_JSON),
    ("garbage",                      FIXTURE_GARBAGE),
    ("reason_then_score_plaintext",  FIXTURE_REASON_THEN_SCORE_PLAINTEXT),
]
