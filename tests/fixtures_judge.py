"""
Fixtures for judge parsing tests.

Each fixture is a dict with:
  - raw:      the raw LLM answer (str)
  - expected: expected result of parse_judge_response (dict)
              fields that should not be set -> the key is missing in expected
"""

# --- 1. Normal case: plain markdown code block (like real model answers) ---
FIXTURE_MARKDOWN_CODEBLOCK = {
    "raw": """\
```json
{
  "FAITHFULNESS": 5,
  "CLARITY": 4,
  "COMPLETENESS": 4,
  "FAITHFULNESS_REASONING": "All three top 3 drivers named correctly.",
  "CLARITY_REASONING": "Everyday language, no jargon.",
  "COMPLETENESS_REASONING": "All three required sections present."
}
```""",
    "expected": {
        "faithfulness": 5,
        "clarity": 4,
        "completeness": 4,
        "faithfulness_reasoning": "All three top 3 drivers named correctly.",
        "clarity_reasoning": "Everyday language, no jargon.",
        "completeness_reasoning": "All three required sections present.",
    },
}

# --- 2. JSON embedded in prose ---
FIXTURE_JSON_IN_FLIESSTEXT = {
    "raw": """\
Here is my evaluation of the explanation:

The explanation is solid overall. My scores:

{"FAITHFULNESS": 3, "CLARITY": 5, "COMPLETENESS": 2,
 "FAITHFULNESS_REASONING": "Drivers partly wrong.",
 "CLARITY_REASONING": "Very understandable.",
 "COMPLETENESS_REASONING": "Recommendation missing."}

I hope this helps.""",
    "expected": {
        "faithfulness": 3,
        "clarity": 5,
        "completeness": 2,
        "faithfulness_reasoning": "Drivers partly wrong.",
        "clarity_reasoning": "Very understandable.",
        "completeness_reasoning": "Recommendation missing.",
    },
}

# --- 3. Clear JSON object without code block, no prose ---
FIXTURE_PLAIN_JSON = {
    "raw": """\
{
  "faithfulness": 4,
  "clarity": 3,
  "completeness": 5,
  "faithfulness_reasoning": "Important features mentioned.",
  "clarity_reasoning": "A bit technical.",
  "completeness_reasoning": "Complete."
}""",
    "expected": {
        "faithfulness": 4,
        "clarity": 3,
        "completeness": 5,
        "faithfulness_reasoning": "Important features mentioned.",
        "clarity_reasoning": "A bit technical.",
        "completeness_reasoning": "Complete.",
    },
}

# --- 4a. Missing fields: only two of three scores present ---
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
        # completeness missing on purpose
    },
}

# --- 4b. Truncated JSON (regex fallback needed) ---
FIXTURE_TRUNCATED_JSON = {
    "raw": """\
{
  "FAITHFULNESS": 1,
  "CLARITY": 2,
  "COMPLETENESS": 3,
  "FAITHFULNESS_REASONING": "Wrong.""",  # no closing }
    "expected": {
        "faithfulness": 1,
        "clarity": 2,
        "completeness": 3,
    },
}

# --- 5. Full garbage: no JSON, no scores extractable ---
FIXTURE_GARBAGE = {
    "raw": "Sorry, I cannot answer this request. Please try again.",
    "expected": {},  # empty dict, no score extractable
}

# --- 6. Reason then score plain text (reasoning before score) ---
FIXTURE_REASON_THEN_SCORE_PLAINTEXT = {
    "raw": """\
FAITHFULNESS_REASONING: Anchor point 5: all three top 3 drivers named correctly, prediction number correct. No deduction. Final score = max(1, 5+0) = 5.
FAITHFULNESS: 5
CLARITY_REASONING: Anchor point 4: one mild technical term present. One deduction. Final score = max(1, 4-1) = 3.
CLARITY: 3
COMPLETENESS_REASONING: Anchor point 5: all three required sections substantially present. No deduction. Final score = max(1, 5+0) = 5.
COMPLETENESS: 5""",
    "expected": {
        "faithfulness": 5,
        "clarity": 3,
        "completeness": 5,
        "faithfulness_reasoning": "Anchor point 5: all three top 3 drivers named correctly, prediction number correct. No deduction. Final score = max(1, 5+0) = 5.",
        "clarity_reasoning": "Anchor point 4: one mild technical term present. One deduction. Final score = max(1, 4-1) = 3.",
        "completeness_reasoning": "Anchor point 5: all three required sections substantially present. No deduction. Final score = max(1, 5+0) = 5.",
    },
}

# --- 7. XML format (primary parsing path) ---
FIXTURE_XML_FULL = {
    "raw": """\
<faithfulness_reasoning>All three top 3 drivers correct; yr sign right. Anchor point 5, no deduction.</faithfulness_reasoning>
<faithfulness>5</faithfulness>
<clarity_reasoning>Everyday language; one technical term, barely. Anchor point 4, no required deduction.</clarity_reasoning>
<clarity>4</clarity>
<completeness_reasoning>All three sections substantially present. Anchor point 5, no deduction.</completeness_reasoning>
<completeness>5</completeness>""",
    "expected": {
        "faithfulness": 5,
        "clarity": 4,
        "completeness": 5,
        "faithfulness_reasoning": "All three top 3 drivers correct; yr sign right. Anchor point 5, no deduction.",
        "clarity_reasoning": "Everyday language; one technical term, barely. Anchor point 4, no required deduction.",
        "completeness_reasoning": "All three sections substantially present. Anchor point 5, no deduction.",
    },
}

# --- 8. XML partial (scores only, no reasoning) ---
FIXTURE_XML_SCORES_ONLY = {
    "raw": """\
<faithfulness>3</faithfulness>
<clarity>2</clarity>
<completeness>4</completeness>""",
    "expected": {
        "faithfulness": 3,
        "clarity": 2,
        "completeness": 4,
    },
}

# --- 9. XML with surrounding text (robust against a preamble) ---
FIXTURE_XML_WITH_PREAMBLE = {
    "raw": """\
Here is my evaluation:

<faithfulness_reasoning>Drivers correct. Anchor point 4.</faithfulness_reasoning>
<faithfulness>4</faithfulness>
<clarity_reasoning>Clear and understandable. Anchor point 5.</clarity_reasoning>
<clarity>5</clarity>
<completeness_reasoning>Recommendation present. Anchor point 5.</completeness_reasoning>
<completeness>5</completeness>

End of evaluation.""",
    "expected": {
        "faithfulness": 4,
        "clarity": 5,
        "completeness": 5,
        "faithfulness_reasoning": "Drivers correct. Anchor point 4.",
        "clarity_reasoning": "Clear and understandable. Anchor point 5.",
        "completeness_reasoning": "Recommendation present. Anchor point 5.",
    },
}

# --- All fixtures as a list for parametrized use ---
ALL_FIXTURES = [
    ("markdown_codeblock",           FIXTURE_MARKDOWN_CODEBLOCK),
    ("json_in_fliesstext",           FIXTURE_JSON_IN_FLIESSTEXT),
    ("plain_json",                   FIXTURE_PLAIN_JSON),
    ("missing_fields",               FIXTURE_MISSING_FIELDS),
    ("truncated_json",               FIXTURE_TRUNCATED_JSON),
    ("garbage",                      FIXTURE_GARBAGE),
    ("reason_then_score_plaintext",  FIXTURE_REASON_THEN_SCORE_PLAINTEXT),
    ("xml_full",                     FIXTURE_XML_FULL),
    ("xml_scores_only",              FIXTURE_XML_SCORES_ONLY),
    ("xml_with_preamble",            FIXTURE_XML_WITH_PREAMBLE),
]
