"""Tests for the rank-parser in utils.rubric.

These lock in the verbal-ordinal phrasings the generator actually produces on the
G2b whole-model track (``ranked third in importance``, ``fifth in importance``,
``ranked last (9th)``, ``lowest importance of all features``, ``third-most
important``, ``ranked sixth``, etc.). A rubric parser that misses these silently
inflates ``rank_stated`` gaps between modalities and drives the ``vision_all``
"rank collapse" number reported in the paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from utils.rubric import _extract_rank


# ---------------------------------------------------------------------------
# 1. numeric rank + digit (regressions -- these already worked)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("rank 1",                            1),
    ("rank 9",                            9),
    ("ranked #2",                         2),
    ("ranks 7th",                         7),
    ("Ranked 1st with an importance of 0.889", 1),
    ("Ranked 3rd with importance 0.197",  3),
    ("(rank 5, importance 0.118)",        5),
    ("7th out of 9",                      7),
    ("2 of 9",                            2),
    ("**rank 7th**",                      7),  # markdown emphasis stripped
    ("This is by far the most important feature (rank 1)", 1),
])
def test_numeric_rank_cues(text, expected):
    assert _extract_rank(text) == expected


# ---------------------------------------------------------------------------
# 2. word ordinal with a rank cue -- "Ranked <word>"
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("Ranked third in importance",                              3),
    ("Ranked fourth, month has a moderate effect",              4),
    ("Ranked fifth, humidity has a modest but real effect",     5),
    ("Ranked sixth, weather situation has a moderate impact",   6),
    ("Ranked seventh, weekday has a low importance",            7),
    ("Ranked eighth, windspeed has a low overall importance",   8),
    ("Humidity ranks sixth in importance",                      6),
    ("Weather situation ranks seventh in the model",            7),
])
def test_word_ordinal_after_rank_cue(text, expected):
    assert _extract_rank(text) == expected


# ---------------------------------------------------------------------------
# 3. "Ranked last (...)" -- always the bottom feature
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "Ranked last (ninth) with the lowest importance score",
    "Ranked last (9th) with an importance of 0.008",
    "Ranked last (9th) with importance 0.012",
    "Ranked last (9th)",
    "Holiday ranks 9th (last) in overall importance",  # digit + "(last)"
])
def test_ranked_last_is_nine(text):
    assert _extract_rank(text) == 9


# ---------------------------------------------------------------------------
# 4. "<ordinal> in importance" -- no explicit "rank" keyword
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("Third in importance (0.217), this feature has a moderately strong effect", 3),
    ("Fourth in importance (0.086); a moderate influence",                       4),
    ("Fifth in importance (0.083), close to month",                              5),
    ("Sixth in importance (0.061); meaningful but smaller in magnitude",         6),
    ("Seventh in importance (0.024); a real but small effect",                   7),
    ("Eighth in importance (0.014); a minor feature",                            8),
])
def test_ordinal_in_importance(text, expected):
    assert _extract_rank(text) == expected


# ---------------------------------------------------------------------------
# 5. Hyphenated ordinals: "third-most important", "fifth-ranked feature",
#    "second-strongest driver"
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("This is the third-most important feature (rank 3, importance 0.197)", 3),
    ("This is the fifth-ranked feature (rank 5, importance 0.118)",         5),
    ("This is the eighth-ranked feature (rank 8, importance 0.020)",        8),
    ("Temperature is the second-strongest driver of demand",                2),
    ("Year is the second-most influential feature",                         2),
])
def test_hyphenated_ordinal_forms(text, expected):
    assert _extract_rank(text) == expected


# ---------------------------------------------------------------------------
# 6. "<ordinal> most X" (with or without "the")
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("This is the second most important feature",                       2),
    ("This is the third most important feature",                        3),
    ("This is the fourth most important feature",                       4),
    ("Temperature is the third most important feature",                 3),
    ("Year is the second most important feature overall",               2),
    ("Windspeed is the eighth most important feature",                  8),
])
def test_ordinal_most_important(text, expected):
    assert _extract_rank(text) == expected


# ---------------------------------------------------------------------------
# 7. Superlatives -> rank 1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "This is by far the most important feature",
    "This is by far the single most influential feature",
    "This is by far the strongest feature (importance 0.889)",
    "This is the single most important feature in the model",
    "This is by far the most influential feature",
    "This is by far the most important feature in the model, "
    "with a horizontal spread several times wider than any other feature",
])
def test_superlative_rank_one(text):
    assert _extract_rank(text) == 1


# ---------------------------------------------------------------------------
# 8. "Lowest / least important / weakest" phrasings -> rank 9
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "Lowest importance of all features (0.008); the effect is negligible",
    "the least important feature in the model (rank 9)",  # 'rank 9' would also catch it
    "This is the least important feature and can be considered negligible",
    "the weakest feature in the model",
    "Holiday is the least important feature in this model",
])
def test_bottom_rank_phrasings(text):
    assert _extract_rank(text) == 9


# ---------------------------------------------------------------------------
# 9. Non-committal phrasings -> None (the parser must NOT invent a rank)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "This is a high-importance feature, with a total vertical range of roughly 1.1 log-units, "
    "ranking among the top three or four features.",
    "This is a moderate-importance feature with a vertical range of roughly 0.6-0.7 log-units, "
    "placing it toward the lower-middle of the ranking.",
    "sitting in the middle of the ranking",
    "This feature has a moderate but real seasonal signal, about half the strength of temperature.",
    "",
])
def test_non_committal_returns_none(text):
    assert _extract_rank(text) is None


# ---------------------------------------------------------------------------
# 10. False-positive guards: mentioning another feature's rank must not leak
# ---------------------------------------------------------------------------
def test_one_third_is_not_a_rank():
    # "one-third the size of temperature" contains 'third' but is not a rank claim
    assert _extract_rank("Ranked fourth, month has a moderate effect, roughly "
                         "one-third the size of temperature's influence") == 4


def test_rank_is_from_first_cue():
    # first-hit wins: description that also mentions another feature's role
    text = ("Ranked 3rd with an importance of 0.197 - nearly as strong as year "
            "(rank 2, importance 0.200).")
    # "rank 3" is caught first because it is the leading cue in the sentence.
    assert _extract_rank(text) == 3


# ---------------------------------------------------------------------------
# 11. Regression tests -- real records the older, fixed-priority parser got
#     wrong. Each phrasing is quoted verbatim from results/global/* or
#     results/global_variance/* (2026-09-28). These pin the "earliest claim
#     in the text wins" behaviour so comparison-feature ranks cannot leak in.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    # tooluse_ebm_hr: subject's own superlative ("single most important")
    # must beat the later mention of "second-ranked feature, temperature".
    ("hr is the single most important feature in the model by a wide margin "
     "— its importance score (0.889) is more than three times that of "
     "the second-ranked feature, temperature (0.253).",                     1),
    # tooluse_ebm_hr_gen2 (variance): " #1" must parse even though '#' has
    # no word boundary before it.
    ("hour of day is by far the #1 most important feature in the model, "
     "with an importance score of 0.889 — more than three times greater "
     "than the second-ranked feature (temperature, at 0.253).",             1),
    # tooluse_xgb_weekday: subject's "a rank of 4 out of 9" comes first;
    # the later "hour of day (rank 1, ...)" must not win.
    ("with a global importance score of 0.173 and a rank of 4 out of 9, "
     "weekday is a meaningful but secondary driver of demand. it is "
     "considerably less influential than the hour of day (rank 1, "
     "importance 0.83).",                                                   4),
    # tooluse_xgb_yr: subject's "2nd most important" comes first; later
    # "(ranked 3rd ...)" and "(ranked 1st ...)" describe comparison features.
    ("year is the 2nd most important feature in the model (out of 9), with "
     "an importance score of 0.200 — very close to temperature "
     "(ranked 3rd at 0.197) but substantially weaker than the hour of the "
     "day (ranked 1st at 0.833).",                                          2),
])
def test_earliest_rank_claim_wins(text, expected):
    assert _extract_rank(text) == expected


# ---------------------------------------------------------------------------
# 12. "Second/third-least" / "-lowest" idiom must map to 10 - k (nine
#     features), not to k. When the same text also carries an explicit
#     "N of 9" claim, the concrete numeric claim wins over the idiom.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    # bare idiom (no numeric confirmation) -> 10 - k
    ("Windspeed is the second-least important feature in the model.",       8),
    ("This is the second least important feature.",                         8),
    ("Weekday is the third-lowest ranked feature in the model.",            7),
    # idiom + confirming numeric (agree) -> numeric wins, still 8
    ("wind speed is the second-least important feature in the model, "
     "ranking 8th out of 9.",                                               8),
    # idiom + contradicting numeric ("7th out of 9") -> trust the numeric
    ("weekday is the second-least important feature in the model, ranked "
     "7th out of 9.",                                                       7),
])
def test_kth_least_idiom(text, expected):
    assert _extract_rank(text) == expected


# ---------------------------------------------------------------------------
# 13. Bare "least important" that is actually part of "second-least
#     important" must not fall through to the pattern-5 superlative and
#     yield rank 9.
# ---------------------------------------------------------------------------
def test_second_least_does_not_leak_to_nine():
    # would previously have matched \b(?:the\s+)?least\s+important\b -> 9
    assert _extract_rank(
        "This is the second-least important feature."
    ) == 8
