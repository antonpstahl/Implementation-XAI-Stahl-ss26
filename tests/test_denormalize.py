"""
Phase 3·2 De-Risking — DRY-Denormalisierung (Smoke- und Golden-Test).

Prüft die einzige Denormalisierungsquelle in utils/explanations.py gegen
bekannte Eingaben. Schlägt fehl, wenn Faktoren, Maps oder Funktionssignaturen
divergieren — schützt den teuren 3b-Lauf vor still verfälschten Payloads.
"""

from __future__ import annotations

import pytest

from utils.explanations import (
    HUM_FACTOR,
    MONTH_NAMES,
    TEMP_FACTOR,
    WEATHER_NAMES,
    WEEKDAY_NAMES,
    WIND_FACTOR,
    build_context_string,
    humanize_feature,
)


# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

def test_factors():
    assert TEMP_FACTOR == 41
    assert HUM_FACTOR  == 100
    assert WIND_FACTOR == 67


def test_weekday_names_length():
    assert len(WEEKDAY_NAMES) == 7
    assert WEEKDAY_NAMES[0] == "Sonntag"
    assert WEEKDAY_NAMES[6] == "Samstag"


def test_month_names_length():
    assert len(MONTH_NAMES) == 13        # Index 0 ist leer
    assert MONTH_NAMES[0] == ""
    assert MONTH_NAMES[1] == "Januar"
    assert MONTH_NAMES[12] == "Dezember"


def test_weather_names_keys():
    assert set(WEATHER_NAMES.keys()) == {1, 2, 3, 4}


# ---------------------------------------------------------------------------
# humanize_feature — Einzelwert-Denormalisierung
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("val,expected", [
    (0.0,  f"~{0.0 * 41:.1f} °C"),
    (0.5,  f"~{0.5 * 41:.1f} °C"),
    (1.0,  f"~{1.0 * 41:.1f} °C"),
])
def test_humanize_temp(val, expected):
    assert humanize_feature("temp", val) == expected


@pytest.mark.parametrize("val,expected", [
    (0.0, "0 %"),
    (0.75, "75 %"),
    (1.0, "100 %"),
])
def test_humanize_hum(val, expected):
    assert humanize_feature("hum", val) == expected


@pytest.mark.parametrize("val,expected", [
    (0.0,  f"{0.0 * 67:.1f} km/h"),
    (0.3,  f"{0.3 * 67:.1f} km/h"),
])
def test_humanize_windspeed(val, expected):
    assert humanize_feature("windspeed", val) == expected


@pytest.mark.parametrize("val,expected", [
    (0,  "00:00 Uhr"),
    (8,  "08:00 Uhr"),
    (23, "23:00 Uhr"),
])
def test_humanize_hr(val, expected):
    assert humanize_feature("hr", val) == expected


@pytest.mark.parametrize("val,expected", [
    (0, "Sonntag"), (1, "Montag"), (5, "Freitag"), (6, "Samstag"),
])
def test_humanize_weekday(val, expected):
    assert humanize_feature("weekday", val) == expected


@pytest.mark.parametrize("val,expected", [
    (1, "Januar"), (6, "Juni"), (12, "Dezember"),
])
def test_humanize_mnth(val, expected):
    assert humanize_feature("mnth", val) == expected


@pytest.mark.parametrize("val,expected", [
    (1, "klar/wenige Wolken"),
    (2, "Nebel/bewölkt"),
    (3, "leichter Regen/Schnee"),
    (4, "Starkregen/Gewitter"),
])
def test_humanize_weathersit(val, expected):
    assert humanize_feature("weathersit", val) == expected


def test_humanize_yr():
    assert humanize_feature("yr", 0) == "2011"
    assert humanize_feature("yr", 1) == "2012"


def test_humanize_holiday():
    assert humanize_feature("holiday", 0) == "kein Feiertag"
    assert humanize_feature("holiday", 1) == "Feiertag"


def test_humanize_unknown_feature():
    assert humanize_feature("season", 2) is None


def test_humanize_bad_value():
    assert humanize_feature("weekday", "not_an_int") is None


# ---------------------------------------------------------------------------
# build_context_string — Golden-Test (NB04 JSON-Payload-Feld)
# ---------------------------------------------------------------------------

_GOLDEN_FV = {
    "hr": 8, "weekday": 3, "mnth": 6, "yr": 0,
    "weathersit": 1, "temp": 0.68, "hum": 0.79,
    "windspeed": 0.22, "holiday": 0,
}

_GOLDEN_EXPECTED = (
    "08:00 Uhr, Mittwoch, Juni, 2011, klar/wenige Wolken, "
    f"~{0.68 * 41:.1f} °C, {0.79 * 100:.0f} % Luftfeuchtigkeit, "
    f"Wind {0.22 * 67:.1f} km/h"
)


def test_build_context_string_golden():
    assert build_context_string(_GOLDEN_FV) == _GOLDEN_EXPECTED


def test_build_context_string_feiertag_included():
    fv = {**_GOLDEN_FV, "holiday": 1}
    result = build_context_string(fv)
    assert result.endswith(", Feiertag")


def test_build_context_string_kein_feiertag_omitted():
    result = build_context_string({**_GOLDEN_FV, "holiday": 0})
    assert "Feiertag" not in result


def test_build_context_string_partial_fv():
    result = build_context_string({"temp": 0.5})
    assert result == f"~{0.5 * 41:.1f} °C"


def test_build_context_string_empty():
    assert build_context_string({}) == ""
