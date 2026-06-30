"""
Denormalisation smoke and golden test.

Checks the single denormalisation source in utils/explanations.py against known
inputs. Fails if factors, maps or function signatures diverge, which protects the
expensive run from silently corrupted payloads.
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
# Constants
# ---------------------------------------------------------------------------

def test_factors():
    assert TEMP_FACTOR == 41
    assert HUM_FACTOR  == 100
    assert WIND_FACTOR == 67


def test_weekday_names_length():
    assert len(WEEKDAY_NAMES) == 7
    assert WEEKDAY_NAMES[0] == "Sunday"
    assert WEEKDAY_NAMES[6] == "Saturday"


def test_month_names_length():
    assert len(MONTH_NAMES) == 13        # index 0 is empty
    assert MONTH_NAMES[0] == ""
    assert MONTH_NAMES[1] == "January"
    assert MONTH_NAMES[12] == "December"


def test_weather_names_keys():
    assert set(WEATHER_NAMES.keys()) == {1, 2, 3, 4}


# ---------------------------------------------------------------------------
# humanize_feature - single value denormalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("val,expected", [
    (0.0,  f"~{0.0 * 41:.1f} C"),
    (0.5,  f"~{0.5 * 41:.1f} C"),
    (1.0,  f"~{1.0 * 41:.1f} C"),
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
    (0,  "00:00"),
    (8,  "08:00"),
    (23, "23:00"),
])
def test_humanize_hr(val, expected):
    assert humanize_feature("hr", val) == expected


@pytest.mark.parametrize("val,expected", [
    (0, "Sunday"), (1, "Monday"), (5, "Friday"), (6, "Saturday"),
])
def test_humanize_weekday(val, expected):
    assert humanize_feature("weekday", val) == expected


@pytest.mark.parametrize("val,expected", [
    (1, "January"), (6, "June"), (12, "December"),
])
def test_humanize_mnth(val, expected):
    assert humanize_feature("mnth", val) == expected


@pytest.mark.parametrize("val,expected", [
    (1, "clear/few clouds"),
    (2, "mist/cloudy"),
    (3, "light rain/snow"),
    (4, "heavy rain/thunderstorm"),
])
def test_humanize_weathersit(val, expected):
    assert humanize_feature("weathersit", val) == expected


def test_humanize_yr():
    assert humanize_feature("yr", 0) == "2011"
    assert humanize_feature("yr", 1) == "2012"


def test_humanize_holiday():
    assert humanize_feature("holiday", 0) == "no holiday"
    assert humanize_feature("holiday", 1) == "holiday"


def test_humanize_unknown_feature():
    assert humanize_feature("season", 2) is None


def test_humanize_bad_value():
    assert humanize_feature("weekday", "not_an_int") is None


# ---------------------------------------------------------------------------
# build_context_string - golden test (NB04b JSON payload field)
# ---------------------------------------------------------------------------

_GOLDEN_FV = {
    "hr": 8, "weekday": 3, "mnth": 6, "yr": 0,
    "weathersit": 1, "temp": 0.68, "hum": 0.79,
    "windspeed": 0.22, "holiday": 0,
}

_GOLDEN_EXPECTED = (
    "08:00, Wednesday, June, 2011, clear/few clouds, "
    f"~{0.68 * 41:.1f} C, {0.79 * 100:.0f} % humidity, "
    f"wind {0.22 * 67:.1f} km/h"
)


def test_build_context_string_golden():
    assert build_context_string(_GOLDEN_FV) == _GOLDEN_EXPECTED


def test_build_context_string_holiday_included():
    fv = {**_GOLDEN_FV, "holiday": 1}
    result = build_context_string(fv)
    assert result.endswith(", holiday")


def test_build_context_string_no_holiday_omitted():
    result = build_context_string({**_GOLDEN_FV, "holiday": 0})
    assert "holiday" not in result


def test_build_context_string_partial_fv():
    result = build_context_string({"temp": 0.5})
    assert result == f"~{0.5 * 41:.1f} C"


def test_build_context_string_empty():
    assert build_context_string({}) == ""
