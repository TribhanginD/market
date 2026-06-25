import pytest

from pipeline.stage3_scenarios import (
    _normalize_scenario_result,
    _normalize_scenarios_structure,
)


def test_stage3_normalizes_percent_point_ev_and_recommendation():
    result = {
        "probability_weighted_return_12m": 10.4,
        "recommendation": "AVOID",
        "bull": {"return_pct": 28.7},
        "bear": {"return_pct": -26.0},
    }
    fundamentals = {"price": 218.0}

    normalized = _normalize_scenario_result(
        result,
        symbol="GROWW",
        company_name="Billionbrains Garage Ventures Ltd.",
        sector="Financial Services",
        fundamentals=fundamentals,
    )

    assert normalized["probability_weighted_return_12m"] == pytest.approx(0.104)
    assert normalized["bull"]["return_pct"] == pytest.approx(0.287)
    assert normalized["bear"]["return_pct"] == pytest.approx(-0.26)
    assert normalized["recommendation"] == "BUY"
    assert normalized["current_price"] == 218.0


def test_stage3_keeps_decimal_ev_scale():
    normalized = _normalize_scenario_result(
        {"probability_weighted_return_12m": 0.086, "recommendation": "AVOID"},
        symbol="FINCABLES",
        company_name="Finolex Cables Ltd.",
        sector="Industrials",
        fundamentals={"currentPrice": 975.0},
    )

    assert normalized["probability_weighted_return_12m"] == 0.086
    assert normalized["recommendation"] == "BUY"
    assert normalized["current_price"] == 975.0


def test_stage3_normalizes_scenarios_list_to_dict():
    result = {
        "scenarios": [
            {"scenario_name": "Bull", "probability": 0.33},
            {"scenario_name": "Base", "probability": 0.50},
            {"scenario_name": "Bear", "probability": 0.17},
        ]
    }

    _normalize_scenarios_structure(result)

    assert isinstance(result["scenarios"], dict)
    assert result["scenarios"]["bull"]["probability"] == pytest.approx(0.33)
    assert result["scenarios"]["base"]["probability"] == pytest.approx(0.50)
    assert result["scenarios"]["bear"]["probability"] == pytest.approx(0.17)
