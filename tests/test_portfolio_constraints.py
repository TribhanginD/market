"""
Portfolio constraint tests — updated for deterministic optimizer.
"""


import config
from pipeline.stage4_construction import (
    Stage4PortfolioConstruction,
    _compute_deterministic_weights,
    _enforce_position_bounds,
    _enforce_sector_caps,
    _normalize_to_100,
)


def test_deterministic_weights_sum_to_100():
    candidates = [
        {"symbol": f"STOCK{i}", "probability_weighted_return_12m": 0.20 - i * 0.01, "conviction": 0.6}
        for i in range(15)
    ]
    result = _compute_deterministic_weights(candidates)
    total = sum(c["allocation_pct"] for c in result)
    assert abs(total - 100.0) < 0.1, f"Expected 100%, got {total}"


def test_position_bounds_enforced():
    positions = [
        {"symbol": "BIG", "allocation_pct": 50.0},
        {"symbol": "SMALL", "allocation_pct": 0.5},
        {"symbol": "OK", "allocation_pct": 8.0},
    ]
    result = _enforce_position_bounds(positions)
    for p in result:
        assert p["allocation_pct"] >= config.MIN_POSITION_PCT * 100
        assert p["allocation_pct"] <= config.MAX_POSITION_PCT * 100


def test_sector_caps_enforced():
    positions = [
        {"symbol": "TECH1", "sector": "Technology", "allocation_pct": 20, "ev_12m_return": 0.12},
        {"symbol": "TECH2", "sector": "Technology", "allocation_pct": 15, "ev_12m_return": 0.10},
        {"symbol": "TECH3", "sector": "Technology", "allocation_pct": 10, "ev_12m_return": 0.08},
        {"symbol": "FIN1", "sector": "Financials", "allocation_pct": 12, "ev_12m_return": 0.09},
        {"symbol": "FIN2", "sector": "Financials", "allocation_pct": 8, "ev_12m_return": 0.07},
        {"symbol": "HLTH1", "sector": "Healthcare", "allocation_pct": 10, "ev_12m_return": 0.06},
        {"symbol": "ENRG1", "sector": "Energy", "allocation_pct": 10, "ev_12m_return": 0.05},
        {"symbol": "CONS1", "sector": "Consumer", "allocation_pct": 8, "ev_12m_return": 0.04},
        {"symbol": "IND1", "sector": "Industrials", "allocation_pct": 7, "ev_12m_return": 0.03},
    ]
    result = _enforce_sector_caps(positions)
    sector_totals = {}
    for p in result:
        sector = p.get("sector", "Unknown")
        sector_totals[sector] = sector_totals.get(sector, 0) + p["allocation_pct"]

    for sector, pct in sector_totals.items():
        assert pct <= config.MAX_SECTOR_PCT * 100 + 1.0, f"{sector} at {pct:.1f}% exceeds cap"


def test_normalize_to_100():
    positions = [
        {"symbol": "A", "allocation_pct": 20},
        {"symbol": "B", "allocation_pct": 30},
        {"symbol": "C", "allocation_pct": 50},
    ]
    result = _normalize_to_100(positions)
    total = sum(p["allocation_pct"] for p in result)
    assert abs(total - 100.0) < 0.5, f"Expected ~100%, got {total}"


def test_normalize_handles_zero_total():
    positions = [
        {"symbol": "A", "allocation_pct": 0},
        {"symbol": "B", "allocation_pct": 0},
    ]
    result = _normalize_to_100(positions)
    total = sum(p["allocation_pct"] for p in result)
    assert abs(total - 100.0) < 0.5


def test_higher_ev_x_conviction_gets_higher_weight():
    candidates = [
        {"symbol": "HIGH", "probability_weighted_return_12m": 0.30, "conviction": 0.9},
        {"symbol": "MED", "probability_weighted_return_12m": 0.15, "conviction": 0.6},
        {"symbol": "LOW", "probability_weighted_return_12m": 0.05, "conviction": 0.3},
    ]
    result = _compute_deterministic_weights(candidates)
    by_sym = {c["symbol"]: c for c in result}
    assert by_sym["HIGH"]["allocation_pct"] > by_sym["MED"]["allocation_pct"]
    assert by_sym["MED"]["allocation_pct"] > by_sym["LOW"]["allocation_pct"]
