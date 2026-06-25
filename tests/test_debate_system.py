"""
Tests for the multi-agent debate system.

Tests:
1. DebateScorer: probability math, edge cases
2. SelectionPressure: EV normalization
3. Portfolio constraints: position bounds, sector caps, weight formula
4. PipelineValidators: constraint checks
"""


import config


# ────────────────────────────────────────────────────────────────────────────
# Debate Scorer Tests
# ────────────────────────────────────────────────────────────────────────────

def test_debate_scorer_all_buy():
    """All agents are BUY → P_bull should dominate."""
    from agents.debate_scorer import score_debate

    round3 = [
        {"agent_type": "growth", "final_stance": "BUY", "final_confidence": 0.9},
        {"agent_type": "value", "final_stance": "BUY", "final_confidence": 0.8},
        {"agent_type": "macro", "final_stance": "BUY", "final_confidence": 0.7},
        {"agent_type": "risk", "final_stance": "BUY", "final_confidence": 0.6},
    ]
    result = score_debate(round3)

    assert result["P_bull"] > 0.5, f"P_bull should dominate: got {result['P_bull']}"
    assert result["P_bear"] == 0.0 or result["P_bear"] < 0.15
    assert abs(result["P_bull"] + result["P_bear"] + result["P_base"] - 1.0) < 0.01
    assert len(result["bull_agents"]) == 4
    assert len(result["bear_agents"]) == 0


def test_debate_scorer_all_sell():
    """All agents are SELL → P_bear should dominate."""
    from agents.debate_scorer import score_debate

    round3 = [
        {"agent_type": "growth", "final_stance": "SELL", "final_confidence": 0.9},
        {"agent_type": "value", "final_stance": "SELL", "final_confidence": 0.8},
        {"agent_type": "macro", "final_stance": "SELL", "final_confidence": 0.7},
        {"agent_type": "risk", "final_stance": "SELL", "final_confidence": 0.6},
    ]
    result = score_debate(round3)

    assert result["P_bear"] > 0.5, f"P_bear should dominate: got {result['P_bear']}"
    assert result["P_bull"] == 0.0 or result["P_bull"] < 0.15
    assert abs(result["P_bull"] + result["P_bear"] + result["P_base"] - 1.0) < 0.01


def test_debate_scorer_mixed_disagreement():
    """2 BUY, 2 SELL → high uncertainty, P_base should be significant."""
    from agents.debate_scorer import score_debate

    round3 = [
        {"agent_type": "growth", "final_stance": "BUY", "final_confidence": 0.8},
        {"agent_type": "value", "final_stance": "SELL", "final_confidence": 0.8},
        {"agent_type": "macro", "final_stance": "BUY", "final_confidence": 0.7},
        {"agent_type": "risk", "final_stance": "SELL", "final_confidence": 0.7},
    ]
    result = score_debate(round3)

    assert abs(result["P_bull"] + result["P_bear"] + result["P_base"] - 1.0) < 0.01
    # Neither side should overwhelmingly dominate
    assert result["P_bull"] < 0.70, f"P_bull too high for mixed debate: {result['P_bull']}"
    assert result["P_bear"] < 0.70, f"P_bear too high for mixed debate: {result['P_bear']}"
    # Consensus strength should be 0.5 (2/4 on each side)
    assert result["consensus_strength"] == 0.5


def test_debate_scorer_all_hold():
    """All agents HOLD → fallback to default split."""
    from agents.debate_scorer import score_debate

    round3 = [
        {"agent_type": "growth", "final_stance": "HOLD", "final_confidence": 0.5},
        {"agent_type": "value", "final_stance": "HOLD", "final_confidence": 0.5},
        {"agent_type": "macro", "final_stance": "HOLD", "final_confidence": 0.5},
        {"agent_type": "risk", "final_stance": "HOLD", "final_confidence": 0.5},
    ]
    result = score_debate(round3)

    # P_base should be large since all agents HOLD
    assert result["P_base"] > 0.3
    assert abs(result["P_bull"] + result["P_bear"] + result["P_base"] - 1.0) < 0.01


def test_debate_scorer_sum_always_one():
    """Probabilities must always sum to 1.0."""
    from agents.debate_scorer import score_debate

    test_configs = [
        [{"agent_type": "g", "final_stance": "BUY", "final_confidence": 1.0}] * 4,
        [{"agent_type": "g", "final_stance": "SELL", "final_confidence": 0.0}] * 4,
        [{"agent_type": "g", "final_stance": "HOLD", "final_confidence": 0.5}] * 4,
        [
            {"agent_type": "g", "final_stance": "BUY", "final_confidence": 0.9},
            {"agent_type": "v", "final_stance": "SELL", "final_confidence": 0.1},
            {"agent_type": "m", "final_stance": "HOLD", "final_confidence": 0.5},
            {"agent_type": "r", "final_stance": "BUY", "final_confidence": 0.3},
        ],
    ]
    for cfg in test_configs:
        result = score_debate(cfg)
        total = result["P_bull"] + result["P_bear"] + result["P_base"]
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, not 1.0"


def test_debate_scorer_with_agent_weights():
    """Agent weights should influence probabilities."""
    from agents.debate_scorer import score_debate

    round3 = [
        {"agent_type": "growth", "final_stance": "BUY", "final_confidence": 0.8},
        {"agent_type": "value", "final_stance": "SELL", "final_confidence": 0.8},
        {"agent_type": "macro", "final_stance": "HOLD", "final_confidence": 0.5},
        {"agent_type": "risk", "final_stance": "HOLD", "final_confidence": 0.5},
    ]

    # Without weights: BUY and SELL should be roughly balanced
    result_no_weight = score_debate(round3)

    # With growth agent weighted higher: P_bull should increase
    result_weighted = score_debate(round3, agent_weights={"growth": 2.0, "value": 0.5})

    assert result_weighted["P_bull"] > result_no_weight["P_bull"], \
        f"Weighted P_bull ({result_weighted['P_bull']}) should exceed unweighted ({result_no_weight['P_bull']})"


def test_conviction_computation():
    """Conviction should reflect agreement level."""
    from agents.debate_scorer import compute_conviction

    # High consensus — all agree
    high_consensus = {
        "consensus_strength": 1.0,
        "agent_details": [
            {"raw_confidence": 0.9},
            {"raw_confidence": 0.85},
            {"raw_confidence": 0.8},
            {"raw_confidence": 0.75},
        ],
    }
    conviction_high = compute_conviction(high_consensus)
    assert conviction_high > 0.7

    # Low consensus — split
    low_consensus = {
        "consensus_strength": 0.5,
        "agent_details": [
            {"raw_confidence": 0.3},
            {"raw_confidence": 0.4},
            {"raw_confidence": 0.3},
            {"raw_confidence": 0.4},
        ],
    }
    conviction_low = compute_conviction(low_consensus)
    assert conviction_low < conviction_high


# ────────────────────────────────────────────────────────────────────────────
# Pipeline Validator Tests
# ────────────────────────────────────────────────────────────────────────────

def test_validator_catches_sector_cap_violation():
    from pipeline.validators import validate_sector_caps

    portfolio = [
        {"symbol": "A", "sector": "Tech", "allocation_pct": 20},
        {"symbol": "B", "sector": "Tech", "allocation_pct": 20},
        {"symbol": "C", "sector": "Finance", "allocation_pct": 30},
        {"symbol": "D", "sector": "Health", "allocation_pct": 30},
    ]
    errors = validate_sector_caps(portfolio, max_sector_pct=0.35)
    # Tech = 40% > 35% cap
    assert len(errors) >= 1
    assert "Tech" in errors[0]


def test_validator_catches_probability_sum_violation():
    from pipeline.validators import validate_probabilities

    scenario_results = [
        {"symbol": "TEST", "P_bull": 0.5, "P_bear": 0.4, "P_base": 0.2},  # sum = 1.1
    ]
    errors = validate_probabilities(scenario_results)
    assert len(errors) >= 1
    assert "PROBABILITY_SUM" in errors[0]


def test_validator_passes_valid_portfolio():
    from pipeline.validators import validate_sector_caps, validate_position_bounds, validate_allocation_sum

    portfolio = [
        {"symbol": "A", "sector": "Tech", "allocation_pct": 10},
        {"symbol": "B", "sector": "Finance", "allocation_pct": 10},
        {"symbol": "C", "sector": "Health", "allocation_pct": 10},
        {"symbol": "D", "sector": "Energy", "allocation_pct": 10},
        {"symbol": "E", "sector": "Consumer", "allocation_pct": 10},
        {"symbol": "F", "sector": "Tech", "allocation_pct": 10},
        {"symbol": "G", "sector": "Materials", "allocation_pct": 10},
        {"symbol": "H", "sector": "Utilities", "allocation_pct": 10},
        {"symbol": "I", "sector": "Telecom", "allocation_pct": 10},
        {"symbol": "J", "sector": "Real Estate", "allocation_pct": 10},
    ]
    assert validate_sector_caps(portfolio, max_sector_pct=0.35) == []
    assert validate_position_bounds(portfolio, min_pct=0.03, max_pct=0.15) == []
    assert validate_allocation_sum(portfolio) == []


def test_validator_catches_position_too_large():
    from pipeline.validators import validate_position_bounds

    portfolio = [
        {"symbol": "A", "allocation_pct": 25},  # > 15%
        {"symbol": "B", "allocation_pct": 75},  # > 15%
    ]
    errors = validate_position_bounds(portfolio, min_pct=0.03, max_pct=0.15)
    assert len(errors) >= 2


def test_validator_catches_ev_not_traceable():
    from pipeline.validators import validate_ev_traceability

    scenario_results = [
        {"symbol": "NOTRACEABLE", "debate_scores": {}},  # empty debate_scores
    ]
    errors = validate_ev_traceability(scenario_results)
    assert len(errors) >= 1
    assert "EV_NOT_TRACEABLE" in errors[0]


def test_validator_passes_traceable_ev():
    from pipeline.validators import validate_ev_traceability

    scenario_results = [
        {
            "symbol": "TRACEABLE",
            "debate_scores": {
                "bull_agents": ["growth", "macro"],
                "bear_agents": ["risk"],
            },
        },
    ]
    errors = validate_ev_traceability(scenario_results)
    assert errors == []


# ────────────────────────────────────────────────────────────────────────────
# Portfolio Construction Constraint Tests (updated for deterministic optimizer)
# ────────────────────────────────────────────────────────────────────────────

def test_deterministic_weight_computation():
    from pipeline.stage4_construction import _compute_deterministic_weights

    candidates = [
        {"symbol": "A", "probability_weighted_return_12m": 0.20, "conviction": 0.8},
        {"symbol": "B", "probability_weighted_return_12m": 0.10, "conviction": 0.6},
        {"symbol": "C", "probability_weighted_return_12m": 0.05, "conviction": 0.4},
    ]
    result = _compute_deterministic_weights(candidates)

    total_alloc = sum(c["allocation_pct"] for c in result)
    assert abs(total_alloc - 100.0) < 0.1, f"Allocations should sum to 100%, got {total_alloc}"

    # Higher EV×conviction should get higher allocation
    by_symbol = {c["symbol"]: c for c in result}
    assert by_symbol["A"]["allocation_pct"] > by_symbol["B"]["allocation_pct"]
    assert by_symbol["B"]["allocation_pct"] > by_symbol["C"]["allocation_pct"]


def test_position_bounds_enforcement():
    from pipeline.stage4_construction import _enforce_position_bounds

    positions = [
        {"symbol": "A", "allocation_pct": 1.0},   # Below min 3%
        {"symbol": "B", "allocation_pct": 50.0},   # Above max 15%
        {"symbol": "C", "allocation_pct": 8.0},    # Within bounds
    ]
    result = _enforce_position_bounds(positions)

    for p in result:
        assert p["allocation_pct"] >= config.MIN_POSITION_PCT * 100
        assert p["allocation_pct"] <= config.MAX_POSITION_PCT * 100


def test_sector_cap_enforcement():
    from pipeline.stage4_construction import _enforce_sector_caps

    positions = [
        {"symbol": "A", "sector": "Tech", "allocation_pct": 25, "ev_12m_return": 0.2},
        {"symbol": "B", "sector": "Tech", "allocation_pct": 25, "ev_12m_return": 0.1},
        {"symbol": "C", "sector": "Finance", "allocation_pct": 25, "ev_12m_return": 0.15},
        {"symbol": "D", "sector": "Health", "allocation_pct": 25, "ev_12m_return": 0.12},
    ]
    result = _enforce_sector_caps(positions)

    tech_total = sum(p["allocation_pct"] for p in result if p["sector"] == "Tech")
    assert tech_total <= config.MAX_SECTOR_PCT * 100 + 1.0, \
        f"Tech sector at {tech_total}% exceeds {config.MAX_SECTOR_PCT*100}% cap"


def test_normalize_to_100():
    from pipeline.stage4_construction import _normalize_to_100

    positions = [
        {"symbol": "A", "allocation_pct": 10},
        {"symbol": "B", "allocation_pct": 20},
        {"symbol": "C", "allocation_pct": 30},
    ]
    result = _normalize_to_100(positions)
    total = sum(p["allocation_pct"] for p in result)
    assert abs(total - 100.0) < 0.5


# ────────────────────────────────────────────────────────────────────────────
# Duplicate Argument Detection
# ────────────────────────────────────────────────────────────────────────────

def test_duplicate_argument_detection():
    from pipeline.validators import validate_duplicate_arguments

    debate_results = [
        {
            "symbol": "TESTSTOCK",
            "round1": [
                {
                    "agent_type": "growth",
                    "arguments": [{"point": "Strong revenue growth of 25% yoy driven by market expansion"}],
                },
                {
                    "agent_type": "value",
                    "arguments": [{"point": "Strong revenue growth of 25% yoy driven by market expansion"}],
                },
            ],
        }
    ]
    errors = validate_duplicate_arguments(debate_results, threshold=0.7)
    assert len(errors) >= 1, "Should detect duplicate arguments across agents"


def test_no_false_positive_duplicates():
    from pipeline.validators import validate_duplicate_arguments

    debate_results = [
        {
            "symbol": "TESTSTOCK",
            "round1": [
                {
                    "agent_type": "growth",
                    "arguments": [{"point": "Revenue growing 25% driven by new market expansion"}],
                },
                {
                    "agent_type": "risk",
                    "arguments": [{"point": "Debt to equity ratio is concerning at 2.1x"}],
                },
            ],
        }
    ]
    errors = validate_duplicate_arguments(debate_results, threshold=0.7)
    assert len(errors) == 0, f"Should not flag unrelated arguments as duplicates: {errors}"
