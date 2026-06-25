"""
Debate Scorer — deterministic conversion of debate output into scenario probabilities.

ZERO LLM calls. Pure math.

Converts agent stances and confidences from the 3-round debate
into P_bull, P_bear, P_base probabilities that sum to exactly 1.0.
"""

import logging
from statistics import variance as _variance

logger = logging.getLogger(__name__)


def _safe_variance(values: list[float]) -> float:
    """Compute variance safely, returning 0 for < 2 values."""
    if len(values) < 2:
        return 0.0
    return _variance(values)


def score_debate(
    round3_outputs: list[dict],
    agent_weights: dict[str, float] | None = None,
) -> dict:
    """
    Convert debate Round 3 outputs into scenario probabilities.

    Args:
        round3_outputs: List of Round 3 final position dicts from each agent.
            Each must have: agent_type, final_stance (BUY/HOLD/SELL), final_confidence (0-1).
        agent_weights: Optional dict of {agent_type: weight_multiplier}.
            Default 1.0 for all agents. Higher weight = more influence on probabilities.

    Returns:
        Dict with P_bull, P_bear, P_base, supporting metadata, and traceability.
    """
    if not agent_weights:
        agent_weights = {}

    buy_agents = []
    sell_agents = []
    hold_agents = []

    for agent_output in round3_outputs:
        agent_type = agent_output.get("agent_type", "unknown")
        stance = (agent_output.get("final_stance") or "HOLD").upper()
        confidence = float(agent_output.get("final_confidence") or 0.5)

        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        # Structural limitations removed. 
        # Agents are now fully profit-maximizing entities.
        # Their raw confidence maps directly to probability.

        # Apply agent weight multiplier
        weight = float(agent_weights.get(agent_type, 1.0))
        weighted_confidence = confidence * weight

        entry = {
            "agent_type": agent_type,
            "stance": stance,
            "raw_confidence": confidence,
            "weight": weight,
            "weighted_confidence": weighted_confidence,
        }

        if stance == "BUY":
            buy_agents.append(entry)
        elif stance == "SELL":
            sell_agents.append(entry)
        else:
            hold_agents.append(entry)

    # Compute scores (normalized by total weights to prevent large counts dominating)
    total_weight = sum(a["weight"] for a in buy_agents + sell_agents + hold_agents) or 1.0
    bull_score = sum(a["weighted_confidence"] for a in buy_agents) / total_weight
    bear_score = sum(a["weighted_confidence"] for a in sell_agents) / total_weight

    # Uncertainty = variance of ALL confidences (including hold).
    # High disagreement → high uncertainty → larger P_base.
    all_confidences = [
        a.get("raw_confidence", 0.5) for a in round3_outputs
        if not a.get("error")
    ]
    uncertainty_score = _safe_variance(all_confidences)

    # If agents strongly disagree (high variance), inflate P_base
    # Variance range: [0, 0.25] for values in [0,1].
    # Scale to something meaningful: multiply by a factor to give it weight.
    # A variance of 0.1 (~moderate disagreement) should add ~0.2 to the denominator.
    uncertainty_weight = uncertainty_score * 2.0

    total = bull_score + bear_score + uncertainty_weight
    if total <= 0:
        # Complete deadlock or all HOLDs with 0 confidence
        P_bull = 0.25
        P_bear = 0.25
        P_base = 0.50
    else:
        P_bull = bull_score / total
        P_bear = bear_score / total
        P_base = 1.0 - (P_bull + P_bear)

    # Ensure P_base is non-negative (can happen if uncertainty_weight is 0)
    if P_base < 0:
        # Normalize bull and bear to leave room for base
        scale = 1.0 / (P_bull + P_bear)
        P_bull *= scale * 0.95
        P_bear *= scale * 0.95
        P_base = 1.0 - (P_bull + P_bear)

    # Clamp minimum base case probability (always some uncertainty)
    MIN_BASE = 0.10
    if P_base < MIN_BASE:
        excess = MIN_BASE - P_base
        if P_bull + P_bear > 0:
            ratio = P_bull / (P_bull + P_bear)
            P_bull -= excess * ratio
            P_bear -= excess * (1 - ratio)
        P_base = MIN_BASE

    # Final normalization to guarantee sum = 1.0
    total_prob = P_bull + P_bear + P_base
    if abs(total_prob - 1.0) > 1e-9:
        P_bull /= total_prob
        P_bear /= total_prob
        P_base /= total_prob

    # Compute conviction (agreement level)
    n_agents = len(round3_outputs)
    if n_agents > 0:
        majority_count = max(len(buy_agents), len(sell_agents), len(hold_agents))
        consensus_strength = majority_count / n_agents
    else:
        consensus_strength = 0.0

    result = {
        "P_bull": round(P_bull, 4),
        "P_bear": round(P_bear, 4),
        "P_base": round(P_base, 4),
        "bull_score": round(bull_score, 4),
        "bear_score": round(bear_score, 4),
        "uncertainty_score": round(uncertainty_score, 6),
        "bull_agents": [a["agent_type"] for a in buy_agents],
        "bear_agents": [a["agent_type"] for a in sell_agents],
        "hold_agents": [a["agent_type"] for a in hold_agents],
        "consensus_strength": round(consensus_strength, 4),
        "agent_details": buy_agents + sell_agents + hold_agents,
        "stance_distribution": {
            "BUY": len(buy_agents),
            "SELL": len(sell_agents),
            "HOLD": len(hold_agents),
        },
    }

    # Validate
    prob_sum = result["P_bull"] + result["P_bear"] + result["P_base"]
    if abs(prob_sum - 1.0) > 0.01:
        logger.error(
            "Probability sum violation: P_bull=%s + P_bear=%s + P_base=%s = %s",
            result["P_bull"], result["P_bear"], result["P_base"], prob_sum,
        )

    return result


def compute_conviction(debate_result: dict) -> float:
    """
    Compute conviction score (0-1) from debate result.
    Higher = more agent agreement. Lower = more disagreement.

    Used as weight modifier in portfolio construction:
      weight = EV * conviction
    """
    consensus = debate_result.get("consensus_strength", 0.5)

    # Also factor in average confidence
    details = debate_result.get("agent_details", [])
    if details:
        avg_confidence = sum(d.get("raw_confidence", 0.5) for d in details) / len(details)
    else:
        avg_confidence = 0.5

    # conviction = average of consensus and average confidence
    conviction = (consensus + avg_confidence) / 2.0

    return round(max(0.0, min(1.0, conviction)), 4)
