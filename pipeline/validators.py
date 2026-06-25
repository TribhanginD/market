"""
Pipeline Validators — hard validation that throws errors if constraints are broken.

Validates:
1. Sector cap compliance
2. Probability sum = 1
3. EV traceability to debate output
4. Duplicate argument detection
5. Position size bounds
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PipelineValidationError(Exception):
    """Raised when pipeline output violates hard constraints."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Pipeline validation failed with {len(errors)} error(s): {'; '.join(errors[:5])}")


def validate_probabilities(scenario_results: list[dict]) -> list[str]:
    """Validate that all scenario probabilities sum to 1."""
    errors = []
    for stock in scenario_results:
        symbol = stock.get("symbol", "?")
        p_bull = float(stock.get("P_bull") or stock.get("debate_scores", {}).get("P_bull") or 0)
        p_bear = float(stock.get("P_bear") or stock.get("debate_scores", {}).get("P_bear") or 0)
        p_base = float(stock.get("P_base") or stock.get("debate_scores", {}).get("P_base") or 0)
        p_sum = p_bull + p_bear + p_base
        if abs(p_sum - 1.0) > 0.02:
            errors.append(f"PROBABILITY_SUM: {symbol} P_bull={p_bull:.3f} + P_bear={p_bear:.3f} + P_base={p_base:.3f} = {p_sum:.3f}")
    return errors


def validate_ev_traceability(scenario_results: list[dict]) -> list[str]:
    """Validate that EV is derived from debate-sourced scenarios."""
    errors = []
    for stock in scenario_results:
        symbol = stock.get("symbol", "?")
        debate_scores = stock.get("debate_scores", {})
        if not debate_scores:
            errors.append(f"EV_NOT_TRACEABLE: {symbol} missing debate_scores")
            continue
        if not debate_scores.get("bull_agents") and not debate_scores.get("bear_agents"):
            errors.append(f"EV_NOT_TRACEABLE: {symbol} no agents cited in bull/bear scenarios")
    return errors


def validate_sector_caps(portfolio: list[dict], max_sector_pct: float) -> list[str]:
    """Validate that no sector exceeds the cap."""
    errors = []
    sector_totals: dict[str, float] = {}
    for p in portfolio:
        sector = p.get("sector", "Unknown")
        sector_totals[sector] = sector_totals.get(sector, 0) + float(p.get("allocation_pct", 0))

    max_pct = max_sector_pct * 100
    for sector, pct in sector_totals.items():
        if pct > max_pct + 0.5:  # 0.5% tolerance for rounding
            errors.append(f"SECTOR_CAP_EXCEEDED: {sector} = {pct:.1f}% (max {max_pct:.0f}%)")
    return errors


def validate_position_bounds(portfolio: list[dict], min_pct: float, max_pct: float) -> list[str]:
    """Validate position size bounds."""
    errors = []
    min_bound = min_pct * 100
    max_bound = max_pct * 100
    for p in portfolio:
        symbol = p.get("symbol", "?")
        alloc = float(p.get("allocation_pct", 0))
        if alloc < min_bound - 0.5:
            errors.append(f"POSITION_TOO_SMALL: {symbol} = {alloc:.1f}% (min {min_bound:.0f}%)")
        if alloc > max_bound + 0.5:
            errors.append(f"POSITION_TOO_LARGE: {symbol} = {alloc:.1f}% (max {max_bound:.0f}%)")
    return errors


def validate_allocation_sum(portfolio: list[dict]) -> list[str]:
    """Validate allocations sum to ~100%."""
    errors = []
    total = sum(float(p.get("allocation_pct", 0)) for p in portfolio)
    if abs(total - 100.0) > 1.0:
        errors.append(f"ALLOCATION_SUM: total = {total:.1f}% (expected 100%)")
    return errors


def validate_duplicate_arguments(debate_results: list[dict], threshold: float = 0.8) -> list[str]:
    """
    Detect agents producing duplicate arguments within the same debate.
    Uses simple word-overlap as a proxy for semantic similarity.
    """
    errors = []
    for debate in debate_results:
        symbol = debate.get("symbol", "?")
        round1 = debate.get("round1", [])

        # Collect all arguments across agents
        all_args_by_agent: dict[str, list[str]] = {}
        for agent_output in round1:
            agent_type = agent_output.get("agent_type", "?")
            args = agent_output.get("arguments", [])
            points = [str(a.get("point", "")).lower().strip() for a in args if isinstance(a, dict)]
            all_args_by_agent[agent_type] = points

        # Cross-agent comparison
        agents = list(all_args_by_agent.keys())
        for i, a1 in enumerate(agents):
            for a2 in agents[i + 1:]:
                for p1 in all_args_by_agent.get(a1, []):
                    for p2 in all_args_by_agent.get(a2, []):
                        if _word_overlap(p1, p2) > threshold:
                            errors.append(
                                f"DUPLICATE_ARGS: {symbol} — {a1} and {a2} share: '{p1[:80]}'"
                            )
    return errors


import math
from collections import Counter

def _word_overlap(text1: str, text2: str) -> float:
    """Compute TF-IDF cosine similarity for semantic duplication check."""
    if not text1 or not text2:
        return 0.0
    
    # Tokenize very simply
    words1 = text1.split()
    words2 = text2.split()
    if not words1 or not words2:
        return 0.0
        
    c1 = Counter(words1)
    c2 = Counter(words2)
    
    # Compute Term Frequency (TF)
    tf1 = {k: v / len(words1) for k, v in c1.items()}
    tf2 = {k: v / len(words2) for k, v in c2.items()}
    
    # Compute Inverse Document Frequency (IDF) over this 2-doc corpus
    all_words = set(c1.keys()) | set(c2.keys())
    idf = {}
    for w in all_words:
        docs_with_w = sum(1 for c in (c1, c2) if w in c)
        idf[w] = math.log(2.0 / docs_with_w) if docs_with_w > 0 else 0
        
    # Compute TF-IDF vectors
    vec1 = {w: tf1.get(w, 0) * idf[w] for w in all_words}
    vec2 = {w: tf2.get(w, 0) * idf[w] for w in all_words}
    
    # Notice: If words are present in both docs, docs_with_w = 2, so log(2/2) = 0.
    # Standard TF-IDF zeros out words in all docs. 
    # For a 2-doc comparison, we need a smoothed IDF or just TF-based Cosine Similarity.
    # Let's use standard Cosine Similarity on raw TF to capture proportional overlap.
    
    dot_product = sum(tf1.get(w, 0) * tf2.get(w, 0) for w in all_words)
    mag1 = math.sqrt(sum(v**2 for v in tf1.values()))
    mag2 = math.sqrt(sum(v**2 for v in tf2.values()))
    
    if mag1 * mag2 == 0:
        return 0.0
        
    return dot_product / (mag1 * mag2)


def validate_pipeline_output(
    portfolio: list[dict],
    scenario_results: list[dict],
    debate_results: list[dict],
    max_sector_pct: float = 0.35,
    min_position_pct: float = 0.03,
    max_position_pct: float = 0.15,
    raise_on_error: bool = True,
) -> list[str]:
    """
    Run all validations on pipeline output.

    Args:
        portfolio: Final portfolio positions
        scenario_results: Stage 3 scenario models
        debate_results: Stage 2 debate transcripts
        max_sector_pct: Max sector allocation (decimal)
        min_position_pct: Min position size (decimal)
        max_position_pct: Max position size (decimal)
        raise_on_error: If True, raise PipelineValidationError on failure

    Returns:
        List of error strings (empty if all valid)

    Raises:
        PipelineValidationError: If raise_on_error=True and errors found
    """
    all_errors = []

    all_errors.extend(validate_probabilities(scenario_results))
    all_errors.extend(validate_ev_traceability(scenario_results))
    all_errors.extend(validate_sector_caps(portfolio, max_sector_pct))
    all_errors.extend(validate_position_bounds(portfolio, min_position_pct, max_position_pct))
    all_errors.extend(validate_allocation_sum(portfolio))
    all_errors.extend(validate_duplicate_arguments(debate_results))

    if all_errors:
        for error in all_errors:
            logger.warning(f"VALIDATION: {error}")
        if raise_on_error:
            raise PipelineValidationError(all_errors)

    return all_errors
