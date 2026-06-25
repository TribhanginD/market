"""
Agent Memory — persistent store for tracking agent predictions, actual outcomes,
accuracy, and calibration error.

Used to weight agents in the debate scorer:
  - High-performing agents get increased weight
  - Poor agents get decreased weight
"""

from pathlib import Path
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)

MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    prediction_date TEXT NOT NULL,
    predicted_stance TEXT NOT NULL,
    predicted_confidence REAL NOT NULL,
    actual_return_30d REAL,
    actual_return_90d REAL,
    actual_return_12m REAL,
    accuracy_30d INTEGER,
    calibration_error REAL,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS agent_weights (
    agent_type TEXT PRIMARY KEY,
    current_weight REAL NOT NULL DEFAULT 1.0,
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    avg_calibration_error REAL DEFAULT 0.0,
    last_updated TEXT
);
"""


@contextmanager
def _connect():
    path = Path(config.DB_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_memory_tables() -> None:
    """Create agent memory tables if they don't exist."""
    with _connect() as conn:
        conn.executescript(MEMORY_SCHEMA)


def record_predictions(
    run_id: str,
    debate_results: list[dict],
) -> int:
    """
    Record agent predictions from debate results for later accuracy tracking.

    Args:
        run_id: Pipeline run ID
        debate_results: List of debate transcripts (one per stock)

    Returns:
        Number of predictions recorded
    """
    init_memory_tables()
    now = datetime.now().isoformat()
    count = 0

    with _connect() as conn:
        for debate in debate_results:
            symbol = debate.get("symbol", "")
            round3 = debate.get("round3", [])

            for agent_output in round3:
                agent_type = agent_output.get("agent_type", "")
                stance = agent_output.get("final_stance", "HOLD")
                confidence = float(agent_output.get("final_confidence", 0.5))

                if not agent_type or not symbol:
                    continue

                conn.execute(
                    """
                    INSERT INTO agent_predictions
                        (run_id, agent_type, symbol, prediction_date,
                         predicted_stance, predicted_confidence, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (run_id, agent_type, symbol, now, stance, confidence, now),
                )
                count += 1

    logger.info(f"Recorded {count} agent predictions for run {run_id}")
    return count


def load_agent_weights() -> dict[str, float]:
    """
    Load current agent weights from the database.

    Returns:
        Dict of {agent_type: weight_multiplier}.
        Returns {agent: 1.0} for all agents if no data exists.
    """
    init_memory_tables()
    default_agents = ["growth", "value", "macro", "risk"]
    weights = {agent: 1.0 for agent in default_agents}

    try:
        with _connect() as conn:
            rows = conn.execute("SELECT agent_type, current_weight FROM agent_weights").fetchall()
            for row in rows:
                agent_type = row["agent_type"]
                weight = float(row["current_weight"])
                # Clamp to bounds
                weight = max(
                    getattr(config, "AGENT_WEIGHT_MIN", 0.5),
                    min(getattr(config, "AGENT_WEIGHT_MAX", 2.0), weight),
                )
                weights[agent_type] = weight
    except Exception as e:
        logger.warning(f"Failed to load agent weights: {e}")

    return weights


def update_outcomes(
    symbol: str,
    actual_return_30d: Optional[float] = None,
    actual_return_90d: Optional[float] = None,
    actual_return_12m: Optional[float] = None,
) -> int:
    """
    Update actual outcomes for a symbol's predictions.
    Called by a scheduled job after 30/90/12m have elapsed.

    Returns:
        Number of predictions updated
    """
    init_memory_tables()
    now = datetime.now().isoformat()
    updated = 0

    with _connect() as conn:
        # Find predictions for this symbol that haven't been scored yet
        rows = conn.execute(
            """
            SELECT id, predicted_stance, predicted_confidence
            FROM agent_predictions
            WHERE symbol = ? AND actual_return_30d IS NULL
            """,
            (symbol,),
        ).fetchall()

        for row in rows:
            pred_id = row["id"]
            stance = row["predicted_stance"]
            confidence = float(row["predicted_confidence"])

            accuracy = None
            cal_error = None  # Using this column for Brier Score now

            if actual_return_30d is not None:
                # Accuracy: did the stance predict the direction correctly?
                if stance == "BUY" and actual_return_30d > 0:
                    accuracy = 1
                elif stance == "SELL" and actual_return_30d < 0:
                    accuracy = 1
                elif stance == "HOLD" and abs(actual_return_30d) < 0.05:
                    accuracy = 1
                else:
                    accuracy = 0

                # Brier Score = (predicted_prob - actual_outcome)^2
                # Lower is better. 0.0 = perfect, 1.0 = completely wrong
                actual_outcome = 1.0 if accuracy else 0.0
                cal_error = (confidence - actual_outcome) ** 2

            conn.execute(
                """
                UPDATE agent_predictions SET
                    actual_return_30d = ?,
                    actual_return_90d = ?,
                    actual_return_12m = ?,
                    accuracy_30d = ?,
                    calibration_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (actual_return_30d, actual_return_90d, actual_return_12m,
                 accuracy, cal_error, now, pred_id),
            )
            updated += 1

    if updated > 0:
        _recompute_agent_weights()

    return updated


def _recompute_agent_weights() -> None:
    """Recompute agent weights based on accumulated accuracy data."""
    init_memory_tables()
    now = datetime.now().isoformat()

    weight_min = getattr(config, "AGENT_WEIGHT_MIN", 0.5)
    weight_max = getattr(config, "AGENT_WEIGHT_MAX", 2.0)

    with _connect() as conn:
        agents = conn.execute(
            """
            SELECT agent_type,
                   COUNT(*) as total,
                   SUM(CASE WHEN accuracy_30d = 1 THEN 1 ELSE 0 END) as correct,
                   AVG(COALESCE(calibration_error, 0.5)) as avg_cal_error
            FROM agent_predictions
            WHERE accuracy_30d IS NOT NULL
            GROUP BY agent_type
            """,
        ).fetchall()

        for row in agents:
            agent_type = row["agent_type"]
            total = int(row["total"])
            correct = int(row["correct"])
            avg_cal_error = float(row["avg_cal_error"])

            if total < 5:
                # Not enough data — keep at 1.0
                weight = 1.0
            else:
                accuracy_rate = correct / total
                # weight = 1.0 + (accuracy_bonus) - (calibration_penalty)
                accuracy_bonus = (accuracy_rate - 0.5) * 1.0  # [-0.5, +0.5]
                calibration_penalty = avg_cal_error * 0.5  # [0, 0.5]
                weight = 1.0 + accuracy_bonus - calibration_penalty

            weight = max(weight_min, min(weight_max, weight))

            conn.execute(
                """
                INSERT INTO agent_weights (agent_type, current_weight, total_predictions,
                                           correct_predictions, avg_calibration_error, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_type) DO UPDATE SET
                    current_weight = excluded.current_weight,
                    total_predictions = excluded.total_predictions,
                    correct_predictions = excluded.correct_predictions,
                    avg_calibration_error = excluded.avg_calibration_error,
                    last_updated = excluded.last_updated
                """,
                (agent_type, weight, total, correct, avg_cal_error, now),
            )

            logger.info(
                f"Agent weight updated: {agent_type} = {weight:.3f} "
                f"(accuracy={correct}/{total}, cal_error={avg_cal_error:.3f})"
            )


def get_agent_performance_summary() -> list[dict]:
    """Return a summary of agent performance for display."""
    init_memory_tables()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT agent_type, current_weight, total_predictions,
                   correct_predictions, avg_calibration_error, last_updated
            FROM agent_weights
            ORDER BY current_weight DESC
            """,
        ).fetchall()
        return [dict(row) for row in rows]
