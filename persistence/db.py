import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import config

logger = logging.getLogger(__name__)


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  stage_start INTEGER,
  stage_end INTEGER,
  status TEXT NOT NULL,
  meta_json TEXT
);

CREATE TABLE IF NOT EXISTS positions (
  run_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  company_name TEXT,
  sector TEXT,
  allocation_pct REAL,
  allocation_inr REAL,
  current_price REAL,
  entry_price REAL,
  ev_12m_return REAL,
  conviction TEXT,
  rationale TEXT,
  entry_note TEXT,
  exit_trigger TEXT,
  last_updated TEXT,
  payload_json TEXT,
  PRIMARY KEY (run_id, symbol)
);

CREATE TABLE IF NOT EXISTS current_positions (
  symbol TEXT PRIMARY KEY,
  company_name TEXT,
  sector TEXT,
  allocation_pct REAL,
  allocation_inr REAL,
  current_price REAL,
  entry_price REAL,
  ev_12m_return REAL,
  conviction TEXT,
  rationale TEXT,
  entry_note TEXT,
  exit_trigger TEXT,
  last_updated TEXT,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  action TEXT NOT NULL,
  symbol TEXT NOT NULL,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS thesis_checks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  checked_at TEXT NOT NULL,
  symbol TEXT NOT NULL,
  thesis_status TEXT,
  alert_level TEXT,
  action_recommended TEXT,
  reason TEXT,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS research_artifacts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  created_at TEXT NOT NULL,
  kind TEXT NOT NULL,
  agent_id INTEGER,
  payload_json TEXT
 );

CREATE TABLE IF NOT EXISTS decision_artifacts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  stage INTEGER NOT NULL,
  symbol TEXT,
  created_at TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS source_payloads (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  category TEXT NOT NULL,
  fetched_at TEXT NOT NULL,
  published_at TEXT,
  symbol TEXT,
  company TEXT,
  url TEXT,
  url_hash TEXT NOT NULL,
  raw_path TEXT NOT NULL,
  content_type TEXT,
  payload_hash TEXT,
  meta_json TEXT,
  UNIQUE(source, category, url_hash, payload_hash)
);

CREATE TABLE IF NOT EXISTS corporate_actions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  symbol TEXT,
  company TEXT,
  subject TEXT,
  ex_date TEXT,
  record_date TEXT,
  published_at TEXT,
  url TEXT,
  raw_path TEXT,
  hash TEXT UNIQUE,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS financial_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  symbol TEXT,
  company TEXT,
  industry TEXT,
  period TEXT,
  relating_to TEXT,
  financial_year TEXT,
  from_date TEXT,
  to_date TEXT,
  filing_date TEXT,
  published_at TEXT,
  xbrl_url TEXT,
  raw_path TEXT,
  hash TEXT UNIQUE,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS bulk_deals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  symbol TEXT,
  company TEXT,
  event_date TEXT,
  client_name TEXT,
  buy_sell TEXT,
  quantity REAL,
  price REAL,
  remarks TEXT,
  raw_path TEXT,
  hash TEXT UNIQUE,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS block_deals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  symbol TEXT,
  company TEXT,
  event_date TEXT,
  client_name TEXT,
  buy_sell TEXT,
  quantity REAL,
  price REAL,
  remarks TEXT,
  raw_path TEXT,
  hash TEXT UNIQUE,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS results_calendar (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  symbol TEXT,
  company TEXT,
  event_date TEXT,
  purpose TEXT,
  raw_path TEXT,
  hash TEXT UNIQUE,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS reference_data (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  category TEXT NOT NULL,
  symbol TEXT,
  company TEXT,
  event_date TEXT,
  url TEXT,
  raw_path TEXT,
  hash TEXT UNIQUE,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS filings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  symbol TEXT,
  company TEXT,
  title TEXT,
  event_date TEXT,
  published_at TEXT,
  url TEXT,
  raw_path TEXT,
  hash TEXT UNIQUE,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS insider_trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  symbol TEXT,
  company TEXT,
  person_name TEXT,
  category_of_person TEXT,
  transaction_type TEXT,
  event_date TEXT,
  reported_at TEXT,
  quantity REAL,
  value REAL,
  url TEXT,
  raw_path TEXT,
  hash TEXT UNIQUE,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS macro_drivers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  category TEXT NOT NULL,
  series_name TEXT,
  event_date TEXT,
  value REAL,
  unit TEXT,
  url TEXT,
  raw_path TEXT,
  hash TEXT UNIQUE,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  source_table TEXT NOT NULL,
  source_row_id INTEGER,
  symbol TEXT,
  company TEXT,
  title TEXT,
  doc_type TEXT NOT NULL,
  url TEXT,
  raw_path TEXT,
  content_type TEXT,
  published_at TEXT,
  event_date TEXT,
  hash TEXT UNIQUE,
  meta_json TEXT,
  extracted_text TEXT,
  text_hash TEXT,
  extraction_status TEXT DEFAULT 'pending',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  char_count INTEGER,
  hash TEXT,
  created_at TEXT NOT NULL,
  UNIQUE(document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS document_jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL,
  job_type TEXT NOT NULL,
  priority INTEGER NOT NULL DEFAULT 50,
  status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0,
  retry_at TEXT,
  model TEXT,
  last_error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(document_id, job_type)
);

CREATE TABLE IF NOT EXISTS document_memos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL,
  memo_type TEXT NOT NULL,
  model TEXT,
  created_at TEXT NOT NULL,
  input_tokens INTEGER DEFAULT 0,
  output_tokens INTEGER DEFAULT 0,
  confidence REAL,
  payload_json TEXT,
  raw_text TEXT,
  hash TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS symbol_memory (
  symbol TEXT PRIMARY KEY,
  company TEXT,
  updated_at TEXT NOT NULL,
  summary_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS etl_packets (
  symbol TEXT NOT NULL,
  as_of_date TEXT NOT NULL,
  created_at TEXT NOT NULL,
  packet_json TEXT NOT NULL,
  hash TEXT NOT NULL,
  PRIMARY KEY (symbol, as_of_date)
);

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

CREATE TABLE IF NOT EXISTS debate_transcripts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  created_at TEXT NOT NULL,
  transcript_json TEXT NOT NULL,
  debate_scores_json TEXT,
  conviction REAL
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


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)


def upsert_run(
    run_id: str,
    *,
    started_at: str,
    stage_start: Optional[int],
    stage_end: Optional[int],
    status: str,
    ended_at: Optional[str] = None,
    meta: Optional[dict[str, Any]] = None,
) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, started_at, ended_at, stage_start, stage_end, status, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
              ended_at=excluded.ended_at,
              stage_start=excluded.stage_start,
              stage_end=excluded.stage_end,
              status=excluded.status,
              meta_json=excluded.meta_json
            """,
            (
                run_id,
                started_at,
                ended_at,
                stage_start,
                stage_end,
                status,
                json.dumps(meta or {}, default=str),
            ),
        )


def save_positions(run_id: str, positions: Iterable[dict[str, Any]]) -> None:
    init_db()
    now = datetime.now().isoformat()
    with _connect() as conn:
        for position in positions:
            symbol = (position.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            conn.execute(
                """
                INSERT INTO positions (
                  run_id, symbol, company_name, sector, allocation_pct, allocation_inr,
                  current_price, entry_price, ev_12m_return, conviction, rationale,
                  entry_note, exit_trigger, last_updated, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, symbol) DO UPDATE SET
                  company_name=excluded.company_name,
                  sector=excluded.sector,
                  allocation_pct=excluded.allocation_pct,
                  allocation_inr=excluded.allocation_inr,
                  current_price=excluded.current_price,
                  entry_price=excluded.entry_price,
                  ev_12m_return=excluded.ev_12m_return,
                  conviction=excluded.conviction,
                  rationale=excluded.rationale,
                  entry_note=excluded.entry_note,
                  exit_trigger=excluded.exit_trigger,
                  last_updated=excluded.last_updated,
                  payload_json=excluded.payload_json
                """,
                (
                    run_id,
                    symbol,
                    position.get("company_name"),
                    position.get("sector"),
                    position.get("allocation_pct"),
                    position.get("allocation_inr"),
                    position.get("current_price"),
                    position.get("entry_price"),
                    position.get("ev_12m_return"),
                    position.get("conviction"),
                    position.get("rationale") or position.get("position_rationale"),
                    position.get("entry_note"),
                    position.get("exit_trigger"),
                    position.get("last_updated") or now,
                    json.dumps(position, default=str),
                ),
            )


def replace_current_positions(positions: Iterable[dict[str, Any]]) -> None:
    init_db()
    now = datetime.now().isoformat()
    with _connect() as conn:
        conn.execute("DELETE FROM current_positions")
        for position in positions:
            symbol = (position.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            conn.execute(
                """
                INSERT INTO current_positions (
                  symbol, company_name, sector, allocation_pct, allocation_inr,
                  current_price, entry_price, ev_12m_return, conviction, rationale,
                  entry_note, exit_trigger, last_updated, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    position.get("company_name"),
                    position.get("sector"),
                    position.get("allocation_pct"),
                    position.get("allocation_inr"),
                    position.get("current_price"),
                    position.get("entry_price"),
                    position.get("ev_12m_return"),
                    position.get("conviction"),
                    position.get("rationale") or position.get("position_rationale"),
                    position.get("entry_note"),
                    position.get("exit_trigger"),
                    position.get("last_updated") or now,
                    json.dumps(position, default=str),
                ),
            )


def append_trades(run_id: str, trades: Iterable[dict[str, Any]]) -> None:
    init_db()
    ts = datetime.now().isoformat()
    with _connect() as conn:
        for trade in trades:
            symbol = (trade.get("symbol") or "").strip().upper()
            action = (trade.get("action") or "").strip().upper()
            if not symbol or not action:
                continue
            conn.execute(
                """
                INSERT INTO trades (run_id, timestamp, action, symbol, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, ts, action, symbol, json.dumps(trade, default=str)),
            )


def append_thesis_checks(results: Iterable[dict[str, Any]]) -> None:
    init_db()
    checked_at = datetime.now().isoformat()
    with _connect() as conn:
        for result in results:
            symbol = (result.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            conn.execute(
                """
                INSERT INTO thesis_checks (
                  checked_at, symbol, thesis_status, alert_level, action_recommended, reason, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checked_at,
                    symbol,
                    result.get("thesis_status"),
                    result.get("alert_level"),
                    result.get("action_recommended"),
                    result.get("reason"),
                    json.dumps(result, default=str),
                ),
            )


def append_research_artifacts(
    run_id: str,
    symbol: str,
    artifacts: Iterable[dict[str, Any]],
) -> None:
    """
    Persist large research outputs (bull/bear cases + synthesis packets) so later stages
    can retrieve them without re-injecting raw text into the LLM context.
    """
    init_db()
    created_at = datetime.now().isoformat()
    sym = (symbol or "").strip().upper()
    if not sym:
        return

    with _connect() as conn:
        for artifact in artifacts:
            kind = (artifact.get("kind") or "").strip()
            if not kind:
                continue
            agent_id = artifact.get("agent_id")
            payload = artifact.get("payload")
            conn.execute(
                """
                INSERT INTO research_artifacts (run_id, symbol, created_at, kind, agent_id, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    sym,
                    created_at,
                    kind,
                    int(agent_id) if agent_id is not None else None,
                    json.dumps(payload, default=str),
                ),
            )


def append_decision_artifacts(
    run_id: str,
    stage: int,
    artifacts: Iterable[dict[str, Any]],
) -> None:
    """
    Persist stage decision commentary and rationale.
    """
    init_db()
    created_at = datetime.now().isoformat()
    with _connect() as conn:
        for artifact in artifacts:
            kind = (artifact.get("kind") or "").strip()
            if not kind:
                continue
            symbol = (artifact.get("symbol") or "").strip().upper() or None
            conn.execute(
                """
                INSERT INTO decision_artifacts (run_id, stage, symbol, created_at, kind, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    int(stage),
                    symbol,
                    created_at,
                    kind,
                    json.dumps(artifact.get("payload"), default=str),
                ),
            )


def load_current_portfolio() -> dict[str, Any]:
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM current_positions ORDER BY allocation_pct DESC").fetchall()
        positions = []
        for row in rows:
            payload = row["payload_json"]
            if payload:
                try:
                    positions.append(json.loads(payload))
                    continue
                except Exception:
                    pass
            positions.append(dict(row))
        return {"positions": positions, "last_run": _latest_run_id(conn)}

def get_recent_flagged_symbols(
    *,
    max_symbols: int = 2,
    lookback_hours: int = 48,
    alert_levels: tuple[str, ...] = ("URGENT", "WATCH"),
) -> list[str]:
    """
    Return a de-duplicated list of symbols recently flagged by the thesis monitor.
    Most recent first. Uses thesis_checks table.
    """
    init_db()
    max_symbols = max(0, int(max_symbols or 0))
    if max_symbols <= 0:
        return []

    levels = tuple((a or "").strip().upper() for a in alert_levels if (a or "").strip())
    if not levels:
        return []

    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT symbol, checked_at, alert_level
            FROM thesis_checks
            WHERE checked_at >= datetime('now', ?)
              AND upper(alert_level) IN ({",".join(["?"] * len(levels))})
            ORDER BY checked_at DESC
            """,
            (f"-{int(lookback_hours)} hours", *levels),
        ).fetchall()

    seen = set()
    out: list[str] = []
    for r in rows:
        sym = (r["symbol"] or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
        if len(out) >= max_symbols:
            break
    return out


def _latest_run_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
    return row["run_id"] if row else ""


def insert_source_payload(record: dict[str, Any]) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO source_payloads (
              source, category, fetched_at, published_at, symbol, company, url, url_hash,
              raw_path, content_type, payload_hash, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("source"),
                record.get("category"),
                record.get("fetched_at"),
                record.get("published_at"),
                record.get("symbol"),
                record.get("company"),
                record.get("url"),
                record.get("url_hash"),
                record.get("raw_path"),
                record.get("content_type"),
                record.get("payload_hash"),
                json.dumps(record.get("meta") or {}, default=str),
            ),
        )


def insert_normalized_rows(table: str, rows: Iterable[dict[str, Any]]) -> int:
    init_db()
    allowed = {
        "corporate_actions": (
            "source", "symbol", "company", "subject", "ex_date", "record_date",
            "published_at", "url", "raw_path", "hash", "payload_json",
        ),
        "financial_results": (
            "source", "symbol", "company", "industry", "period", "relating_to",
            "financial_year", "from_date", "to_date", "filing_date", "published_at",
            "xbrl_url", "raw_path", "hash", "payload_json",
        ),
        "bulk_deals": (
            "source", "symbol", "company", "event_date", "client_name", "buy_sell",
            "quantity", "price", "remarks", "raw_path", "hash", "payload_json",
        ),
        "block_deals": (
            "source", "symbol", "company", "event_date", "client_name", "buy_sell",
            "quantity", "price", "remarks", "raw_path", "hash", "payload_json",
        ),
        "results_calendar": (
            "source", "symbol", "company", "event_date", "purpose", "raw_path",
            "hash", "payload_json",
        ),
        "reference_data": (
            "source", "category", "symbol", "company", "event_date", "url",
            "raw_path", "hash", "payload_json",
        ),
        "filings": (
            "source", "symbol", "company", "title", "event_date", "published_at",
            "url", "raw_path", "hash", "payload_json",
        ),
        "insider_trades": (
            "source", "symbol", "company", "person_name", "category_of_person",
            "transaction_type", "event_date", "reported_at", "quantity", "value",
            "url", "raw_path", "hash", "payload_json",
        ),
        "macro_drivers": (
            "source", "category", "series_name", "event_date", "value", "unit",
            "url", "raw_path", "hash", "payload_json",
        ),
    }
    columns = allowed.get(table)
    if not columns:
        raise ValueError(f"Unsupported table: {table}")

    placeholders = ", ".join(["?"] * len(columns))
    inserted = 0
    with _connect() as conn:
        for row in rows:
            payload = dict(row)
            if "payload_json" not in payload:
                payload["payload_json"] = json.dumps(row, default=str)
            before = conn.total_changes
            conn.execute(
                f"INSERT OR IGNORE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                tuple(payload.get(col) for col in columns),
            )
            inserted += conn.total_changes - before
    return inserted


def upsert_document(record: dict[str, Any]) -> Optional[int]:
    init_db()
    now = datetime.now().isoformat()
    payload = dict(record)
    payload.setdefault("created_at", now)
    payload.setdefault("updated_at", now)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO documents (
              source, source_table, source_row_id, symbol, company, title, doc_type, url,
              raw_path, content_type, published_at, event_date, hash, meta_json,
              extracted_text, text_hash, extraction_status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hash) DO UPDATE SET
              symbol=excluded.symbol,
              company=excluded.company,
              title=excluded.title,
              doc_type=excluded.doc_type,
              url=excluded.url,
              raw_path=excluded.raw_path,
              content_type=excluded.content_type,
              published_at=excluded.published_at,
              event_date=excluded.event_date,
              meta_json=excluded.meta_json,
              updated_at=excluded.updated_at
            """,
            (
                payload.get("source"),
                payload.get("source_table"),
                payload.get("source_row_id"),
                payload.get("symbol"),
                payload.get("company"),
                payload.get("title"),
                payload.get("doc_type"),
                payload.get("url"),
                payload.get("raw_path"),
                payload.get("content_type"),
                payload.get("published_at"),
                payload.get("event_date"),
                payload.get("hash"),
                json.dumps(payload.get("meta") or {}, default=str),
                payload.get("extracted_text"),
                payload.get("text_hash"),
                payload.get("extraction_status") or "pending",
                payload.get("created_at"),
                payload.get("updated_at"),
            ),
        )
        row = conn.execute("SELECT id FROM documents WHERE hash = ?", (payload.get("hash"),)).fetchone()
        return int(row["id"]) if row else None


def update_document_extraction(document_id: int, *, extracted_text: str, text_hash: str, extraction_status: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE documents
            SET extracted_text = ?, text_hash = ?, extraction_status = ?, updated_at = ?
            WHERE id = ?
            """,
            (extracted_text, text_hash, extraction_status, datetime.now().isoformat(), int(document_id)),
        )


def replace_document_chunks(document_id: int, chunks: Iterable[dict[str, Any]]) -> None:
    init_db()
    now = datetime.now().isoformat()
    with _connect() as conn:
        conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (int(document_id),))
        for chunk in chunks:
            conn.execute(
                """
                INSERT INTO document_chunks (document_id, chunk_index, text, char_count, hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(document_id),
                    int(chunk.get("chunk_index") or 0),
                    chunk.get("text") or "",
                    int(chunk.get("char_count") or 0),
                    chunk.get("hash"),
                    now,
                ),
            )


def upsert_document_job(
    document_id: int,
    *,
    job_type: str,
    priority: int = 50,
    model: Optional[str] = None,
    status: str = "pending",
    retry_at: Optional[str] = None,
    last_error: Optional[str] = None,
) -> None:
    init_db()
    now = datetime.now().isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO document_jobs (
              document_id, job_type, priority, status, attempts, retry_at, model, last_error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id, job_type) DO UPDATE SET
              priority=excluded.priority,
              model=COALESCE(excluded.model, document_jobs.model),
              status=CASE
                WHEN document_jobs.status = 'done' THEN document_jobs.status
                ELSE excluded.status
              END,
              retry_at=excluded.retry_at,
              last_error=excluded.last_error,
              updated_at=excluded.updated_at
            """,
            (int(document_id), job_type, int(priority), status, retry_at, model, last_error, now, now),
        )


def update_document_job(
    job_id: int,
    *,
    status: str,
    attempts_increment: bool = False,
    retry_at: Optional[str] = None,
    last_error: Optional[str] = None,
) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            f"""
            UPDATE document_jobs
            SET status = ?,
                attempts = attempts + {'1' if attempts_increment else '0'},
                retry_at = ?,
                last_error = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (status, retry_at, last_error, datetime.now().isoformat(), int(job_id)),
        )


def insert_document_memo(record: dict[str, Any]) -> None:
    init_db()
    payload = dict(record)
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO document_memos (
              document_id, memo_type, model, created_at, input_tokens, output_tokens,
              confidence, payload_json, raw_text, hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(payload.get("document_id")),
                payload.get("memo_type"),
                payload.get("model"),
                payload.get("created_at") or datetime.now().isoformat(),
                int(payload.get("input_tokens") or 0),
                int(payload.get("output_tokens") or 0),
                payload.get("confidence"),
                json.dumps(payload.get("payload") or {}, default=str),
                payload.get("raw_text"),
                payload.get("hash"),
            ),
        )


def upsert_symbol_memory(symbol: str, company: Optional[str], summary: dict[str, Any]) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO symbol_memory (symbol, company, updated_at, summary_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
              company=excluded.company,
              updated_at=excluded.updated_at,
              summary_json=excluded.summary_json
            """,
            (
                (symbol or "").strip().upper(),
                company,
                datetime.now().isoformat(),
                json.dumps(summary, default=str),
            ),
        )


def upsert_etl_packet(symbol: str, as_of_date: str, packet: dict[str, Any], packet_hash: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO etl_packets (symbol, as_of_date, created_at, packet_json, hash)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol, as_of_date) DO UPDATE SET
              created_at=excluded.created_at,
              packet_json=excluded.packet_json,
              hash=excluded.hash
            """,
            (
                (symbol or "").strip().upper(),
                as_of_date,
                datetime.now().isoformat(),
                json.dumps(packet, default=str),
                packet_hash,
            ),
        )


def fetch_rows(query: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(r) for r in rows]


def mark_stale_runs(cutoff_hours: int = 2) -> int:
    """
    Mark runs stuck in 'running' status as 'stale'.
    Returns count of runs updated.
    Any run still showing 'running' after cutoff_hours is assumed crashed/interrupted.
    """
    from datetime import timedelta
    init_db()
    cutoff = (datetime.now() - timedelta(hours=cutoff_hours)).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE runs SET status='stale', ended_at=?
            WHERE status='running' AND started_at < ?
            """,
            (datetime.now().isoformat(), cutoff),
        )
        count = cur.rowcount
    if count:
        import logging
        logging.getLogger(__name__).warning("Marked %d stale run(s) (stuck > %dh)", count, cutoff_hours)
    return count
