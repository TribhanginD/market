"""
Custom dashboard server for the Indian AI Portfolio.

This intentionally avoids Streamlit so the UI can be fully controlled with
HTML/CSS/JS while still reading the existing pipeline artifacts and yfinance
live data from Python.
"""

from __future__ import annotations

from pathlib import Path

import argparse
import json
import math
import mimetypes
import sqlite3
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd


import config
from data.fetcher import get_live_prices_batch
from data.nifty500 import _yf_symbol

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"


def _read_json(path: Path, fallback):
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        return fallback
    return fallback


def _safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def pct_value(value) -> float:
    val = _safe_float(value, 0.0) or 0.0
    return val * 100 if abs(val) <= 1.5 else val


def latest_run_dirs() -> list[Path]:
    return sorted([p for p in config.PIPELINE_RUNS_DIR.glob("*") if p.is_dir()], reverse=True)


def latest_run_dir() -> Path | None:
    runs = latest_run_dirs()
    return runs[0] if runs else None


def artifact(run_dir: Path | None, filename: str) -> dict:
    if not run_dir:
        return {}
    return _read_json(run_dir / filename, {})


def _run_date_label(run_id: str) -> str:
    raw = str(run_id or "")
    if len(raw) >= 8 and raw[:8].isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw


def _summarize_selected_run(run_summary: dict, stage5: dict, stage3: dict) -> dict:
    trades = stage5.get("trades", []) if isinstance(stage5, dict) else []
    actions = [str(t.get("action") or "").upper() for t in trades]
    warnings = run_summary.get("validation_warnings", []) if isinstance(run_summary, dict) else []
    positive_ev = ((stage3 or {}).get("positive_ev_count")) or ((run_summary.get("stage3") or {}).get("positive_ev_count"))
    return {
        "run_id": run_summary.get("run_id") or "",
        "date": _run_date_label(run_summary.get("run_id") or ""),
        "started_at": run_summary.get("start_time") or "",
        "duration_human": run_summary.get("duration_human") or "",
        "duration_seconds": _safe_float(run_summary.get("duration_seconds"), 0.0) or 0.0,
        "input_tokens": int(((run_summary.get("total_tokens") or {}).get("input")) or 0),
        "output_tokens": int(((run_summary.get("total_tokens") or {}).get("output")) or 0),
        "token_total": int((((run_summary.get("total_tokens") or {}).get("input")) or 0) + (((run_summary.get("total_tokens") or {}).get("output")) or 0)),
        "cost_usd": _safe_float(((run_summary.get("total_tokens") or {}).get("cost_usd")), 0.0) or 0.0,
        "trades": len(trades),
        "buys": actions.count("BUY"),
        "sells": actions.count("SELL"),
        "holds": actions.count("HOLD"),
        "warnings": len(warnings),
        "warning_items": warnings[:8],
        "positive_ev_count": int(positive_ev or 0),
        "positions_selected": int((((run_summary.get("stage4") or {}).get("positions_selected")) or 0)),
        "portfolio_ev_pct": pct_value(((run_summary.get("stage4") or {}).get("portfolio_ev"))),
        "actions_taken": [t for t in trades if str(t.get("action") or "").upper() in {"BUY", "SELL", "HOLD"}][:10],
        "notes": stage5.get("rebalance_notes") or "",
    }


def build_run_catalog(run_dirs: list[Path]) -> list[dict]:
    catalog = []
    for run_dir in run_dirs:
        run_summary = artifact(run_dir, "run_summary.json")
        stage5 = artifact(run_dir, "stage5_output.json")
        stage3 = artifact(run_dir, "stage3_output.json")
        trades = stage5.get("trades", []) if isinstance(stage5, dict) else []
        summary = run_summary.get("total_tokens") or {}
        stage_keys = [key.upper() for key in ("stage1", "stage2", "stage3", "stage4", "stage5") if run_summary.get(key)]
        actions = [str(t.get("action") or "").upper() for t in trades]
        top_symbols = []
        for trade in trades:
            symbol = str(trade.get("symbol") or "").upper().strip()
            if symbol and symbol not in top_symbols:
                top_symbols.append(symbol)
            if len(top_symbols) >= 5:
                break
        catalog.append(
            {
                "run_id": run_dir.name,
                "date": _run_date_label(run_dir.name),
                "started_at": run_summary.get("start_time") or "",
                "duration_human": run_summary.get("duration_human") or "",
                "duration_seconds": _safe_float(run_summary.get("duration_seconds"), 0.0) or 0.0,
                "stages": stage_keys,
                "input_tokens": int(summary.get("input") or 0),
                "output_tokens": int(summary.get("output") or 0),
                "token_total": int((summary.get("input") or 0) + (summary.get("output") or 0)),
                "cost_usd": _safe_float(summary.get("cost_usd"), 0.0) or 0.0,
                "positions_selected": int((((run_summary.get("stage4") or {}).get("positions_selected")) or 0)),
                "positive_ev_count": int((((run_summary.get("stage3") or {}).get("positive_ev_count")) or ((stage3 or {}).get("positive_ev_count")) or 0)),
                "portfolio_ev_pct": pct_value((((run_summary.get("stage4") or {}).get("portfolio_ev")))),
                "trades": len(trades),
                "buys": actions.count("BUY"),
                "sells": actions.count("SELL"),
                "holds": actions.count("HOLD"),
                "warnings": len(run_summary.get("validation_warnings", []) or []),
                "top_symbols": top_symbols,
                "stage3_modeled_count": int((((run_summary.get("stage3") or {}).get("modeled_count")) or 0)),
                "stage2_stocks": int((((run_summary.get("stage2") or {}).get("stocks_researched")) or 0)),
                "notes": short_text(stage5.get("rebalance_notes") or "", 180),
            }
        )
    return catalog


def short_text(value, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def scenario_entries(model: dict) -> list[dict]:
    if isinstance(model.get("scenarios"), list):
        return model["scenarios"]
    entries = []
    for key in ("bull", "base", "bear", "bull_scenario", "base_scenario", "bear_scenario"):
        raw = model.get(key)
        if not isinstance(raw, dict):
            continue
        entries.append(
            {
                "case": key.replace("_scenario", "").upper(),
                "scenario": raw.get("scenario") or raw.get("summary") or "",
                "probability": pct_value(raw.get("probability")),
                "target": _safe_float(raw.get("target") or raw.get("target_price")),
                "return_pct": pct_value(raw.get("return_pct") or raw.get("return")),
            }
        )
    return entries


def load_db_counts() -> dict:
    path = Path(config.DB_FILE)
    if not path.exists():
        return {}
    tables = [
        "runs",
        "current_positions",
        "trades",
        "thesis_checks",
        "decision_artifacts",
        "source_payloads",
        "corporate_actions",
        "financial_results",
        "filings",
        "documents",
        "document_memos",
        "etl_packets",
    ]
    out = {}
    try:
        with sqlite3.connect(path) as conn:
            for table in tables:
                try:
                    out[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                except Exception:
                    out[table] = 0
            row = conn.execute(
                "SELECT run_id, status, started_at, ended_at FROM runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            if row:
                out["latest_run"] = {
                    "run_id": row[0],
                    "status": row[1],
                    "started_at": row[2],
                    "ended_at": row[3],
                }
    except Exception as exc:
        out["error"] = str(exc)
    return out


def live_quotes(symbols: list[str]) -> dict:
    clean = [s.strip().upper() for s in symbols if s.strip()]
    yf_symbols = [_yf_symbol(s) for s in clean]
    raw = get_live_prices_batch(yf_symbols)
    out = {}
    for sym in clean:
        yf_sym = _yf_symbol(sym)
        out[sym] = raw.get(yf_sym, {"yf_symbol": yf_sym, "price": None, "timestamp": ""})
    return out


def build_positions(portfolio: dict, quotes: dict) -> list[dict]:
    portfolio_value = _safe_float(portfolio.get("portfolio_value_inr"), config.PAPER_PORTFOLIO_VALUE_INR) or 0
    rows = []
    for p in portfolio.get("positions", []):
        symbol = str(p.get("symbol") or "").upper()
        target_value = _safe_float(p.get("allocation_inr"), 0.0) or 0.0
        target_pct = _safe_float(p.get("allocation_pct"), 0.0) or 0.0
        entry_price = _safe_float(p.get("entry_price")) or _safe_float(p.get("current_price"))
        quote = quotes.get(symbol, {})
        live_price = _safe_float(quote.get("price")) or _safe_float(p.get("current_price")) or entry_price
        shares = target_value / entry_price if entry_price and entry_price > 0 else None
        live_value = shares * live_price if shares and live_price else target_value
        live_pct = live_value / portfolio_value * 100 if portfolio_value else target_pct
        row = dict(p)
        row.update(
            {
                "symbol": symbol,
                "target_value": target_value,
                "target_pct": target_pct,
                "entry_price": entry_price,
                "live_price": live_price,
                "live_value": live_value,
                "live_pct": live_pct,
                "drift_pct": live_pct - target_pct,
                "quote_time": quote.get("timestamp", ""),
                "ev_12m_pct": pct_value(p.get("ev_12m_return")),
                "rationale": p.get("rationale") or p.get("position_rationale") or "",
            }
        )
        rows.append(row)
    return rows


def normalize_stage4_portfolio(stage4: dict) -> dict:
    positions = []
    for raw in (stage4 or {}).get("portfolio", []) or []:
        item = dict(raw)
        item["allocation_pct"] = item.get("allocation_pct") or item.get("target_allocation_pct") or 0
        item["allocation_inr"] = item.get("allocation_inr") or item.get("target_allocation_inr") or 0
        item["rationale"] = item.get("rationale") or item.get("position_rationale") or ""
        positions.append(item)
    return {
        "positions": positions,
        "portfolio_value_inr": config.PAPER_PORTFOLIO_VALUE_INR,
        "last_run": (stage4 or {}).get("run_time") or "",
    }


def flatten_trades(trades_log: list[dict]) -> list[dict]:
    rows = []
    for entry in trades_log:
        for trade in entry.get("trades", []):
            symbol = str(trade.get("symbol") or "").upper()
            if not symbol:
                continue
            row = dict(trade)
            row.update(
                {
                    "date": entry.get("date"),
                    "run_id": entry.get("run_id"),
                    "action": (trade.get("action") or "").upper(),
                    "symbol": symbol,
                    "target_pct": trade.get("target_allocation_pct") or trade.get("new_target_allocation_pct"),
                    "current_pct": trade.get("current_allocation_pct"),
                    "change_pct": trade.get("allocation_change_pct"),
                    "ev_12m_pct": pct_value(trade.get("ev_12m_return")),
                    "reason": trade.get("rationale") or trade.get("status_note") or "",
                    "counterargument": trade.get("counterargument") or "",
                }
            )
            rows.append(row)
    return rows


def history_payload(symbols: list[str], period: str) -> dict:
    if yf is None or not symbols:
        return {"dates": [], "series": []}
    clean = [s.strip().upper() for s in symbols if s.strip()]
    tickers = [_yf_symbol(s) for s in clean]
    if config.BENCHMARK_SYMBOL not in tickers:
        tickers.append(config.BENCHMARK_SYMBOL)
    try:
        hist = yf.download(
            tickers,
            period=period,
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True,
            auto_adjust=True,
        )
    except Exception as exc:
        return {"dates": [], "series": [], "error": str(exc)}
    if hist is None or hist.empty:
        return {"dates": [], "series": []}

    close = pd.DataFrame()
    if hasattr(hist.columns, "nlevels") and hist.columns.nlevels == 2:
        level0 = set(map(str, hist.columns.get_level_values(0)))
        if any(t in level0 for t in tickers):
            for ticker in tickers:
                try:
                    close[ticker] = hist[(ticker, "Close")]
                except Exception:
                    pass
        else:
            for ticker in tickers:
                try:
                    close[ticker] = hist[("Close", ticker)]
                except Exception:
                    pass
    else:
        if len(tickers) == 1 and "Close" in hist:
            close[tickers[0]] = hist["Close"]

    close = close.ffill().dropna(how="all")
    if close.empty:
        return {"dates": [], "series": []}
    normalized = close / close.iloc[0] * 100
    dates = [idx.strftime("%Y-%m-%d") for idx in normalized.index]
    series = []
    reverse = {_yf_symbol(s): s for s in clean}
    reverse[config.BENCHMARK_SYMBOL] = config.BENCHMARK_NAME
    for col in normalized.columns:
        values = [_safe_float(v) for v in normalized[col].tolist()]
        series.append({"name": reverse.get(col, col), "values": values})
    return {"dates": dates, "series": series}


def snapshot(run_id: str | None = None, include_live: bool = True) -> dict:
    runs = latest_run_dirs()
    run_dir = None
    if run_id:
        run_dir = next((r for r in runs if r.name == run_id), None)
    run_dir = run_dir or latest_run_dir()

    current_portfolio = _read_json(config.PORTFOLIO_FILE, {})
    trades_log = _read_json(config.TRADES_LOG_FILE, [])
    thesis_alerts = _read_json(config.THESIS_ALERTS_FILE, {})
    stage3 = artifact(run_dir, "stage3_output.json")
    stage4 = artifact(run_dir, "stage4_output.json")
    stage5 = artifact(run_dir, "stage5_output.json")
    run_summary = artifact(run_dir, "run_summary.json")
    run_catalog = build_run_catalog(runs)
    portfolio = normalize_stage4_portfolio(stage4) if stage4.get("portfolio") else current_portfolio
    selected_trades_log_entry = next((entry for entry in trades_log if entry.get("run_id") == (run_dir.name if run_dir else "")), None)
    selected_trades = []
    if stage5.get("trades"):
        selected_trades = stage5.get("trades", [])
    elif selected_trades_log_entry:
        selected_trades = selected_trades_log_entry.get("trades", [])
    symbols = [str(p.get("symbol") or "").upper() for p in portfolio.get("positions", []) if p.get("symbol")]
    quotes = live_quotes(symbols) if include_live else {}
    positions = build_positions(portfolio, quotes)
    trades = flatten_trades(trades_log)
    selected_trades_flat = flatten_trades([selected_trades_log_entry]) if selected_trades_log_entry else []
    if stage5.get("trades"):
        selected_trades_flat = flatten_trades([{
            "run_id": run_dir.name if run_dir else "",
            "date": _run_date_label(run_dir.name if run_dir else ""),
            "trades": stage5.get("trades", []),
        }])

    scenario_models = []
    for model in stage3.get("scenario_models", []) if isinstance(stage3, dict) else []:
        scenario_models.append(
            {
                "symbol": model.get("symbol"),
                "company_name": model.get("company_name"),
                "sector": model.get("sector"),
                "recommendation": model.get("recommendation"),
                "ev_12m_pct": pct_value(model.get("probability_weighted_return_12m")),
                "cases": scenario_entries(model),
                "debate_log": model.get("debate_log") or {},
            }
        )

    target_total = sum(_safe_float(p.get("target_pct"), 0.0) or 0.0 for p in positions)
    weighted_ev = (
        sum((_safe_float(p.get("target_pct"), 0.0) or 0.0) * (_safe_float(p.get("ev_12m_pct"), 0.0) or 0.0) for p in positions)
        / target_total
        if target_total
        else 0.0
    )
    live_value = sum(_safe_float(p.get("live_value"), 0.0) or 0.0 for p in positions)
    portfolio_value = _safe_float(portfolio.get("portfolio_value_inr"), config.PAPER_PORTFOLIO_VALUE_INR) or 0.0
    quote_times = [q.get("timestamp") for q in quotes.values() if q.get("timestamp")]
    today_key = datetime.now().strftime("%Y-%m-%d")
    today_alerts = thesis_alerts.get(today_key, {}) if isinstance(thesis_alerts, dict) else {}

    return {
        "as_of": datetime.now().isoformat(),
        "runs": [r.name for r in runs],
        "selected_run": run_dir.name if run_dir else "",
        "portfolio": portfolio,
        "current_portfolio": current_portfolio,
        "positions": positions,
        "trades": trades,
        "selected_trades": selected_trades_flat,
        "trades_log": trades_log,
        "thesis_alerts": thesis_alerts,
        "scenario_models": scenario_models,
        "stage4": stage4,
        "stage5": stage5,
        "run_summary": run_summary,
        "run_catalog": run_catalog,
        "selected_run_summary": _summarize_selected_run(run_summary, stage5, stage3),
        "db": load_db_counts(),
        "rules": {
            "capital_inr": config.PAPER_PORTFOLIO_VALUE_INR,
            "max_positions": config.MAX_POSITIONS,
            "min_positions": config.MIN_POSITIONS,
            "max_position_pct": config.MAX_POSITION_PCT * 100,
            "max_sector_pct": config.MAX_SECTOR_PCT * 100,
            "swap_hurdle_pct": config.MIN_EV_IMPROVEMENT_TO_SWAP * 100,
            "turnover_cap_pct": config.MAX_PORTFOLIO_TURNOVER_PER_CYCLE * 100,
            "benchmark": config.BENCHMARK_NAME,
        },
        "metrics": {
            "portfolio_value": portfolio_value,
            "live_value": live_value,
            "live_return_pct": ((live_value / portfolio_value) - 1) * 100 if portfolio_value else 0.0,
            "weighted_ev_pct": weighted_ev,
            "positions": len(positions),
            "urgent_alerts": len(today_alerts.get("urgent", [])) if isinstance(today_alerts, dict) else 0,
            "quote_time": max(quote_times) if quote_times else "",
            "last_run": portfolio.get("last_run", ""),
        },
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "MarketDashboard/1.0"

    def log_message(self, fmt, *args):
        print(f"[dashboard] {self.address_string()} - {fmt % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/snapshot":
            qs = parse_qs(parsed.query)
            run_id = (qs.get("run") or [None])[0]
            live = (qs.get("live") or ["1"])[0] != "0"
            self._json(snapshot(run_id=run_id, include_live=live))
            return
        if parsed.path == "/api/history":
            qs = parse_qs(parsed.query)
            symbols = ",".join(qs.get("symbols", []))
            period = (qs.get("period") or ["3mo"])[0]
            self._json(history_payload(symbols.split(","), period))
            return
        self._static(parsed.path)

    def do_HEAD(self):
        parsed = urlparse(self.path)
        if parsed.path in ("", "/"):
            path = "/index.html"
        else:
            path = parsed.path
        target = (STATIC_DIR / path.lstrip("/")).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.exists() or not target.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(target.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _json(self, payload: dict):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, path: str):
        if path in ("", "/"):
            path = "/index.html"
        target = (STATIC_DIR / path.lstrip("/")).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.exists() or not target.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run custom portfolio dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args(argv)
    httpd = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
