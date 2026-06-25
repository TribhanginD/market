#!/usr/bin/env python3
"""
Indian AI Portfolio Pipeline — CLI Entrypoint

Usage:
  python run.py --mode pipeline      # Full 5-stage run (takes ~2-3 hours)
  python run.py --mode pipeline --stages 1-3   # Run only stages 1–3
  python run.py --mode monitor       # Daily thesis check only
  python run.py --mode status        # Show current portfolio + last run summary
  python run.py --mode schedule      # Start 3-day auto-scheduler
  python run.py --mode test-data     # Test data fetching (no Claude calls)
  python run.py --mode sync-sources  # Sync verified official NSE sources into SQLite
  python run.py --mode dashboard     # Launch custom trading dashboard
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Set
from zoneinfo import ZoneInfo
from pathlib import Path

# ── Rich console for pretty output ──────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
import config

# ── Logging Setup ────────────────────────────────────────────────────────────
def setup_logging(level: str = "INFO") -> None:
    if RICH_AVAILABLE:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(message)s",
            datefmt="[%H:%M:%S]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
    else:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

logger = logging.getLogger("run")
console = Console() if RICH_AVAILABLE else None


# ── Mode: pipeline ─────────────────────────────────────────────────────────
def run_pipeline(stage_range: str = "1-5", stage1_file: str | None = None, stage2_file: str | None = None) -> None:
    """Run the full (or partial) pipeline."""
    parts = stage_range.split("-")
    stage_start = int(parts[0]) if parts else 1
    stage_end = int(parts[1]) if len(parts) > 1 else 5
    stage1_data = None
    stage2_data = None
    if stage1_file:
        p = Path(stage1_file).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Stage 1 file not found: {p}")
        with open(p) as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            loaded = loaded.get("top50") or loaded.get("stocks") or loaded.get("results")
        if not isinstance(loaded, list):
            raise ValueError("Stage 1 file must contain a JSON list, or a dict with top50/stocks/results.")
        stage1_data = loaded
    if stage2_file:
        p = Path(stage2_file).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Stage 2 file not found: {p}")
        with open(p) as f:
            loaded = json.load(f)
        if isinstance(loaded, dict) and "research" in loaded:
            loaded = loaded["research"]
        if not isinstance(loaded, dict):
            raise ValueError("Stage 2 file must contain a research dict, or a dict with a research key.")
        stage2_data = loaded

    if console:
        console.print(Panel(
            f"[bold cyan]Starting Pipeline: Stages {stage_start}–{stage_end}[/bold cyan]\n"
            f"Portfolio: ₹{config.PAPER_PORTFOLIO_VALUE_INR:,.0f} | "
            f"Target: {config.MAX_POSITIONS} positions | "
            f"Benchmark: {config.BENCHMARK_NAME}",
            title="🤖 Indian AI Portfolio",
            border_style="cyan",
        ))

    from pipeline.orchestrator import run_pipeline as _run
    results = _run(stage_start=stage_start, stage_end=stage_end, mock_stage1=stage1_data, mock_stage2=stage2_data)

    if console and results:
        console.print(Panel(
            f"[bold green]✅ Pipeline complete![/bold green]\n"
            f"Duration: {results.get('duration_human', '?')}\n"
            f"Est. cost: ${results.get('total_tokens', {}).get('cost_usd', 0):.2f} USD\n"
            f"Output: {results.get('run_dir', '?')}",
            title="Pipeline Complete",
            border_style="green",
        ))

def run_daily_lite() -> None:
    """
    Daily-lite pipeline:
    - Runs Stage 1 full quant sweep
    - Then runs Stage 2/3 only on a tiny shortlist:
      up to DAILY_LITE_FLAGGED_HOLDINGS (from last thesis monitor) + DAILY_LITE_NEW_CANDIDATES (from Stage 1 top50 not held)
    """
    if console:
        console.print(Panel(
            f"[bold cyan]Daily-lite run (Stages 1–3)[/bold cyan]\n"
            f"Shortlist: flagged={config.DAILY_LITE_FLAGGED_HOLDINGS}, new={config.DAILY_LITE_NEW_CANDIDATES}\n"
            f"Models: fast={config.MODEL_FAST} smart={config.MODEL_SMART}",
            title="🧠 Daily Lite",
            border_style="cyan",
        ))

    from pipeline.orchestrator import run_pipeline as _run
    results = _run(stage_start=1, stage_end=3, daily_lite=True)

    if console and results:
        dl = results.get("daily_lite") or {}
        console.print(Panel(
            f"[bold green]✅ Daily-lite complete![/bold green]\n"
            f"Flagged: {', '.join(dl.get('flagged_symbols', []) or []) or 'None'}\n"
            f"New: {', '.join(dl.get('new_symbols', []) or []) or 'None'}\n"
            f"Duration: {results.get('duration_human', '?')}\n"
            f"Tokens: in={results.get('total_tokens',{}).get('input',0)} out={results.get('total_tokens',{}).get('output',0)}",
            title="Daily Lite Complete",
            border_style="green",
        ))


# ── Mode: monitor ──────────────────────────────────────────────────────────
def run_monitor() -> None:
    """Run daily thesis monitor on current holdings."""
    current_portfolio = _load_portfolio()

    if not current_portfolio.get("positions"):
        logger.warning("No portfolio found. Run 'python run.py --mode pipeline' first.")
        return

    if console:
        n = len(current_portfolio["positions"])
        console.print(Panel(
            f"[bold yellow]Checking {n} holdings for thesis integrity...[/bold yellow]",
            title="📰 Thesis Monitor",
            border_style="yellow",
        ))

    from pipeline.thesis_monitor import ThesisMonitor
    monitor = ThesisMonitor()
    results = monitor.run(current_portfolio)

    if console:
        urgent = [r for r in results if r.get("alert_level") == "URGENT"]
        watch  = [r for r in results if r.get("alert_level") == "WATCH"]

        if urgent:
            console.print(f"[bold red]⚠️  {len(urgent)} URGENT alerts![/bold red]")
            for r in urgent:
                console.print(f"  [red]• {r['symbol']}: {r.get('reason', '')}[/red]")
        elif watch:
            console.print(f"[bold yellow]👁  {len(watch)} WATCH alerts[/bold yellow]")
        else:
            console.print("[bold green]✅ All theses intact[/bold green]")


# ── Mode: status ───────────────────────────────────────────────────────────
def run_status() -> None:
    """Display the current portfolio and last run summary."""
    portfolio = _load_portfolio()
    trades_log = _load_trades_log()

    if not portfolio.get("positions"):
        if console:
            console.print("[yellow]No portfolio yet. Run: python run.py --mode pipeline[/yellow]")
        else:
            print("No portfolio yet. Run: python run.py --mode pipeline")
        return

    # ── Portfolio Table ──
    positions = portfolio["positions"]
    last_run = portfolio.get("last_run", "Unknown")

    if console:
        table = Table(
            title=f"📊 Current Portfolio — ₹{portfolio.get('portfolio_value_inr', 0):,.0f} | Last Run: {last_run[:16]}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#",     style="dim",   width=4)
        table.add_column("Symbol",               width=14)
        table.add_column("Company",              width=28)
        table.add_column("Sector",               width=18)
        table.add_column("Alloc%",  justify="right", width=8)
        table.add_column("Alloc ₹", justify="right", width=12)
        table.add_column("EV 12m",  justify="right", width=9)
        table.add_column("Conviction",            width=10)

        for i, p in enumerate(positions, 1):
            ev = p.get("ev_12m_return", 0) or 0
            ev_str = f"[green]+{ev*100:.1f}%[/green]" if ev > 0 else f"[red]{ev*100:.1f}%[/red]"
            table.add_row(
                str(i),
                p.get("symbol", ""),
                (p.get("company_name") or "")[:27],
                (p.get("sector") or "")[:17],
                f"{p.get('allocation_pct', 0):.1f}%",
                f"₹{p.get('allocation_inr', 0):,.0f}",
                ev_str,
                p.get("conviction", "Medium"),
            )

        console.print(table)

        # Summary
        portfolio_ev = portfolio.get("summary", {}).get("weighted_avg_ev_return", 0) or 0
        alpha = portfolio.get("summary", {}).get("vs_nifty50_alpha", 0) or 0
        console.print(
            f"\n[bold]Portfolio EV:[/bold] [green]+{portfolio_ev*100:.1f}%[/green]  |  "
            f"[bold]Alpha vs Nifty 50:[/bold] [green]+{alpha*100:.1f}%[/green]  |  "
            f"[bold]Positions:[/bold] {len(positions)}"
        )

    else:
        # Plain text fallback
        print(f"\nPortfolio ({len(positions)} positions) — Last run: {last_run}")
        print(f"{'#':>3} {'Symbol':12} {'Alloc%':>7} {'EV 12m':>8}")
        print("-" * 40)
        for i, p in enumerate(positions, 1):
            ev = (p.get("ev_12m_return") or 0) * 100
            print(f"{i:>3} {p.get('symbol',''):12} {p.get('allocation_pct',0):>6.1f}%  {ev:>+7.1f}%")

    # ── Last run summary ──
    if trades_log:
        last_run_entry = trades_log[-1]
        trades = last_run_entry.get("trades", [])
        buys  = [t for t in trades if t.get("action") == "BUY"]
        sells = [t for t in trades if t.get("action") == "SELL"]
        holds = [t for t in trades if t.get("action") == "HOLD"]

        if console:
            console.print(
                f"\n[bold]Last rebalance:[/bold] {last_run_entry.get('date', '?')}  |  "
                f"Buys: [green]{len(buys)}[/green]  Sells: [red]{len(sells)}[/red]  Holds: {len(holds)}"
            )
            if last_run_entry.get("notes"):
                console.print(f"[dim]{last_run_entry['notes'][:200]}[/dim]")
        else:
            print(f"\nLast rebalance: {last_run_entry.get('date','?')} — Buys: {len(buys)} Sells: {len(sells)}")


# ── Mode: schedule ─────────────────────────────────────────────────────────
def run_scheduler() -> None:
    """Start the 3-day auto-scheduler with daily thesis monitoring."""
    try:
        import schedule
    except ImportError:
        logger.error("Install 'schedule': pip install schedule")
        return

    if console:
        console.print(Panel(
            f"[bold cyan]Auto-scheduler started[/bold cyan]\n"
            f"• Full pipeline: every {config.REBALANCE_EVERY_DAYS} days\n"
            f"• Thesis monitor: daily at {config.THESIS_MONITOR_HOUR:02d}:00 IST\n"
            f"Press Ctrl+C to stop.",
            title="⏰ Scheduler",
            border_style="cyan",
        ))

    def pipeline_job():
        logger.info("⏰ Scheduled pipeline run triggered")
        run_pipeline()

    def monitor_job():
        logger.info("⏰ Scheduled thesis monitor triggered")
        run_monitor()

    schedule.every(config.REBALANCE_EVERY_DAYS).days.do(pipeline_job)
    schedule.every().day.at(f"{config.THESIS_MONITOR_HOUR:02d}:00").do(monitor_job)

    logger.info(f"Scheduler running. Next pipeline run in {config.REBALANCE_EVERY_DAYS} days.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped.")


def run_daemon() -> None:
    """
    Timezone-aware autonomous runner:
    - Official source sync daily at DAILY_SOURCE_SYNC_TIME
    - Thesis monitor daily at DAILY_MONITOR_TIME
    - Full pipeline daily at DAILY_PIPELINE_TIME (or restricted to PIPELINE_DAYS)
    - Off-cycle pipeline run if monitor raises URGENT alerts (cooldown-gated)
    """
    tz = ZoneInfo(config.SCHEDULER_TIMEZONE)
    pipeline_days = _parse_days(config.PIPELINE_DAYS)
    source_h, source_m = _parse_hhmm(config.DAILY_SOURCE_SYNC_TIME)
    pipeline_h, pipeline_m = _parse_hhmm(config.DAILY_PIPELINE_TIME)
    monitor_h, monitor_m = _parse_hhmm(config.DAILY_MONITOR_TIME)

    logger.info(
        "Daemon started (tz=%s). Source sync @ %02d:%02d. Pipeline=%s @ %02d:%02d. Monitor @ %02d:%02d.",
        config.SCHEDULER_TIMEZONE,
        source_h,
        source_m,
        config.PIPELINE_DAYS or "DAILY",
        pipeline_h,
        pipeline_m,
        monitor_h,
        monitor_m,
    )

    last_pipeline_end: Optional[datetime] = None
    next_source_sync = _next_scheduled_time(datetime.now(tz), source_h, source_m, None)
    next_pipeline = _next_scheduled_time(datetime.now(tz), pipeline_h, pipeline_m, pipeline_days)
    next_monitor = _next_scheduled_time(datetime.now(tz), monitor_h, monitor_m, None)

    try:
        while True:
            now = datetime.now(tz)
            enabled_events = [next_pipeline, next_monitor]
            if config.DAILY_SOURCE_SYNC_ENABLED:
                enabled_events.append(next_source_sync)
            next_event = min(enabled_events)
            sleep_s = max(1.0, (next_event - now).total_seconds())
            time.sleep(min(sleep_s, 60.0))
            now = datetime.now(tz)

            if config.DAILY_SOURCE_SYNC_ENABLED and now >= next_source_sync:
                logger.info("⏰ Daily source sync triggered (%s)", now.isoformat())
                run_sync_sources_silent()
                next_source_sync = _next_scheduled_time(now, source_h, source_m, None)

            if now >= next_monitor:
                logger.info("⏰ Thesis monitor triggered (%s)", now.isoformat())
                results = []
                try:
                    results = _run_monitor_and_return()
                except Exception as e:
                    logger.error("Thesis monitor failed: %s", e, exc_info=True)

                urgent = [r for r in results if r.get("alert_level") == "URGENT"]
                if urgent and config.OFFCYCLE_REBALANCE_ON_URGENT:
                    cooldown_h = config.OFFCYCLE_MIN_HOURS_BETWEEN_PIPELINES
                    can_run = (
                        last_pipeline_end is None
                        or (now - last_pipeline_end).total_seconds() >= cooldown_h * 3600
                    )
                    if can_run:
                        logger.warning("⚠️ URGENT alerts detected (%s). Triggering off-cycle pipeline.", len(urgent))
                        try:
                            run_pipeline("1-5")
                            last_pipeline_end = datetime.now(tz)
                        except Exception as e:
                            logger.error("Off-cycle pipeline failed: %s", e, exc_info=True)
                    else:
                        logger.info("Off-cycle pipeline suppressed by cooldown (%.1fh).", cooldown_h)

                next_monitor = _next_scheduled_time(now, monitor_h, monitor_m, None)

            if now >= next_pipeline:
                logger.info("⏰ Scheduled pipeline triggered (%s)", now.isoformat())
                try:
                    # Mon–Thu: daily-lite. Fri: full rebalance. Weekends: skip unless PIPELINE_DAYS includes them.
                    if now.weekday() == 4:  # Friday
                        run_pipeline("1-5")
                    else:
                        run_daily_lite()
                    last_pipeline_end = datetime.now(tz)
                except Exception as e:
                    logger.error("Scheduled pipeline failed: %s", e, exc_info=True)
                next_pipeline = _next_scheduled_time(now, pipeline_h, pipeline_m, pipeline_days)
    except KeyboardInterrupt:
        logger.info("Daemon stopped.")


def _run_monitor_and_return() -> list[dict]:
    current_portfolio = _load_portfolio()
    if not current_portfolio.get("positions"):
        logger.warning("No portfolio found; skipping thesis monitor.")
        return []
    from pipeline.thesis_monitor import ThesisMonitor
    monitor = ThesisMonitor()
    return monitor.run(current_portfolio)


def _parse_hhmm(value: str) -> tuple[int, int]:
    parts = (value or "").strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {value!r} (expected HH:MM)")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid time: {value!r}")
    return hour, minute


def _parse_days(value: str) -> Optional[Set[int]]:
    raw = (value or "").strip()
    if not raw:
        return None
    mapping = {
        "MON": 0,
        "TUE": 1,
        "WED": 2,
        "THU": 3,
        "FRI": 4,
        "SAT": 5,
        "SUN": 6,
    }
    out = set()
    for token in raw.replace(" ", "").upper().split(","):
        if not token:
            continue
        if token not in mapping:
            raise ValueError(f"Invalid day token: {token!r} (use MON,TUE,...)")
        out.add(mapping[token])
    return out or None


def _next_scheduled_time(now: datetime, hour: int, minute: int, days: Optional[Set[int]]) -> datetime:
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate = candidate.replace(day=candidate.day) + timedelta(days=1)

    if days is None:
        return candidate

    while candidate.weekday() not in days:
        candidate = candidate + timedelta(days=1)
    return candidate


# ── Mode: test-data ────────────────────────────────────────────────────────
def run_test_data() -> None:
    """Quick data layer smoke test — no Claude calls, no API cost."""
    if console:
        console.print(Panel("[bold]Running data layer tests (no Claude calls)...[/bold]", title="🔍 Data Test"))

    from data.nifty500 import get_nifty500
    from data.fetcher import get_fundamentals, get_news_for_stock, get_macro_context, compute_composite_score

    # 1. Universe
    logger.info("Test 1: Nifty 500 universe...")
    try:
        # Test-mode allows partial universe for diagnostics; pipeline runs remain strict.
        old = getattr(config, "REQUIRE_FULL_UNIVERSE", False)
        config.REQUIRE_FULL_UNIVERSE = False
        df = get_nifty500()
        logger.info(f"  ✅ Loaded {len(df)} stocks across {df['sector'].nunique()} sectors")
    finally:
        config.REQUIRE_FULL_UNIVERSE = old

    # 2. Fundamentals
    test_symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    logger.info("Test 2: Fundamentals fetch...")
    for sym in test_symbols:
        data = get_fundamentals(sym)
        score = compute_composite_score(data)
        price = data.get("price") or data.get("currentPrice") or "N/A"
        pe = data.get("trailingPE") or "N/A"
        logger.info(f"  ✅ {sym}: price=₹{price} | PE={pe} | composite_score={score['composite']:.1f}")

    # 3. News
    logger.info("Test 3: News fetch...")
    news = get_news_for_stock("RELIANCE", "Reliance Industries", days=7)
    logger.info(f"  ✅ Found {len(news)} articles for RELIANCE (last 7 days)")
    for a in news[:2]:
        logger.info(f"     [{a.get('source','')}] {a.get('title','')[:70]}")

    # 4. Macro
    logger.info("Test 4: Macro context...")
    macro = get_macro_context()
    logger.info(f"  ✅ Nifty 50: {macro.get('nifty50_current','N/A')} | USD/INR: {macro.get('usd_inr','N/A')} | VIX: {macro.get('india_vix','N/A')}")

    # 5. Official sources
    logger.info("Test 5: Official-source sync...")
    from data.official_sources import sync_verified_sources
    sync = sync_verified_sources(force_refresh=False)
    logger.info(
        "  ✅ Official sources payloads=%s normalized_rows=%s",
        sync.get("totals", {}).get("payloads", 0),
        sync.get("totals", {}).get("normalized_rows", 0),
    )

    if console:
        console.print("[bold green]✅ All data tests passed![/bold green]")


def run_sync_sources(*, headless: Optional[bool] = None, pause_for_user: Optional[bool] = None) -> None:
    """Sync verified official NSE sources into raw payload storage + SQLite."""
    if headless is not None:
        config.OFFICIAL_SOURCE_BROWSER_HEADLESS = bool(headless)
    if pause_for_user is not None:
        config.OFFICIAL_SOURCE_BROWSER_PAUSE_FOR_USER = bool(pause_for_user)

    if console:
        console.print(Panel("[bold]Syncing verified official sources...[/bold]", title="🏛️ Source Sync"))

    from data.official_sources import sync_verified_sources

    results = sync_verified_sources(force_refresh=True)
    totals = results.get("totals", {})
    logger.info(
        "Official sources synced: payloads=%s normalized_rows=%s",
        totals.get("payloads", 0),
        totals.get("normalized_rows", 0),
    )
    for name, item in (results.get("sources") or {}).items():
        status = item.get("error") or f"rows={item.get('normalized_rows', 0)}"
        logger.info("  %s -> %s", name, status)

    if console:
        console.print("[bold green]✅ Official-source sync complete[/bold green]")


def run_sync_sources_silent() -> None:
    """Daemon-safe daily source sync."""
    try:
        from data.official_sources import sync_verified_sources
        results = sync_verified_sources(force_refresh=True)
        totals = results.get("totals", {})
        logger.info(
            "Daily source sync complete: payloads=%s normalized_rows=%s",
            totals.get("payloads", 0),
            totals.get("normalized_rows", 0),
        )
    except Exception as e:
        logger.error("Daily source sync failed: %s", e, exc_info=True)

    if not getattr(config, "DAILY_BS_RESEARCH_SYNC_ENABLED", True):
        return

    try:
        from scripts.bs_research_scrape import ScrapeOptions, run_sync
        rc = run_sync(
            ScrapeOptions(
                out_dir=str(config.BS_RESEARCH_OUT_DIR),
                max_pdfs=config.BS_RESEARCH_SYNC_MAX_PDFS,
                days=config.BS_RESEARCH_SYNC_DAYS,
                incremental=True,
                user_data_dir=str(config.BS_RESEARCH_USER_DATA_DIR),
                headless=config.BS_RESEARCH_SYNC_HEADLESS,
                debug=False,
                pause_for_user=False,
                slow_mo_ms=config.BS_RESEARCH_SYNC_SLOW_MO_MS,
                timezone=config.SCHEDULER_TIMEZONE,
                scrolls=config.BS_RESEARCH_SYNC_SCROLLS,
                wait_ms=config.BS_RESEARCH_SYNC_WAIT_MS,
            )
        )
        if rc == 0:
            logger.info("Daily Business Standard research sync complete.")
        else:
            logger.warning("Daily Business Standard research sync returned code %s.", rc)
    except Exception as e:
        logger.warning("Daily Business Standard research sync failed: %s", e, exc_info=True)

    if getattr(config, "DAILY_ETL_ENABLED", True):
        try:
            run_etl(silent=True)
        except Exception as e:
            logger.error("Daily ETL run failed: %s", e, exc_info=True)


def run_etl(*, silent: bool = False) -> dict | None:
    if not silent and console:
        console.print(Panel("[bold]Running ETL pipeline...[/bold]", title="🧠 ETL"))

    from pipeline.etl_pipeline import ETLPipeline

    pipeline = ETLPipeline()
    results = pipeline.run()
    logger.info(
        "ETL complete: docs_ingested=%s extracted=%s queued=%s processed=%s failed=%s symbols=%s packets=%s in_tokens=%s out_tokens=%s",
        results.get("ingested_documents", 0),
        results.get("extracted_documents", 0),
        results.get("queued_jobs", 0),
        results.get("processed_jobs", 0),
        results.get("failed_jobs", 0),
        results.get("updated_symbols", 0),
        results.get("packets_built", 0),
        results.get("llm_input_tokens", 0),
        results.get("llm_output_tokens", 0),
    )

    if not silent and console:
        console.print("[bold green]✅ ETL complete[/bold green]")
    return results

# ── Mode: dashboard ────────────────────────────────────────────────────────
def run_dashboard() -> None:
    """Launch the custom dashboard server."""
    from dashboard.server import main as dashboard_main

    logger.info("Launching dashboard at http://127.0.0.1:8765 ...")
    dashboard_main(["--host", "127.0.0.1", "--port", "8765"])


# ── Helpers ────────────────────────────────────────────────────────────────
def _load_portfolio() -> dict:
    try:
        from persistence import db as pdb
        portfolio = pdb.load_current_portfolio()
        if portfolio.get("positions"):
            return portfolio
    except Exception:
        pass

    if config.PORTFOLIO_FILE.exists():
        with open(config.PORTFOLIO_FILE) as f:
            return json.load(f)
    return {}


def _load_trades_log() -> list:
    if config.TRADES_LOG_FILE.exists():
        with open(config.TRADES_LOG_FILE) as f:
            return json.load(f)
    return []


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Indian AI Portfolio — Multi-Agent Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "daily", "monitor", "status", "schedule", "daemon", "test-data", "sync-sources", "etl", "dashboard", "backtest"],
        default="status",
        help="Operation mode (default: status)",
    )
    parser.add_argument(
        "--stages",
        default="1-5",
        help="Stage range to run, e.g. '1-3' or '4-5' (pipeline mode only)",
    )
    parser.add_argument(
        "--stage1-file",
        help="Path to an existing stage1_output.json to use when starting at Stage 2 or later",
    )
    parser.add_argument(
        "--stage2-file",
        help="Path to an existing stage2_output.json to use when starting at Stage 3 or later",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show the Playwright browser for source sync fallback flows",
    )
    parser.add_argument(
        "--pause-for-user",
        action="store_true",
        help="Pause during browser-based source sync so you can complete login/CAPTCHA if needed",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=365,
        help="Backtest forward-return horizon in days (default 365)",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    if console:
        console.print(f"[dim]Indian AI Portfolio | Mode: {args.mode} | {datetime.now().strftime('%Y-%m-%d %H:%M')}[/dim]")

    dispatch = {
        "pipeline":   lambda: run_pipeline(args.stages, stage1_file=args.stage1_file, stage2_file=args.stage2_file),
        "daily":      run_daily_lite,
        "monitor":    run_monitor,
        "status":     run_status,
        "schedule":   run_scheduler,
        "daemon":     run_daemon,
        "test-data":  run_test_data,
        "sync-sources": lambda: run_sync_sources(headless=not args.show_browser, pause_for_user=args.pause_for_user),
        "etl":        run_etl,
        "dashboard":  run_dashboard,
        "backtest":   lambda: __import__("pipeline.backtest", fromlist=["run_backtest"]).run_backtest(horizon_days=args.horizon_days),
    }

    try:
        dispatch[args.mode]()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
