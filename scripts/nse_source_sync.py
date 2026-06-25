#!/usr/bin/env python3
"""
NSE official source sync via real browser session.

This mirrors the Business Standard scraper workflow:
- persistent Playwright profile
- headed/headless mode
- optional manual pause for login/CAPTCHA/challenge
- debug HTML/screenshots/status per endpoint

It does not bypass access controls. It automates a real browser session and then
hands the fetched payloads to the existing normalization/storage layer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

import sys

import config
from data.source_registry import get_source_registry
from data.official_sources import sync_verified_sources


async def _debug_endpoint(page, out_dir: Path, name: str, url: str) -> dict:
    endpoint_dir = out_dir / name
    endpoint_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_url = config.NSE_EVENT_CALENDAR_PAGE_URL if "event-calendar" in url else config.NSE_FILINGS_ANNOUNCEMENTS_PAGE_URL
    resp = await page.goto(bootstrap_url, wait_until="domcontentloaded", timeout=90_000)
    status = resp.status if resp else None

    if config.OFFICIAL_SOURCE_BROWSER_WAIT_MS > 0:
        await page.wait_for_timeout(config.OFFICIAL_SOURCE_BROWSER_WAIT_MS)

    result = await page.evaluate(
        """async (targetUrl) => {
            const res = await fetch(targetUrl, {
              method: 'GET',
              credentials: 'include',
              headers: { 'Accept': 'application/json,text/plain,*/*' }
            });
            const text = await res.text();
            return {
              status: res.status,
              headers: Object.fromEntries(res.headers.entries()),
              text
            };
        }""",
        url,
    )

    (endpoint_dir / "bootstrap.html").write_text(await page.content(), encoding="utf-8")
    try:
        await page.screenshot(path=str(endpoint_dir / "bootstrap.png"), full_page=False, timeout=10_000)
    except Exception:
        pass
    (endpoint_dir / "response.txt").write_text(result.get("text") or "", encoding="utf-8")
    (endpoint_dir / "meta.json").write_text(
        json.dumps(
            {
                "bootstrap_url": bootstrap_url,
                "bootstrap_status": status,
                "api_url": url,
                "api_status": result.get("status"),
                "headers": result.get("headers") or {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "name": name,
        "bootstrap_status": status,
        "api_status": result.get("status"),
        "debug_dir": str(endpoint_dir),
    }


async def main_async(args) -> int:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    user_data_dir = Path(args.user_data_dir).resolve()
    user_data_dir.mkdir(parents=True, exist_ok=True)

    registry = get_source_registry()
    selected = {}
    for name, meta in registry.items():
        if not name.startswith("nse_"):
            continue
        if args.source and name != args.source:
            continue
        selected[name] = meta

    if not selected:
        print("No matching NSE sources selected.")
        return 1

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            headless=args.headless,
            slow_mo=args.slow_mo_ms if args.slow_mo_ms > 0 else None,
            locale="en-US",
            timezone_id="Asia/Kolkata",
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            extra_http_headers={
                "Accept": "application/json,text/plain,*/*",
                "Origin": "https://www.nseindia.com",
            },
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()

        if args.pause_for_user:
            await page.goto(config.NSE_FILINGS_ANNOUNCEMENTS_PAGE_URL, wait_until="domcontentloaded", timeout=90_000)
            print("Paused. Complete any login / CAPTCHA in the opened browser window, then press Enter here to continue...")
            await asyncio.get_event_loop().run_in_executor(None, input)

        debug = []
        for name, meta in selected.items():
            print(f"Checking {name} ...")
            try:
                item = await _debug_endpoint(page, out_dir, name, meta["url"])
            except Exception as e:
                item = {"name": name, "error": str(e)}
            debug.append(item)
            print(json.dumps(item))

        (out_dir / "summary.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")
        await ctx.close()

    print("Running normalized SQLite sync...")
    results = sync_verified_sources(force_refresh=True)
    print(json.dumps(results, indent=2, default=str))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="NSE official source sync/debugger")
    parser.add_argument("--source", help="Single source key from data/source_registry.py, e.g. nse_corporate_actions")
    parser.add_argument("--out-dir", default=str(config.STORAGE_DIR / "nse_source_debug"))
    parser.add_argument("--user-data-dir", default=str(config.STORAGE_DIR / "nse_profile"))
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--pause-for-user", action="store_true", help="Pause for manual login/challenge handling")
    parser.add_argument("--slow-mo-ms", type=int, default=150)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
