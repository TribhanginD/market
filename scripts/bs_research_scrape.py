#!/usr/bin/env python3
"""
Business Standard research-report scraper (Playwright).

Important:
- This does NOT bypass access controls. It automates a real browser session.
- If the site requires login/subscription, use a persistent profile so you can sign in once.

Outputs:
- Downloads visible PDF links (bsmedia.business-standard.com/*.pdf)
- Writes a small index JSONL with extracted table rows (company/action/target/broker/date/pdf_url)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import async_playwright


START_URL = "https://www.business-standard.com/markets/research-report"


@dataclass
class Row:
    company: str
    action: str
    target: str
    broker: str
    date: str
    company_url: str
    pdf_url: str


@dataclass
class ScrapeOptions:
    out_dir: str = "storage/bs_research"
    max_pdfs: int = 20
    days: int = 30
    incremental: bool = True
    user_data_dir: str = ""
    headless: bool = True
    debug: bool = False
    pause_for_user: bool = False
    slow_mo_ms: int = 0
    timezone: str = "Asia/Kolkata"
    scrolls: int = 5
    wait_ms: int = 3000


def clean_filename(name: str) -> str:
    name = re.sub(r"[^\w.\- ]+", "_", name).strip()
    name = re.sub(r"\s+", " ", name)
    return (name[:150] or "report") + ".pdf" if not name.lower().endswith(".pdf") else name[:150]


def is_pdf(url: str) -> bool:
    try:
        return urlparse(url).path.lower().endswith(".pdf")
    except Exception:
        return False


def _parse_bs_date(s: str) -> datetime | None:
    """
    Business Standard table uses e.g. '20-Apr-2026'.
    """
    raw = (s or "").strip()
    if not raw:
        return None
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            continue
    return None


def _load_existing_pdf_urls(index_path: Path) -> set[str]:
    out: set[str] = set()
    if not index_path.exists():
        return out
    try:
        for line in index_path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            u = (obj.get("pdf_url") or "").strip()
            if u:
                out.add(u)
    except Exception:
        return out
    return out


async def extract_pdf_links(page) -> set[str]:
    links = await page.locator("a[href]").evaluate_all(
        """els => els.map(a => ({
            href: a.href || "",
            text: (a.innerText || a.textContent || "").trim()
        }))"""
    )
    pdfs: set[str] = set()
    for item in links:
        href = (item.get("href") or "").strip()
        if href and is_pdf(href):
            pdfs.add(href)
    return pdfs


async def extract_table_rows(page) -> list[Row]:
    """
    Best-effort: find rows with a company link and a pdf link.
    """
    rows = []
    # Evaluate in page context for speed and to avoid fragile selectors.
    raw = await page.evaluate(
        """() => {
          const out = [];
          const trs = Array.from(document.querySelectorAll('tr'));
          for (const tr of trs) {
            const tds = Array.from(tr.querySelectorAll('td'));
            if (tds.length < 6) continue;
            const companyA = tds[0].querySelector('a[href]');
            const pdfA = tds[5].querySelector('a[href]');
            const pdf = pdfA ? (pdfA.href || '') : '';
            if (!pdf || !pdf.toLowerCase().endsWith('.pdf')) continue;
            out.push({
              company: (companyA ? companyA.textContent : tds[0].textContent || '').trim(),
              company_url: (companyA ? companyA.href : ''),
              action: (tds[1].textContent || '').trim(),
              target: (tds[2].textContent || '').trim(),
              broker: (tds[3].textContent || '').trim(),
              date: (tds[4].textContent || '').trim(),
              pdf_url: pdf
            });
          }
          return out;
        }"""
    )
    for r in raw or []:
        rows.append(
            Row(
                company=r.get("company", ""),
                action=r.get("action", ""),
                target=r.get("target", ""),
                broker=r.get("broker", ""),
                date=r.get("date", ""),
                company_url=r.get("company_url", ""),
                pdf_url=r.get("pdf_url", ""),
            )
        )
    return rows


async def main_async(args) -> int:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"
    existing_pdf_urls = _load_existing_pdf_urls(index_path) if args.incremental else set()

    cutoff_dt = None
    if args.days > 0:
        cutoff_dt = datetime.now() - timedelta(days=args.days)

    async with async_playwright() as p:
        browser_type = p.chromium

        if args.user_data_dir:
            ctx = await browser_type.launch_persistent_context(
                user_data_dir=args.user_data_dir,
                headless=args.headless,
                slow_mo=args.slow_mo_ms if args.slow_mo_ms > 0 else None,
                locale="en-US",
                timezone_id=args.timezone,
            )
        else:
            browser = await browser_type.launch(
                headless=args.headless,
                slow_mo=args.slow_mo_ms if args.slow_mo_ms > 0 else None,
            )
            ctx = await browser.new_context()

        page = await ctx.new_page()
        resp = await page.goto(START_URL, wait_until="domcontentloaded", timeout=90_000)
        try:
            status = resp.status if resp else None
        except Exception:
            status = None
        if args.debug:
            print("goto_status:", status)

        if args.wait_ms > 0:
            await page.wait_for_timeout(args.wait_ms)

        # Optional manual pause for interactive login / verification.
        if args.pause_for_user:
            print("Paused. Complete any login / CAPTCHA in the opened browser window, then press Enter here to continue...")
            try:
                await asyncio.get_event_loop().run_in_executor(None, input)
            except Exception:
                pass

        # Scroll to trigger lazy load
        for _ in range(max(0, args.scrolls)):
            await page.mouse.wheel(0, 4000)
            await page.wait_for_timeout(800)

        rows = await extract_table_rows(page)
        pdf_links = await extract_pdf_links(page)
        if args.debug:
            print("page_title:", await page.title())
            print("rows_found:", len(rows))
            print("pdf_links_found:", len(pdf_links))
            try:
                (out_dir / "page.html").write_text(await page.content(), encoding="utf-8")
                # Full-page screenshots can hang on font loading; keep it optional and bounded.
                await page.screenshot(path=str(out_dir / "page.png"), full_page=False, timeout=10_000)
            except Exception as e:
                print("debug_save_failed:", str(e)[:160])

        # Prefer table pdfs (keep metadata)
        pdfs_ordered = []
        table_pdf_set = {r.pdf_url for r in rows if r.pdf_url}
        pdfs_ordered.extend([r.pdf_url for r in rows if r.pdf_url])
        pdfs_ordered.extend([u for u in sorted(pdf_links) if u not in table_pdf_set])

        # Filter rows by cutoff and newness.
        filtered_rows: list[Row] = []
        for r in rows:
            if cutoff_dt is not None:
                dt = _parse_bs_date(r.date)
                if dt is None or dt < cutoff_dt:
                    continue
            if args.incremental and r.pdf_url in existing_pdf_urls:
                continue
            filtered_rows.append(r)

        saved = 0
        # Append only new rows if incremental; otherwise overwrite.
        mode = "a" if args.incremental else "w"
        with index_path.open(mode, encoding="utf-8") as f:
            for r in filtered_rows:
                f.write(json.dumps(asdict(r), ensure_ascii=True) + "\n")

        # Download only PDFs referenced by filtered rows (most useful).
        pdfs_to_download = [r.pdf_url for r in filtered_rows if r.pdf_url]
        # If table parsing fails, fall back to discovered pdf_links.
        if not pdfs_to_download:
            pdfs_to_download = [u for u in pdfs_ordered if (not args.incremental or u not in existing_pdf_urls)]

        for i, pdf_url in enumerate(pdfs_to_download, start=1):
            if args.max_pdfs and saved >= args.max_pdfs:
                break
            if args.incremental and pdf_url in existing_pdf_urls:
                continue
            try:
                resp = await ctx.request.get(pdf_url)
                if not resp.ok:
                    continue
                body = await resp.body()
                filename = os.path.basename(urlparse(pdf_url).path) or f"report_{i}.pdf"
                filename = clean_filename(filename)
                (out_dir / filename).write_bytes(body)
                saved += 1
            except Exception:
                continue

        if args.user_data_dir:
            await ctx.close()
        else:
            await ctx.close()
            await browser.close()

        print(f"Saved PDFs: {saved}")
        print(f"Index: {index_path}")
        return 0


def run_sync(options: ScrapeOptions | None = None) -> int:
    opts = options or ScrapeOptions()
    return asyncio.run(main_async(opts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="storage/bs_research", help="Output directory for PDFs + index.jsonl")
    ap.add_argument("--max-pdfs", type=int, default=20, help="Max PDFs to download (0 = unlimited)")
    ap.add_argument("--days", type=int, default=30, help="Only keep rows within the last N days (0 = no cutoff)")
    ap.add_argument("--incremental", action="store_true", help="Only download PDFs not already present in index.jsonl (append new rows)")
    ap.add_argument("--user-data-dir", default="", help="Playwright persistent profile dir (recommended for login)")
    ap.add_argument("--headless", action="store_true", help="Run headless")
    ap.add_argument("--debug", action="store_true", help="Save page.html/page.png and print diagnostics")
    ap.add_argument("--pause-for-user", action="store_true", help="Pause after initial load so you can login in the browser window")
    ap.add_argument("--slow-mo-ms", type=int, default=0, help="Slow down Playwright actions (ms) for stability")
    ap.add_argument("--timezone", default="Asia/Kolkata", help="Browser timezone id (e.g. Asia/Kolkata)")
    ap.add_argument("--scrolls", type=int, default=5, help="Number of scrolls to load more rows")
    ap.add_argument("--wait-ms", type=int, default=3000, help="Initial wait after load (ms)")
    args = ap.parse_args()
    return run_sync(ScrapeOptions(**vars(args)))


if __name__ == "__main__":
    raise SystemExit(main())
