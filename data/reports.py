"""
NiftyIndices reports collector (research papers / daily reports / monthly reports).

Goal: enrich the pipeline with non-LLM macro/market context from official index provider PDFs.
We:
  - scrape report listing pages
  - download PDFs (cached)
  - extract text excerpts (cached)
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)


try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None


@dataclass
class ReportItem:
    category: str
    title: str
    url: str
    published_at: str | None = None
    local_path: str | None = None
    text_excerpt: str | None = None


def _http_get(url: str, *, timeout_s: int = 20) -> requests.Response:
    # Light retry/backoff: niftyindices.com can be slow.
    last_exc: Exception | None = None
    retries = max(0, int(getattr(config, "REPORTS_HTTP_RETRIES", 1)))
    for attempt in range(retries + 1):
        try:
            return requests.get(
                url,
                headers={
                    "User-Agent": "market-research-bot/1.0 (+https://github.com/)",
                    "Accept": "text/html,application/pdf,*/*",
                },
                timeout=timeout_s,
            )
        except requests.RequestException as e:
            last_exc = e
            time.sleep(0.8 + attempt * 1.2)
            continue
    raise last_exc  # type: ignore[misc]


def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


def _is_pdf_url(url: str) -> bool:
    p = urlparse(url)
    return p.path.lower().endswith(".pdf")

def _jina_url(url: str) -> str:
    # r.jina.ai works as a read-only fetch proxy for HTML pages.
    return "https://r.jina.ai/" + url

def _scrape_pdf_links_from_text(text: str, *, base_url: str, category: str) -> list[ReportItem]:
    if not text:
        return []
    # Extract any PDF URLs (absolute or relative)
    urls = set()
    # 1) absolute URLs
    for m in re.findall(r"https?://[^\\s\"'>]+?\\.pdf", text, flags=re.IGNORECASE):
        urls.add(m)
    # 2) any relative-ish token containing a .pdf path
    for m in re.findall(r"[^\\s\"'>]+?\\.pdf", text, flags=re.IGNORECASE):
        if m.lower().startswith("http"):
            continue
        urls.add(urljoin(base_url, m))
    items: list[ReportItem] = []
    for u in sorted(urls):
        title = Path(urlparse(u).path).name
        items.append(ReportItem(category=category, title=title[:200], url=u, published_at=None))
    return items

def _extract_date(text: str) -> str | None:
    """
    Best-effort date extraction from titles like:
      "Daily Report - 20 Apr 2026" or "April 2026" etc.
    We keep it simple; reliability isn't critical for now.
    """
    t = (text or "").strip()
    if not t:
        return None
    m = re.search(r"(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})", t)
    if m:
        return m.group(1)
    m = re.search(r"([A-Za-z]{3,9}\s+\d{4})", t)
    if m:
        return m.group(1)
    return None


def scrape_niftyindices_report_list(url: str, *, category: str) -> list[ReportItem]:
    """
    Scrape a NiftyIndices reports page and return discovered PDF links.
    """
    try:
        resp = _http_get(url, timeout_s=getattr(config, "REPORTS_HTTP_TIMEOUT_SECONDS", 8))
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        items: list[ReportItem] = []
        for a in soup.select("a[href]"):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            full = urljoin(url, href)
            if not _is_pdf_url(full):
                continue
            title = (a.get_text(" ", strip=True) or "").strip() or Path(urlparse(full).path).name
            items.append(ReportItem(category=category, title=title[:200], url=full, published_at=_extract_date(title)))
    except Exception as e:
        logger.warning("Direct scrape failed (%s). Trying jina proxy: %s", category, str(e)[:160])
        # Fallback: jina proxy returns a plain-text rendering; extract PDF links via regex.
        j = _jina_url(url)
        resp2 = _http_get(j, timeout_s=getattr(config, "REPORTS_HTTP_TIMEOUT_SECONDS", 8))
        text = resp2.text if hasattr(resp2, "text") else ""
        items = _scrape_pdf_links_from_text(text, base_url=url, category=category)

    # De-dupe by url (preserve order)
    seen = set()
    out: list[ReportItem] = []
    for it in items:
        if it.url in seen:
            continue
        seen.add(it.url)
        out.append(it)

    return out[: config.REPORTS_MAX_PER_CATEGORY]


def download_report_pdf(item: ReportItem, *, force: bool = False) -> Path:
    """
    Download the PDF to local cache and return the path.
    """
    cache_id = _cache_key(item.url)
    dest = config.REPORTS_CACHE_DIR / f"{item.category}_{cache_id}.pdf"
    if dest.exists() and not force:
        return dest

    resp = _http_get(item.url, timeout_s=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def extract_pdf_text_excerpt(pdf_path: Path, *, max_chars: int = 2000) -> str:
    if PdfReader is None:
        return "PDF text extraction unavailable (install pypdf)."
    try:
        reader = PdfReader(str(pdf_path))
        parts: list[str] = []
        for page in reader.pages[:6]:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        text = "\n".join(parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text[:max_chars]
    except Exception as e:
        return f"PDF extract failed: {e}"


def get_latest_reports_snapshot(*, force_refresh: bool = False) -> dict[str, Any]:
    """
    Returns a snapshot dict with a few latest items in each category.
    Cached on disk to avoid re-downloading PDFs on every pipeline run.
    """
    cache_path = config.REPORTS_CACHE_DIR / "reports_snapshot.json"
    now = time.time()

    if not force_refresh and cache_path.exists():
        age = now - cache_path.stat().st_mtime
        if age < config.CACHE_TTL_REPORTS:
            try:
                return {"asof": datetime.now().isoformat(), "cached": True, **(cache_path.read_text() and __import__("json").loads(cache_path.read_text()))}
            except Exception:
                pass

    try:
        categories = [
            ("research", config.NIFTYINDICES_RESEARCH_PAPERS_URL),
            ("daily", config.NIFTYINDICES_DAILY_REPORTS_URL),
            ("monthly", config.NIFTYINDICES_MONTHLY_REPORTS_URL),
        ]
        out: dict[str, Any] = {"asof": datetime.now().isoformat(), "cached": False, "categories": {}}

        for cat, url in categories:
            try:
                lst = scrape_niftyindices_report_list(url, category=cat)
            except Exception as e:
                logger.warning("Reports scrape failed (%s): %s", cat, str(e)[:160])
                out["categories"][cat] = {"source_url": url, "error": str(e), "items": []}
                continue

            enriched = []
            for it in lst:
                try:
                    pdf_path = download_report_pdf(it, force=False)
                    excerpt = extract_pdf_text_excerpt(pdf_path, max_chars=config.REPORTS_TEXT_EXCERPT_CHARS)
                    enriched.append(
                        {
                            "title": it.title,
                            "url": it.url,
                            "published_at": it.published_at,
                            "local_path": str(pdf_path),
                            "text_excerpt": excerpt,
                        }
                    )
                except Exception as e:
                    enriched.append({"title": it.title, "url": it.url, "published_at": it.published_at, "error": str(e)})

            out["categories"][cat] = {"source_url": url, "items": enriched}

        try:
            import json
            cache_path.write_text(json.dumps(out, indent=2, default=str))
        except Exception:
            pass

        return out
    except Exception as e:
        return {
            "asof": datetime.now().isoformat(),
            "cached": False,
            "error": f"reports_snapshot_failed: {e}",
            "categories": {},
        }
