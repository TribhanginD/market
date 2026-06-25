"""
Analyst/Broker reports collector (RSS/Atom).

Many broker reports are behind logins; the most robust non-LLM option is to ingest
RSS/Atom feeds you provide (or public research feeds when available).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import re
import requests
from bs4 import BeautifulSoup

import config
import data.cache as cache

try:
    import feedparser
except Exception:  # pragma: no cover
    feedparser = None

logger = logging.getLogger(__name__)

def _http_get(url: str, *, timeout_s: int = 15) -> str:
    resp = requests.get(
        url,
        headers={
            "User-Agent": "market-analyst-research/1.0",
            "Accept": "text/html,*/*",
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.text

def _http_get_bs(url: str, *, timeout_s: int = 20) -> str:
    """
    Business Standard is frequently 403-blocked. If BUSINESS_STANDARD_COOKIE is provided,
    attach it to requests.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    cookie = getattr(config, "BUSINESS_STANDARD_COOKIE", "") or ""
    if cookie:
        headers["Cookie"] = cookie
    resp = requests.get(url, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    return resp.text

def _download_pdf(url: str, *, timeout_s: int = 30) -> bytes:
    resp = requests.get(
        url,
        headers={
            "User-Agent": "market-analyst-research/1.0",
            "Accept": "application/pdf,*/*",
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.content

def _extract_pdf_excerpt(pdf_bytes: bytes, *, max_chars: int) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return "PDF text extraction unavailable (install pypdf)."
    try:
        import io
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
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

def _bsmedia_pdf_links_from_text(text: str) -> list[str]:
    if not text:
        return []
    urls = re.findall(r"https?://bsmedia\\.business-standard\\.com/[^\\s\"'>]+?\\.pdf", text, flags=re.IGNORECASE)
    # de-dupe preserve order
    seen = set()
    out = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out

def _fetch_business_standard_brokertips_pdfs(days: int = 14) -> list[dict[str, Any]]:
    """
    Fetch latest broker tips PDFs linked from Business Standard research-report area.
    Uses jina proxy to avoid JS/paywall issues, then downloads PDFs directly from bsmedia.
    """
    cache_key = f"bs_brokertips_pdfs:{days}"
    cached = cache.get("analyst_pdfs", cache_key, ttl=getattr(config, "ANALYST_PDF_CACHE_TTL", 12 * 3600))
    if cached is not None:
        return cached

    base = "https://www.business-standard.com/markets/research-report"
    items: list[dict[str, Any]] = []
    try:
        # jina proxy gives us a text rendering that often includes the raw bsmedia PDF URLs.
        txt = _http_get("https://r.jina.ai/" + base, timeout_s=25)
        pdf_urls = _bsmedia_pdf_links_from_text(txt)
    except Exception:
        pdf_urls = []

    max_n = int(getattr(config, "ANALYST_PDF_MAX", 3))
    for u in pdf_urls[:max_n]:
        try:
            pdf_bytes = _download_pdf(u, timeout_s=30)
            excerpt = _extract_pdf_excerpt(pdf_bytes, max_chars=int(getattr(config, "ANALYST_PDF_EXCERPT_CHARS", 1500)))
            items.append(
                {
                    "title": u.split("/")[-1],
                    "url": u,
                    "published_at": None,
                    "source": "business-standard:bsmedia:pdf",
                    "summary": excerpt,
                }
            )
        except Exception as e:
            items.append(
                {
                    "title": u.split("/")[-1],
                    "url": u,
                    "published_at": None,
                    "source": "business-standard:bsmedia:pdf",
                    "error": str(e)[:200],
                }
            )

    cache.set("analyst_pdfs", cache_key, items)
    return items


def _parse_bs_research_listing(html: str) -> list[dict[str, Any]]:
    """
    Best-effort parser for Business Standard /markets/research-report listing.
    Returns items with title,url,published_at,source,summary.
    """
    soup = BeautifulSoup(html, "lxml")
    items: list[dict[str, Any]] = []

    # Grab links that look like article cards in the listing area.
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if "/markets/" not in href and "/article/" not in href:
            continue
        title = (a.get_text(" ", strip=True) or "").strip()
        if not title or len(title) < 12:
            continue

        if href.startswith("/"):
            url = "https://www.business-standard.com" + href
        else:
            url = href

        items.append(
            {
                "title": title[:300],
                "url": url,
                "published_at": None,
                "source": "business-standard:research-report",
                "summary": "",
            }
        )

    # de-dupe by url, preserve order
    seen = set()
    out = []
    for it in items:
        u = it["url"]
        if u in seen:
            continue
        seen.add(u)
        out.append(it)
    return out[:80]

def _parse_bs_research_table(html: str) -> list[dict[str, Any]]:
    """
    Parse the broker tips table when available server-side.
    Extracts: company, rating, target, broker, date, pdf_url (bsmedia).
    """
    soup = BeautifulSoup(html, "lxml")
    out: list[dict[str, Any]] = []
    for tr in soup.select("tr"):
        tds = tr.find_all("td")
        if len(tds) < 6:
            continue
        company_a = tds[0].find("a")
        company = company_a.get_text(" ", strip=True) if company_a else tds[0].get_text(" ", strip=True)
        action = tds[1].get_text(" ", strip=True)
        target = tds[2].get_text(" ", strip=True)
        broker = tds[3].get_text(" ", strip=True)
        date = tds[4].get_text(" ", strip=True)
        pdf_a = tds[5].find("a", href=True)
        pdf_url = (pdf_a["href"].strip() if pdf_a and pdf_a.get("href") else "")
        if not pdf_url or "bsmedia.business-standard.com" not in pdf_url:
            continue
        if not company or len(company) < 2:
            continue
        out.append(
            {
                "title": f"{company} | {action} | {broker} | {date} | target {target}",
                "url": pdf_url,
                "published_at": date,
                "source": "business-standard:research-report:table",
                "summary": "",
                "meta": {
                    "company": company,
                    "action": action,
                    "target": target,
                    "broker": broker,
                    "date": date,
                    "company_url": (company_a["href"] if company_a and company_a.get("href") else ""),
                },
            }
        )
    return out

def _parse_markdown_links(text: str) -> list[dict[str, Any]]:
    """
    Extract [title](url) style links from jina markdown output.
    """
    items: list[dict[str, Any]] = []
    if not text:
        return items
    for m in re.findall(r"\[([^\]]{8,200})\]\((https?://[^)]+)\)", text):
        title, url = m[0].strip(), m[1].strip()
        if "business-standard.com" not in url:
            continue
        if "/markets/" not in url and "/article/" not in url:
            continue
        items.append(
            {
                "title": title[:300],
                "url": url,
                "published_at": None,
                "source": "business-standard:research-report",
                "summary": "",
            }
        )
    # de-dupe
    seen = set()
    out = []
    for it in items:
        if it["url"] in seen:
            continue
        seen.add(it["url"])
        out.append(it)
    return out[:80]


def _fetch_business_standard_research(days: int = 14) -> list[dict[str, Any]]:
    """
    Pull a small recent set of Business Standard research-report articles.
    We don't hard-filter by date (site markup varies); we rely on recency of listing + days arg.
    """
    cache_key = f"bs_research:{days}"
    cached = cache.get("analyst_html", cache_key, ttl=6 * 3600)
    if cached is not None:
        return cached

    try:
        url = "https://www.business-standard.com/markets/research-report"
        # Try direct fetch (may require cookie); if blocked, fallback to other methods.
        html = _http_get_bs(url, timeout_s=25)
        items = _parse_bs_research_table(html)
        if not items:
            items = _parse_bs_research_listing(html)
        if not items:
            # JS-heavy page: fallback to jina proxy markdown and extract links.
            md = _http_get("https://r.jina.ai/" + url, timeout_s=25)
            items = _parse_markdown_links(md)
    except Exception as e:
        logger.debug("Business Standard scrape failed: %s", str(e)[:160])
        items = []

    cache.set("analyst_html", cache_key, items)
    return items


def _fetch_business_standard_markets_rss(days: int = 14) -> list[dict[str, Any]]:
    """
    Practical fallback: Business Standard markets RSS often contains broker calls/research.
    We treat only research-like headlines as analyst reports.
    """
    cache_key = f"bs_markets_rss:{days}"
    cached = cache.get("analyst_rss", cache_key, ttl=3600)
    if cached is not None:
        return cached

    url = "https://www.business-standard.com/rss/markets-106.rss"
    # BS blocks direct RSS in many environments (403). Use jina proxy as a fetcher.
    proxy_url = "https://r.jina.ai/" + url
    cutoff = datetime.now() - timedelta(days=int(days or 14))
    out: list[dict[str, Any]] = []
    def _jina(url: str) -> str:
        return "https://r.jina.ai/" + url

    def _title_from_jina_article(text: str) -> str:
        # jina format starts with "Title: ..."
        for ln in (text or "").splitlines()[:40]:
            if ln.startswith("Title:"):
                t = ln.split("Title:", 1)[1].strip()
                return t
        return ""

    try:
        md_text = _http_get(proxy_url, timeout_s=20)
        # jina proxy renders RSS items as markdown with bare links. Extract the article URLs.
        urls = re.findall(r"https?://www\\.business-standard\\.com/[^\\s\\)\\]]+", md_text)
        seen = set()
        for link in urls[:60]:
            if link in seen:
                continue
            seen.add(link)
            # Fetch real title (cheap-ish via jina; capped)
            title = ""
            try:
                article_txt = _http_get(_jina(link), timeout_s=15)
                title = _title_from_jina_article(article_txt)
            except Exception:
                title = ""
            if not title:
                slug = link.rstrip("/").split("/")[-1]
                title = slug.replace("_", " ").replace("-", " ")[:180]
            summary = ""
            blob = (title + " " + summary).lower()

            # Keep only items likely to be "research/brokerage/calls" (heuristic).
            if not any(k in blob for k in ("broker", "brokerage", "target", "upgrade", "downgrade", "research", "buy", "sell")):
                continue

            out.append(
                {
                    "title": title[:300],
                    "url": link,
                    "published_at": None,
                    "source": "business-standard:rss:markets",
                    "summary": summary[:500],
                }
            )
    except Exception as e:
        logger.debug("BS markets RSS fetch failed: %s", str(e)[:160])

    cache.set("analyst_rss", cache_key, out)
    return out


def _google_news_rss(query: str) -> list[dict[str, Any]]:
    url = "https://news.google.com/rss/search?q=" + requests.utils.quote(query)
    out: list[dict[str, Any]] = []

    if feedparser is not None:
        try:
            parsed = feedparser.parse(url)
            entries = getattr(parsed, "entries", []) or []
            for e in entries[:30]:
                out.append(
                    {
                        "title": str(e.get("title") or "")[:300],
                        "url": str(e.get("link") or ""),
                        "published_at": None,
                        "source": f"google_news_rss:{query[:60]}",
                        "summary": str(e.get("summary") or "")[:500],
                    }
                )
            if out and not getattr(parsed, "bozo", False):
                return out
            logger.debug(
                "Google News feedparser fallback for query=%s (entries=%s bozo=%s status=%s)",
                query[:80],
                len(out),
                getattr(parsed, "bozo", False),
                getattr(parsed, "status", None),
            )
        except Exception as e:
            logger.debug("Google News feedparser parse failed for %s: %s", query[:80], str(e)[:160])

    try:
        xml_text = _http_get(
            url,
            timeout_s=15,
        )
        root = ET.fromstring(xml_text)
        for item in root.findall(".//item")[:30]:
            title = (item.findtext("title") or "")[:300]
            link = item.findtext("link") or ""
            summary = (item.findtext("description") or "")[:500]
            if not title and not link:
                continue
            out.append(
                {
                    "title": title,
                    "url": link,
                    "published_at": None,
                    "source": f"google_news_rss:{query[:60]}",
                    "summary": summary,
                }
            )
    except Exception as e:
        logger.debug("Google News RSS fallback failed for %s: %s", query[:80], str(e)[:160])

    return out


def _fetch_business_standard_analyst_like_via_google(symbol: str, company_name: str, days: int = 14) -> list[dict[str, Any]]:
    """
    Practical, non-scrape way to find broker/analyst-like notes on Business Standard:
    Google News RSS restricted to site + keywords.
    """
    sym = (symbol or "").strip().upper()
    cname = (company_name or "").strip()
    cache_key = f"bs_google:{sym}:{cname}:{days}"
    cached = cache.get("analyst_google", cache_key, ttl=3600)
    if cached is not None:
        return cached

    terms = f"({cname} OR {sym}) (brokerage OR target OR upgrade OR downgrade OR buy OR sell) site:business-standard.com"
    items = _google_news_rss(terms)

    # If the RSS indicates bsmedia PDFs (often via the title), resolve the redirect chain
    # from the Google News RSS article link to find the final PDF URL.
    pdf_urls: list[str] = []
    for it in items:
        title = str(it.get("title") or "")
        if "bsmedia.business-standard.com" not in title.lower():
            continue
        link = str(it.get("url") or "")
        if not link:
            continue
        try:
            r = requests.get(link, headers={"User-Agent": "Mozilla/5.0"}, timeout=20, allow_redirects=True)
            final = str(getattr(r, "url", "") or "")
            if final.lower().endswith(".pdf") and "bsmedia.business-standard.com" in final.lower():
                pdf_urls.append(final)
        except Exception:
            continue

    seen = set()
    max_n = int(getattr(config, "ANALYST_PDF_MAX", 3))
    for u in pdf_urls:
        if len(seen) >= max_n:
            break
        if u in seen:
            continue
        seen.add(u)
        try:
            pdf_bytes = _download_pdf(u, timeout_s=30)
            excerpt = _extract_pdf_excerpt(pdf_bytes, max_chars=int(getattr(config, "ANALYST_PDF_EXCERPT_CHARS", 1500)))
            items.append(
                {
                    "title": u.split("/")[-1],
                    "url": u,
                    "published_at": None,
                    "source": "business-standard:bsmedia:pdf",
                    "summary": excerpt,
                }
            )
        except Exception as e:
            items.append(
                {
                    "title": u.split("/")[-1],
                    "url": u,
                    "published_at": None,
                    "source": "business-standard:bsmedia:pdf",
                    "error": str(e)[:200],
                }
            )
    cache.set("analyst_google", cache_key, items)
    return items


def get_analyst_reports_for_stock(symbol: str, company_name: str = "", days: int = 14) -> list[dict[str, Any]]:
    """
    Pull recent items from configured ANALYST_RSS_FEEDS and keyword-match the symbol/company.
    Returns list of {title, url, published_at, source, summary}.
    """
    feeds = getattr(config, "ANALYST_RSS_FEEDS", []) or []
    html_sources = getattr(config, "ANALYST_HTML_SOURCES", []) or []
    if not feeds and not html_sources:
        return []

    sym = (symbol or "").strip().upper()
    cname = (company_name or "").strip().lower()
    key = f"{sym}:{cname}:{days}:{hash((tuple(feeds), tuple(html_sources)))}"

    cached = cache.get("analyst_reports", key, ttl=3600)
    if cached is not None:
        return cached

    cutoff = datetime.now() - timedelta(days=int(days or 14))
    out: list[dict[str, Any]] = []

    # RSS/Atom feeds (if provided)
    if feeds and feedparser is not None:
        for url in feeds:
            try:
                parsed = feedparser.parse(url)
                entries = getattr(parsed, "entries", []) or []
                for e in entries[:50]:
                    title = str(e.get("title") or "")
                    link = str(e.get("link") or e.get("id") or "")
                    summary = str(e.get("summary") or e.get("description") or "")
                    blob = (title + " " + summary).lower()

                    if sym and sym.lower() not in blob and (cname and cname not in blob):
                        continue

                    published = None
                    try:
                        if e.get("published_parsed"):
                            published = datetime(*e["published_parsed"][:6]).isoformat()
                    except Exception:
                        published = None

                    # If we can parse a date and it's too old, skip.
                    try:
                        if published:
                            dt = datetime.fromisoformat(published)
                            if dt < cutoff:
                                continue
                    except Exception:
                        pass

                    out.append(
                        {
                            "title": title[:300],
                            "url": link,
                            "published_at": published,
                            "source": f"analyst_rss:{url}",
                            "summary": summary[:500],
                        }
                    )
            except Exception as ex:
                logger.debug("Analyst feed parse failed: %s (%s)", url, str(ex)[:120])
                continue

    # HTML sources (best-effort scrapes)
    for src in html_sources:
        if "business-standard.com/markets/research-report" in src:
            items = _fetch_business_standard_research(days=days)
            if not items:
                items = _fetch_business_standard_markets_rss(days=days)
            if not items:
                # Final fallback: google RSS for analyst-like BS articles
                items = _fetch_business_standard_analyst_like_via_google(sym, company_name, days=days)
            # Additionally: ingest broker-tip PDFs (bsmedia) and match by symbol/company in excerpt
            pdf_items = _fetch_business_standard_brokertips_pdfs(days=days)
            for pit in pdf_items:
                blob = (str(pit.get("title") or "") + " " + str(pit.get("summary") or "")).lower()
                if sym and sym.lower() in blob:
                    out.append(pit)
                elif cname and cname in blob:
                    out.append(pit)
        else:
            # Unknown source: skip for now.
            items = []

        # keyword match using title only (listing doesn't include summary)
        for it in items:
            title = (it.get("title") or "").lower()
            if sym and sym.lower() not in title and (cname and cname not in title):
                continue
            out.append(it)

    # Optional: offline-ingested BS PDFs via Playwright script (no cookies stored in app).
    # If present, we can match by PDF excerpt text once extracted externally.
    try:
        bs_dir = Path("storage/bs_research")
        idx = bs_dir / "index.jsonl"
        if idx.exists():
            for line in idx.read_text(encoding="utf-8").splitlines()[-500:]:
                try:
                    obj = __import__("json").loads(line)
                except Exception:
                    continue
                title = str(obj.get("company") or obj.get("title") or "").lower()
                if sym and sym.lower() not in title and (cname and cname not in title):
                    continue
                out.append(
                    {
                        "title": f"{obj.get('company','')} | {obj.get('action','')} | {obj.get('broker','')} | {obj.get('date','')}",
                        "url": obj.get("pdf_url") or "",
                        "published_at": obj.get("date"),
                        "source": "business-standard:offline:index",
                        "summary": "",
                        "meta": obj,
                    }
                )
    except Exception:
        pass

    # de-dupe by url
    seen = set()
    deduped = []
    for it in out:
        u = it.get("url")
        if not u or u in seen:
            continue
        seen.add(u)
        deduped.append(it)

    cache.set("analyst_reports", key, deduped)
    return deduped
