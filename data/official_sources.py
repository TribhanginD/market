"""
Official-source ingestion for NSE/BSE-adjacent market datasets.

Stores raw payloads exactly as fetched and writes normalized records into SQLite.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

import config
import data.cache as cache
from data.fetcher import _get_nse_session
from persistence import db as pdb

logger = logging.getLogger(__name__)


def sync_verified_sources(force_refresh: bool = False) -> dict[str, Any]:
    """
    Pull all currently verified official NSE datasets, persist raw payloads,
    and normalize them into source tables.
    """
    if not getattr(config, "SOURCES_ENABLED", True):
        return {"disabled": True}

    tasks = [
        _source_spec(
            name="nse_corporate_actions",
            url=config.NSE_CORPORATE_ACTIONS_URL,
            category="corporate_actions",
            fmt="json",
            normalizer=_normalize_corporate_actions,
            table="corporate_actions",
        ),
        _source_spec(
            name="nse_financial_results",
            url=config.NSE_FINANCIAL_RESULTS_URL,
            category="financial_results",
            fmt="json",
            normalizer=_normalize_financial_results,
            table="financial_results",
        ),
        _source_spec(
            name="nse_bulk_deals",
            url=config.NSE_BULK_DEALS_URL,
            category="bulk_deals",
            fmt="json",
            normalizer=_normalize_bulk_deals,
            table="bulk_deals",
        ),
        _source_spec(
            name="nse_block_deals",
            url=config.NSE_BLOCK_DEALS_URL,
            category="block_deals",
            fmt="json",
            normalizer=_normalize_block_deals,
            table="block_deals",
        ),
        _source_spec(
            name="nse_index_names",
            url=config.NSE_INDEX_NAMES_URL,
            category="reference:index_names",
            fmt="json",
            normalizer=_normalize_reference_rows,
            table="reference_data",
        ),
        _source_spec(
            name="nse_equity_master",
            url=config.NSE_EQUITY_MASTER_URL,
            category="reference:equity_master",
            fmt="json",
            normalizer=_normalize_reference_rows,
            table="reference_data",
        ),
        _source_spec(
            name="nse_underlying_information",
            url=config.NSE_UNDERLYING_INFO_URL,
            category="reference:underlying_information",
            fmt="json",
            normalizer=_normalize_reference_rows,
            table="reference_data",
        ),
        _source_spec(
            name="nse_shareholding_sdd",
            url="https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern-sdd",
            category="filings",
            fmt="html",
            normalizer=lambda payload, raw_path: _normalize_link_documents_html(
                payload, raw_path,
                source="nse",
                company="NSE India",
                title_prefix="NSE Shareholding",
                include_patterns=("shareholding", "sdd", "xbrl", "ixbrl", "compliance", "filings"),
                base_url="https://www.nseindia.com/",
            ),
            table="filings",
        ),
        _source_spec(
            name="bse_insider_trading",
            url="https://www.bseindia.com/corporates/Insider_Trading_new.aspx",
            category="insider_trades",
            fmt="html",
            normalizer=_normalize_bse_insider_html,
            table="insider_trades",
        ),
        _source_spec(
            name="bse_announcements",
            url="https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w",
            category="filings",
            fmt="json",
            normalizer=_normalize_bse_announcements,
            table="filings",
            request_params=_default_bse_announcement_params(),
        ),
        _source_spec(
            name="bse_forthcoming_results",
            url="https://api.bseindia.com/BseIndiaAPI/api/Corpforthresults/w",
            category="results_calendar",
            fmt="json",
            normalizer=_normalize_bse_forthcoming_results,
            table="results_calendar",
        ),
        _source_spec(
            name="bse_corporate_actions",
            url="https://api.bseindia.com/BseIndiaAPI/api/DefaultData/w",
            category="corporate_actions",
            fmt="json",
            normalizer=_normalize_bse_corporate_actions,
            table="corporate_actions",
            request_params={
                "scripcode": "",
                "Fdate": "",
                "TDate": "",
                "Purposecode": "",
                "strSearch": "D",
                "ddlindustrys": "",
                "ddlcategorys": "E",
                "segment": "0",
            },
        ),
        _source_spec(
            name="bse_shareholding_page",
            url="https://www.bseindia.com/static/about/Clause_35_Shareholding_Pattern.aspx",
            category="filings",
            fmt="html",
            normalizer=lambda payload, raw_path: _normalize_link_documents_html(
                payload, raw_path,
                source="bse",
                company="BSE India",
                title_prefix="BSE Shareholding",
                include_patterns=("shareholding", "pattern", "clause", "quarter", "pdf"),
                base_url="https://www.bseindia.com/",
            ),
            table="filings",
        ),
        _source_spec(
            name="bse_compliance_calendar",
            url="https://www.bseindia.com/corporates/compliancecalendar.aspx",
            category="filings",
            fmt="html",
            normalizer=lambda payload, raw_path: _normalize_link_documents_html(
                payload, raw_path,
                source="bse",
                company="BSE India",
                title_prefix="BSE Compliance",
                include_patterns=("compliance", "calendar", "corporate", "shareholding", "result", "pdf"),
                base_url="https://www.bseindia.com/",
            ),
            table="filings",
        ),
        _source_spec(
            name="amfi_research_information",
            url="https://www.amfiindia.com/research-information",
            category="filings",
            fmt="html",
            normalizer=_normalize_amfi_research_html,
            table="filings",
        ),
        _source_spec(
            name="amfi_amfi_data",
            url="https://www.amfiindia.com/research-information/amfi-data",
            category="filings",
            fmt="html",
            normalizer=lambda payload, raw_path: _normalize_link_documents_html(
                payload, raw_path,
                source="amfi",
                company="AMFI",
                title_prefix="AMFI Data",
                include_patterns=("amfi", "aum", "folio", "data", "pdf", "uploads"),
                base_url="https://www.amfiindia.com/",
            ),
            table="filings",
        ),
        _source_spec(
            name="amfi_otherdata",
            url="https://www.amfiindia.com/otherdata",
            category="filings",
            fmt="html",
            normalizer=lambda payload, raw_path: _normalize_link_documents_html(
                payload, raw_path,
                source="amfi",
                company="AMFI",
                title_prefix="AMFI Other Data",
                include_patterns=("amfi", "otherdata", "data", "pdf", "uploads"),
                base_url="https://www.amfiindia.com/",
            ),
            table="filings",
        ),
        _source_spec(
            name="amfi_sif_research_information",
            url="https://www.amfiindia.com/sif/research-information",
            category="filings",
            fmt="html",
            normalizer=lambda payload, raw_path: _normalize_link_documents_html(
                payload, raw_path,
                source="amfi",
                company="AMFI",
                title_prefix="AMFI SIF",
                include_patterns=("sif", "research", "data", "pdf", "uploads"),
                base_url="https://www.amfiindia.com/",
            ),
            table="filings",
        ),
        _source_spec(
            name="rbi_sectoral_credit",
            url="https://www.rbi.org.in/Scripts/Data_Sectoral_Deployment.aspx",
            category="filings",
            fmt="html",
            normalizer=_normalize_rbi_sectoral_credit_html,
            table="filings",
        ),
        _source_spec(
            name="rbi_dbie",
            url="https://data.rbi.org.in/",
            category="filings",
            fmt="html",
            normalizer=lambda payload, raw_path: _normalize_link_documents_html(
                payload, raw_path,
                source="rbi",
                company="Reserve Bank of India",
                title_prefix="RBI DBIE",
                include_patterns=("dbie", "data", "statistics", "dataset", "release"),
                base_url="https://data.rbi.org.in/",
            ),
            table="filings",
        ),
    ]

    results: dict[str, Any] = {
        "fetched_at": datetime.now().isoformat(),
        "sources": {},
        "totals": {"payloads": 0, "normalized_rows": 0},
    }

    for task in tasks:
        result = _sync_source(task, force_refresh=force_refresh)
        results["sources"][task["name"]] = result
        results["totals"]["payloads"] += result.get("payloads_saved", 0)
        results["totals"]["normalized_rows"] += result.get("normalized_rows", 0)

    # Candidate-but-not-yet-verified source: try it, but keep it clearly marked.
    calendar_result = _try_event_calendar(force_refresh=force_refresh)
    results["sources"]["nse_event_calendar_candidate"] = calendar_result
    results["totals"]["payloads"] += calendar_result.get("payloads_saved", 0)
    results["totals"]["normalized_rows"] += calendar_result.get("normalized_rows", 0)
    return results


def _source_spec(
    *,
    name: str,
    url: str,
    category: str,
    fmt: str,
    normalizer: Callable[[Any, str], list[dict[str, Any]]],
    table: str,
    request_params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "url": url,
        "category": category,
        "fmt": fmt,
        "normalizer": normalizer,
        "table": table,
        "request_params": request_params or {},
    }


def _sync_source(spec: dict[str, Any], *, force_refresh: bool) -> dict[str, Any]:
    cached = None if force_refresh else cache.get("official_source", spec["name"], config.CACHE_TTL_OFFICIAL_SOURCES)
    if cached is not None:
        return cached

    session = _get_nse_session()
    fetched_at = datetime.now().isoformat()
    out = {
        "source": spec["name"],
        "url": spec["url"],
        "category": spec["category"],
        "fetched_at": fetched_at,
        "payloads_saved": 0,
        "normalized_rows": 0,
    }
    try:
        if spec["name"].startswith(("bse_", "amfi_", "rbi_")):
            response = _plain_get(spec["url"], params=spec.get("request_params") or None)
        else:
            response = _nse_get(session, spec["url"])
        raw_text = response.text
        content_type = response.headers.get("content-type", "")
        raw_path = _persist_raw_payload(
            source=spec["name"],
            category=spec["category"],
            content=raw_text.encode("utf-8", errors="ignore"),
            ext=_guess_ext(spec["fmt"], content_type),
            fetched_at=fetched_at,
        )
        payload = _parse_payload(raw_text, spec["fmt"])
        normalized_rows = spec["normalizer"](payload, str(raw_path))

        meta = {
            "row_count": _rough_len(payload),
            "normalized_count": len(normalized_rows),
        }
        pdb.insert_source_payload(
            {
                "source": spec["name"],
                "category": spec["category"],
                "fetched_at": fetched_at,
                "url": spec["url"],
                "url_hash": _hash_text(spec["url"]),
                "raw_path": str(raw_path),
                "content_type": content_type,
                "payload_hash": _hash_bytes(raw_text.encode("utf-8", errors="ignore")),
                "meta": meta,
            }
        )
        inserted = pdb.insert_normalized_rows(spec["table"], normalized_rows)
        out["payloads_saved"] = 1
        out["normalized_rows"] = inserted
        out["raw_path"] = str(raw_path)
        out["row_count"] = meta["row_count"]
        cache.set("official_source", spec["name"], out)
    except Exception as e:
        out["error"] = str(e)
        logger.warning("Official source sync failed for %s: %s", spec["name"], e)
    return out


def _try_event_calendar(*, force_refresh: bool) -> dict[str, Any]:
    name = "nse_event_calendar"
    cached = None if force_refresh else cache.get("official_source", name, config.CACHE_TTL_OFFICIAL_SOURCES)
    if cached is not None:
        return cached

    session = _get_nse_session()
    fetched_at = datetime.now().isoformat()
    out = {
        "source": name,
        "url": config.NSE_EVENT_CALENDAR_URL,
        "category": "results_calendar",
        "status": "candidate",
        "fetched_at": fetched_at,
        "payloads_saved": 0,
        "normalized_rows": 0,
    }
    try:
        response = _nse_get(session, config.NSE_EVENT_CALENDAR_URL)
        content = response.content
        text = response.text
        content_type = response.headers.get("content-type", "")
        raw_path = _persist_raw_payload(
            source=name,
            category="results_calendar",
            content=content,
            ext=_guess_ext("csv", content_type),
            fetched_at=fetched_at,
        )
        rows = _parse_event_calendar_csv(text)
        normalized = _normalize_event_calendar(rows, str(raw_path))
        pdb.insert_source_payload(
            {
                "source": name,
                "category": "results_calendar",
                "fetched_at": fetched_at,
                "url": config.NSE_EVENT_CALENDAR_URL,
                "url_hash": _hash_text(config.NSE_EVENT_CALENDAR_URL),
                "raw_path": str(raw_path),
                "content_type": content_type,
                "payload_hash": _hash_bytes(content),
                "meta": {"row_count": len(rows), "verified": False},
            }
        )
        inserted = pdb.insert_normalized_rows("results_calendar", normalized)
        out["payloads_saved"] = 1
        out["normalized_rows"] = inserted
        out["raw_path"] = str(raw_path)
        out["row_count"] = len(rows)
        out["status"] = "candidate_ingested"
        cache.set("official_source", name, out)
    except Exception as e:
        out["error"] = str(e)
        logger.info("Event calendar candidate source not yet stable: %s", e)
    return out


def _nse_get(session: requests.Session, url: str) -> requests.Response:
    session.get("https://www.nseindia.com", timeout=config.HTTP_TIMEOUT_SECONDS)
    response = session.get(url, timeout=config.HTTP_TIMEOUT_SECONDS)
    if response.status_code < 400:
        return response
    if response.status_code == 403 and getattr(config, "OFFICIAL_SOURCE_BROWSER_FALLBACK", True):
        browser_resp = _nse_get_via_browser(url)
        if browser_resp is not None:
            return browser_resp
    response.raise_for_status()
    return response


def _plain_get(url: str, params: Optional[dict[str, Any]] = None) -> requests.Response:
    response = requests.get(
        url,
        params=params,
        timeout=config.HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.bseindia.com/"},
    )
    response.raise_for_status()
    return response


_PLAYWRIGHT_MISSING_LOGGED = False


def _nse_get_via_browser(url: str) -> Optional[requests.Response]:
    if sync_playwright is None:
        global _PLAYWRIGHT_MISSING_LOGGED
        if not _PLAYWRIGHT_MISSING_LOGGED:
            logger.warning(
                "playwright not installed — browser-based NSE fetch disabled. "
                "Install with `pip install playwright && playwright install chromium` to enable."
            )
            _PLAYWRIGHT_MISSING_LOGGED = True
        return None
    try:
        with sync_playwright() as p:
            browser_type = p.chromium
            user_data_dir = Path(config.OFFICIAL_SOURCE_BROWSER_USER_DATA_DIR)
            user_data_dir.mkdir(parents=True, exist_ok=True)
            context = browser_type.launch_persistent_context(
                user_data_dir=str(user_data_dir),
                headless=config.OFFICIAL_SOURCE_BROWSER_HEADLESS,
                slow_mo=config.OFFICIAL_SOURCE_BROWSER_SLOW_MO_MS if config.OFFICIAL_SOURCE_BROWSER_SLOW_MO_MS > 0 else None,
                locale="en-US",
                timezone_id="Asia/Kolkata",
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                extra_http_headers={
                    "Accept": "application/json,text/plain,*/*",
                    "Referer": _bootstrap_page_for_url(url),
                    "Origin": "https://www.nseindia.com",
                },
            )
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(_bootstrap_page_for_url(url), wait_until="domcontentloaded", timeout=config.HTTP_TIMEOUT_SECONDS * 1000)
            if config.OFFICIAL_SOURCE_BROWSER_WAIT_MS > 0:
                page.wait_for_timeout(config.OFFICIAL_SOURCE_BROWSER_WAIT_MS)
            if config.OFFICIAL_SOURCE_BROWSER_PAUSE_FOR_USER:
                print("Paused. Complete any login / CAPTCHA in the opened browser window, then press Enter here to continue...")
                try:
                    input()
                except EOFError:
                    pass
            result = page.evaluate(
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
            context.close()
            r = requests.Response()
            r.status_code = int(result.get("status") or 0)
            r._content = (result.get("text") or "").encode("utf-8", errors="ignore")
            headers = {str(k).lower(): str(v) for k, v in (result.get("headers") or {}).items()}
            r.headers = headers
            r.url = url
            if r.status_code >= 400:
                return None
            return r
    except Exception as e:
        logger.info("Browser fallback failed for %s: %s", url, e)
        return None


def _persist_raw_payload(*, source: str, category: str, content: bytes, ext: str, fetched_at: str) -> Path:
    ts = fetched_at.replace(":", "").replace("-", "").replace(".", "")
    source_dir = config.SOURCE_PAYLOADS_DIR / source / category.replace(":", "_")
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / f"{ts}.{ext}"
    path.write_bytes(content)
    return path


def _parse_payload(text: str, fmt: str) -> Any:
    if fmt == "json":
        return json.loads(text)
    if fmt == "csv":
        return _csv_dicts(text)
    if fmt == "html":
        return text
    raise ValueError(f"Unsupported payload format: {fmt}")


def _parse_event_calendar_csv(text: str) -> list[dict[str, Any]]:
    return _csv_dicts(text)


def _normalize_corporate_actions(payload: Any, raw_path: str) -> list[dict[str, Any]]:
    rows = payload if isinstance(payload, list) else payload.get("data", [])
    out = []
    for row in rows[: config.SOURCE_MAX_ROWS_PER_TABLE]:
        symbol = _clean_symbol(row.get("symbol"))
        company = row.get("comp")
        subject = row.get("subject")
        ex_date = _iso_date(row.get("exDate"))
        record_date = _iso_date(row.get("recDate"))
        hash_key = _stable_hash("corporate_action", symbol, subject, ex_date, row.get("isin"))
        out.append(
            {
                "source": "nse",
                "symbol": symbol,
                "company": company,
                "subject": subject,
                "ex_date": ex_date,
                "record_date": record_date,
                "published_at": _iso_datetime(row.get("caBroadcastDate")),
                "url": config.NSE_CORPORATE_ACTIONS_URL,
                "raw_path": raw_path,
                "hash": hash_key,
                "payload_json": json.dumps(row, default=str),
            }
        )
    return out


def _normalize_financial_results(payload: Any, raw_path: str) -> list[dict[str, Any]]:
    rows = payload if isinstance(payload, list) else payload.get("data", [])
    out = []
    for row in rows[: config.SOURCE_MAX_ROWS_PER_TABLE]:
        symbol = _clean_symbol(row.get("symbol"))
        filing_date = _iso_datetime(row.get("filingDate"))
        out.append(
            {
                "source": "nse",
                "symbol": symbol,
                "company": row.get("companyName"),
                "industry": row.get("industry"),
                "period": row.get("period"),
                "relating_to": row.get("relatingTo"),
                "financial_year": row.get("financialYear"),
                "from_date": _iso_date(row.get("fromDate")),
                "to_date": _iso_date(row.get("toDate")),
                "filing_date": filing_date,
                "published_at": _iso_datetime(row.get("broadCastDate")) or filing_date,
                "xbrl_url": row.get("xbrl"),
                "raw_path": raw_path,
                "hash": _stable_hash("financial_result", symbol, row.get("period"), row.get("fromDate"), row.get("toDate"), row.get("xbrl")),
                "payload_json": json.dumps(row, default=str),
            }
        )
    return out


def _normalize_bulk_deals(payload: Any, raw_path: str) -> list[dict[str, Any]]:
    rows = []
    if isinstance(payload, dict):
        rows = payload.get("BULK_DEALS_DATA") or payload.get("data") or []
    elif isinstance(payload, list):
        rows = payload
    return _normalize_deal_rows(rows, raw_path, "nse_bulk")


def _normalize_block_deals(payload: Any, raw_path: str) -> list[dict[str, Any]]:
    rows = payload.get("data", []) if isinstance(payload, dict) else payload
    return _normalize_deal_rows(rows, raw_path, "nse_block")


def _normalize_deal_rows(rows: Any, raw_path: str, source_label: str) -> list[dict[str, Any]]:
    out = []
    if not isinstance(rows, list):
        return out
    for row in rows[: config.SOURCE_MAX_ROWS_PER_TABLE]:
        symbol = _clean_symbol(row.get("symbol") or row.get("Symbol"))
        company = row.get("name") or row.get("companyName") or row.get("company")
        event_date = _iso_date(row.get("date") or row.get("tradeDate"))
        client_name = row.get("clientName") or row.get("client")
        buy_sell = row.get("buySell") or row.get("action")
        quantity = _to_float(row.get("qty") or row.get("quantity"))
        price = _to_float(row.get("watp") or row.get("price"))
        remarks = row.get("remarks")
        out.append(
            {
                "source": "nse",
                "symbol": symbol,
                "company": company,
                "event_date": event_date,
                "client_name": client_name,
                "buy_sell": buy_sell,
                "quantity": quantity,
                "price": price,
                "remarks": remarks,
                "raw_path": raw_path,
                "hash": _stable_hash(source_label, symbol, event_date, client_name, buy_sell, quantity, price),
                "payload_json": json.dumps(row, default=str),
            }
        )
    return out


def _normalize_event_calendar(rows: list[dict[str, Any]], raw_path: str) -> list[dict[str, Any]]:
    out = []
    for row in rows[: config.SOURCE_MAX_ROWS_PER_TABLE]:
        symbol = _clean_symbol(row.get("symbol") or row.get("Symbol"))
        company = row.get("companyName") or row.get("company") or row.get("Company")
        event_date = _iso_date(row.get("eventDate") or row.get("date") or row.get("Date"))
        purpose = row.get("purpose") or row.get("event") or row.get("subject")
        out.append(
            {
                "source": "nse",
                "symbol": symbol,
                "company": company,
                "event_date": event_date,
                "purpose": purpose,
                "raw_path": raw_path,
                "hash": _stable_hash("results_calendar", symbol, event_date, purpose),
                "payload_json": json.dumps(row, default=str),
            }
        )
    return out


def _normalize_reference_rows(payload: Any, raw_path: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("data") or payload.get("indexList") or payload.get("underlyingInformation") or payload.get("value") or payload
    else:
        rows = payload

    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        return []

    out = []
    for row in rows[: config.SOURCE_MAX_ROWS_PER_TABLE]:
        symbol = _clean_symbol(row.get("symbol"))
        company = row.get("underlying") or row.get("companyName") or row.get("name") or row.get("index")
        category = "reference"
        if "underlying" in row:
            category = "underlying_information"
        elif "index" in row or "indexName" in row:
            category = "index_names"
        elif "series" in row or "isin" in row:
            category = "equity_master"
        out.append(
            {
                "source": "nse",
                "category": category,
                "symbol": symbol,
                "company": company,
                "event_date": None,
                "url": None,
                "raw_path": raw_path,
                "hash": _stable_hash(category, symbol, company, json.dumps(row, sort_keys=True, default=str)),
                "payload_json": json.dumps(row, default=str),
            }
        )
    return out


def _normalize_bse_insider_html(payload: Any, raw_path: str) -> list[dict[str, Any]]:
    html = str(payload or "")
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for tr in soup.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(cells) != 16:
            continue
        security_code = cells[0].strip()
        if not security_code.isdigit():
            continue
        security_name = cells[1].strip()
        person_name = cells[2].strip()
        category = cells[3].strip()
        quantity = _to_float(cells[6])
        value = _to_float(cells[7])
        transaction_type = cells[8].strip()
        period_text = cells[10].strip()
        event_date = _extract_first_date(period_text)
        reported_at = _iso_date(cells[15])
        out.append(
            {
                "source": "bse",
                "symbol": security_code,
                "company": security_name,
                "person_name": person_name,
                "category_of_person": category,
                "transaction_type": transaction_type,
                "event_date": event_date,
                "reported_at": reported_at,
                "quantity": quantity,
                "value": value,
                "url": "https://www.bseindia.com/corporates/Insider_Trading_new.aspx",
                "raw_path": raw_path,
                "hash": _stable_hash("bse_insider", security_code, person_name, transaction_type, event_date, reported_at, quantity, value),
                "payload_json": json.dumps(
                    {
                        "security_code": security_code,
                        "security_name": security_name,
                        "person_name": person_name,
                        "category": category,
                        "pre_holding": cells[4],
                        "security_type": cells[5],
                        "number": cells[6],
                        "value": cells[7],
                        "transaction_type": transaction_type,
                        "post_holding": cells[9],
                        "period": period_text,
                        "mode": cells[11],
                        "contract_type": cells[12],
                        "buy_value_units": cells[13],
                        "sale_value_units": cells[14],
                        "reported_to_exchange": cells[15],
                    },
                    default=str,
                ),
            }
        )
    return out


def _normalize_amfi_research_html(payload: Any, raw_path: str) -> list[dict[str, Any]]:
    html = str(payload or "")
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        title = a.get_text(" ", strip=True)
        href = a["href"].strip()
        if not href:
            continue
        if "research-information" not in href and "otherdata" not in href:
            continue
        url = urljoin("https://www.amfiindia.com/", href)
        out.append(
            {
                "source": "amfi",
                "symbol": None,
                "company": "AMFI",
                "title": title or href,
                "event_date": None,
                "published_at": None,
                "url": url,
                "raw_path": raw_path,
                "hash": _stable_hash("amfi_research", title, url),
                "payload_json": json.dumps({"title": title, "url": url}, default=str),
            }
        )
    return out


def _normalize_link_documents_html(
    payload: Any,
    raw_path: str,
    *,
    source: str,
    company: str,
    title_prefix: str,
    include_patterns: tuple[str, ...],
    base_url: str,
) -> list[dict[str, Any]]:
    html = str(payload or "")
    soup = BeautifulSoup(html, "html.parser")
    out = []
    seen: set[str] = set()
    pats = tuple((p or "").lower() for p in include_patterns if p)
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        title = a.get_text(" ", strip=True)
        if not href:
            continue
        url = urljoin(base_url, href)
        hay = f"{title} {href}".lower()
        if pats and not any(p in hay for p in pats):
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append(
            {
                "source": source,
                "symbol": None,
                "company": company,
                "title": title or f"{title_prefix} link",
                "event_date": None,
                "published_at": None,
                "url": url,
                "raw_path": raw_path,
                "hash": _stable_hash(source, title_prefix, url),
                "payload_json": json.dumps({"title": title, "url": url}, default=str),
            }
        )
    return out


def _normalize_rbi_sectoral_credit_html(payload: Any, raw_path: str) -> list[dict[str, Any]]:
    html = str(payload or "")
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        title = a.get_text(" ", strip=True)
        href = a["href"].strip()
        if not title or "Sectoral Deployment of Bank Credit" not in title:
            continue
        url = urljoin("https://www.rbi.org.in/Scripts/", href)
        event_date = _extract_month_year(title)
        out.append(
            {
                "source": "rbi",
                "symbol": None,
                "company": "Reserve Bank of India",
                "title": title,
                "event_date": event_date,
                "published_at": None,
                "url": url,
                "raw_path": raw_path,
                "hash": _stable_hash("rbi_sectoral_credit", title, url),
                "payload_json": json.dumps({"title": title, "url": url}, default=str),
            }
        )
    return out


def _normalize_bse_announcements(payload: Any, raw_path: str) -> list[dict[str, Any]]:
    rows = payload.get("Table", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows[: config.SOURCE_MAX_ROWS_PER_TABLE]:
        news_dt = row.get("NEWS_DT") or row.get("DT_TM")
        attachment = _build_bse_announcement_attachment_url(row)
        out.append(
            {
                "source": "bse",
                "symbol": str(row.get("SCRIP_CD") or "").strip() or None,
                "company": row.get("SLONGNAME"),
                "title": row.get("NEWSSUB") or row.get("HEADLINE"),
                "event_date": _iso_datetime(news_dt),
                "published_at": _iso_datetime(row.get("DissemDT")) or _iso_datetime(news_dt),
                "url": attachment or row.get("NSURL") or "https://www.bseindia.com/corporates/ann.html",
                "raw_path": raw_path,
                "hash": _stable_hash("bse_announcement", row.get("NEWSID"), row.get("SCRIP_CD"), row.get("ATTACHMENTNAME")),
                "payload_json": json.dumps(row, default=str),
            }
        )
    return out


def _normalize_bse_forthcoming_results(payload: Any, raw_path: str) -> list[dict[str, Any]]:
    rows = payload if isinstance(payload, list) else payload.get("Table", [])
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows[: config.SOURCE_MAX_ROWS_PER_TABLE]:
        out.append(
            {
                "source": "bse",
                "symbol": str(row.get("scrip_Code") or "").strip() or None,
                "company": row.get("Long_Name") or row.get("short_name"),
                "event_date": _iso_date(row.get("meeting_date")),
                "purpose": "Forthcoming Results",
                "raw_path": raw_path,
                "hash": _stable_hash("bse_forthcoming_results", row.get("scrip_Code"), row.get("meeting_date")),
                "payload_json": json.dumps(row, default=str),
            }
        )
    return out


def _normalize_bse_corporate_actions(payload: Any, raw_path: str) -> list[dict[str, Any]]:
    rows = payload if isinstance(payload, list) else payload.get("Table", [])
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows[: config.SOURCE_MAX_ROWS_PER_TABLE]:
        out.append(
            {
                "source": "bse",
                "symbol": str(row.get("scrip_code") or "").strip() or None,
                "company": row.get("long_name") or row.get("short_name"),
                "subject": row.get("Purpose"),
                "ex_date": _iso_date(row.get("Ex_date")) or _iso_date(row.get("exdate")),
                "record_date": _iso_date(row.get("RD_Date")),
                "published_at": None,
                "url": "https://www.bseindia.com/corporates/corporates_act.html",
                "raw_path": raw_path,
                "hash": _stable_hash("bse_corporate_action", row.get("scrip_code"), row.get("Purpose"), row.get("Ex_date")),
                "payload_json": json.dumps(row, default=str),
            }
        )
    return out


def _clean_symbol(value: Any) -> Optional[str]:
    text = str(value).strip().upper() if value is not None else ""
    return text or None


def _to_float(value: Any) -> Optional[float]:
    if value in (None, "", "-"):
        return None
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return None


def _iso_date(value: Any) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%d-%b-%Y", "%d %b %Y", "%d %B %Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%b-%y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except Exception:
            continue
    return text


def _iso_datetime(value: Any) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%d-%b-%Y %H:%M", "%d-%b-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).isoformat()
        except Exception:
            continue
    return text


def _extract_first_date(text: str) -> Optional[str]:
    match = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", text or "")
    if not match:
        return None
    return _iso_date(match.group(1))


def _extract_month_year(text: str) -> Optional[str]:
    normalized = (text or "").replace("–", " ").replace("-", " ")
    months = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    match = re.search(months + r"\s+(\d{4})", normalized, re.IGNORECASE)
    if not match:
        return None
    try:
        month = match.group(1).title()
        year = match.group(2)
        dt = datetime.strptime(f"01 {month} {year}", "%d %B %Y")
        return dt.date().isoformat()
    except Exception:
        return None


def _default_bse_announcement_params() -> dict[str, Any]:
    today = datetime.now().strftime("%Y%m%d")
    return {
        "strScrip": "",
        "strCat": "-1",
        "strPrevDate": today,
        "strToDate": today,
        "strSearch": "P",
        "strType": "C",
        "pageno": "1",
        "subcategory": "-1",
    }


def _build_bse_announcement_attachment_url(row: dict[str, Any]) -> Optional[str]:
    attachment = (row.get("ATTACHMENTNAME") or "").strip()
    if not attachment:
        return None
    news_dt = str(row.get("NEWS_DT") or "")
    parts = news_dt.split("-")
    if len(parts) < 2:
        return None
    year = parts[0]
    month = parts[1].lstrip("0") or parts[1]
    return f"https://www.bseindia.com/xml-data/corpfiling/CorpAttachment/{year}/{month}/{attachment}"
    try:
        dt = datetime.strptime(f"01 {match.group(1)} {match.group(2)}", "%d %B %Y")
        return dt.date().isoformat()
    except Exception:
        return None


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_hash(*parts: Any) -> str:
    joined = "|".join("" if p is None else str(p) for p in parts)
    return _hash_text(joined)


def _guess_ext(fmt: str, content_type: str) -> str:
    if "csv" in content_type or fmt == "csv":
        return "csv"
    if "json" in content_type or fmt == "json":
        return "json"
    if "html" in content_type:
        return "html"
    return "txt"


def _rough_len(payload: Any) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("data", "BULK_DEALS_DATA", "indexList", "underlyingInformation"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
        return len(payload)
    return 0


def _bootstrap_page_for_url(url: str) -> str:
    if "event-calendar" in url:
        return config.NSE_EVENT_CALENDAR_PAGE_URL
    return config.NSE_FILINGS_ANNOUNCEMENTS_PAGE_URL


def _csv_dicts(text: str) -> list[dict[str, Any]]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    cleaned = "\n".join(lines)
    if cleaned.startswith("\ufeff"):
        cleaned = cleaned.lstrip("\ufeff")
    return list(csv.DictReader(io.StringIO(cleaned)))
