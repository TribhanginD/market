from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

import config
from agents.base_agent import BaseAgent
from data.nifty500 import get_nifty500
from llm.providers import parse_model_provider
from persistence import db as pdb

logger = logging.getLogger(__name__)

ALLOWED_IMPORTANCE = {"low", "medium", "high", "critical"}
ALLOWED_THESIS_CHANGE = {"strengthens", "weakens", "neutral", "possible", "unclear"}


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _now_iso() -> str:
    return datetime.now().isoformat()


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _date_str(dt: Optional[datetime]) -> str:
    return dt.strftime("%Y-%m-%d") if dt else datetime.now().strftime("%Y-%m-%d")


def _clip(text: Any, max_len: int) -> str:
    value = str(text or "").strip()
    return value[:max_len]


def _as_list_of_str(value: Any, *, max_items: int, max_item_len: int) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    out: list[str] = []
    seen = set()
    for item in items:
        text = _clip(item, max_item_len)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def _as_float_01(value: Any, default: float) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    return max(0.0, min(1.0, x))


def _first_match(pattern: str, text: str, flags: int = re.I) -> str:
    m = re.search(pattern, text or "", flags)
    return m.group(1).strip() if m else ""


def _norm_company(name: str) -> str:
    x = (name or "").upper()
    x = re.sub(r"[^A-Z0-9 ]+", " ", x)
    for token in (
        " LIMITED", " LTD", " PRIVATE", " PVT", " PLC", " INC", " CORPORATION",
        " CORP", " COMPANY", " CO", " INDUSTRIES", " INDUSTRY"
    ):
        x = x.replace(token, " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


class EntityMapper:
    def __init__(self) -> None:
        self.df = get_nifty500(force_refresh=False)
        self.symbols = set(self.df["symbol"].astype(str).str.upper())
        self.by_company: dict[str, tuple[str, str]] = {}
        for _, row in self.df.iterrows():
            symbol = str(row["symbol"]).strip().upper()
            company = str(row.get("company_name") or "").strip()
            norm = _norm_company(company)
            if norm and norm not in self.by_company:
                self.by_company[norm] = (symbol, company)

    def resolve(self, symbol: str = "", company: str = "", title: str = "") -> tuple[str, str]:
        sym = (symbol or "").strip().upper()
        company = (company or "").strip()
        if sym and sym in self.symbols:
            if not company:
                company = self.df.loc[self.df["symbol"] == sym, "company_name"].iloc[0]
            return sym, company

        for candidate in (company, title):
            norm = _norm_company(candidate)
            if not norm:
                continue
            if norm in self.by_company:
                return self.by_company[norm]
            for key, value in self.by_company.items():
                if len(norm) >= 10 and (norm in key or key in norm):
                    return value
        return sym, company


@dataclass
class ETLResult:
    ingested_documents: int = 0
    extracted_documents: int = 0
    queued_jobs: int = 0
    processed_jobs: int = 0
    failed_jobs: int = 0
    updated_symbols: int = 0
    packets_built: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class ETLPipeline:
    def __init__(self) -> None:
        self.mapper = EntityMapper()
        self.docs_dir = config.STORAGE_DIR / "documents_raw"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.memo_model = getattr(config, "ETL_MEMO_MODEL", "groq:openai/gpt-oss-20b")
        self.memo_max_tokens = int(getattr(config, "ETL_MEMO_MAX_TOKENS", config.MAX_TOKENS))
        self.chunk_chars = int(getattr(config, "ETL_CHUNK_CHARS", 2800))
        self.max_chunks = int(getattr(config, "ETL_MAX_CHUNKS_PER_DOC", 4))
        self.lookback_days = int(getattr(config, "ETL_LOOKBACK_DAYS", 35))
        self.max_docs_per_run = int(getattr(config, "ETL_MAX_DOCS_PER_RUN", 150))
        self.max_jobs_per_run = int(getattr(config, "ETL_MAX_JOBS_PER_RUN", 30))

    def run(self, *, max_docs: Optional[int] = None, max_jobs: Optional[int] = None) -> dict[str, Any]:
        result = ETLResult()
        doc_limit = int(max_docs or self.max_docs_per_run)
        job_limit = int(max_jobs or self.max_jobs_per_run)

        ingested = self.ingest_documents(limit=doc_limit)
        result.ingested_documents = ingested

        extracted_ids = self.extract_pending_documents(limit=doc_limit)
        result.extracted_documents = len(extracted_ids)

        queued = self.enqueue_memo_jobs(limit=doc_limit)
        result.queued_jobs = queued

        processed = self.process_jobs(limit=job_limit)
        result.processed_jobs = processed["done"]
        result.failed_jobs = processed["failed"]
        result.llm_input_tokens = processed["input_tokens"]
        result.llm_output_tokens = processed["output_tokens"]

        touched_symbols = self._recent_document_symbols(limit_days=self.lookback_days)
        result.updated_symbols = self.rebuild_symbol_memory(touched_symbols)
        result.packets_built = self.build_packets(touched_symbols)
        return result.to_dict()

    def ingest_documents(self, *, limit: int) -> int:
        inserted = 0
        rows = self._source_documents(limit=limit)
        for row in rows:
            document = self._document_from_source_row(row)
            if not document:
                continue
            before = len(pdb.fetch_rows("SELECT id FROM documents WHERE hash = ?", (document["hash"],)))
            doc_id = pdb.upsert_document(document)
            if doc_id and before == 0:
                inserted += 1
        return inserted

    def extract_pending_documents(self, *, limit: int) -> list[int]:
        rows = pdb.fetch_rows(
            """
            SELECT *
            FROM documents
            WHERE extraction_status IN ('pending', 'failed')
            ORDER BY COALESCE(published_at, event_date, created_at) DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        extracted_ids: list[int] = []
        for row in rows:
            doc_id = int(row["id"])
            try:
                text = self._extract_document_text(row)
                text_hash = _sha(text)
                pdb.update_document_extraction(doc_id, extracted_text=text, text_hash=text_hash, extraction_status="done")
                chunks = self._chunk_text(text)
                pdb.replace_document_chunks(doc_id, chunks)
                extracted_ids.append(doc_id)
            except Exception as e:
                logger.warning("ETL extraction failed for doc_id=%s: %s", doc_id, str(e)[:200])
                pdb.update_document_extraction(doc_id, extracted_text="", text_hash="", extraction_status="failed")
        return extracted_ids

    def enqueue_memo_jobs(self, *, limit: int) -> int:
        rows = pdb.fetch_rows(
            """
            SELECT d.id, d.doc_type
            FROM documents d
            LEFT JOIN document_memos m ON m.document_id = d.id AND m.memo_type = 'research_memo'
            WHERE d.extraction_status = 'done' AND m.id IS NULL
            ORDER BY COALESCE(d.published_at, d.event_date, d.created_at) DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        count = 0
        for row in rows:
            pdb.upsert_document_job(
                int(row["id"]),
                job_type="research_memo",
                priority=self._priority_for_doc_type(str(row["doc_type"] or "")),
                model=self.memo_model,
                status="pending",
            )
            count += 1
        return count

    def process_jobs(self, *, limit: int) -> dict[str, int]:
        rows = pdb.fetch_rows(
            """
            SELECT j.*, d.symbol, d.company, d.title, d.doc_type, d.url, d.raw_path, d.published_at, d.event_date, d.meta_json
            FROM document_jobs j
            JOIN documents d ON d.id = j.document_id
            WHERE j.job_type = 'research_memo'
              AND j.status IN ('pending', 'retry')
              AND (j.retry_at IS NULL OR j.retry_at <= ?)
            ORDER BY j.priority DESC, j.created_at ASC
            LIMIT ?
            """,
            (_now_iso(), int(limit)),
        )
        done = failed = input_tokens = output_tokens = 0
        for row in rows:
            job_id = int(row["id"])
            try:
                memo, usage = self._generate_research_memo(row)
                payload = memo if isinstance(memo, dict) else {"summary": str(memo)}
                hash_key = _sha(f"{row['document_id']}|research_memo|{json.dumps(payload, sort_keys=True, default=str)}")
                pdb.insert_document_memo(
                    {
                        "document_id": int(row["document_id"]),
                        "memo_type": "research_memo",
                        "model": row.get("model") or self.memo_model,
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "confidence": payload.get("confidence"),
                        "payload": payload,
                        "raw_text": json.dumps(payload, default=str),
                        "hash": hash_key,
                    }
                )
                pdb.update_document_job(job_id, status="done", attempts_increment=True, retry_at=None, last_error=None)
                done += 1
                input_tokens += int(usage.get("input_tokens") or 0)
                output_tokens += int(usage.get("output_tokens") or 0)
            except Exception as e:
                msg = str(e)[:500]
                retryable = ("429" in msg) or ("rate limit" in msg.lower()) or ("503" in msg)
                logger.warning("ETL memo job failed job_id=%s: %s", job_id, msg)
                if retryable:
                    retry_at = (datetime.now() + timedelta(minutes=10)).isoformat()
                    pdb.update_document_job(job_id, status="retry", attempts_increment=True, retry_at=retry_at, last_error=msg)
                else:
                    pdb.update_document_job(job_id, status="failed", attempts_increment=True, retry_at=None, last_error=msg)
                    failed += 1
        return {"done": done, "failed": failed, "input_tokens": input_tokens, "output_tokens": output_tokens}

    def rebuild_symbol_memory(self, symbols: Iterable[str]) -> int:
        count = 0
        for sym in sorted({(s or "").strip().upper() for s in symbols if (s or "").strip()}):
            docs = pdb.fetch_rows(
                """
                SELECT d.id, d.company, d.title, d.doc_type, d.url, d.published_at, d.event_date,
                       m.payload_json
                FROM documents d
                LEFT JOIN document_memos m
                  ON m.document_id = d.id AND m.memo_type = 'research_memo'
                WHERE d.symbol = ?
                ORDER BY COALESCE(d.published_at, d.event_date, d.created_at) DESC
                LIMIT 20
                """,
                (sym,),
            )
            if not docs:
                continue
            company = docs[0].get("company")
            memos = []
            for doc in docs[:10]:
                payload = {}
                try:
                    payload = json.loads(doc.get("payload_json") or "{}")
                except Exception:
                    payload = {}
                memos.append(
                    {
                        "title": _clip(doc.get("title"), 300),
                        "doc_type": _clip(doc.get("doc_type"), 60),
                        "published_at": doc.get("published_at"),
                        "event_date": doc.get("event_date"),
                        "url": doc.get("url"),
                        "memo": self._normalize_memo_payload(payload),
                    }
                )
            summary = self._normalize_symbol_memory(
                {
                "symbol": sym,
                "company": company,
                "updated_at": _now_iso(),
                "recent_documents": memos,
                "event_counts": self._event_counts(sym),
                "latest_events": self._latest_structured_events(sym),
                }
            )
            pdb.upsert_symbol_memory(sym, company, summary)
            count += 1
        return count

    def build_packets(self, symbols: Iterable[str]) -> int:
        count = 0
        as_of_date = datetime.now().strftime("%Y-%m-%d")
        for sym in sorted({(s or "").strip().upper() for s in symbols if (s or "").strip()}):
            rows = pdb.fetch_rows("SELECT company, summary_json FROM symbol_memory WHERE symbol = ?", (sym,))
            if not rows:
                continue
            summary = json.loads(rows[0]["summary_json"])
            packet = self._normalize_etl_packet({
                "symbol": sym,
                "company": rows[0].get("company"),
                "as_of_date": as_of_date,
                "research_memory": summary,
                "top_recent_memos": summary.get("recent_documents", [])[:5],
                "event_counts": summary.get("event_counts", {}),
                "latest_events": summary.get("latest_events", {}),
            })
            packet_hash = _sha(json.dumps(packet, sort_keys=True, default=str))
            pdb.upsert_etl_packet(sym, as_of_date, packet, packet_hash)
            count += 1
        return count

    def _source_documents(self, *, limit: int) -> list[dict[str, Any]]:
        cutoff = (datetime.now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        conn = sqlite3.connect(str(config.DB_FILE))
        conn.row_factory = sqlite3.Row
        try:
            queries = [
                """
                SELECT 'filings' AS source_table, id AS source_row_id, source, symbol, company, title, url, raw_path,
                       published_at, event_date, payload_json, NULL AS content_type, 'filing' AS doc_type
                FROM filings
                WHERE COALESCE(event_date, substr(published_at, 1, 10), '9999-12-31') >= ?
                """,
                """
                SELECT 'financial_results' AS source_table, id AS source_row_id, source, symbol, company,
                       (company || ' ' || COALESCE(relating_to, '') || ' results ' || COALESCE(to_date, '')) AS title,
                       xbrl_url AS url, raw_path, filing_date AS published_at, to_date AS event_date, payload_json,
                       'application/xml' AS content_type, 'financial_result' AS doc_type
                FROM financial_results
                WHERE COALESCE(to_date, substr(filing_date, 1, 10), '9999-12-31') >= ?
                """,
                """
                SELECT 'corporate_actions' AS source_table, id AS source_row_id, source, symbol, company,
                       subject AS title, url, raw_path, published_at, ex_date AS event_date, payload_json,
                       NULL AS content_type, 'corporate_action' AS doc_type
                FROM corporate_actions
                WHERE COALESCE(ex_date, record_date, substr(published_at, 1, 10), '9999-12-31') >= ?
                """,
                """
                SELECT 'insider_trades' AS source_table, id AS source_row_id, source, symbol, company,
                       (COALESCE(person_name, '') || ' ' || COALESCE(transaction_type, '') || ' ' || COALESCE(company, '')) AS title,
                       url, raw_path, reported_at AS published_at, event_date, payload_json,
                       NULL AS content_type, 'insider_trade' AS doc_type
                FROM insider_trades
                WHERE COALESCE(event_date, substr(reported_at, 1, 10), '9999-12-31') >= ?
                """,
                """
                SELECT 'bulk_deals' AS source_table, id AS source_row_id, source, symbol, company,
                       (COALESCE(buy_sell, '') || ' bulk deal ' || COALESCE(company, '')) AS title,
                       NULL AS url, raw_path, event_date AS published_at, event_date, payload_json,
                       NULL AS content_type, 'bulk_deal' AS doc_type
                FROM bulk_deals
                WHERE COALESCE(event_date, '9999-12-31') >= ?
                """,
                """
                SELECT 'results_calendar' AS source_table, id AS source_row_id, source, symbol, company,
                       (COALESCE(company, '') || ' ' || COALESCE(purpose, 'Results')) AS title,
                       NULL AS url, raw_path, event_date AS published_at, event_date, payload_json,
                       NULL AS content_type, 'results_calendar' AS doc_type
                FROM results_calendar
                WHERE COALESCE(event_date, '9999-12-31') >= ?
                """,
            ]
            out: list[dict[str, Any]] = []
            for query in queries:
                out.extend([dict(r) for r in conn.execute(query, (cutoff,)).fetchall()])
        finally:
            conn.close()

        out.extend(self._bs_index_documents())
        out.sort(key=lambda r: (r.get("published_at") or r.get("event_date") or ""), reverse=True)
        return out[:limit]

    def _bs_index_documents(self) -> list[dict[str, Any]]:
        index_path = Path(config.BS_RESEARCH_OUT_DIR) / "index.jsonl"
        if not index_path.exists():
            return []
        cutoff = datetime.now() - timedelta(days=self.lookback_days)
        out: list[dict[str, Any]] = []
        for line in index_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            published = rec.get("date") or rec.get("published_at")
            dt = _parse_dt(published)
            if dt and dt < cutoff:
                continue
            local_path = rec.get("local_path") or self._resolve_bs_local_path(rec)
            out.append(
                {
                    "source_table": "bs_analyst_reports",
                    "source_row_id": None,
                    "source": "business-standard",
                    "symbol": rec.get("symbol"),
                    "company": rec.get("company"),
                    "title": rec.get("title") or rec.get("company") or "Business Standard Analyst Report",
                    "url": rec.get("pdf_url") or rec.get("url"),
                    "raw_path": local_path,
                    "published_at": published,
                    "event_date": published,
                    "payload_json": json.dumps(rec, default=str),
                    "content_type": "application/pdf",
                    "doc_type": "analyst_report",
                }
            )
        return out

    def _resolve_bs_local_path(self, rec: dict[str, Any]) -> Optional[str]:
        pdf_url = (rec.get("pdf_url") or rec.get("url") or "").strip()
        if not pdf_url:
            return None
        filename = Path(urlparse(pdf_url).path).name
        if not filename:
            return None
        local_path = Path(config.BS_RESEARCH_OUT_DIR) / filename
        if local_path.exists():
            return str(local_path)
        return None

    def _document_from_source_row(self, row: dict[str, Any]) -> Optional[dict[str, Any]]:
        sym, company = self.mapper.resolve(row.get("symbol") or "", row.get("company") or "", row.get("title") or "")
        title = (row.get("title") or company or row.get("doc_type") or "Document").strip()
        source_table = row.get("source_table") or "unknown"
        source_row_id = row.get("source_row_id")
        url = row.get("url")
        raw_path = row.get("raw_path")
        payload = row.get("payload_json")
        doc_hash = _sha(f"{source_table}|{source_row_id}|{title}|{url}|{raw_path}|{payload}")
        return {
            "source": row.get("source") or "",
            "source_table": source_table,
            "source_row_id": source_row_id,
            "symbol": sym or None,
            "company": company or row.get("company"),
            "title": title[:500],
            "doc_type": row.get("doc_type") or "document",
            "url": url,
            "raw_path": raw_path,
            "content_type": row.get("content_type"),
            "published_at": row.get("published_at"),
            "event_date": row.get("event_date"),
            "hash": doc_hash,
            "meta": {"source_payload": payload},
            "extraction_status": "pending",
        }

    def _extract_document_text(self, row: dict[str, Any]) -> str:
        meta = {}
        try:
            meta = json.loads(row.get("meta_json") or "{}")
        except Exception:
            meta = {}
        payload_text = meta.get("source_payload") or ""
        primary_path = self._materialize_preferred_path(row)

        if primary_path and Path(primary_path).exists():
            suffix = Path(primary_path).suffix.lower()
            if suffix == ".pdf":
                text = self._extract_pdf_text(Path(primary_path))
            elif suffix in (".html", ".htm", ".xml", ".xhtml"):
                text = self._extract_html_text(Path(primary_path).read_text(encoding="utf-8", errors="ignore"))
            elif suffix == ".json":
                text = Path(primary_path).read_text(encoding="utf-8", errors="ignore")
            else:
                text = Path(primary_path).read_text(encoding="utf-8", errors="ignore")
        else:
            text = ""

        if len(text.strip()) < 200 and payload_text:
            text = f"{row.get('title')}\n\n{payload_text}"

        if len(text.strip()) < 50:
            text = f"{row.get('title')}\n\nCompany: {row.get('company')}\nSymbol: {row.get('symbol')}\nURL: {row.get('url') or ''}"
        return text[:20000]

    def _materialize_preferred_path(self, row: dict[str, Any]) -> Optional[str]:
        raw_path = row.get("raw_path")
        url = (row.get("url") or "").strip()
        if raw_path and Path(str(raw_path)).exists():
            raw_suffix = Path(str(raw_path)).suffix.lower()
            if raw_suffix in (".pdf", ".html", ".htm", ".json", ".xml", ".xhtml", ".txt"):
                return str(raw_path)
        if not url or url == "-" or url.endswith("/-"):
            return str(raw_path) if raw_path and Path(str(raw_path)).exists() else None
        if url and (url.lower().endswith(".pdf") or "bsmedia.business-standard.com" in url.lower()):
            bs_local = self._resolve_bs_local_path({"pdf_url": url})
            if bs_local:
                return bs_local
            cached = self.docs_dir / f"{row['hash']}.pdf"
            if not cached.exists():
                self._download_to(url, cached)
            return str(cached)
        if url and any(token in url.lower() for token in (".html", ".htm", ".xml", "ixbrl", "xbrl")) and " - " not in url:
            cached = self.docs_dir / f"{row['hash']}.html"
            if not cached.exists():
                self._download_to(url, cached)
            return str(cached)
        if raw_path and Path(str(raw_path)).exists():
            return str(raw_path)
        return None

    def _download_to(self, url: str, path: Path) -> None:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        }
        resp = requests.get(url, headers=headers, timeout=25)
        resp.raise_for_status()
        path.write_bytes(resp.content)

    def _extract_pdf_text(self, path: Path) -> str:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages[:12]:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(parts)

    def _extract_html_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _chunk_text(self, text: str) -> list[dict[str, Any]]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
        chunks: list[dict[str, Any]] = []
        buf = ""
        idx = 0
        for para in paras:
            if len(buf) + len(para) + 2 <= self.chunk_chars:
                buf = f"{buf}\n\n{para}".strip()
                continue
            if buf:
                chunks.append(
                    {
                        "chunk_index": idx,
                        "text": buf,
                        "char_count": len(buf),
                        "hash": _sha(buf),
                    }
                )
                idx += 1
                if idx >= self.max_chunks:
                    return chunks
            buf = para
        if buf and idx < self.max_chunks:
            chunks.append(
                {
                    "chunk_index": idx,
                    "text": buf,
                    "char_count": len(buf),
                    "hash": _sha(buf),
                }
            )
        if not chunks:
            tiny = (text or "")[: self.chunk_chars]
            chunks.append({"chunk_index": 0, "text": tiny, "char_count": len(tiny), "hash": _sha(tiny)})
        return chunks

    def _generate_research_memo(self, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
        chunks = pdb.fetch_rows(
            "SELECT chunk_index, text FROM document_chunks WHERE document_id = ? ORDER BY chunk_index ASC",
            (int(row["document_id"]),),
        )
        text = "\n\n".join(chunk["text"] for chunk in chunks[: self.max_chunks])
        chunk_refs = [int(chunk["chunk_index"]) for chunk in chunks[: self.max_chunks]]
        meta = {
            "symbol": row.get("symbol"),
            "company": row.get("company"),
            "title": row.get("title"),
            "doc_type": row.get("doc_type"),
            "published_at": row.get("published_at"),
            "event_date": row.get("event_date"),
            "url": row.get("url"),
            "document_id": row.get("document_id"),
            "chunk_refs": chunk_refs,
        }
        if not self._llm_available():
            raise RuntimeError("ETL memo model is unavailable")

        system = (
            "You are a buy-side ETL document reader for Indian equities. "
            "Read one document and return strict JSON only. No markdown. No code fences. No prose outside JSON. "
            "Focus on what happened, why it matters, bullish implications, bearish implications, catalysts, risks, "
            "and whether this likely changes a stock thesis. Keep outputs compact and factual. "
            "Use only these exact fields and enums. "
            "importance must be one of: low, medium, high, critical. "
            "thesis_change must be one of: strengthens, weakens, neutral, possible, unclear."
        )
        prompt = (
            "Return valid minified JSON with exactly these keys: "
            "summary, thesis, importance, confidence, analyst_stance, estimate_changes, bullish_points, bearish_points, catalysts, risks, thesis_change, key_facts, numeric_facts, source_chunk_refs.\n"
            "Rules:\n"
            "- summary: one concise sentence\n"
            "- thesis: 1-3 sentences explaining the investment takeaway\n"
            "- confidence: number between 0 and 1\n"
            "- analyst_stance: object with keys rating, target_price_inr, previous_target_price_inr, action\n"
            "- estimate_changes: array of short strings\n"
            "- bullish_points/bearish_points/catalysts/risks/key_facts: arrays of short strings\n"
            "- numeric_facts: array of short strings with numbers that matter\n"
            "- source_chunk_refs: array of integers from provided chunk_refs that support the memo\n"
            "- If the document is an analyst report, key_facts should include rating/target/revision if present\n\n"
            f"Metadata:\n{json.dumps(meta, default=str)}\n\n"
            f"Document text:\n{text[:12000]}"
        )
        agent = BaseAgent(
            system_prompt=system,
            tools=[],
            model=row.get("model") or self.memo_model,
            max_tokens=self.memo_max_tokens,
            temperature=0.1,
        )
        raw = agent.run(prompt)
        usage = agent.get_token_usage()
        try:
            parsed = self._parse_json(raw)
        except Exception:
            raise RuntimeError("ETL memo model returned invalid JSON")
        if not isinstance(parsed, dict):
            raise RuntimeError("ETL memo generation did not return JSON")
        normalized = self._normalize_memo_payload(parsed)
        if meta.get("doc_type") == "analyst_report" and self._memo_needs_analyst_override(normalized):
            raise RuntimeError("ETL analyst report memo failed validation")
        normalized["source_chunk_refs"] = self._resolve_supporting_chunk_refs(
            memo=normalized,
            chunks=chunks,
            default_refs=chunk_refs,
        )
        return normalized, usage

    def _parse_json(self, text: str) -> Any:
        raw = (text or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            return json.loads(raw)
        except Exception:
            code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S | re.I)
            if code_block:
                return json.loads(code_block.group(1))
            nested_summary = re.search(r'"summary"\s*:\s*"```json\\n(\{.*?\})"', raw, flags=re.S)
            if nested_summary:
                inner = nested_summary.group(1).replace('\\"', '"')
                try:
                    return json.loads(inner)
                except Exception:
                    pass
            match = re.search(r"\{.*\}", raw, flags=re.S)
            if match:
                return json.loads(match.group(0))
        raise RuntimeError("Invalid JSON from memo model")

    def _heuristic_memo(self, meta: dict[str, Any], text: str) -> dict[str, Any]:
        if meta.get("doc_type") == "analyst_report":
            return self._heuristic_analyst_report_memo(meta, text)
        lower = (text or "").lower()
        bullish = []
        bearish = []
        catalysts = []
        risks = []
        for kw in ("dividend", "buyback", "acquisition", "order win", "record date", "results"):
            if kw in lower:
                bullish.append(f"Mentions {kw}")
        for kw in ("sell", "penalty", "downgrade", "pledge", "warning", "loss"):
            if kw in lower:
                bearish.append(f"Mentions {kw}")
        if meta.get("doc_type") == "financial_result":
            catalysts.append("Fresh financial result filing")
        if meta.get("doc_type") == "analyst_report":
            catalysts.append("New analyst view published")
        if meta.get("doc_type") in ("insider_trade", "bulk_deal"):
            risks.append("Needs confirmation whether flow is thesis-relevant or temporary")
        summary = (text or "").strip().replace("\n", " ")[:350]
        return self._normalize_memo_payload({
            "summary": summary,
            "thesis": summary,
            "importance": "high" if meta.get("doc_type") in ("financial_result", "analyst_report") else "medium",
            "confidence": 0.45,
            "analyst_stance": {"rating": "", "target_price_inr": None, "previous_target_price_inr": None, "action": ""},
            "estimate_changes": [],
            "bullish_points": bullish[:4],
            "bearish_points": bearish[:4],
            "catalysts": catalysts[:4],
            "risks": risks[:4],
            "thesis_change": "possible" if bullish or bearish else "unclear",
            "key_facts": [
                f"doc_type={meta.get('doc_type')}",
                f"title={meta.get('title')}",
            ],
            "numeric_facts": [],
            "source_chunk_refs": list(meta.get("chunk_refs") or [])[:3],
        })

    def _heuristic_analyst_report_memo(self, meta: dict[str, Any], text: str) -> dict[str, Any]:
        flat = re.sub(r"\s+", " ", text or "").strip()
        rating = _first_match(r"Rating[:\s]+([A-Z]{3,16})", flat)
        rating = {
            "AND": "ADD",  # OCR/scan noise frequently turns ADD into AND.
            "ACCUMULATE": "BUY",
            "NEUTRAL": "HOLD",
            "OVERWEIGHT": "BUY",
            "UNDERWEIGHT": "SELL",
            "OUTPERFORM": "BUY",
            "UNDERPERFORM": "SELL",
        }.get(rating, rating)
        target = _first_match(
            r"(?:12 month price target \(INR\)|Target price \(12-mth\)|Target Price: INR)\s*[:\-]?\s*([\d,]+(?:\.\d+)?)",
            flat,
        )
        target_revision = re.search(r"TP of INR[\s]*([\d,]+(?:\.\d+)?)\s*\(from INR[\s]*([\d,]+(?:\.\d+)?)\)", flat, flags=re.I)
        summary = self._extract_analyst_report_summary(text)

        key_facts = []
        if rating:
            key_facts.append(f"rating={rating}")
        if target:
            key_facts.append(f"target_inr={target.replace(',', '')}")
        if target_revision:
            key_facts.append(f"target_revision_to={target_revision.group(1).replace(',', '')}")
            key_facts.append(f"target_revision_from={target_revision.group(2).replace(',', '')}")
            estimate_changes = [f"target_price: {target_revision.group(2)} -> {target_revision.group(1)}"]
        else:
            estimate_changes = []

        revisions = re.findall(r"(Revenue|EBITDA|Adjusted profit|Diluted EPS \(INR\))\s+[\d,.\-]+\s+[\d,.\-]+\s+([\-+]?\d+(?:\.\d+)?%)\s+([\-+]?\d+(?:\.\d+)?%)", flat, flags=re.I)
        for name, fy26, fy27 in revisions[:4]:
            key_facts.append(f"{name.lower()}_rev={fy26}/{fy27}")
            estimate_changes.append(f"{name}: {fy26}/{fy27}")
        mg = _first_match(r"margin guidance by ~?([\d.]+bp) for FY26E", flat)
        if mg:
            estimate_changes.append(f"margin_guidance_fy26e=-{mg}")

        bullish = []
        bearish = []
        catalysts = []
        risks = []
        numeric_facts = []

        price = _first_match(r"Price \(INR\)\s*([\d,]+(?:\.\d+)?)", flat)
        if price:
            numeric_facts.append(f"price_inr={price.replace(',', '')}")
        for patt, label in (
            (r"Revenue/EBITDA grew ([\d.]+%/[\d.]+%) YoY", "rev_ebitda_growth"),
            (r"revenue/EBITDA grew ([\d.]+%/[\d.]+%) YoY", "rev_ebitda_growth"),
            (r"Q2FY26 revenue/EBITDA grew ([\d.]+%/[\d.]+%) YoY", "rev_ebitda_growth"),
            (r"India volumes inched up ([\d.]+%) YoY", "india_volume_growth"),
            (r"HPC grew ([\d.]+%) YoY", "hpc_growth"),
            (r"R&D spend.*?\(([\d.]+%) of sales\)", "rnd_pct_sales"),
            (r"margin guidance by ~?([\d.]+bp) for FY26E", "margin_guidance_change"),
        ):
            m = _first_match(patt, flat)
            if m:
                numeric_facts.append(f"{label}={m}")

        for phrase in (
            "revenue beat", "outpaced industry", "market share", "new-age digital-first",
            "growth lever", "launches", "volume CAGR", "exports revenue", "benefit"
        ):
            if phrase in flat.lower():
                bullish.append(phrase)
        for phrase in (
            "guidance cut", "margin contraction", "erosion", "competition", "underperformance",
            "costs", "trimming", "cutting", "declined"
        ):
            if phrase in flat.lower():
                bearish.append(phrase)

        if "launch" in flat.lower():
            catalysts.append("Product launches/refresh cycle")
        if "md & ceo" in flat.lower() or "ceo" in flat.lower():
            catalysts.append("Management transition")
        if "la nina" in flat.lower():
            catalysts.append("Weather-driven demand tailwind")
        if "competition" in flat.lower() or "erosion" in flat.lower():
            risks.append("Competitive pressure / pricing erosion")
        if "guidance cut" in flat.lower() or "margin contraction" in flat.lower():
            risks.append("Margin pressure")

        thesis_change = "neutral"
        if rating == "BUY":
            thesis_change = "strengthens"
        elif rating == "HOLD":
            thesis_change = "neutral"
        elif rating in ("REDUCE", "SELL"):
            thesis_change = "weakens"
        if any(x in flat.lower() for x in ("guidance cut", "trimming", "cutting fy", "downgrade")):
            thesis_change = "possible" if thesis_change == "strengthens" else "weakens"

        return self._normalize_memo_payload({
            "summary": summary,
            "thesis": summary,
            "importance": "high",
            "confidence": 0.72,
            "analyst_stance": {
                "rating": rating,
                "target_price_inr": float(target.replace(",", "")) if target else None,
                "previous_target_price_inr": float(target_revision.group(2).replace(",", "")) if target_revision else None,
                "action": "maintained" if rating in ("BUY", "HOLD", "SELL", "REDUCE") else "",
            },
            "estimate_changes": estimate_changes[:6],
            "bullish_points": bullish[:5],
            "bearish_points": bearish[:5],
            "catalysts": catalysts[:5],
            "risks": risks[:5],
            "thesis_change": thesis_change,
            "key_facts": key_facts[:8] + [f"doc_type={meta.get('doc_type')}", f"title={meta.get('title')}"],
            "numeric_facts": numeric_facts[:8],
            "source_chunk_refs": list(meta.get("chunk_refs") or [])[:3],
        })

    def _memo_needs_analyst_override(self, memo: dict[str, Any]) -> bool:
        summary = str((memo or {}).get("summary") or "")
        key_facts = [str(x) for x in ((memo or {}).get("key_facts") or [])]
        if summary.startswith("Nuvama Research is also available"):
            return True
        if "KEY DATA Rating" in summary:
            return True
        if any("llm_non_json_response" in x for x in key_facts):
            return True
        if not any(x.startswith("rating=") or x.startswith("Rating:") for x in key_facts):
            return True
        if not any("target_inr=" in x or "Target Price:" in x for x in key_facts):
            return True
        return False

    def _extract_analyst_report_summary(self, text: str) -> str:
        lines = [ln.strip() for ln in (text or "").splitlines()]
        start_idx = 0
        for i, ln in enumerate(lines):
            if "Pledge" in ln:
                start_idx = i + 1
                break
        candidate_lines: list[str] = []
        for ln in lines[start_idx:]:
            if not ln:
                continue
            if any(stop in ln for stop in ("FINANCIALS", "PRICE PERFORMANCE", "Financial Statements", "Income Statement")):
                break
            if ln.startswith("Nuvama Research is also available"):
                continue
            if len(ln) < 12:
                continue
            candidate_lines.append(ln)
            if sum(len(x) for x in candidate_lines) >= 280:
                break
        summary = " ".join(candidate_lines).strip()
        summary = re.sub(r"\s+", " ", summary)
        # Prefer the first substantive sentence after the research header / key data block.
        if summary:
            summary = re.sub(r"^Please refer to important disclosures at the end of this report\s*", "", summary, flags=re.I)
            summary = re.sub(r"^(ADD|BUY|HOLD|REDUCE|SELL)\s*\(.*?\)\s*", "", summary, flags=re.I)
            summary = re.sub(r"^CMP:\s*INR\s*[\d,]+.*?\s*", "", summary, flags=re.I)
        if len(summary) < 40:
            summary = flat = re.sub(r"\s+", " ", text or "").strip()[:320]
        return summary[:420]

    def _normalize_memo_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        p = dict(payload or {})
        importance = str(p.get("importance") or "medium").strip().lower()
        if importance not in ALLOWED_IMPORTANCE:
            importance = "medium"
        thesis_change = str(p.get("thesis_change") or "unclear").strip().lower()
        if thesis_change not in ALLOWED_THESIS_CHANGE:
            thesis_change = "unclear"
        summary = _clip(p.get("summary"), 600)
        if not summary:
            summary = "No usable summary extracted."
        thesis = _clip(p.get("thesis") or summary, 1200)
        analyst = p.get("analyst_stance") if isinstance(p.get("analyst_stance"), dict) else {}
        target_price = analyst.get("target_price_inr")
        prev_target = analyst.get("previous_target_price_inr")
        try:
            target_price = float(target_price) if target_price not in (None, "", "null") else None
        except Exception:
            target_price = None
        try:
            prev_target = float(prev_target) if prev_target not in (None, "", "null") else None
        except Exception:
            prev_target = None
        memo = {
            "summary": summary,
            "thesis": thesis,
            "importance": importance,
            "confidence": round(_as_float_01(p.get("confidence"), 0.4), 3),
            "analyst_stance": {
                "rating": _clip(analyst.get("rating"), 30).upper(),
                "target_price_inr": target_price,
                "previous_target_price_inr": prev_target,
                "action": _clip(analyst.get("action"), 40).lower(),
            },
            "estimate_changes": _as_list_of_str(p.get("estimate_changes"), max_items=8, max_item_len=180),
            "bullish_points": _as_list_of_str(p.get("bullish_points"), max_items=5, max_item_len=180),
            "bearish_points": _as_list_of_str(p.get("bearish_points"), max_items=5, max_item_len=180),
            "catalysts": _as_list_of_str(p.get("catalysts"), max_items=6, max_item_len=180),
            "risks": _as_list_of_str(p.get("risks"), max_items=6, max_item_len=180),
            "thesis_change": thesis_change,
            "key_facts": _as_list_of_str(p.get("key_facts"), max_items=8, max_item_len=160),
            "numeric_facts": _as_list_of_str(p.get("numeric_facts"), max_items=10, max_item_len=120),
            "source_chunk_refs": [int(x) for x in (p.get("source_chunk_refs") or []) if str(x).strip().isdigit()][:6],
        }
        return memo

    def _resolve_supporting_chunk_refs(
        self,
        *,
        memo: dict[str, Any],
        chunks: list[dict[str, Any]],
        default_refs: list[int],
    ) -> list[int]:
        if memo.get("source_chunk_refs"):
            refs = [int(x) for x in memo.get("source_chunk_refs", []) if str(x).strip().isdigit()]
            return refs[:6]

        scored: list[tuple[int, int]] = []
        needles = []
        needles.extend(memo.get("key_facts") or [])
        needles.extend(memo.get("numeric_facts") or [])
        needles.extend(memo.get("bullish_points") or [])
        needles.extend(memo.get("bearish_points") or [])
        needles.extend(memo.get("catalysts") or [])
        needles.extend(memo.get("risks") or [])
        needles.append(memo.get("summary") or "")
        needles.append(memo.get("thesis") or "")
        needles = [str(x).lower() for x in needles if str(x or "").strip()]

        for chunk in chunks:
            idx = int(chunk["chunk_index"])
            text = str(chunk.get("text") or "").lower()
            score = 0
            for needle in needles:
                if not needle:
                    continue
                tokens = [t for t in re.findall(r"[a-z0-9%.-]+", needle) if len(t) >= 4]
                if not tokens:
                    continue
                local = sum(1 for tok in tokens[:10] if tok in text)
                score += local
            scored.append((score, idx))

        scored.sort(key=lambda x: (-x[0], x[1]))
        refs = [idx for score, idx in scored if score > 0][:3]
        if not refs:
            refs = list(default_refs[:3])
        return refs

    def _normalize_symbol_memory(self, payload: dict[str, Any]) -> dict[str, Any]:
        p = dict(payload or {})
        recent_docs = []
        for item in p.get("recent_documents") or []:
            if not isinstance(item, dict):
                continue
            recent_docs.append(
                {
                    "title": _clip(item.get("title"), 300),
                    "doc_type": _clip(item.get("doc_type"), 60),
                    "published_at": item.get("published_at"),
                    "event_date": item.get("event_date"),
                    "url": _clip(item.get("url"), 500),
                    "memo": self._normalize_memo_payload(item.get("memo") or {}),
                }
            )
            if len(recent_docs) >= 10:
                break
        event_counts = {}
        for k, v in (p.get("event_counts") or {}).items():
            try:
                event_counts[str(k)] = max(0, int(v))
            except Exception:
                event_counts[str(k)] = 0
        latest_events = p.get("latest_events") or {}
        if not isinstance(latest_events, dict):
            latest_events = {}
        return {
            "symbol": _clip(p.get("symbol"), 24).upper(),
            "company": _clip(p.get("company"), 200),
            "updated_at": p.get("updated_at") or _now_iso(),
            "recent_documents": recent_docs,
            "event_counts": event_counts,
            "latest_events": latest_events,
        }

    def _normalize_etl_packet(self, payload: dict[str, Any]) -> dict[str, Any]:
        p = dict(payload or {})
        research_memory = self._normalize_symbol_memory(p.get("research_memory") or {})
        top_recent_memos = []
        for item in p.get("top_recent_memos") or []:
            if not isinstance(item, dict):
                continue
            top_recent_memos.append(
                {
                    "title": _clip(item.get("title"), 300),
                    "doc_type": _clip(item.get("doc_type"), 60),
                    "published_at": item.get("published_at"),
                    "event_date": item.get("event_date"),
                    "url": _clip(item.get("url"), 500),
                    "memo": self._normalize_memo_payload(item.get("memo") or {}),
                }
            )
            if len(top_recent_memos) >= 5:
                break
        return {
            "schema_version": 1,
            "symbol": _clip(p.get("symbol"), 24).upper(),
            "company": _clip(p.get("company"), 200),
            "as_of_date": _clip(p.get("as_of_date"), 10),
            "research_memory": research_memory,
            "top_recent_memos": top_recent_memos,
            "event_counts": research_memory.get("event_counts", {}),
            "latest_events": research_memory.get("latest_events", {}),
        }

    def _llm_available(self) -> bool:
        provider, _ = parse_model_provider(self.memo_model or "")
        if provider == "groq":
            return bool(config.GROQ_API_KEY)
        if provider == "gemini":
            return bool(config.GEMINI_API_KEY)
        if provider == "mistral":
            return bool(config.MISTRAL_API_KEY)
        if provider == "openai":
            return bool(config.OPENAI_COMPAT_BASE_URL)
        return bool(config.ANTHROPIC_API_KEY)

    def _priority_for_doc_type(self, doc_type: str) -> int:
        mapping = {
            "analyst_report": 100,
            "financial_result": 95,
            "filing": 80,
            "corporate_action": 75,
            "insider_trade": 70,
            "bulk_deal": 65,
            "results_calendar": 55,
        }
        return mapping.get((doc_type or "").strip().lower(), 50)

    def _event_counts(self, symbol: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for table in ("documents", "corporate_actions", "financial_results", "insider_trades", "bulk_deals", "results_calendar"):
            try:
                rows = pdb.fetch_rows(f"SELECT COUNT(*) AS c FROM {table} WHERE symbol = ?", (symbol,))
                out[table] = int(rows[0]["c"]) if rows else 0
            except Exception:
                out[table] = 0
        return out

    def _latest_structured_events(self, symbol: str) -> dict[str, list[dict[str, Any]]]:
        return {
            "corporate_actions": pdb.fetch_rows(
                """
                SELECT subject, ex_date, record_date, published_at
                FROM corporate_actions WHERE symbol = ?
                ORDER BY COALESCE(ex_date, record_date, published_at) DESC LIMIT 3
                """,
                (symbol,),
            ),
            "financial_results": pdb.fetch_rows(
                """
                SELECT relating_to, to_date, filing_date, xbrl_url
                FROM financial_results WHERE symbol = ?
                ORDER BY COALESCE(to_date, filing_date) DESC LIMIT 3
                """,
                (symbol,),
            ),
            "insider_trades": pdb.fetch_rows(
                """
                SELECT person_name, transaction_type, event_date, value
                FROM insider_trades WHERE symbol = ?
                ORDER BY COALESCE(event_date, reported_at) DESC LIMIT 3
                """,
                (symbol,),
            ),
            "bulk_deals": pdb.fetch_rows(
                """
                SELECT client_name, buy_sell, event_date, quantity, price
                FROM bulk_deals WHERE symbol = ?
                ORDER BY event_date DESC LIMIT 3
                """,
                (symbol,),
            ),
        }

    def _recent_document_symbols(self, *, limit_days: int) -> list[str]:
        cutoff = (datetime.now() - timedelta(days=limit_days)).strftime("%Y-%m-%d")
        rows = pdb.fetch_rows(
            """
            SELECT DISTINCT symbol
            FROM documents
            WHERE symbol IS NOT NULL
              AND symbol != ''
              AND COALESCE(substr(published_at, 1, 10), event_date, substr(created_at, 1, 10)) >= ?
            """,
            (cutoff,),
        )
        return [r["symbol"] for r in rows if r.get("symbol")]
