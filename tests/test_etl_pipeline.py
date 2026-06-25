import json
from pathlib import Path

import pandas as pd

import config
from persistence import db as pdb
from pipeline.etl_pipeline import ETLPipeline


def test_etl_chunking_and_heuristic_memo(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DB_FILE", str(tmp_path / "etl.sqlite3"))
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(
        __import__("pipeline.etl_pipeline", fromlist=["get_nifty500"]),
        "get_nifty500",
        lambda force_refresh=False: pd.DataFrame(
            [{"symbol": "RELIANCE", "company_name": "Reliance Industries Limited"}]
        ),
    )

    etl = ETLPipeline()
    chunks = etl._chunk_text("para1\n\n" + ("x" * 4000) + "\n\npara3")
    assert len(chunks) >= 2

    memo = etl._heuristic_memo(
        {"doc_type": "financial_result", "title": "Q4 results"},
        "Company announced dividend and results. No downgrade.",
    )
    assert memo["importance"] == "high"
    assert memo["catalysts"]


def test_etl_end_to_end_builds_packet(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DB_FILE", str(tmp_path / "etl.sqlite3"))
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(config, "BS_RESEARCH_OUT_DIR", str(tmp_path / "bs_research"))
    monkeypatch.setattr(config, "ETL_MAX_DOCS_PER_RUN", 20)
    monkeypatch.setattr(config, "ETL_MAX_JOBS_PER_RUN", 20)
    monkeypatch.setattr(config, "ETL_LOOKBACK_DAYS", 9999)
    monkeypatch.setattr(
        __import__("pipeline.etl_pipeline", fromlist=["get_nifty500"]),
        "get_nifty500",
        lambda force_refresh=False: pd.DataFrame(
            [{"symbol": "RELIANCE", "company_name": "Reliance Industries Limited"}]
        ),
    )

    raw = tmp_path / "filing.html"
    raw.write_text("<html><body>Reliance Industries announced dividend and results.</body></html>", encoding="utf-8")

    pdb.init_db()
    pdb.insert_normalized_rows(
        "filings",
        [
            {
                "source": "nse",
                "symbol": "RELIANCE",
                "company": "Reliance Industries Limited",
                "title": "Dividend update",
                "event_date": "2026-04-21",
                "published_at": "2026-04-21T12:00:00",
                "url": "https://example.com/reliance.html",
                "raw_path": str(raw),
                "hash": "hash-rel-filing",
                "payload_json": json.dumps({"title": "Dividend update"}),
            }
        ],
    )

    etl = ETLPipeline()
    results = etl.run(max_docs=10, max_jobs=10)
    assert results["ingested_documents"] >= 1
    assert results["extracted_documents"] >= 1
    assert results["packets_built"] >= 1

    packets = pdb.fetch_rows("SELECT symbol, packet_json FROM etl_packets WHERE symbol = ?", ("RELIANCE",))
    assert len(packets) == 1
    packet = json.loads(packets[0]["packet_json"])
    assert packet["symbol"] == "RELIANCE"
    assert packet["top_recent_memos"]


def test_etl_normalizes_memo_and_packet_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DB_FILE", str(tmp_path / "etl.sqlite3"))
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(
        __import__("pipeline.etl_pipeline", fromlist=["get_nifty500"]),
        "get_nifty500",
        lambda force_refresh=False: pd.DataFrame(
            [{"symbol": "RELIANCE", "company_name": "Reliance Industries Limited"}]
        ),
    )
    etl = ETLPipeline()

    memo = etl._normalize_memo_payload(
        {
            "summary": "x" * 1000,
            "thesis": "t" * 1400,
            "importance": "VERY_HIGH",
            "confidence": 9,
            "analyst_stance": {"rating": "buy", "target_price_inr": "1715", "previous_target_price_inr": "1725", "action": "maintained"},
            "estimate_changes": ["e1", "e2"],
            "bullish_points": ["a", "a", "", "b"],
            "bearish_points": "single risk",
            "catalysts": ["c1", "c2", "c3", "c4", "c5", "c6", "c7"],
            "risks": None,
            "thesis_change": "MAYBE",
            "key_facts": ["k1", "k2"],
            "numeric_facts": ["n1", "n2"],
            "source_chunk_refs": [0, "1", "x"],
        }
    )
    assert set(memo.keys()) == {
        "summary", "thesis", "importance", "confidence", "analyst_stance", "estimate_changes",
        "bullish_points", "bearish_points", "catalysts", "risks", "thesis_change", "key_facts",
        "numeric_facts", "source_chunk_refs"
    }
    assert memo["importance"] == "medium"
    assert memo["thesis_change"] == "unclear"
    assert memo["confidence"] == 1.0
    assert len(memo["summary"]) == 600
    assert len(memo["thesis"]) == 1200
    assert memo["bearish_points"] == ["single risk"]
    assert len(memo["catalysts"]) == 6
    assert memo["analyst_stance"]["rating"] == "BUY"
    assert memo["analyst_stance"]["target_price_inr"] == 1715.0
    assert memo["source_chunk_refs"] == [0, 1]

    packet = etl._normalize_etl_packet(
        {
            "symbol": "reliance",
            "company": "Reliance Industries Limited",
            "as_of_date": "2026-04-22",
            "research_memory": {
                "symbol": "reliance",
                "company": "Reliance Industries Limited",
                "recent_documents": [
                    {"title": "Doc", "doc_type": "filing", "url": "https://example.com", "memo": {"summary": "s"}}
                ],
                "event_counts": {"documents": "3"},
                "latest_events": {"x": [{"a": 1}]},
            },
            "top_recent_memos": [
                {"title": "Doc", "doc_type": "filing", "url": "https://example.com", "memo": {"summary": "s"}}
            ],
        }
    )
    assert packet["schema_version"] == 1
    assert packet["symbol"] == "RELIANCE"
    assert packet["research_memory"]["symbol"] == "RELIANCE"
    assert packet["event_counts"]["documents"] == 3
    assert packet["top_recent_memos"][0]["memo"]["importance"] == "medium"
    assert packet["top_recent_memos"][0]["url"] == "https://example.com"


def test_etl_heuristic_analyst_report_preserves_key_details(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DB_FILE", str(tmp_path / "etl.sqlite3"))
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(
        __import__("pipeline.etl_pipeline", fromlist=["get_nifty500"]),
        "get_nifty500",
        lambda force_refresh=False: pd.DataFrame(
            [{"symbol": "CIPLA", "company_name": "Cipla Ltd."}]
        ),
    )
    etl = ETLPipeline()
    text = """
    KEY DATA
    Rating HOLD
    12 month price target (INR) 1,715
    Succession finalised; EBITDA guidance cut
    Cipla’s Q2FY26 revenue beat consensus estimates by 2% while EBITDA/PAT were in line.
    Mr Achin Gupta would take charge as Cipla’s MD & CEO from Apr-26.
    It has lowered margin guidance by ~50bp for FY26E.
    We reckon entry of competition in Lanreotide in CY26.
    Retain HOLD with a TP of INR1,715 (from INR1,725).
    """
    memo = etl._heuristic_analyst_report_memo({"doc_type": "analyst_report", "title": "Cipla"}, text)
    facts = " | ".join(memo["key_facts"])
    assert "rating=HOLD" in facts
    assert "target_inr=1715" in facts
    assert memo["summary"]
    assert any("Management transition" == x for x in memo["catalysts"])
    assert any("Margin pressure" == x for x in memo["risks"])
    assert memo["thesis_change"] in {"neutral", "weakens", "possible"}
    assert "target_revision_from=1725" in " | ".join(memo["key_facts"])
    assert "Succession finalised" in memo["summary"] or "revenue beat" in memo["summary"]
    assert memo["analyst_stance"]["rating"] == "HOLD"
    assert memo["analyst_stance"]["target_price_inr"] == 1715.0
    assert memo["estimate_changes"]


def test_etl_detects_low_quality_analyst_memo_for_override(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DB_FILE", str(tmp_path / "etl.sqlite3"))
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(
        __import__("pipeline.etl_pipeline", fromlist=["get_nifty500"]),
        "get_nifty500",
        lambda force_refresh=False: pd.DataFrame(
            [{"symbol": "HYUNDAI", "company_name": "Hyundai Motor India Ltd."}]
        ),
    )
    etl = ETLPipeline()
    assert etl._memo_needs_analyst_override(
        {
            "summary": "Nuvama Research is also available on www.nuvamaresearch.com KEY DATA Rating BUY",
            "key_facts": ["doc_type=analyst_report", "title=Hyundai"],
        }
    )


def test_etl_heuristic_summary_extracts_real_thesis_line(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DB_FILE", str(tmp_path / "etl.sqlite3"))
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(
        __import__("pipeline.etl_pipeline", fromlist=["get_nifty500"]),
        "get_nifty500",
        lambda force_refresh=False: pd.DataFrame(
            [{"symbol": "HYUNDAI", "company_name": "Hyundai Motor India Ltd."}]
        ),
    )
    etl = ETLPipeline()
    text = """
    KEY DATA
    Rating BUY
    12 month price target (INR) 2,900
    Pledge 0.00% 0.00% 0.00%

    Steady ride in Q2; launches to drive growth
    Revenue/EBITDA grew 1%/10% YoY to INR174.6bn/24.3bn, broadly in line with our estimates.
    Factoring in higher costs relating to the new Talegaon plant, we are cutting FY26-28E EPS by up to 10%.
    """
    summary = etl._extract_analyst_report_summary(text)
    assert summary.startswith("Steady ride in Q2")


def test_etl_resolves_supporting_chunk_refs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DB_FILE", str(tmp_path / "etl.sqlite3"))
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(
        __import__("pipeline.etl_pipeline", fromlist=["get_nifty500"]),
        "get_nifty500",
        lambda force_refresh=False: pd.DataFrame(
            [{"symbol": "DABUR", "company_name": "Dabur India Ltd."}]
        ),
    )
    etl = ETLPipeline()
    memo = {
        "summary": "Revenue and EBITDA growth with target price revision",
        "thesis": "Target revised to 605 from 625 after weak beverages and GST hit.",
        "bullish_points": ["market share"],
        "bearish_points": ["weak beverages"],
        "catalysts": ["weather-driven demand tailwind"],
        "risks": ["competitive pressure"],
        "key_facts": ["target_inr=605", "target_revision_from=625"],
        "numeric_facts": ["rev_ebitda_growth=5.4%/6.5%"],
        "source_chunk_refs": [],
    }
    chunks = [
        {"chunk_index": 0, "text": "Intro and header only"},
        {"chunk_index": 1, "text": "Revenue/EBITDA grew 5.4%/6.5% YoY. TP of INR605 (from INR625). Weak beverages due to competition."},
        {"chunk_index": 2, "text": "Some unrelated appendix"},
    ]
    refs = etl._resolve_supporting_chunk_refs(memo=memo, chunks=chunks, default_refs=[0, 1, 2])
    assert refs[0] == 1
