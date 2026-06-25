import json

from data.official_sources import (
    _normalize_amfi_research_html,
    _normalize_bse_announcements,
    _normalize_bse_corporate_actions,
    _normalize_bse_forthcoming_results,
    _normalize_bulk_deals,
    _normalize_bse_insider_html,
    _normalize_corporate_actions,
    _normalize_financial_results,
    _normalize_link_documents_html,
    _normalize_rbi_sectoral_credit_html,
)


def test_normalize_corporate_actions_basic():
    rows = _normalize_corporate_actions(
        [
            {
                "symbol": "SANOFI",
                "comp": "Sanofi India Limited",
                "subject": "Dividend - Rs 48 Per Share",
                "exDate": "22-Apr-2026",
                "recDate": "22-Apr-2026",
            }
        ],
        "raw.json",
    )
    assert len(rows) == 1
    assert rows[0]["symbol"] == "SANOFI"
    assert rows[0]["ex_date"] == "2026-04-22"


def test_normalize_financial_results_basic():
    rows = _normalize_financial_results(
        [
            {
                "symbol": "UTKARSHBNK",
                "companyName": "Utkarsh Small Finance Bank Limited",
                "period": "Quarterly",
                "relatingTo": "Third Quarter",
                "financialYear": "01-Apr-2024 To 31-Mar-2025",
                "fromDate": "01-Oct-2024",
                "toDate": "31-Dec-2024",
                "filingDate": "17-Feb-2025 12:17",
                "xbrl": "https://example.com/result.xml",
            }
        ],
        "raw.json",
    )
    assert len(rows) == 1
    assert rows[0]["symbol"] == "UTKARSHBNK"
    assert rows[0]["filing_date"].startswith("2025-02-17T12:17")


def test_normalize_bulk_deals_uses_nested_payload():
    rows = _normalize_bulk_deals(
        {
            "BULK_DEALS_DATA": [
                {
                    "date": "21-Apr-2026",
                    "symbol": "ROLEXRINGS",
                    "name": "Rolex Rings Limited",
                    "clientName": "ABC CAPITAL",
                    "buySell": "BUY",
                    "qty": "1808380",
                    "watp": "158.47",
                }
            ]
        },
        "raw.json",
    )
    assert len(rows) == 1
    assert rows[0]["quantity"] == 1808380.0
    assert json.loads(rows[0]["payload_json"])["symbol"] == "ROLEXRINGS"


def test_normalize_bse_insider_html_basic():
    html = """
    <table>
      <tr><td>540376</td><td>Avenue Supermarts Ltd</td><td>Elvin Elias Machado</td><td>Director</td>
      <td>333000 (0.05)</td><td>Equity Shares</td><td>20,000</td><td>5980000.00</td>
      <td>Acquisition</td><td>353000 (0.05)</td><td>16/04/2026 16/04/2026</td><td>ESOP</td>
      <td></td><td></td><td></td><td>21/04/2026</td></tr>
    </table>
    """
    rows = _normalize_bse_insider_html(html, "raw.html")
    assert len(rows) == 1
    assert rows[0]["symbol"] == "540376"
    assert rows[0]["event_date"] == "2026-04-16"
    assert rows[0]["reported_at"] == "2026-04-21"


def test_normalize_amfi_research_html_links():
    html = """
    <a href="research-information/amfi-data">AMFI Monthly / Quarterly Data</a>
    <a href="otherdata">Other Data</a>
    <a href="/aboutamfi">About</a>
    """
    rows = _normalize_amfi_research_html(html, "raw.html")
    assert len(rows) == 2
    assert rows[0]["source"] == "amfi"


def test_normalize_rbi_sectoral_credit_html_links():
    html = '<a href="BS_PressReleaseDisplay.aspx?prid=62467">Sectoral Deployment of Bank Credit – February 2026</a>'
    rows = _normalize_rbi_sectoral_credit_html(html, "raw.html")
    assert len(rows) == 1
    assert rows[0]["source"] == "rbi"
    assert rows[0]["event_date"] == "2026-02-01"


def test_normalize_bse_announcements_basic():
    payload = {
        "Table": [
            {
                "NEWSID": "abc",
                "SCRIP_CD": 543065,
                "NEWSSUB": "SM Auto Stamping Ltd - 543065 - Announcement Under Reg 30",
                "NEWS_DT": "2026-04-21T23:47:42.963",
                "DissemDT": "2026-04-21T23:47:42.963",
                "ATTACHMENTNAME": "file.pdf",
                "SLONGNAME": "SM Auto Stamping Ltd",
            }
        ]
    }
    rows = _normalize_bse_announcements(payload, "raw.json")
    assert len(rows) == 1
    assert rows[0]["symbol"] == "543065"
    assert rows[0]["url"].endswith("/2026/4/file.pdf")


def test_normalize_bse_forthcoming_results_basic():
    payload = [
        {
            "scrip_Code": "506597",
            "short_name": "AMAL",
            "Long_Name": "Amal Ltd",
            "meeting_date": "22 Apr 2026",
            "URL": "https://www.bseindia.com/stock-share-price/amal-ltd/amal/506597/",
        }
    ]
    rows = _normalize_bse_forthcoming_results(payload, "raw.json")
    assert len(rows) == 1
    assert rows[0]["event_date"] == "2026-04-22"
    assert rows[0]["purpose"] == "Forthcoming Results"


def test_normalize_bse_corporate_actions_basic():
    payload = [
        {
            "scrip_code": 500674,
            "short_name": "SANOFI",
            "Ex_date": "22 Apr 2026",
            "Purpose": "Final Dividend - Rs. - 48.0000",
            "RD_Date": "22 Apr 2026",
            "long_name": "Sanofi India Ltd",
        }
    ]
    rows = _normalize_bse_corporate_actions(payload, "raw.json")
    assert len(rows) == 1
    assert rows[0]["symbol"] == "500674"
    assert rows[0]["record_date"] == "2026-04-22"


def test_normalize_link_documents_html_filters_links():
    html = """
    <a href="/doc1.pdf">Shareholding Pattern Q1</a>
    <a href="/foo">Ignore</a>
    <a href="https://example.com/ixbrl/test.html">iXBRL filing</a>
    """
    rows = _normalize_link_documents_html(
        html,
        "raw.html",
        source="nse",
        company="NSE India",
        title_prefix="NSE Shareholding",
        include_patterns=("shareholding", "ixbrl"),
        base_url="https://www.nseindia.com/",
    )
    assert len(rows) == 2
    assert rows[0]["source"] == "nse"
