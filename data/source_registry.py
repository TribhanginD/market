"""
Registry of free official and near-official market data sources for India.

This keeps integration targets explicit and prevents endpoint knowledge from
being scattered across the codebase.
"""

from __future__ import annotations

from typing import Any

import config


SOURCE_REGISTRY: dict[str, dict[str, Any]] = {
    "nse_corporate_actions": {
        "url": config.NSE_CORPORATE_ACTIONS_URL,
        "source": "NSE",
        "category": "corporate_actions",
        "status": "verified_api",
        "normalized_table": "corporate_actions",
    },
    "nse_financial_results": {
        "url": config.NSE_FINANCIAL_RESULTS_URL,
        "source": "NSE",
        "category": "financial_results",
        "status": "verified_api",
        "normalized_table": "financial_results",
    },
    "nse_bulk_deals": {
        "url": config.NSE_BULK_DEALS_URL,
        "source": "NSE",
        "category": "bulk_deals",
        "status": "verified_api",
        "normalized_table": "bulk_deals",
    },
    "nse_block_deals": {
        "url": config.NSE_BLOCK_DEALS_URL,
        "source": "NSE",
        "category": "block_deals",
        "status": "verified_api",
        "normalized_table": "block_deals",
    },
    "nse_index_names": {
        "url": config.NSE_INDEX_NAMES_URL,
        "source": "NSE",
        "category": "reference_data",
        "status": "verified_api",
        "normalized_table": "reference_data",
    },
    "nse_equity_master": {
        "url": config.NSE_EQUITY_MASTER_URL,
        "source": "NSE",
        "category": "reference_data",
        "status": "verified_api",
        "normalized_table": "reference_data",
    },
    "nse_underlying_information": {
        "url": config.NSE_UNDERLYING_INFO_URL,
        "source": "NSE",
        "category": "reference_data",
        "status": "verified_api",
        "normalized_table": "reference_data",
    },
    "nse_event_calendar": {
        "url": config.NSE_EVENT_CALENDAR_URL,
        "source": "NSE",
        "category": "results_calendar",
        "status": "candidate_api",
        "normalized_table": "results_calendar",
    },
    "nse_shareholding_sdd": {
        "url": "https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern-sdd",
        "source": "NSE",
        "category": "filings",
        "status": "verified_page",
        "normalized_table": "filings",
    },
    "bse_corporates_hub": {
        "url": "https://www.bseindia.com/corporates.html",
        "source": "BSE",
        "category": "filings",
        "status": "candidate_page",
        "normalized_table": "filings",
    },
    "bse_insider_trading": {
        "url": "https://www.bseindia.com/corporates/Insider_Trading_new.aspx",
        "source": "BSE",
        "category": "insider_trades",
        "status": "verified_page",
        "normalized_table": "insider_trades",
    },
    "bse_announcements": {
        "url": "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w",
        "source": "BSE",
        "category": "filings",
        "status": "verified_api",
        "normalized_table": "filings",
    },
    "bse_forthcoming_results": {
        "url": "https://api.bseindia.com/BseIndiaAPI/api/Corpforthresults/w",
        "source": "BSE",
        "category": "results_calendar",
        "status": "verified_api",
        "normalized_table": "results_calendar",
    },
    "bse_corporate_actions": {
        "url": "https://api.bseindia.com/BseIndiaAPI/api/DefaultData/w",
        "source": "BSE",
        "category": "corporate_actions",
        "status": "verified_api",
        "normalized_table": "corporate_actions",
    },
    "bse_shareholding_page": {
        "url": "https://www.bseindia.com/static/about/Clause_35_Shareholding_Pattern.aspx",
        "source": "BSE",
        "category": "filings",
        "status": "verified_page",
        "normalized_table": "filings",
    },
    "bse_compliance_calendar": {
        "url": "https://www.bseindia.com/corporates/compliancecalendar.aspx",
        "source": "BSE",
        "category": "filings",
        "status": "verified_page",
        "normalized_table": "filings",
    },
    "amfi_research_information": {
        "url": "https://www.amfiindia.com/research-information",
        "source": "AMFI",
        "category": "filings",
        "status": "verified_page",
        "normalized_table": "filings",
    },
    "amfi_amfi_data": {
        "url": "https://www.amfiindia.com/research-information/amfi-data",
        "source": "AMFI",
        "category": "filings",
        "status": "verified_page",
        "normalized_table": "filings",
    },
    "amfi_otherdata": {
        "url": "https://www.amfiindia.com/otherdata",
        "source": "AMFI",
        "category": "filings",
        "status": "verified_page",
        "normalized_table": "filings",
    },
    "amfi_sif_research_information": {
        "url": "https://www.amfiindia.com/sif/research-information",
        "source": "AMFI",
        "category": "filings",
        "status": "verified_page",
        "normalized_table": "filings",
    },
    "amfi_fund_performance": {
        "url": "https://www.amfiindia.com/otherdata/fund-performance",
        "source": "AMFI",
        "category": "macro_drivers",
        "status": "candidate_page",
        "normalized_table": "macro_drivers",
    },
    "rbi_dbie": {
        "url": "https://data.rbi.org.in/",
        "source": "RBI",
        "category": "filings",
        "status": "verified_page",
        "normalized_table": "filings",
    },
    "rbi_sectoral_credit": {
        "url": "https://www.rbi.org.in/Scripts/Data_Sectoral_Deployment.aspx",
        "source": "RBI",
        "category": "filings",
        "status": "verified_page",
        "normalized_table": "filings",
    },
    "siam_statistical_services": {
        "url": "https://www.siam.in/statistical-services/statistical-profile",
        "source": "SIAM",
        "category": "sector_drivers",
        "status": "candidate_page",
        "normalized_table": "macro_drivers",
    },
}


def get_source_registry() -> dict[str, dict[str, Any]]:
    return SOURCE_REGISTRY
