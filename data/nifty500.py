"""
Nifty 500 universe manager.
Fetches and caches the full list of Nifty 500 constituents from NSE.
"""

import pandas as pd
import requests
import logging
from datetime import datetime
from typing import Optional

# Add parent to path for imports
import config

logger = logging.getLogger(__name__)

# Sector name normalization map (NSE uses verbose names)
SECTOR_MAP = {
    "Financial Services": "Financials",
    "Information Technology": "Technology",
    "Consumer Goods": "Consumer Staples",
    "Consumer Services": "Consumer Discretionary",
    "Automobile and Auto Components": "Automobile",
    "Oil Gas & Consumable Fuels": "Energy",
    "Fast Moving Consumer Goods": "FMCG",
    "Healthcare": "Healthcare",
    "Capital Goods": "Capital Goods",
    "Metals & Mining": "Metals",
    "Chemicals": "Chemicals",
    "Realty": "Realty",
    "Telecommunication": "Telecom",
    "Construction Materials": "Construction",
    "Power": "Power",
    "Services": "Services",
    "Textiles": "Textiles",
    "Media Entertainment & Publication": "Media",
    "Construction": "Construction",
    "Diversified": "Diversified",
}


def _yf_symbol(nse_symbol: str) -> str:
    """Convert NSE symbol to yfinance format (append .NS)."""
    # Handle special cases
    symbol = nse_symbol.strip().upper()
    # Remove any existing suffix
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    return f"{symbol}.NS"


def get_nifty500(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch Nifty 500 constituents from NSE.
    
    Returns DataFrame with columns:
    - symbol: NSE symbol (e.g., RELIANCE)
    - yf_symbol: yfinance symbol (e.g., RELIANCE.NS)
    - company_name: Full company name
    - sector: Normalized sector name
    - industry: Industry group
    
    Data is cached for CACHE_TTL_UNIVERSE seconds.
    """
    cache_file = config.CACHE_DIR / "nifty500.parquet"
    
    # Check cache
    if not force_refresh and cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age_seconds = (datetime.now() - mtime).total_seconds()
        if age_seconds < config.CACHE_TTL_UNIVERSE:
            logger.info(f"Loading Nifty 500 from cache (age: {age_seconds/3600:.1f}h)")
            cached = pd.read_parquet(cache_file)
            if _is_valid_universe(cached):
                return cached
            logger.warning("Cached Nifty 500 universe failed validation. Refreshing live source...")
    
    logger.info("Fetching Nifty 500 constituents from NSE...")
    
    df_raw = _fetch_primary_source()
    if df_raw is None or df_raw.empty:
        df_raw = _fetch_mirrors()

    if df_raw is None or df_raw.empty:
        raise RuntimeError(
            "Could not fetch Nifty 500 constituents from live sources.\n"
            "- The official endpoint is currently returning HTTP 403 in this environment.\n"
            "- Mirror sources also failed.\n"
            "Refusing to proceed (no hardcoded fallback universe)."
        )
    
    # Normalize columns
    df = _normalize_universe(df_raw)
    if not _is_valid_universe(df):
        raise RuntimeError(
            "Universe validation failed (likely partial/blocked download). "
            "Refusing to proceed without a full, diversified universe."
        )
    
    # Cache it
    df.to_parquet(cache_file)
    logger.info(f"Cached {len(df)} Nifty 500 stocks to {cache_file}")
    
    return df


def _fetch_primary_source() -> Optional[pd.DataFrame]:
    """Fetch the live constituent CSV from the primary NSE endpoint."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/csv,application/csv,text/plain,*/*",
            "Referer": "https://www.niftyindices.com/",
        }
        response = requests.get(config.NIFTY500_CSV_URL, headers=headers, timeout=30)
        response.raise_for_status()

        from io import StringIO
        return pd.read_csv(StringIO(response.text))
    except Exception as e:
        logger.warning(f"Failed to fetch from NSE: {e}")
        return None


def _fetch_mirrors() -> Optional[pd.DataFrame]:
    """Try live mirrors when niftyindices.com blocks automation."""
    urls = getattr(config, "NIFTY500_MIRROR_URLS", []) or []
    if not urls:
        return None
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "text/csv,text/plain,*/*",
    }
    for url in urls:
        try:
            logger.info("Trying mirror universe source: %s", url)
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            from io import StringIO
            text = response.text or ""
            if "<html" in text.lower() or "access denied" in text.lower():
                logger.warning("Mirror returned HTML/AccessDenied (%s).", url)
                continue

            df_raw = _read_csv_robust(text)
            if df_raw is not None and not df_raw.empty:
                return df_raw
        except Exception as e:
            logger.warning("Mirror fetch failed (%s): %s", url, e)
    return None


def _read_csv_robust(text: str) -> Optional[pd.DataFrame]:
    """
    Some mirrors occasionally serve headerless or partially parsed CSVs.
    Try a few strategies and return the best-looking DataFrame.
    """
    from io import StringIO

    candidates: list[pd.DataFrame] = []
    for kwargs in (
        {"engine": "python"},
        {"engine": "python", "encoding_errors": "ignore"},
    ):
        try:
            candidates.append(pd.read_csv(StringIO(text), **kwargs))
        except Exception:
            continue

    # Headerless fallback.
    try:
        candidates.append(pd.read_csv(StringIO(text), header=None, engine="python"))
    except Exception:
        pass

    best = None
    best_score = -1
    for df in candidates:
        if df is None or df.empty:
            continue
        # Prefer frames with 3+ columns and 400+ rows.
        score = 0
        if df.shape[1] >= 3:
            score += 2
        if df.shape[0] >= 400:
            score += 3
        if df.shape[0] >= 500:
            score += 2
        if score > best_score:
            best = df
            best_score = score

    return best


def _normalize_universe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw NSE CSV into clean DataFrame."""
    # Column names vary by NSE CSV version — handle both
    col_map = {}
    cols_lower = {c.lower().strip(): c for c in df_raw.columns}
    
    for key, candidates in {
        "symbol": ["symbol", "ticker", "nsesymbol"],
        "company_name": ["company name", "security name", "companyname", "name"],
        "sector": ["sector", "industry sector"],
        "industry": ["industry", "sub industry", "macro-economic sector name"],
    }.items():
        for c in candidates:
            if c in cols_lower:
                col_map[cols_lower[c]] = key
                break
    
    df = df_raw.rename(columns=col_map)
    
    # Ensure required columns exist
    for col in ["symbol", "company_name"]:
        if col not in df.columns:
            logger.error(f"Could not find column '{col}' in NSE CSV. Columns: {list(df_raw.columns)}")
            raise ValueError(f"Missing required column: {col}")
    
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    if "industry" not in df.columns:
        df["industry"] = "Unknown"
    
    # Clean up
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["company_name"] = df["company_name"].str.strip()
    df["sector"] = df["sector"].str.strip().map(lambda x: SECTOR_MAP.get(x, x))
    df["industry"] = df["industry"].str.strip()
    
    # Add yfinance symbol
    df["yf_symbol"] = df["symbol"].apply(_yf_symbol)
    
    # Drop duplicates and NaN symbols
    df = df.dropna(subset=["symbol"])
    df = df.drop_duplicates(subset=["symbol"])
    
    # Final column order
    df = df[["symbol", "yf_symbol", "company_name", "sector", "industry"]].reset_index(drop=True)
    
    return df


def _is_valid_universe(df: Optional[pd.DataFrame]) -> bool:
    """Reject malformed caches and partial fetches before they infect later stages."""
    if df is None or df.empty:
        return False

    required = {"symbol", "yf_symbol", "company_name", "sector", "industry"}
    if not required.issubset(df.columns):
        return False

    row_count = len(df)
    unique_symbols = df["symbol"].nunique(dropna=True)

    if getattr(config, "REQUIRE_FULL_UNIVERSE", False):
        min_size = getattr(config, "MIN_UNIVERSE_SIZE", 500)
    else:
        min_size = 100
    if row_count < min_size:
        return False
    if unique_symbols < min_size:
        return False
    # Mirrors often don't have sector; we fill sector later from yfinance fundamentals.
    # Only enforce sector diversity if it looks populated.
    try:
        sector_values = set(str(s).strip() for s in df["sector"].dropna().unique().tolist())
        if sector_values and not (len(sector_values) == 1 and "Unknown" in sector_values):
            if len(sector_values) < 8:
                return False
    except Exception:
        pass
    return True


def _fetch_fallback() -> Optional[pd.DataFrame]:
    """
    Fallback: use a hardcoded partial list of top Nifty 500 stocks
    if the live fetch fails. This ensures the system can still run.
    """
    logger.warning("Using fallback Nifty 500 list (top 100 by market cap, hardcoded).")
    
    # Top ~100 Nifty 500 stocks as fallback — covers most of market cap
    FALLBACK_STOCKS = [
        ("RELIANCE", "Reliance Industries Ltd", "Energy", "Oil & Gas"),
        ("TCS", "Tata Consultancy Services Ltd", "Technology", "IT Services"),
        ("HDFCBANK", "HDFC Bank Ltd", "Financials", "Private Banks"),
        ("BHARTIARTL", "Bharti Airtel Ltd", "Telecom", "Telecom Services"),
        ("ICICIBANK", "ICICI Bank Ltd", "Financials", "Private Banks"),
        ("INFOSYS", "Infosys Ltd", "Technology", "IT Services"),
        ("SBIN", "State Bank of India", "Financials", "Public Banks"),
        ("HINDUNILVR", "Hindustan Unilever Ltd", "FMCG", "Personal Products"),
        ("ITC", "ITC Ltd", "FMCG", "Cigarettes"),
        ("KOTAKBANK", "Kotak Mahindra Bank Ltd", "Financials", "Private Banks"),
        ("LT", "Larsen & Toubro Ltd", "Capital Goods", "Construction"),
        ("BAJFINANCE", "Bajaj Finance Ltd", "Financials", "NBFCs"),
        ("HCLTECH", "HCL Technologies Ltd", "Technology", "IT Services"),
        ("AXISBANK", "Axis Bank Ltd", "Financials", "Private Banks"),
        ("MARUTI", "Maruti Suzuki India Ltd", "Automobile", "Cars"),
        ("ASIANPAINT", "Asian Paints Ltd", "Chemicals", "Paints"),
        ("TITAN", "Titan Company Ltd", "Consumer Discretionary", "Jewellery"),
        ("SUNPHARMA", "Sun Pharmaceutical Industries Ltd", "Healthcare", "Pharma"),
        ("ULTRACEMCO", "UltraTech Cement Ltd", "Construction", "Cement"),
        ("WIPRO", "Wipro Ltd", "Technology", "IT Services"),
        ("POWERGRID", "Power Grid Corporation of India Ltd", "Power", "Power"),
        ("NTPC", "NTPC Ltd", "Power", "Power"),
        ("TATAMOTORS", "Tata Motors Ltd", "Automobile", "Cars"),
        ("BAJAJFINSV", "Bajaj Finserv Ltd", "Financials", "NBFCs"),
        ("M&M", "Mahindra & Mahindra Ltd", "Automobile", "Cars"),
        ("TECHM", "Tech Mahindra Ltd", "Technology", "IT Services"),
        ("ONGC", "Oil and Natural Gas Corporation Ltd", "Energy", "Oil & Gas"),
        ("GRASIM", "Grasim Industries Ltd", "Diversified", "Cement"),
        ("ADANIENT", "Adani Enterprises Ltd", "Diversified", "Conglomerate"),
        ("ADANIPORTS", "Adani Ports & SEZ Ltd", "Services", "Ports"),
        ("COALINDIA", "Coal India Ltd", "Metals", "Mining"),
        ("JSWSTEEL", "JSW Steel Ltd", "Metals", "Steel"),
        ("TATASTEEL", "Tata Steel Ltd", "Metals", "Steel"),
        ("HINDALCO", "Hindalco Industries Ltd", "Metals", "Aluminium"),
        ("DRREDDY", "Dr. Reddy's Laboratories Ltd", "Healthcare", "Pharma"),
        ("CIPLA", "Cipla Ltd", "Healthcare", "Pharma"),
        ("DIVISLAB", "Divi's Laboratories Ltd", "Healthcare", "Pharma"),
        ("APOLLOHOSP", "Apollo Hospitals Enterprise Ltd", "Healthcare", "Hospitals"),
        ("SBILIFE", "SBI Life Insurance Company Ltd", "Financials", "Insurance"),
        ("HDFCLIFE", "HDFC Life Insurance Company Ltd", "Financials", "Insurance"),
        ("ICICIPRULIFE", "ICICI Prudential Life Insurance Co Ltd", "Financials", "Insurance"),
        ("BAJAJ-AUTO", "Bajaj Auto Ltd", "Automobile", "Two-Wheelers"),
        ("EICHERMOT", "Eicher Motors Ltd", "Automobile", "Two-Wheelers"),
        ("HEROMOTOCO", "Hero MotoCorp Ltd", "Automobile", "Two-Wheelers"),
        ("NESTLEIND", "Nestle India Ltd", "FMCG", "Food"),
        ("BRITANNIA", "Britannia Industries Ltd", "FMCG", "Food"),
        ("GODREJCP", "Godrej Consumer Products Ltd", "FMCG", "Personal Products"),
        ("MARICO", "Marico Ltd", "FMCG", "Personal Products"),
        ("PIDILITIND", "Pidilite Industries Ltd", "Chemicals", "Adhesives"),
        ("BERGEPAINT", "Berger Paints India Ltd", "Chemicals", "Paints"),
        ("HAVELLS", "Havells India Ltd", "Capital Goods", "Electricals"),
        ("DMART", "Avenue Supermarts Ltd (DMart)", "Consumer Discretionary", "Retail"),
        ("TRENT", "Trent Ltd", "Consumer Discretionary", "Retail"),
        ("NAUKRI", "Info Edge (India) Ltd", "Technology", "Internet"),
        ("PAYTM", "One97 Communications Ltd", "Technology", "Fintech"),
        ("ZOMATO", "Zomato Ltd", "Consumer Discretionary", "Food Delivery"),
        ("INDHOTEL", "Indian Hotels Company Ltd", "Consumer Discretionary", "Hotels"),
        ("IRCTC", "Indian Railway Catering and Tourism Corp", "Services", "Tourism"),
        ("BANDHANBNK", "Bandhan Bank Ltd", "Financials", "Private Banks"),
        ("FEDERALBNK", "The Federal Bank Ltd", "Financials", "Private Banks"),
        ("IDFCFIRSTB", "IDFC First Bank Ltd", "Financials", "Private Banks"),
        ("PNB", "Punjab National Bank", "Financials", "Public Banks"),
        ("BANKBARODA", "Bank of Baroda", "Financials", "Public Banks"),
        ("CANBK", "Canara Bank", "Financials", "Public Banks"),
        ("RECLTD", "REC Limited", "Financials", "Power Finance"),
        ("PFC", "Power Finance Corporation Ltd", "Financials", "Power Finance"),
        ("IRFC", "Indian Railway Finance Corporation Ltd", "Financials", "Finance"),
        ("CHOLAFIN", "Cholamandalam Investment & Finance", "Financials", "NBFCs"),
        ("MUTHOOTFIN", "Muthoot Finance Ltd", "Financials", "NBFCs"),
        ("SHRIRAMFIN", "Shriram Finance Ltd", "Financials", "NBFCs"),
        ("LTIM", "LTIMindtree Ltd", "Technology", "IT Services"),
        ("MPHASIS", "Mphasis Ltd", "Technology", "IT Services"),
        ("PERSISTENT", "Persistent Systems Ltd", "Technology", "IT Services"),
        ("COFORGE", "Coforge Ltd", "Technology", "IT Services"),
        ("LTTS", "L&T Technology Services Ltd", "Technology", "IT Services"),
        ("TATACONSUM", "Tata Consumer Products Ltd", "FMCG", "Food"),
        ("COLPAL", "Colgate-Palmolive (India) Ltd", "FMCG", "Personal Products"),
        ("DABUR", "Dabur India Ltd", "FMCG", "Personal Products"),
        ("EMAMILTD", "Emami Ltd", "FMCG", "Personal Products"),
        ("TORNTPHARM", "Torrent Pharmaceuticals Ltd", "Healthcare", "Pharma"),
        ("BIOCON", "Biocon Ltd", "Healthcare", "Biotech"),
        ("AUROPHARMA", "Aurobindo Pharma Ltd", "Healthcare", "Pharma"),
        ("LUPIN", "Lupin Ltd", "Healthcare", "Pharma"),
        ("ALKEM", "Alkem Laboratories Ltd", "Healthcare", "Pharma"),
        ("MAXHEALTH", "Max Healthcare Institute Ltd", "Healthcare", "Hospitals"),
        ("FORTIS", "Fortis Healthcare Ltd", "Healthcare", "Hospitals"),
        ("ZYDUSLIFE", "Zydus Lifesciences Ltd", "Healthcare", "Pharma"),
        ("CGPOWER", "CG Power & Industrial Solutions Ltd", "Capital Goods", "Electricals"),
        ("ABB", "ABB India Ltd", "Capital Goods", "Electricals"),
        ("SIEMENS", "Siemens Ltd", "Capital Goods", "Electricals"),
        ("BHEL", "Bharat Heavy Electricals Ltd", "Capital Goods", "Engineering"),
        ("BEL", "Bharat Electronics Ltd", "Capital Goods", "Defence"),
        ("HAL", "Hindustan Aeronautics Ltd", "Capital Goods", "Defence"),
        ("TATAPOWER", "Tata Power Company Ltd", "Power", "Power"),
        ("ADANITRANS", "Adani Energy Solutions Ltd", "Power", "Transmission"),
        ("APLAPOLLO", "APL Apollo Tubes Ltd", "Metals", "Steel"),
        ("JINDALSAW", "Jindal Saw Ltd", "Metals", "Steel"),
        ("SAIL", "Steel Authority of India Ltd", "Metals", "Steel"),
        ("HINDZINC", "Hindustan Zinc Ltd", "Metals", "Zinc"),
        ("VEDL", "Vedanta Ltd", "Metals", "Mining"),
        ("NATIONALUM", "National Aluminium Company Ltd", "Metals", "Aluminium"),
    ]
    
    rows = []
    for symbol, name, sector, industry in FALLBACK_STOCKS:
        rows.append({
            "symbol": symbol,
            "yf_symbol": f"{symbol}.NS",
            "company_name": name,
            "sector": sector,
            "industry": industry,
        })
    
    return pd.DataFrame(rows)


def get_sector_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return sector counts and percentages for the universe."""
    counts = df["sector"].value_counts().reset_index()
    counts.columns = ["sector", "count"]
    counts["pct"] = counts["count"] / counts["count"].sum() * 100
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = get_nifty500()
    print(f"\nNifty 500 Universe: {len(df)} stocks")
    print(df.head(10).to_string())
    print("\nSector Distribution:")
    print(get_sector_distribution(df).to_string())
