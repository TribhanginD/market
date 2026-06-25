import pandas as pd

from data.nifty500 import _is_valid_universe
import config


def test_universe_validation_rejects_single_sector_cache():
    config.REQUIRE_FULL_UNIVERSE = False
    df = pd.DataFrame(
        {
            "symbol": [f"S{i}" for i in range(500)],
            "yf_symbol": [f"S{i}.NS" for i in range(500)],
            "company_name": [f"Stock {i}" for i in range(500)],
            "sector": ["Unknown"] * 500,
            "industry": ["Unknown"] * 500,
        }
    )
    # Sector may be Unknown when the universe source doesn't provide it; we enrich later from yfinance.
    assert _is_valid_universe(df) is True


def test_universe_validation_accepts_diversified_universe():
    config.REQUIRE_FULL_UNIVERSE = False
    sectors = [f"Sector{i % 10}" for i in range(200)]
    df = pd.DataFrame(
        {
            "symbol": [f"S{i}" for i in range(200)],
            "yf_symbol": [f"S{i}.NS" for i in range(200)],
            "company_name": [f"Stock {i}" for i in range(200)],
            "sector": sectors,
            "industry": [f"Industry{i % 20}" for i in range(200)],
        }
    )
    assert _is_valid_universe(df) is True
