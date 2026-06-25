from data.source_registry import get_source_registry


def test_nse_registry_has_verified_sources():
    registry = get_source_registry()
    assert "nse_corporate_actions" in registry
    assert registry["nse_corporate_actions"]["source"] == "NSE"
    assert registry["nse_financial_results"]["normalized_table"] == "financial_results"
