"""
Tests for src.data.symbol_resolver
====================================

Validates the SymbolResolver, SymbolCache, and the
two-tier lookup (cache → fallback).

Coding Standards: PEP 8, PEP 257, PEP 484, Google Python Style Guide.
"""

import json
from pathlib import Path

import pytest

from unittest.mock import patch, MagicMock

from src.data.symbol_resolver import (
    AssetClass,
    SymbolCache,
    SymbolResolver,
    _FALLBACK_TICKERS,
    _CRYPTO_SYMBOLS,
    _FX_SYMBOLS,
    resolve_symbol,
    get_resolver,
    lookup_symbol_name,
)


# ══════════════════════════════════════════════════════════════════════
#  SymbolCache
# ══════════════════════════════════════════════════════════════════════

class TestSymbolCache:
    """Tests for the JSON-backed SymbolCache."""

    def test_put_and_get(self, tmp_path: Path) -> None:
        cache = SymbolCache(cache_path=tmp_path / "cache.json")
        cache.put("apple", "AAPL")
        assert cache.get("apple") == "AAPL"
        assert cache.get("Apple") == "AAPL"  # case-insensitive

    def test_get_missing(self, tmp_path: Path) -> None:
        cache = SymbolCache(cache_path=tmp_path / "cache.json")
        assert cache.get("nonexistent") is None

    def test_put_many(self, tmp_path: Path) -> None:
        cache = SymbolCache(cache_path=tmp_path / "cache.json")
        cache.put_many({"google": "GOOGL", "tesla": "TSLA"})
        assert cache.get("google") == "GOOGL"
        assert cache.get("tesla") == "TSLA"
        assert cache.size() == 2

    def test_persistence(self, tmp_path: Path) -> None:
        # Write cache
        path = tmp_path / "cache.json"
        cache1 = SymbolCache(cache_path=path)
        cache1.put("microsoft", "MSFT")

        # Reload from same file
        cache2 = SymbolCache(cache_path=path)
        assert cache2.get("microsoft") == "MSFT"

    def test_all_entries(self, tmp_path: Path) -> None:
        cache = SymbolCache(cache_path=tmp_path / "cache.json")
        cache.put_many({"a": "AA", "b": "BB"})
        entries = cache.all_entries()
        assert isinstance(entries, dict)
        assert entries == {"a": "AA", "b": "BB"}

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        path.write_text("")
        cache = SymbolCache(cache_path=path)
        assert cache.size() == 0


# ══════════════════════════════════════════════════════════════════════
#  SymbolResolver
# ══════════════════════════════════════════════════════════════════════

class TestSymbolResolver:
    """Tests for the main SymbolResolver."""

    def _make_resolver(self, tmp_path: Path) -> SymbolResolver:
        return SymbolResolver(
            cache_path=tmp_path / "test_cache.json",
        )

    def test_resolve_from_fallback(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("google") == "GOOGL"
        assert resolver.resolve("apple") == "AAPL"
        assert resolver.resolve("tesla") == "TSLA"
        assert resolver.resolve("microsoft") == "MSFT"

    def test_resolve_case_insensitive(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("Google") == "GOOGL"
        assert resolver.resolve("GOOGLE") == "GOOGL"

    def test_resolve_ticker_passthrough(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("AAPL") == "AAPL"
        assert resolver.resolve("MSFT") == "MSFT"

    def test_resolve_empty_returns_none(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("") is None
        assert resolver.resolve("  ") is None

    def test_resolve_unknown_no_api(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("xyznonexistent") is None

    def test_resolve_many(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        results = resolver.resolve_many(["google", "apple", "unknown_co"])
        assert results["google"] == "GOOGL"
        assert results["apple"] == "AAPL"
        assert results["unknown_co"] is None

    def test_cache_seeded_on_init(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.cache_size >= len(_FALLBACK_TICKERS)

    def test_extended_fallback_entries(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        # Test some of the extended entries
        assert resolver.resolve("visa") == "V"
        assert resolver.resolve("starbucks") == "SBUX"
        assert resolver.resolve("boeing") == "BA"
        assert resolver.resolve("ford") == "F"
        assert resolver.resolve("nike") == "NKE"

    def test_generate_training_pairs(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        pairs = resolver.generate_training_pairs()
        assert isinstance(pairs, list)
        assert len(pairs) > 0
        # Each pair should have company, ticker, nl_text, fix_snippet
        for p in pairs[:5]:
            assert "company" in p
            assert "ticker" in p
            assert "nl_text" in p
            assert "fix_snippet" in p
            assert "55=" in p["fix_snippet"]

    def test_populate_from_list(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        results = resolver.populate_from_list(["google", "apple", "tesla"])
        assert results["google"] == "GOOGL"
        assert results["apple"] == "AAPL"
        assert results["tesla"] == "TSLA"


# ══════════════════════════════════════════════════════════════════════
#  Module-level convenience functions
# ══════════════════════════════════════════════════════════════════════

class TestModuleFunctions:
    """Tests for resolve_symbol() and get_resolver()."""

    def test_resolve_symbol_function(self) -> None:
        # Uses singleton resolver
        result = resolve_symbol("google")
        assert result == "GOOGL"

    def test_get_resolver_singleton(self) -> None:
        r1 = get_resolver(use_api=False)
        r2 = get_resolver(use_api=False)
        # Should return same instance (singleton)
        assert r1 is r2


# ══════════════════════════════════════════════════════════════════════
#  Integration with NL parser
# ══════════════════════════════════════════════════════════════════════

class TestNLParserIntegration:
    """Verify _parse_nl_for_demo uses the SymbolResolver."""

    def test_google_resolves(self) -> None:
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("buy 1000 shares of google")
        assert result["symbol"] == "GOOGL"

    def test_tesla_resolves(self) -> None:
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("sell 500 tesla shares at 250.50")
        assert result["symbol"] == "TSLA"

    def test_apple_resolves(self) -> None:
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("buy apple stock")
        assert result["symbol"] == "AAPL"

    def test_explicit_ticker(self) -> None:
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("buy 100 shares of MSFT")
        assert result["symbol"] == "MSFT"

    def test_starbucks_resolves(self) -> None:
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("buy 200 shares of starbucks")
        assert result["symbol"] == "SBUX"

    def test_boeing_resolves(self) -> None:
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("sell boeing shares")
        assert result["symbol"] == "BA"

    def test_nike_resolves(self) -> None:
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("market order to buy 100 nike shares")
        assert result["symbol"] == "NKE"


# ══════════════════════════════════════════════════════════════════════
#  Asset class detection
# ══════════════════════════════════════════════════════════════════════

class TestAssetClassDetection:
    """Tests for SymbolResolver.detect_asset_class()."""

    def test_equity_company_names(self) -> None:
        assert SymbolResolver.detect_asset_class("google") == AssetClass.EQUITY
        assert SymbolResolver.detect_asset_class("apple") == AssetClass.EQUITY
        assert SymbolResolver.detect_asset_class("AAPL") == AssetClass.EQUITY

    def test_fx_pair_slashed(self) -> None:
        assert SymbolResolver.detect_asset_class("EUR/USD") == AssetClass.FX
        assert SymbolResolver.detect_asset_class("GBP/JPY") == AssetClass.FX
        assert SymbolResolver.detect_asset_class("usd/cad") == AssetClass.FX

    def test_fx_pair_joined(self) -> None:
        assert SymbolResolver.detect_asset_class("eurusd") == AssetClass.FX
        assert SymbolResolver.detect_asset_class("GBPJPY") == AssetClass.FX
        assert SymbolResolver.detect_asset_class("usdjpy") == AssetClass.FX

    def test_fx_nicknames(self) -> None:
        assert SymbolResolver.detect_asset_class("cable") == AssetClass.FX
        assert SymbolResolver.detect_asset_class("swissy") == AssetClass.FX
        assert SymbolResolver.detect_asset_class("loonie") == AssetClass.FX

    def test_crypto_names(self) -> None:
        assert SymbolResolver.detect_asset_class("bitcoin") == AssetClass.CRYPTO
        assert SymbolResolver.detect_asset_class("ethereum") == AssetClass.CRYPTO
        assert SymbolResolver.detect_asset_class("solana") == AssetClass.CRYPTO

    def test_crypto_tickers(self) -> None:
        assert SymbolResolver.detect_asset_class("btc") == AssetClass.CRYPTO
        assert SymbolResolver.detect_asset_class("eth") == AssetClass.CRYPTO
        assert SymbolResolver.detect_asset_class("sol") == AssetClass.CRYPTO

    def test_crypto_pairs(self) -> None:
        assert SymbolResolver.detect_asset_class("btc/usd") == AssetClass.CRYPTO
        assert SymbolResolver.detect_asset_class("eth/usdt") == AssetClass.CRYPTO


# ══════════════════════════════════════════════════════════════════════
#  Crypto resolution
# ══════════════════════════════════════════════════════════════════════

class TestCryptoResolution:
    """Tests for cryptocurrency symbol resolution."""

    def _make_resolver(self, tmp_path: Path) -> SymbolResolver:
        return SymbolResolver(
            cache_path=tmp_path / "crypto_cache.json",
            use_api=False,
        )

    def test_bitcoin(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("bitcoin") == "BTC"

    def test_ethereum(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("ethereum") == "ETH"

    def test_ether(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("ether") == "ETH"

    def test_solana(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("solana") == "SOL"

    def test_dogecoin(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("dogecoin") == "DOGE"

    def test_ripple(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("ripple") == "XRP"

    def test_cardano(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("cardano") == "ADA"

    def test_crypto_ticker_shorthand(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("btc") == "BTC"
        assert resolver.resolve("eth") == "ETH"
        assert resolver.resolve("sol") == "SOL"

    def test_crypto_pair(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("btc/usd") == "BTC/USD"
        assert resolver.resolve("eth/usdt") == "ETH/USDT"

    def test_stablecoin(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("tether") == "USDT"
        assert resolver.resolve("usdc") == "USDC"

    def test_defi_tokens(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("aave") == "AAVE"
        assert resolver.resolve("uniswap") == "UNI"
        assert resolver.resolve("maker") == "MKR"

    def test_shiba_inu(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("shiba inu") == "SHIB"
        assert resolver.resolve("shib") == "SHIB"


# ══════════════════════════════════════════════════════════════════════
#  FX resolution
# ══════════════════════════════════════════════════════════════════════

class TestFXResolution:
    """Tests for forex symbol resolution."""

    def _make_resolver(self, tmp_path: Path) -> SymbolResolver:
        return SymbolResolver(
            cache_path=tmp_path / "fx_cache.json",
            use_api=False,
        )

    def test_eurusd_joined(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("eurusd") == "EUR/USD"

    def test_eurusd_slashed(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("eur/usd") == "EUR/USD"

    def test_gbpusd(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("gbpusd") == "GBP/USD"

    def test_usdjpy(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("usdjpy") == "USD/JPY"

    def test_fx_nickname_cable(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("cable") == "GBP/USD"

    def test_fx_nickname_swissy(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("swissy") == "USD/CHF"

    def test_fx_nickname_loonie(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("loonie") == "USD/CAD"

    def test_fx_crosses(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("eurjpy") == "EUR/JPY"
        assert resolver.resolve("gbpjpy") == "GBP/JPY"
        assert resolver.resolve("eurchf") == "EUR/CHF"

    def test_fx_emerging(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("usdmxn") == "USD/MXN"
        assert resolver.resolve("usdinr") == "USD/INR"

    def test_currency_names(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.resolve("euro") == "EUR"
        assert resolver.resolve("pound") == "GBP"
        assert resolver.resolve("yen") == "JPY"
        assert resolver.resolve("yuan") == "CNY"
        assert resolver.resolve("rupee") == "INR"


# ══════════════════════════════════════════════════════════════════════
#  Reverse lookup — lookup_name
# ══════════════════════════════════════════════════════════════════════

class TestLookupName:
    """Tests for SymbolResolver.lookup_name and lookup_symbol_name."""

    def _make_resolver(self, tmp_path: Path) -> SymbolResolver:
        return SymbolResolver(cache_path=tmp_path / "cache.json")

    def test_lookup_name_none_for_empty(self, tmp_path: Path) -> None:
        resolver = self._make_resolver(tmp_path)
        assert resolver.lookup_name("") is None
        assert resolver.lookup_name(None) is None  # type: ignore[arg-type]

    def test_lookup_name_with_synced_store(self, tmp_path: Path) -> None:
        """When a synced SymbolStore exists, reverse map is used."""
        resolver = self._make_resolver(tmp_path)

        fake_reverse = {"AAPL": "Apple Inc.", "BTC": "Bitcoin"}
        mock_store = MagicMock()
        mock_store.is_synced.return_value = True
        mock_store.build_reverse_map.return_value = fake_reverse

        with patch("src.data.twelve_data.SymbolStore", return_value=mock_store):
            # Force re-build of reverse map by clearing cached attr
            if hasattr(resolver, "_reverse_map"):
                del resolver._reverse_map

            result = resolver.lookup_name("AAPL")
            assert result == "Apple Inc."

    def test_lookup_name_case_insensitive_input(self, tmp_path: Path) -> None:
        """Input ticker is normalised to uppercase before lookup."""
        resolver = self._make_resolver(tmp_path)

        fake_reverse = {"TSLA": "Tesla, Inc."}
        mock_store = MagicMock()
        mock_store.is_synced.return_value = True
        mock_store.build_reverse_map.return_value = fake_reverse

        with patch("src.data.twelve_data.SymbolStore", return_value=mock_store):
            if hasattr(resolver, "_reverse_map"):
                del resolver._reverse_map

            assert resolver.lookup_name("tsla") == "Tesla, Inc."
            assert resolver.lookup_name("Tsla") == "Tesla, Inc."

    def test_lookup_name_unknown_ticker(self, tmp_path: Path) -> None:
        """Returns None for a ticker not in the reverse map or cache."""
        resolver = self._make_resolver(tmp_path)

        fake_reverse = {"AAPL": "Apple Inc."}
        mock_store = MagicMock()
        mock_store.is_synced.return_value = True
        mock_store.build_reverse_map.return_value = fake_reverse

        with patch("src.data.twelve_data.SymbolStore", return_value=mock_store):
            if hasattr(resolver, "_reverse_map"):
                del resolver._reverse_map

            assert resolver.lookup_name("XYZQQQ") is None

    def test_lookup_name_fallback_to_cache(self, tmp_path: Path) -> None:
        """When reverse map has no entry, falls back to cache scan."""
        resolver = self._make_resolver(tmp_path)
        # Pre-populate cache with descriptive key
        resolver._cache.put("apple inc.", "AAPL")

        # Empty reverse map
        mock_store = MagicMock()
        mock_store.is_synced.return_value = True
        mock_store.build_reverse_map.return_value = {}

        with patch("src.data.twelve_data.SymbolStore", return_value=mock_store):
            if hasattr(resolver, "_reverse_map"):
                del resolver._reverse_map

            result = resolver.lookup_name("AAPL")
            assert result is not None
            assert "apple" in result.lower()

    def test_lookup_name_store_not_synced(self, tmp_path: Path) -> None:
        """When store is not synced, reverse map is empty — uses fallback."""
        resolver = self._make_resolver(tmp_path)
        resolver._cache.put("microsoft corporation", "MSFT")

        mock_store = MagicMock()
        mock_store.is_synced.return_value = False

        with patch("src.data.twelve_data.SymbolStore", return_value=mock_store):
            if hasattr(resolver, "_reverse_map"):
                del resolver._reverse_map

            result = resolver.lookup_name("MSFT")
            # Falls back to cache scan
            assert result is not None
            assert "microsoft" in result.lower()


# ══════════════════════════════════════════════════════════════════════
#  NL parser — crypto & FX integration
# ══════════════════════════════════════════════════════════════════════

class TestNLParserCryptoFX:
    """Verify _parse_nl_for_demo handles crypto and FX queries."""

    def test_buy_bitcoin(self) -> None:
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("buy 5 bitcoin at market price")
        assert result["symbol"] == "BTC"

    def test_sell_ethereum(self) -> None:
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("sell 10 ethereum")
        assert result["symbol"] == "ETH"

    def test_buy_solana(self) -> None:
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("buy solana")
        assert result["symbol"] == "SOL"

    def test_dogecoin(self) -> None:
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("buy 1000 dogecoin")
        assert result["symbol"] == "DOGE"

    def test_fx_eurusd(self) -> None:
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("buy EUR/USD")
        assert result["symbol"] == "EUR/USD"

    def test_fx_gbpusd(self) -> None:
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("sell gbpusd at 1.2650")
        assert result["symbol"] == "GBP/USD"

    def test_fx_usdjpy(self) -> None:
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("buy usdjpy")
        assert result["symbol"] == "USD/JPY"


# ══════════════════════════════════════════════════════════════════════
#  NL parser — "one buy/sell" & quantity edge cases
# ══════════════════════════════════════════════════════════════════════

class TestNLParserEdgeCases:
    """Verify edge-case phrasing is parsed correctly."""

    def test_one_buy_ford(self) -> None:
        """'one' before buy should not resolve as ticker ONE."""
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("one buy 100 ford stock at market price")
        assert result["symbol"] == "F"
        assert result["qty"] == "100"
        assert result["side"] == "Buy"
        assert result["price"] is None

    def test_one_sell_ford_for_units(self) -> None:
        """'for 300 units' should set qty=300, not grab the price."""
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("one sell ford stock at 210.00 for 300 units")
        assert result["symbol"] == "F"
        assert result["qty"] == "300"
        assert result["side"] == "Sell"
        assert result["price"] == "210.00"

    def test_sell_units_of(self) -> None:
        """'sell 300 units of ford' — 'units' should not be the symbol."""
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("sell 300 units of ford at 210.00")
        assert result["symbol"] == "F"
        assert result["qty"] == "300"

    def test_price_infers_limit(self) -> None:
        """Price with 'at' but no explicit 'limit' should infer Limit."""
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("sell ford at 210.00 for 300 units")
        assert result["ord_type_code"] == "2"  # Limit

    def test_explicit_market_ignores_price(self) -> None:
        """'at market price' should be Market, not Limit."""
        from src.api.routes.core import _parse_nl_for_demo
        result = _parse_nl_for_demo("buy 100 ford stock at market price")
        assert result["ord_type_code"] == "1"  # Market
        assert result["price"] is None


# ══════════════════════════════════════════════════════════════════════
#  Multi-order splitting
# ══════════════════════════════════════════════════════════════════════

class TestMultiOrderSplitting:
    """Verify _split_multi_orders splits compound prompts."""

    def test_single_order_unchanged(self) -> None:
        from src.api.routes.core import _split_multi_orders
        parts = _split_multi_orders("buy 100 ford stock")
        assert len(parts) == 1
        assert parts[0] == "buy 100 ford stock"

    def test_and_separator(self) -> None:
        from src.api.routes.core import _split_multi_orders
        parts = _split_multi_orders(
            "one buy 100 ford stock at market price "
            "and one sell ford stock at 210.00 for 300 units"
        )
        assert len(parts) == 2
        assert "buy" in parts[0].lower()
        assert "sell" in parts[1].lower()

    def test_newline_separator(self) -> None:
        from src.api.routes.core import _split_multi_orders
        parts = _split_multi_orders(
            "buy 200 apple stock\nsell 50 tesla at 250"
        )
        assert len(parts) == 2

    def test_and_without_buy_sell_does_not_split(self) -> None:
        """'and' without buy/sell after it should NOT split."""
        from src.api.routes.core import _split_multi_orders
        parts = _split_multi_orders("buy 100 shares of simon and schuster")
        assert len(parts) == 1

    def test_multi_parse_produces_correct_symbols(self) -> None:
        from src.api.routes.core import _split_multi_orders, _parse_nl_for_demo
        parts = _split_multi_orders(
            "buy 200 apple stock and sell 50 tesla at 250"
        )
        r1 = _parse_nl_for_demo(parts[0])
        r2 = _parse_nl_for_demo(parts[1])
        assert r1["symbol"] == "AAPL"
        assert r1["qty"] == "200"
        assert r2["symbol"] == "TSLA"
        assert r2["qty"] == "50"
