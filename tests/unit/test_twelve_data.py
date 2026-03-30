"""
Tests for src.data.twelve_data
================================

Validates the TwelveDataClient, SymbolStore, and the sync + mapping
build pipeline.

Coding Standards: PEP 8, PEP 257, PEP 484, Google Python Style Guide.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.twelve_data import (
    TwelveDataClient,
    SymbolStore,
    get_symbol_store,
)


# ══════════════════════════════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def store_dir(tmp_path: Path) -> Path:
    """Temporary directory for symbol store files."""
    d = tmp_path / "symbols"
    d.mkdir()
    return d


@pytest.fixture
def sample_stocks() -> list:
    """Small sample of stock data matching Twelve Data format."""
    return [
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "currency": "USD",
            "exchange": "NASDAQ",
            "type": "Common Stock",
        },
        {
            "symbol": "GOOGL",
            "name": "Alphabet Inc.",
            "currency": "USD",
            "exchange": "NASDAQ",
            "type": "Common Stock",
        },
        {
            "symbol": "MSFT",
            "name": "Microsoft Corporation",
            "currency": "USD",
            "exchange": "NASDAQ",
            "type": "Common Stock",
        },
        {
            "symbol": "JPM",
            "name": "JP Morgan Chase & Co.",
            "currency": "USD",
            "exchange": "NYSE",
            "type": "Common Stock",
        },
    ]


@pytest.fixture
def sample_forex() -> list:
    """Small sample of FX data matching Twelve Data format."""
    return [
        {
            "symbol": "EUR/USD",
            "currency_group": "Major",
            "currency_base": "Euro",
            "currency_quote": "US Dollar",
        },
        {
            "symbol": "GBP/USD",
            "currency_group": "Major",
            "currency_base": "British Pound",
            "currency_quote": "US Dollar",
        },
        {
            "symbol": "USD/JPY",
            "currency_group": "Major",
            "currency_base": "US Dollar",
            "currency_quote": "Japanese Yen",
        },
    ]


@pytest.fixture
def sample_crypto() -> list:
    """Small sample of crypto data matching Twelve Data format."""
    return [
        {
            "symbol": "BTC/USD",
            "available_exchanges": ["Binance", "Coinbase"],
            "currency_base": "Bitcoin",
            "currency_quote": "US Dollar",
        },
        {
            "symbol": "ETH/USD",
            "available_exchanges": ["Binance", "Coinbase"],
            "currency_base": "Ethereum",
            "currency_quote": "US Dollar",
        },
        {
            "symbol": "SOL/USD",
            "available_exchanges": ["Binance"],
            "currency_base": "Solana",
            "currency_quote": "US Dollar",
        },
        {
            "symbol": "DOGE/USDT",
            "available_exchanges": ["Binance"],
            "currency_base": "Dogecoin",
            "currency_quote": "Tether",
        },
    ]


# ══════════════════════════════════════════════════════════════════════
#  TwelveDataClient
# ══════════════════════════════════════════════════════════════════════

class TestTwelveDataClient:
    """Tests for the TwelveDataClient HTTP wrapper."""

    def test_default_api_key(self) -> None:
        client = TwelveDataClient()
        assert client.api_key  # Should have the built-in default

    def test_custom_api_key(self) -> None:
        client = TwelveDataClient(api_key="my_key_123")
        assert client.api_key == "my_key_123"

    def test_env_api_key(self) -> None:
        with patch.dict(os.environ, {"TWELVE_DATA_API_KEY": "env_key_456"}):
            client = TwelveDataClient()
            assert client.api_key == "env_key_456"

    @patch("src.data.twelve_data.requests")
    def test_fetch_stocks_success(self, mock_requests: MagicMock, sample_stocks: list) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": sample_stocks, "status": "ok"}
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        client = TwelveDataClient(api_key="test")
        result = client.fetch_stocks()
        assert len(result) == 4
        assert result[0]["symbol"] == "AAPL"

    @patch("src.data.twelve_data.requests")
    def test_fetch_forex_success(self, mock_requests: MagicMock, sample_forex: list) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": sample_forex, "status": "ok"}
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        client = TwelveDataClient(api_key="test")
        result = client.fetch_forex_pairs()
        assert len(result) == 3
        assert result[0]["symbol"] == "EUR/USD"

    @patch("src.data.twelve_data.requests")
    def test_fetch_crypto_success(self, mock_requests: MagicMock, sample_crypto: list) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": sample_crypto, "status": "ok"}
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        client = TwelveDataClient(api_key="test")
        result = client.fetch_cryptocurrencies()
        assert len(result) == 4
        assert result[0]["symbol"] == "BTC/USD"

    @patch("src.data.twelve_data.requests")
    def test_api_error_raises(self, mock_requests: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "error",
            "message": "Invalid API key",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        client = TwelveDataClient(api_key="bad_key")
        with pytest.raises(RuntimeError, match="Invalid API key"):
            client.fetch_stocks()


# ══════════════════════════════════════════════════════════════════════
#  SymbolStore — persistence and mapping
# ══════════════════════════════════════════════════════════════════════

class TestSymbolStore:
    """Tests for the SymbolStore persistence layer."""

    def test_not_synced_initially(self, store_dir: Path) -> None:
        store = SymbolStore(store_dir=store_dir)
        assert not store.is_synced()

    def test_load_empty_returns_lists(self, store_dir: Path) -> None:
        store = SymbolStore(store_dir=store_dir)
        assert store.load_stocks() == []
        assert store.load_forex() == []
        assert store.load_crypto() == []

    def test_write_and_load_stocks(
        self, store_dir: Path, sample_stocks: list,
    ) -> None:
        # Manually write stock data
        stocks_file = store_dir / "stocks.json"
        with open(stocks_file, "w") as f:
            json.dump(sample_stocks, f)

        store = SymbolStore(store_dir=store_dir)
        loaded = store.load_stocks()
        assert len(loaded) == 4
        assert loaded[0]["symbol"] == "AAPL"

    def test_write_and_load_forex(
        self, store_dir: Path, sample_forex: list,
    ) -> None:
        fx_file = store_dir / "forex.json"
        with open(fx_file, "w") as f:
            json.dump(sample_forex, f)

        store = SymbolStore(store_dir=store_dir)
        loaded = store.load_forex()
        assert len(loaded) == 3

    def test_write_and_load_crypto(
        self, store_dir: Path, sample_crypto: list,
    ) -> None:
        crypto_file = store_dir / "crypto.json"
        with open(crypto_file, "w") as f:
            json.dump(sample_crypto, f)

        store = SymbolStore(store_dir=store_dir)
        loaded = store.load_crypto()
        assert len(loaded) == 4

    def test_sync_meta(self, store_dir: Path) -> None:
        meta = {
            "last_sync": "2026-02-17T02:00:00+00:00",
            "counts": {"stocks": 100, "forex": 50, "crypto": 75},
        }
        meta_file = store_dir / "sync_meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f)

        store = SymbolStore(store_dir=store_dir)
        assert store.is_synced()
        loaded_meta = store.load_sync_meta()
        assert loaded_meta["counts"]["stocks"] == 100


# ══════════════════════════════════════════════════════════════════════
#  SymbolStore — mapping builders
# ══════════════════════════════════════════════════════════════════════

class TestSymbolStoreMapping:
    """Tests for the build_*_map methods."""

    def test_build_stock_map(
        self, store_dir: Path, sample_stocks: list,
    ) -> None:
        stocks_file = store_dir / "stocks.json"
        with open(stocks_file, "w") as f:
            json.dump(sample_stocks, f)

        store = SymbolStore(store_dir=store_dir)
        mapping = store.build_stock_map()

        # Full name
        assert mapping["apple inc."] == "AAPL"
        # Shortened (without " inc.")
        assert mapping["apple"] == "AAPL"
        # Ticker as key
        assert mapping["aapl"] == "AAPL"
        # Microsoft Corporation → stripped
        assert mapping["microsoft"] == "MSFT"
        # Alphabet
        assert mapping["alphabet inc."] == "GOOGL"
        assert mapping["alphabet"] == "GOOGL"

    def test_build_forex_map(
        self, store_dir: Path, sample_forex: list,
    ) -> None:
        fx_file = store_dir / "forex.json"
        with open(fx_file, "w") as f:
            json.dump(sample_forex, f)

        store = SymbolStore(store_dir=store_dir)
        mapping = store.build_forex_map()

        # Slashed pair
        assert mapping["eur/usd"] == "EUR/USD"
        # Joined pair
        assert mapping["eurusd"] == "EUR/USD"
        # Currency name from base
        assert mapping["euro"] == "EUR"
        # Nicknames
        assert mapping["cable"] == "GBP/USD"
        assert mapping["swissy"] == "USD/CHF"

    def test_build_crypto_map(
        self, store_dir: Path, sample_crypto: list,
    ) -> None:
        crypto_file = store_dir / "crypto.json"
        with open(crypto_file, "w") as f:
            json.dump(sample_crypto, f)

        store = SymbolStore(store_dir=store_dir)
        mapping = store.build_crypto_map()

        # Full pair
        assert mapping["btc/usd"] == "BTC/USD"
        # Joined pair
        assert mapping["btcusd"] == "BTC/USD"
        # Base ticker
        assert mapping["btc"] == "BTC"
        # Base name
        assert mapping["bitcoin"] == "BTC"
        assert mapping["ethereum"] == "ETH"
        assert mapping["solana"] == "SOL"
        # Dogecoin pair
        assert mapping["doge/usdt"] == "DOGE/USDT"

    def test_build_combined_map(
        self,
        store_dir: Path,
        sample_stocks: list,
        sample_forex: list,
        sample_crypto: list,
    ) -> None:
        for name, data in [
            ("stocks.json", sample_stocks),
            ("forex.json", sample_forex),
            ("crypto.json", sample_crypto),
        ]:
            with open(store_dir / name, "w") as f:
                json.dump(data, f)

        store = SymbolStore(store_dir=store_dir)
        combined = store.build_combined_map()

        # Equities
        assert combined["apple"] == "AAPL"
        # FX
        assert combined["eur/usd"] == "EUR/USD"
        # Crypto
        assert combined["bitcoin"] == "BTC"
        # Total should be non-trivial
        assert len(combined) > 20

    # ── Reverse map ───────────────────────────────────────────────

    def test_build_reverse_map_stocks(
        self, store_dir: Path, sample_stocks: list,
    ) -> None:
        """Reverse map returns company name keyed by ticker."""
        with open(store_dir / "stocks.json", "w") as f:
            json.dump(sample_stocks, f)
        with open(store_dir / "forex.json", "w") as f:
            json.dump([], f)
        with open(store_dir / "crypto.json", "w") as f:
            json.dump([], f)

        store = SymbolStore(store_dir=store_dir)
        rev = store.build_reverse_map()

        assert rev["AAPL"] == "Apple Inc."
        assert rev["GOOGL"] == "Alphabet Inc."
        assert rev["MSFT"] == "Microsoft Corporation"
        assert rev["JPM"] == "JP Morgan Chase & Co."

    def test_build_reverse_map_forex(
        self, store_dir: Path, sample_forex: list,
    ) -> None:
        """Reverse map returns readable 'Base / Quote' labels for FX."""
        with open(store_dir / "stocks.json", "w") as f:
            json.dump([], f)
        with open(store_dir / "forex.json", "w") as f:
            json.dump(sample_forex, f)
        with open(store_dir / "crypto.json", "w") as f:
            json.dump([], f)

        store = SymbolStore(store_dir=store_dir)
        rev = store.build_reverse_map()

        assert rev["EUR/USD"] == "Euro / US Dollar"
        assert rev["GBP/USD"] == "British Pound / US Dollar"
        assert rev["USD/JPY"] == "US Dollar / Japanese Yen"

    def test_build_reverse_map_crypto(
        self, store_dir: Path, sample_crypto: list,
    ) -> None:
        """Reverse map returns base-currency name for crypto pairs."""
        with open(store_dir / "stocks.json", "w") as f:
            json.dump([], f)
        with open(store_dir / "forex.json", "w") as f:
            json.dump([], f)
        with open(store_dir / "crypto.json", "w") as f:
            json.dump(sample_crypto, f)

        store = SymbolStore(store_dir=store_dir)
        rev = store.build_reverse_map()

        assert rev["BTC/USD"] == "Bitcoin"
        assert rev["BTC"] == "Bitcoin"        # base ticker shortcut
        assert rev["ETH/USD"] == "Ethereum"
        assert rev["ETH"] == "Ethereum"
        assert rev["SOL/USD"] == "Solana"
        assert rev["SOL"] == "Solana"

    def test_build_reverse_map_combined(
        self,
        store_dir: Path,
        sample_stocks: list,
        sample_forex: list,
        sample_crypto: list,
    ) -> None:
        """Reverse map works across all three asset classes."""
        for name, data in [
            ("stocks.json", sample_stocks),
            ("forex.json", sample_forex),
            ("crypto.json", sample_crypto),
        ]:
            with open(store_dir / name, "w") as f:
                json.dump(data, f)

        store = SymbolStore(store_dir=store_dir)
        rev = store.build_reverse_map()

        assert rev["AAPL"] == "Apple Inc."
        assert rev["EUR/USD"] == "Euro / US Dollar"
        assert rev["BTC"] == "Bitcoin"
        assert len(rev) >= 10  # stocks + fx + crypto entries


# ══════════════════════════════════════════════════════════════════════
#  SymbolStore — sync with mocked API
# ══════════════════════════════════════════════════════════════════════

class TestSymbolStoreSync:
    """Tests for the sync method with mocked API calls."""

    @patch("src.data.twelve_data.requests")
    def test_sync_writes_files(
        self,
        mock_requests: MagicMock,
        store_dir: Path,
        sample_stocks: list,
        sample_forex: list,
        sample_crypto: list,
    ) -> None:
        responses = iter([
            _mock_response({"data": sample_stocks, "status": "ok"}),
            _mock_response({"data": sample_forex, "status": "ok"}),
            _mock_response({"data": sample_crypto, "status": "ok"}),
        ])
        mock_requests.get.side_effect = lambda *a, **kw: next(responses)

        store = SymbolStore(store_dir=store_dir, api_key="test_key")
        counts = store.sync()

        assert counts["stocks"] == 4
        assert counts["forex"] == 3
        assert counts["crypto"] == 4
        assert store.is_synced()

        # Verify files written
        assert (store_dir / "stocks.json").exists()
        assert (store_dir / "forex.json").exists()
        assert (store_dir / "crypto.json").exists()
        assert (store_dir / "sync_meta.json").exists()

    @patch("src.data.twelve_data.requests")
    def test_sync_selective(
        self,
        mock_requests: MagicMock,
        store_dir: Path,
        sample_crypto: list,
    ) -> None:
        mock_requests.get.return_value = _mock_response(
            {"data": sample_crypto, "status": "ok"}
        )

        store = SymbolStore(store_dir=store_dir, api_key="test_key")
        counts = store.sync(stocks=False, forex=False, crypto=True)

        assert "stocks" not in counts
        assert "forex" not in counts
        assert counts["crypto"] == 4


# ══════════════════════════════════════════════════════════════════════
#  Resolver integration with Twelve Data store
# ══════════════════════════════════════════════════════════════════════

class TestResolverTwelveDataIntegration:
    """Verify that SymbolResolver seeds from the Twelve Data store."""

    def test_resolver_seeds_from_store(
        self,
        tmp_path: Path,
        store_dir: Path,
        sample_stocks: list,
        sample_forex: list,
        sample_crypto: list,
    ) -> None:
        """When synced data exists, resolver should use it."""
        # Pre-populate the store
        for name, data in [
            ("stocks.json", sample_stocks),
            ("forex.json", sample_forex),
            ("crypto.json", sample_crypto),
        ]:
            with open(store_dir / name, "w") as f:
                json.dump(data, f)

        # Write sync meta so is_synced() returns True
        meta = {"last_sync": "2026-02-17T00:00:00Z", "counts": {}}
        with open(store_dir / "sync_meta.json", "w") as f:
            json.dump(meta, f)

        from src.data.symbol_resolver import SymbolResolver

        # Patch SymbolStore where it's imported inside _seed_cache
        with patch("src.data.twelve_data.SymbolStore") as MockStore:
            mock_store_instance = SymbolStore(store_dir=store_dir)
            MockStore.return_value = mock_store_instance

            resolver = SymbolResolver(
                cache_path=tmp_path / "test_cache.json",
                use_api=False,
            )

            # Should have entries from the store
            assert resolver.resolve("apple") == "AAPL"
            assert resolver.resolve("bitcoin") == "BTC"
            assert resolver.resolve("eur/usd") == "EUR/USD"


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

def _mock_response(json_data: dict) -> MagicMock:
    """Create a mock requests response."""
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp
