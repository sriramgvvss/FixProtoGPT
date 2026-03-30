"""
Module: src.data.twelve_data
=============================

Twelve Data API client for fetching equity, cryptocurrency, and
forex symbols.  Replaces hardcoded fallback dictionaries with live
market data that can be refreshed on a weekly schedule.

API reference: https://twelvedata.com/docs

Endpoints used
--------------
- ``GET /stocks``          — list equities (filter by exchange, country, type)
- ``GET /forex_pairs``     — list all forex pairs with currency metadata
- ``GET /cryptocurrencies`` — list crypto pairs with exchange availability

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from src.utils import paths

logger = logging.getLogger(__name__)

# ── Default config ────────────────────────────────────────────────

_DEFAULT_API_KEY = "1eda2cb290524c0486502231c36e680f"

# Where the synced data lives on disk
_SYMBOL_STORE_DIR = paths.symbols_dir()


# ══════════════════════════════════════════════════════════════════
#  Twelve Data API Client
# ══════════════════════════════════════════════════════════════════

class TwelveDataClient:
    """Low-level HTTP client for the Twelve Data reference-data APIs.

    Handles authentication, request building, and retry logic.

    Attributes:
        api_key: Twelve Data API key.
        base_url: API base URL.
    """

    BASE_URL = "https://api.twelvedata.com"

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialise the client.

        Args:
            api_key: Twelve Data API key.  Falls back to the
                     ``TWELVE_DATA_API_KEY`` env var, then to the
                     built-in default key.
        """
        self.api_key = (
            api_key
            or os.environ.get("TWELVE_DATA_API_KEY", "")
            or _DEFAULT_API_KEY
        )

    # ── Internal helpers ──────────────────────────────────────────

    def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Issue a GET request and return the parsed JSON.

        Args:
            endpoint: API path (e.g. ``"/stocks"``).
            params:   Extra query parameters.
            timeout:  Request timeout in seconds.

        Returns:
            Parsed JSON response dict.

        Raises:
            RuntimeError: If the ``requests`` library is missing or the
                          API returns an error status.
        """
        if not HAS_REQUESTS:
            raise RuntimeError(
                "The 'requests' library is required for Twelve Data API calls. "
                "Install it with: pip install requests"
            )

        url = f"{self.BASE_URL}{endpoint}"
        query: Dict[str, str] = {"apikey": self.api_key}
        if params:
            query.update(params)

        resp = requests.get(url, params=query, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "error":
            raise RuntimeError(
                f"Twelve Data API error on {endpoint}: "
                f"{data.get('message', 'unknown error')}"
            )

        return data

    # ── Public API methods ────────────────────────────────────────

    def fetch_stocks(
        self,
        country: str = "United States",
        stock_type: str = "Common Stock",
        exchange: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Fetch equity symbols from Twelve Data.

        Args:
            country:    Filter by country (default US).
            stock_type: Instrument type filter.
            exchange:   Optional exchange filter (e.g. ``"NYSE"``).

        Returns:
            List of stock dicts with keys: ``symbol``, ``name``,
            ``currency``, ``exchange``, ``type``, etc.
        """
        params: Dict[str, str] = {
            "country": country,
            "type": stock_type,
        }
        if exchange:
            params["exchange"] = exchange

        data = self._get("/stocks", params)
        items = data.get("data", [])
        logger.info(
            "Fetched %d stocks (country=%s, type=%s, exchange=%s)",
            len(items), country, stock_type, exchange,
        )
        return items

    def fetch_forex_pairs(self) -> List[Dict[str, str]]:
        """Fetch all available forex pairs.

        Returns:
            List of forex dicts with keys: ``symbol``,
            ``currency_group``, ``currency_base``, ``currency_quote``.
        """
        data = self._get("/forex_pairs")
        items = data.get("data", [])
        logger.info("Fetched %d forex pairs", len(items))
        return items

    def fetch_cryptocurrencies(self) -> List[Dict[str, str]]:
        """Fetch all available cryptocurrency pairs.

        Returns:
            List of crypto dicts with keys: ``symbol``,
            ``available_exchanges``, ``currency_base``,
            ``currency_quote``.
        """
        data = self._get("/cryptocurrencies")
        items = data.get("data", [])
        logger.info("Fetched %d cryptocurrency pairs", len(items))
        return items

    def fetch_etfs(
        self,
        country: str = "United States",
    ) -> List[Dict[str, str]]:
        """Fetch ETF symbols.

        Args:
            country: Filter by country.

        Returns:
            List of ETF dicts.
        """
        data = self._get("/etf", {"country": country})
        items = data.get("data", [])
        logger.info("Fetched %d ETFs (country=%s)", len(items), country)
        return items


# ══════════════════════════════════════════════════════════════════
#  Symbol Store — persisted fetch results
# ══════════════════════════════════════════════════════════════════

class SymbolStore:
    """Manages the on-disk cache of symbols fetched from Twelve Data.

    Data is stored as JSON files under
    ``model_store/data/<version>/symbols/``.

    The :meth:`sync` method fetches fresh data from the API and writes
    it to disk.  Other modules read from the store via :meth:`load_*`
    methods which never hit the network.

    Attributes:
        store_dir: Directory where symbol JSON files are persisted.
    """

    def __init__(
        self,
        store_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialise the store.

        Args:
            store_dir: Override directory for symbol files.
            api_key:   Twelve Data API key.
        """
        self.store_dir = Path(store_dir) if store_dir else _SYMBOL_STORE_DIR
        self._client = TwelveDataClient(api_key)

    # ── Sync (write) ──────────────────────────────────────────────

    def sync(
        self,
        stocks: bool = True,
        forex: bool = True,
        crypto: bool = True,
        etfs: bool = False,
    ) -> Dict[str, int]:
        """Fetch symbols from Twelve Data and persist to disk.

        This is the method called by the weekly batch script.

        Args:
            stocks: Whether to sync equities.
            forex:  Whether to sync forex pairs.
            crypto: Whether to sync crypto pairs.
            etfs:   Whether to sync ETFs.

        Returns:
            Dict with counts: ``{"stocks": N, "forex": N, "crypto": N}``.
        """
        self.store_dir.mkdir(parents=True, exist_ok=True)
        counts: Dict[str, int] = {}

        if stocks:
            stock_data = self._client.fetch_stocks()
            self._write(self.store_dir / "stocks.json", stock_data)
            counts["stocks"] = len(stock_data)

        if forex:
            fx_data = self._client.fetch_forex_pairs()
            self._write(self.store_dir / "forex.json", fx_data)
            counts["forex"] = len(fx_data)

        if crypto:
            crypto_data = self._client.fetch_cryptocurrencies()
            self._write(self.store_dir / "crypto.json", crypto_data)
            counts["crypto"] = len(crypto_data)

        if etfs:
            etf_data = self._client.fetch_etfs()
            etf_file = self.store_dir / "etfs.json"
            self._write(etf_file, etf_data)
            counts["etfs"] = len(etf_data)

        # Write sync metadata
        meta = {
            "last_sync": datetime.now(timezone.utc).isoformat(),
            "counts": counts,
            "api_key_last4": self._client.api_key[-4:] if self._client.api_key else "",
        }
        self._write(self.store_dir / "sync_meta.json", meta)

        logger.info("Symbol sync complete: %s", counts)
        return counts

    # ── Load (read-only, no network) ──────────────────────────────

    def load_stocks(self) -> List[Dict[str, str]]:
        """Load persisted stock symbols (no API call).

        Returns:
            List of stock dicts, or empty list if not yet synced.
        """
        return self._read_list(
            self.store_dir / "stocks.json"
        )

    def load_forex(self) -> List[Dict[str, str]]:
        """Load persisted forex pairs (no API call).

        Returns:
            List of forex pair dicts.
        """
        return self._read_list(
            self.store_dir / "forex.json"
        )

    def load_crypto(self) -> List[Dict[str, str]]:
        """Load persisted crypto pairs (no API call).

        Returns:
            List of crypto pair dicts.
        """
        return self._read_list(
            self.store_dir / "crypto.json"
        )

    def load_sync_meta(self) -> Dict[str, Any]:
        """Load sync metadata.

        Returns:
            Dict with ``last_sync``, ``counts``, etc.  Empty dict if
            never synced.
        """
        meta_path = self.store_dir / "sync_meta.json"
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def is_synced(self) -> bool:
        """Check whether a sync has been performed at least once."""
        meta_path = self.store_dir / "sync_meta.json"
        return meta_path.exists()

    # ── Derived mappings for the resolver ─────────────────────────

    def build_stock_map(self) -> Dict[str, str]:
        """Build a name→ticker mapping from synced stock data.

        Creates entries for both the full company name and common
        abbreviations (lowercase → uppercase ticker).  Ticker
        passthrough entries (``"aapl"`` → ``"AAPL"``) are added last
        and never overwrite a company-name mapping.

        Returns:
            Dict mapping lowercase name → ticker symbol.
        """
        name_mappings: Dict[str, str] = {}
        ticker_passthroughs: Dict[str, str] = {}

        for item in self.load_stocks():
            symbol = item.get("symbol", "").strip()
            name = item.get("name", "").strip()
            if not symbol or not name:
                continue

            name_lower = name.lower()
            name_mappings[name_lower] = symbol

            # Also add shortened variants
            # "Apple Inc." → "apple", "Tesla, Inc." → "tesla"
            # Strip trailing commas/periods before suffix matching
            short_name = name_lower.rstrip(".,;:")
            for suffix in (
                " inc.", " inc", " corp.", " corp", " corporation",
                " company", " co.", " co",
                " ltd.", " ltd", " limited",
                " plc", " sa", " ag", " nv", " se",
                " group", " holdings", " partners",
                " & co.", " & co",
                " class a", " class b", " class c",
                " cl a", " cl b", " cl c",
            ):
                if short_name.endswith(suffix):
                    short_name = short_name[: -len(suffix)].strip()

            # Clean any remaining trailing punctuation
            short_name = short_name.rstrip(".,;:")

            if short_name != name_lower and short_name and len(short_name) > 1:
                name_mappings[short_name] = symbol

            # For multi-word names, also map just the first word
            # (e.g. "ford motor" → "ford", "boeing" stays "boeing")
            # Only when first word is ≥ 4 chars to avoid collisions
            effective = short_name or name_lower
            first_word = effective.split()[0].rstrip(".,;:")
            if (
                len(first_word) >= 4
                and first_word != effective
                and first_word not in name_mappings
            ):
                name_mappings[first_word] = symbol

            # Ticker passthrough — added later to avoid clobbering names
            ticker_passthroughs[symbol.lower()] = symbol

        # Merge: name mappings take priority over ticker passthroughs
        mapping: Dict[str, str] = {}
        mapping.update(ticker_passthroughs)
        mapping.update(name_mappings)

        logger.info("Built stock mapping: %d entries", len(mapping))
        return mapping

    def build_forex_map(self) -> Dict[str, str]:
        """Build name→pair mapping from synced forex data.

        Creates entries for:
        - Pair symbol as-is (``"eur/usd"`` → ``"EUR/USD"``)
        - Joined form (``"eurusd"`` → ``"EUR/USD"``)
        - Currency base/quote names

        Returns:
            Dict mapping lowercase key → normalised pair string.
        """
        mapping: Dict[str, str] = {}
        for item in self.load_forex():
            symbol = item.get("symbol", "").strip()       # "EUR/USD"
            base_name = item.get("currency_base", "")     # "Euro"

            if not symbol or "/" not in symbol:
                continue

            sym_upper = symbol.upper()
            sym_lower = symbol.lower()

            # "eur/usd" → "EUR/USD"
            mapping[sym_lower] = sym_upper

            # "eurusd" → "EUR/USD"
            joined = sym_lower.replace("/", "")
            mapping[joined] = sym_upper

            # Currency base name → base ISO code
            if base_name:
                base_code = symbol.split("/")[0].upper()
                mapping[base_name.lower()] = base_code

        # Add well-known FX nicknames
        _FX_NICKNAMES: Dict[str, str] = {
            "cable": "GBP/USD",
            "swissy": "USD/CHF",
            "loonie": "USD/CAD",
            "loony": "USD/CAD",
            "aussie dollar": "AUD/USD",
            "kiwi dollar": "NZD/USD",
            "dollar yen": "USD/JPY",
            "euro dollar": "EUR/USD",
            "pound dollar": "GBP/USD",
            "sterling dollar": "GBP/USD",
            "dollar swiss": "USD/CHF",
            "euro": "EUR",
            "pound": "GBP",
            "sterling": "GBP",
            "yen": "JPY",
            "swiss franc": "CHF",
            "aussie": "AUD",
            "kiwi": "NZD",
            "yuan": "CNY",
            "renminbi": "CNY",
            "rmb": "CNY",
            "rupee": "INR",
        }
        mapping.update(_FX_NICKNAMES)

        logger.info("Built forex mapping: %d entries", len(mapping))
        return mapping

    def build_crypto_map(self) -> Dict[str, str]:
        """Build name→symbol mapping from synced crypto data.

        Creates entries for:
        - Base currency name (``"bitcoin"`` → ``"BTC"``)
        - Base currency ticker (``"btc"`` → ``"BTC"``)
        - Full pair (``"btc/usd"`` → ``"BTC/USD"``)

        Returns:
            Dict mapping lowercase key → symbol string.
        """
        mapping: Dict[str, str] = {}
        seen_bases: set = set()

        for item in self.load_crypto():
            symbol = item.get("symbol", "").strip()       # "BTC/USD"
            base_name = item.get("currency_base", "")     # "Bitcoin"

            if not symbol:
                continue

            sym_upper = symbol.upper()
            sym_lower = symbol.lower()

            # Full pair: "btc/usd" → "BTC/USD"
            mapping[sym_lower] = sym_upper

            # Joined pair: "btcusd" → "BTC/USD"
            if "/" in symbol:
                joined = sym_lower.replace("/", "")
                mapping[joined] = sym_upper

            # Base currency ticker: "btc" → "BTC"
            if "/" in symbol:
                base_ticker = symbol.split("/")[0].strip().upper()
                if base_ticker and base_ticker not in seen_bases:
                    mapping[base_ticker.lower()] = base_ticker
                    seen_bases.add(base_ticker)

                    # Base currency name: "bitcoin" → "BTC"
                    if base_name:
                        mapping[base_name.lower()] = base_ticker

        # Common aliases not always in the API data
        _CRYPTO_ALIASES: Dict[str, str] = {
            "ether": "ETH",
            "doge": "DOGE",
            "shib": "SHIB",
            "shiba": "SHIB",
            "ripple": "XRP",
        }
        for alias, ticker in _CRYPTO_ALIASES.items():
            mapping.setdefault(alias, ticker)

        logger.info("Built crypto mapping: %d entries", len(mapping))
        return mapping

    def build_reverse_map(self) -> Dict[str, str]:
        """Build a reverse ticker→display-name map from synced data.

        Useful for enriching explanations with human-readable names.

        Priority (last write wins):
          1. Stocks  (``AAPL`` → ``Apple Inc.``)
          2. Forex   (``EUR/USD`` → ``Euro / US Dollar``)
          3. Crypto  (``BTC`` → ``Bitcoin``)

        Returns:
            Dict mapping uppercase ticker → display name.
        """
        reverse: Dict[str, str] = {}

        # Stocks: ticker → company name
        for item in self.load_stocks():
            sym = item.get("symbol", "").strip()
            name = item.get("name", "").strip()
            if sym and name:
                reverse[sym] = name

        # Forex: pair → readable label
        for item in self.load_forex():
            sym = item.get("symbol", "").strip()          # "EUR/USD"
            base = item.get("currency_base", "").strip()  # "Euro"
            quote = item.get("currency_quote", "").strip()  # "US Dollar"
            if sym and base and quote:
                reverse[sym] = f"{base} / {quote}"

        # Crypto: base ticker → name
        for item in self.load_crypto():
            sym = item.get("symbol", "").strip()          # "BTC/USD"
            base = item.get("currency_base", "").strip()  # "Bitcoin"
            if sym and base:
                reverse[sym] = base
                # Also map base ticker alone
                base_ticker = sym.split("/")[0] if "/" in sym else sym
                reverse.setdefault(base_ticker, base)

        logger.info("Built reverse mapping: %d entries", len(reverse))
        return reverse

    def build_combined_map(self) -> Dict[str, str]:
        """Build a combined mapping across all asset classes.

        Merge order (last write wins):
          1. Stocks  – lowest priority
          2. Crypto  – overrides stocks for shared names
          3. Forex   – highest priority so traditional currency names
                       (rupee → INR, yen → JPY) are not overwritten by
                       minor crypto tokens that share the same name.

        Returns:
            Dict mapping lowercase key → symbol.
        """
        combined: Dict[str, str] = {}
        combined.update(self.build_stock_map())
        combined.update(self.build_crypto_map())
        combined.update(self.build_forex_map())
        logger.info("Built combined mapping: %d total entries", len(combined))
        return combined

    # ── Private helpers ───────────────────────────────────────────

    @staticmethod
    def _write(path: Path, data: Any) -> None:
        """Write data as JSON to *path*."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _read_list(path: Path) -> List[Dict[str, str]]:
        """Read a JSON list from *path*, returning ``[]`` on error."""
        if not path.exists():
            return []
        try:
            with open(path) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return []


# ── Module-level convenience ──────────────────────────────────────

def get_symbol_store(
    api_key: Optional[str] = None,
    store_dir: Optional[Path] = None,
) -> SymbolStore:
    """Create a :class:`SymbolStore` with sensible defaults.

    Args:
        api_key:   Override API key.
        store_dir: Override storage directory.

    Returns:
        A configured :class:`SymbolStore`.
    """
    return SymbolStore(store_dir=store_dir, api_key=api_key)
