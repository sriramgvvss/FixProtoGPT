"""
Module: src.data.symbol_resolver
=================================

Resolve company names, crypto, and FX symbols using a local cache
seeded from the Twelve Data API.

The resolver provides a two-tier lookup strategy:
    1. **Local cache** — instant, no network calls.  Seeded from
       Twelve Data symbol store (~63 K entries).
    2. **Hardcoded fallback** — minimal bootstrap set used only
       when no Twelve Data sync has been performed.

Usage::

    from src.data.symbol_resolver import SymbolResolver

    resolver = SymbolResolver()
    ticker = resolver.resolve("google")       # → "GOOGL"
    ticker = resolver.resolve("apple")        # → "AAPL"
    ticker = resolver.resolve("MSFT")         # → "MSFT" (already a ticker)

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
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils import paths

logger = logging.getLogger(__name__)


# ── Asset class enum ──────────────────────────────────────────────────

class AssetClass:
    """Simple string constants for supported asset classes."""
    EQUITY = "equity"
    CRYPTO = "crypto"
    FX = "fx"


# ── Hardcoded fallback (common company names) ────────────────────────

_FALLBACK_TICKERS: Dict[str, str] = {
    "apple": "AAPL", "google": "GOOGL", "alphabet": "GOOGL",
    "microsoft": "MSFT", "amazon": "AMZN", "meta": "META",
    "facebook": "META", "tesla": "TSLA", "nvidia": "NVDA",
    "netflix": "NFLX", "amd": "AMD", "intel": "INTC",
    "disney": "DIS", "ibm": "IBM", "oracle": "ORCL",
    "salesforce": "CRM", "adobe": "ADBE", "spotify": "SPOT",
    "uber": "UBER", "lyft": "LYFT", "snap": "SNAP",
    "twitter": "TWTR", "paypal": "PYPL", "shopify": "SHOP",
    "zoom": "ZM", "coinbase": "COIN", "palantir": "PLTR",
    "berkshire": "BRK.B", "jpmorgan": "JPM", "goldman": "GS",
    "boeing": "BA", "walmart": "WMT", "costco": "COST",
    "coca-cola": "KO", "pepsi": "PEP", "pepsico": "PEP",
    "johnson & johnson": "JNJ", "procter & gamble": "PG",
    "visa": "V", "mastercard": "MA",
    "chevron": "CVX", "exxon": "XOM", "exxonmobil": "XOM",
    "pfizer": "PFE", "moderna": "MRNA",
    "starbucks": "SBUX", "mcdonald's": "MCD", "mcdonalds": "MCD",
    "nike": "NKE", "home depot": "HD",
    "at&t": "T", "verizon": "VZ", "t-mobile": "TMUS",
    "cisco": "CSCO", "qualcomm": "QCOM", "broadcom": "AVGO",
    "morgan stanley": "MS", "bank of america": "BAC",
    "citigroup": "C", "wells fargo": "WFC",
    "general electric": "GE", "general motors": "GM",
    "ford": "F", "lockheed": "LMT", "lockheed martin": "LMT",
    "raytheon": "RTX", "northrop": "NOC",
    "caterpillar": "CAT", "deere": "DE", "john deere": "DE",
    "3m": "MMM", "honeywell": "HON",
    "ups": "UPS", "fedex": "FDX",
    "target": "TGT", "kroger": "KR",
    "airbnb": "ABNB", "doordash": "DASH",
    "robinhood": "HOOD", "block": "SQ", "square": "SQ",
    "roblox": "RBLX", "unity": "U",
    "snowflake": "SNOW", "datadog": "DDOG",
    "crowdstrike": "CRWD", "palo alto": "PANW",
    "twilio": "TWLO", "okta": "OKTA",
    "rivian": "RIVN", "lucid": "LCID",
    "amc": "AMC", "gamestop": "GME",
    "micron": "MU", "applied materials": "AMAT",
    "lam research": "LRCX", "asml": "ASML",
    "arm": "ARM", "super micro": "SMCI",
    "dell": "DELL", "hp": "HPQ", "hewlett": "HPE",
}


# ── Crypto fallback (name → symbol) ──────────────────────────────────

_CRYPTO_SYMBOLS: Dict[str, str] = {
    # Major cryptocurrencies
    "bitcoin": "BTC", "btc": "BTC",
    "ethereum": "ETH", "ether": "ETH", "eth": "ETH",
    "ripple": "XRP", "xrp": "XRP",
    "solana": "SOL", "sol": "SOL",
    "cardano": "ADA", "ada": "ADA",
    "dogecoin": "DOGE", "doge": "DOGE",
    "polkadot": "DOT", "dot": "DOT",
    "avalanche": "AVAX", "avax": "AVAX",
    "chainlink": "LINK", "link": "LINK",
    "polygon": "MATIC", "matic": "MATIC",
    "litecoin": "LTC", "ltc": "LTC",
    "shiba inu": "SHIB", "shib": "SHIB", "shiba": "SHIB",
    "tron": "TRX", "trx": "TRX",
    "uniswap": "UNI", "uni": "UNI",
    "stellar": "XLM", "xlm": "XLM",
    "cosmos": "ATOM", "atom": "ATOM",
    "near": "NEAR", "near protocol": "NEAR",
    "algorand": "ALGO", "algo": "ALGO",
    "aptos": "APT", "apt": "APT",
    "arbitrum": "ARB", "arb": "ARB",
    "optimism": "OP",
    "sui": "SUI",
    "filecoin": "FIL", "fil": "FIL",
    "aave": "AAVE",
    "maker": "MKR", "mkr": "MKR",
    "pepe": "PEPE",
    "bonk": "BONK",
    "render": "RNDR", "rndr": "RNDR",
    "injective": "INJ", "inj": "INJ",
    "sei": "SEI",
    "celestia": "TIA", "tia": "TIA",
    "jupiter": "JUP", "jup": "JUP",
    # Stablecoins
    "tether": "USDT", "usdt": "USDT",
    "usdc": "USDC", "usd coin": "USDC",
    "dai": "DAI",
    "busd": "BUSD",
    # Wrapped / derivative
    "wrapped bitcoin": "WBTC", "wbtc": "WBTC",
    "wrapped ether": "WETH", "weth": "WETH",
    # Common trading pairs (kept as base symbol)
    "btc/usd": "BTC/USD", "btc/usdt": "BTC/USDT",
    "eth/usd": "ETH/USD", "eth/usdt": "ETH/USDT",
    "sol/usd": "SOL/USD", "sol/usdt": "SOL/USDT",
    "xrp/usd": "XRP/USD", "xrp/usdt": "XRP/USDT",
    "doge/usd": "DOGE/USD", "doge/usdt": "DOGE/USDT",
    "ada/usd": "ADA/USD", "ada/usdt": "ADA/USDT",
}


# ── FX fallback (currency name → ISO pair) ───────────────────────────

_FX_SYMBOLS: Dict[str, str] = {
    # Major FX pairs
    "eurusd": "EUR/USD", "eur/usd": "EUR/USD",
    "euro dollar": "EUR/USD", "euro usd": "EUR/USD",
    "gbpusd": "GBP/USD", "gbp/usd": "GBP/USD",
    "pound dollar": "GBP/USD", "sterling dollar": "GBP/USD",
    "cable": "GBP/USD",
    "usdjpy": "USD/JPY", "usd/jpy": "USD/JPY",
    "dollar yen": "USD/JPY",
    "usdchf": "USD/CHF", "usd/chf": "USD/CHF",
    "dollar swiss": "USD/CHF", "swissy": "USD/CHF",
    "audusd": "AUD/USD", "aud/usd": "AUD/USD",
    "aussie dollar": "AUD/USD", "australian dollar": "AUD/USD",
    "nzdusd": "NZD/USD", "nzd/usd": "NZD/USD",
    "kiwi dollar": "NZD/USD",
    "usdcad": "USD/CAD", "usd/cad": "USD/CAD",
    "dollar cad": "USD/CAD", "loonie": "USD/CAD",
    # Crosses
    "eurgbp": "EUR/GBP", "eur/gbp": "EUR/GBP",
    "eurjpy": "EUR/JPY", "eur/jpy": "EUR/JPY",
    "gbpjpy": "GBP/JPY", "gbp/jpy": "GBP/JPY",
    "eurchf": "EUR/CHF", "eur/chf": "EUR/CHF",
    "audjpy": "AUD/JPY", "aud/jpy": "AUD/JPY",
    "euraud": "EUR/AUD", "eur/aud": "EUR/AUD",
    "gbpaud": "GBP/AUD", "gbp/aud": "GBP/AUD",
    "nzdjpy": "NZD/JPY", "nzd/jpy": "NZD/JPY",
    "cadjpy": "CAD/JPY", "cad/jpy": "CAD/JPY",
    "chfjpy": "CHF/JPY", "chf/jpy": "CHF/JPY",
    # Emerging-market pairs
    "usdmxn": "USD/MXN", "usd/mxn": "USD/MXN",
    "usdzar": "USD/ZAR", "usd/zar": "USD/ZAR",
    "usdtry": "USD/TRY", "usd/try": "USD/TRY",
    "usdinr": "USD/INR", "usd/inr": "USD/INR",
    "usdsgd": "USD/SGD", "usd/sgd": "USD/SGD",
    "usdhkd": "USD/HKD", "usd/hkd": "USD/HKD",
    "usdcny": "USD/CNY", "usd/cny": "USD/CNY",
    # Nicknames — individual currencies
    "euro": "EUR", "eur": "EUR",
    "pound": "GBP", "sterling": "GBP", "gbp": "GBP",
    "yen": "JPY", "jpy": "JPY",
    "swiss franc": "CHF", "chf": "CHF",
    "aussie": "AUD", "aud": "AUD",
    "kiwi": "NZD", "nzd": "NZD",
    "loony": "CAD", "cad": "CAD",
    "yuan": "CNY", "renminbi": "CNY", "rmb": "CNY",
    "rupee": "INR", "inr": "INR",
}




# ══════════════════════════════════════════════════════════════════════
#  Symbol Cache
# ══════════════════════════════════════════════════════════════════════

class SymbolCache:
    """Persistent JSON-backed cache of company→ticker mappings.

    The cache file is stored alongside model data.  Thread-safe for
    concurrent reads and writes.

    Attributes:
        cache_path: Path to the JSON cache file.
    """

    def __init__(self, cache_path: Optional[Path] = None) -> None:
        """Initialise the cache.

        Args:
            cache_path: Override path for the cache file.
                        Defaults to ``model_store/data/symbol_cache.json``.
        """
        if cache_path:
            self.cache_path = Path(cache_path)
        else:
            self.cache_path = paths.symbol_cache_path()
        self._lock = threading.Lock()
        self._cache: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._cache = {k.lower(): v for k, v in data.items()}
                    logger.info("Loaded %d cached symbols from %s",
                                len(self._cache), self.cache_path)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load symbol cache: %s", exc)
                self._cache = {}

    def _save(self) -> None:
        """Persist cache to disk."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=2, sort_keys=True)
        except OSError as exc:
            logger.warning("Failed to save symbol cache: %s", exc)

    def get(self, key: str) -> Optional[str]:
        """Look up a ticker by company name.

        Args:
            key: Company name or keyword (case-insensitive).

        Returns:
            Ticker symbol or ``None`` if not cached.
        """
        with self._lock:
            return self._cache.get(key.lower())

    def put(self, key: str, ticker: str) -> None:
        """Store a company→ticker mapping.

        Args:
            key:    Company name or keyword (case-insensitive).
            ticker: Ticker symbol (stored as-is, typically uppercase).
        """
        with self._lock:
            self._cache[key.lower()] = ticker
            self._save()

    def put_many(self, mappings: Dict[str, str]) -> None:
        """Store multiple mappings at once.

        Args:
            mappings: Dict of company name → ticker symbol.
        """
        with self._lock:
            for key, ticker in mappings.items():
                self._cache[key.lower()] = ticker
            self._save()

    def size(self) -> int:
        """Number of cached entries."""
        return len(self._cache)

    def all_entries(self) -> Dict[str, str]:
        """Return a copy of all cached entries."""
        with self._lock:
            return dict(self._cache)


# ══════════════════════════════════════════════════════════════════════
#  Symbol Resolver  (main public API)
# ══════════════════════════════════════════════════════════════════════

class SymbolResolver:
    """Resolve names to symbols across equities, crypto, and FX.

    Two-tier resolution:
        1. Local cache (JSON file) — instant, no network.  Seeded
           from the Twelve Data symbol store (~16 K equities,
           1 400+ FX pairs, 2 000+ crypto pairs).
        2. Hardcoded bootstrap fallback — minimal set used only
           when no Twelve Data sync has been performed.

    Automatically detects the asset class from the query:
        - FX pairs (``"EUR/USD"``, ``"eurusd"``, ``"cable"``)
        - Crypto (``"bitcoin"``, ``"eth"``, ``"BTC/USDT"``)
        - Equities (``"google"``, ``"AAPL"``)

    Thread-safe: can be shared across Flask request threads.

    Example::

        resolver = SymbolResolver()
        print(resolver.resolve("google"))     # "GOOGL"
        print(resolver.resolve("bitcoin"))    # "BTC"
        print(resolver.resolve("eurusd"))     # "EUR/USD"
        print(resolver.resolve("BTC/USDT"))   # "BTC/USDT"
    """

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        **_kwargs: Any,
    ) -> None:
        """Initialise the resolver.

        Args:
            cache_path: Override path for the symbol cache file.
            **_kwargs:  Accepted for backward compatibility (ignored).
        """
        self._cache = SymbolCache(cache_path)

        # Seed the cache — prefer Twelve Data store, fall back
        # to the hardcoded bootstrap dicts.
        if self._cache.size() == 0:
            self._seed_cache()

    def _seed_cache(self) -> None:
        """Populate the cache from the best available data source.

        Priority:
            1. Twelve Data symbol store (synced weekly).
            2. Hardcoded bootstrap dicts (minimal ~200 entries).
        """
        try:
            from src.data.twelve_data import SymbolStore

            store = SymbolStore()
            if store.is_synced():
                combined = store.build_combined_map()
                if combined:
                    self._cache.put_many(combined)
                    logger.info(
                        "Seeded symbol cache with %d entries from "
                        "Twelve Data store",
                        len(combined),
                    )
                    return
        except Exception as exc:
            logger.debug(
                "Twelve Data store unavailable, using bootstrap "
                "fallback: %s", exc,
            )

        # Fall back to hardcoded bootstrap
        all_fallbacks: Dict[str, str] = {}
        all_fallbacks.update(_FALLBACK_TICKERS)
        all_fallbacks.update(_CRYPTO_SYMBOLS)
        all_fallbacks.update(_FX_SYMBOLS)
        self._cache.put_many(all_fallbacks)
        logger.info(
            "Seeded symbol cache with %d bootstrap fallback entries "
            "(equity=%d, crypto=%d, fx=%d)",
            len(all_fallbacks),
            len(_FALLBACK_TICKERS),
            len(_CRYPTO_SYMBOLS),
            len(_FX_SYMBOLS),
        )

    @staticmethod
    def detect_asset_class(query: str) -> str:
        """Detect the asset class of a query string.

        Args:
            query: The user's input (name, ticker, or pair).

        Returns:
            One of :attr:`AssetClass.EQUITY`, :attr:`AssetClass.CRYPTO`,
            or :attr:`AssetClass.FX`.
        """
        q = query.strip().lower()

        # ── FX detection ─────────────────────────────────────────
        # Explicit pair notation: EUR/USD, GBPUSD, etc.
        if re.match(r'^[a-z]{3}/?[a-z]{3}$', q):
            base = q.replace('/', '')[:3].upper()
            # ISO 4217 currency codes (major + common emerging)
            _CURRENCIES = {
                "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD",
                "CAD", "HKD", "SGD", "MXN", "ZAR", "TRY", "INR",
                "CNY", "SEK", "NOK", "DKK", "PLN", "CZK", "HUF",
                "BRL", "KRW", "TWD", "THB", "ILS", "RUB",
            }
            if base in _CURRENCIES:
                return AssetClass.FX
        # FX nicknames
        if q in _FX_SYMBOLS:
            return AssetClass.FX
        # Keywords
        if any(kw in q for kw in ("forex", "fx ", "currency")):
            return AssetClass.FX

        # ── Crypto detection ─────────────────────────────────────
        # Explicit crypto pair: BTC/USD, ETH/USDT
        if re.match(r'^[a-z]{2,6}/[a-z]{2,6}$', q):
            base = q.split('/')[0].upper()
            if base in {s.upper() for s in _CRYPTO_SYMBOLS.values()
                        if '/' not in s}:
                return AssetClass.CRYPTO
        if q in _CRYPTO_SYMBOLS:
            return AssetClass.CRYPTO
        if any(kw in q for kw in ("crypto", "coin", "token", "defi",
                                   "blockchain", "nft")):
            return AssetClass.CRYPTO

        return AssetClass.EQUITY

    def resolve(self, query: str) -> Optional[str]:
        """Resolve a name or symbol across equities, crypto, and FX.

        Automatically detects the asset class and looks up the
        appropriate fallback dictionary.

        Args:
            query: Company name (``"google"``), crypto (``"bitcoin"``),
                   FX pair (``"eurusd"``), or raw ticker (``"AAPL"``).

        Returns:
            Symbol string, or ``None`` if resolution fails.
        """
        if not query or not query.strip():
            return None

        clean = query.strip()
        normalised = clean.lower()
        asset_class = self.detect_asset_class(clean)

        # ── FX pair resolution ───────────────────────────────────
        if asset_class == AssetClass.FX:
            # Direct lookup in cache / FX fallback
            cached = self._cache.get(normalised)
            if cached:
                return cached
            fx = _FX_SYMBOLS.get(normalised)
            if fx:
                self._cache.put(normalised, fx)
                return fx
            # Normalise joined pair → slashed: "eurusd" → "EUR/USD"
            if re.match(r'^[a-z]{6}$', normalised):
                pair = normalised[:3].upper() + "/" + normalised[3:].upper()
                self._cache.put(normalised, pair)
                return pair
            # Already slashed
            if re.match(r'^[a-z]{3}/[a-z]{3}$', normalised):
                pair = normalised[:3].upper() + "/" + normalised[4:].upper()
                self._cache.put(normalised, pair)
                return pair
            return None

        # ── Crypto resolution ────────────────────────────────────
        if asset_class == AssetClass.CRYPTO:
            cached = self._cache.get(normalised)
            if cached:
                return cached
            crypto = _CRYPTO_SYMBOLS.get(normalised)
            if crypto:
                self._cache.put(normalised, crypto)
                return crypto
            # Explicit pair passthrough: "BTC/USDT" → "BTC/USDT"
            if '/' in clean:
                pair = clean.upper()
                self._cache.put(normalised, pair)
                return pair
            return None

        # ── Equity resolution (original three-tier) ──────────────
        # If it looks like it's already a ticker (all uppercase, 1-5 chars),
        # validate it against the cache and return as-is
        if re.match(r'^[A-Z]{1,5}$', clean):
            cached = self._cache.get(clean)
            if cached:
                return cached
            return clean

        # 1. Check local cache
        cached = self._cache.get(normalised)
        if cached:
            return cached

        # 2. Check fallback dict (in case cache was reset)
        fallback = _FALLBACK_TICKERS.get(normalised)
        if fallback:
            self._cache.put(normalised, fallback)
            return fallback

        return None

    def resolve_many(self, queries: List[str]) -> Dict[str, Optional[str]]:
        """Resolve multiple company names.

        Args:
            queries: List of company names or tickers.

        Returns:
            Dict mapping each query to its resolved ticker (or ``None``).
        """
        return {q: self.resolve(q) for q in queries}

    def populate_from_list(
        self,
        company_names: List[str],
        skip_cached: bool = True,
    ) -> Dict[str, Optional[str]]:
        """Bulk-populate the cache from a list of company names.

        Useful for pre-warming the cache with common names during
        data preparation or training.

        Args:
            company_names: List of company names to resolve.
            skip_cached:   Whether to skip names already in the cache.

        Returns:
            Dict mapping each name to its resolved ticker (or ``None``).
        """
        results: Dict[str, Optional[str]] = {}
        for name in company_names:
            norm = name.lower().strip()
            if skip_cached:
                cached = self._cache.get(norm)
                if cached:
                    results[name] = cached
                    continue
            ticker = self.resolve(name)
            results[name] = ticker
        return results

    def generate_training_pairs(self) -> List[Dict[str, str]]:
        """Generate training data pairs from the cached mappings.

        Returns a list of dicts suitable for training the model on
        company name → ticker symbol resolution.

        Returns:
            List of dicts with keys: ``company``, ``ticker``, ``template``.
        """
        pairs: List[Dict[str, str]] = []
        entries = self._cache.all_entries()

        templates = [
            "buy {qty} shares of {company}",
            "sell {qty} {company} shares",
            "place a limit order for {company} at {price}",
            "market order to buy {company}",
            "purchase {qty} shares of {company} stock",
            "sell {company} at market price",
            "buy {company} stock",
            "short sell {qty} shares of {company}",
            "order for {company}",
            "trade {company} shares",
        ]

        import random

        for company, ticker in entries.items():
            if not ticker or company == ticker.lower():
                continue
            for template in templates:
                qty = str(random.choice([100, 200, 500, 1000, 2500, 5000]))
                price = f"{random.uniform(10, 500):.2f}"
                nl_text = template.format(
                    company=company.title(),
                    qty=qty,
                    price=price,
                )
                fix_snippet = f"55={ticker}|54=1|38={qty}"
                pairs.append({
                    "company": company,
                    "ticker": ticker,
                    "nl_text": nl_text,
                    "fix_snippet": fix_snippet,
                })

        logger.info("Generated %d training pairs from %d cached symbols",
                    len(pairs), len(entries))
        return pairs

    def lookup_name(self, ticker: str) -> Optional[str]:
        """Return the human-readable name for a ticker symbol.

        Uses a lazily-built reverse map from the Twelve Data store.
        Falls back to the forward cache for simple matches.

        Args:
            ticker: Uppercase ticker (e.g. ``"AAPL"``, ``"EUR/USD"``).

        Returns:
            Display name (e.g. ``"Apple Inc."``) or ``None``.
        """
        if not ticker:
            return None

        upper = ticker.strip().upper()

        # Lazy-load the reverse map once
        if not hasattr(self, "_reverse_map"):
            self._reverse_map: Dict[str, str] = {}
            try:
                from src.data.twelve_data import SymbolStore

                store = SymbolStore()
                if store.is_synced():
                    self._reverse_map = store.build_reverse_map()
            except Exception:
                pass

        name = self._reverse_map.get(upper)
        if name:
            return name

        # Fallback: scan forward cache for a descriptive key
        entries = self._cache.all_entries()
        for key, val in entries.items():
            if val == upper and key != upper.lower() and len(key) > len(upper):
                return key.title()

        return None

    @property
    def cache_size(self) -> int:
        """Number of entries in the symbol cache."""
        return self._cache.size()

    @property
    def cached_entries(self) -> Dict[str, str]:
        """All cached company→ticker mappings."""
        return self._cache.all_entries()


# ── Module-level convenience ──────────────────────────────────────────

_resolver_instance: Optional[SymbolResolver] = None
_resolver_lock = threading.Lock()


def get_resolver(**kwargs: Any) -> SymbolResolver:
    """Get or create the singleton ``SymbolResolver`` instance.

    Thread-safe lazy initialisation.

    Args:
        **kwargs: Forwarded to :class:`SymbolResolver` on first call.

    Returns:
        The singleton :class:`SymbolResolver`.
    """
    global _resolver_instance
    if _resolver_instance is None:
        with _resolver_lock:
            if _resolver_instance is None:
                _resolver_instance = SymbolResolver(**kwargs)
    return _resolver_instance


def resolve_symbol(query: str) -> Optional[str]:
    """Convenience function: resolve a company name to a ticker.

    Uses the singleton resolver with default settings.

    Args:
        query: Company name or ticker symbol.

    Returns:
        Ticker symbol string, or ``None``.
    """
    return get_resolver().resolve(query)


def lookup_symbol_name(ticker: str) -> Optional[str]:
    """Convenience function: look up the display name for a ticker.

    Args:
        ticker: Ticker symbol (e.g. ``"AAPL"``).

    Returns:
        Company / instrument name, or ``None``.
    """
    return get_resolver().lookup_name(ticker)


# ── Shared fallback list ──────────────────────────────────────────

_FALLBACK_SYMBOLS: List[str] = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA",
    "JPM", "BAC", "GS", "MS", "C", "WFC", "BRK.B", "V",
    "MA", "UNH", "HD", "PG", "JNJ",
]


def get_training_symbols(*, max_symbols: int = 0) -> List[str]:
    """Return symbols suitable for synthetic FIX message generation.

    Uses a two-tier strategy:
    1. Pull unique tickers from the SymbolResolver cache (Twelve Data).
    2. Fall back to the hardcoded ``_FALLBACK_SYMBOLS`` list.

    Args:
        max_symbols: Cap the returned list at this length (0 = no cap).

    Returns:
        List of ticker strings.
    """
    try:
        resolver = get_resolver(use_api=False)
        cached = resolver.cached_entries
        cache_tickers = list(set(cached.values()))
        if len(cache_tickers) >= 20:
            symbols = cache_tickers
        else:
            symbols = list(set(cache_tickers + _FALLBACK_SYMBOLS))
    except Exception:
        symbols = list(_FALLBACK_SYMBOLS)

    if max_symbols > 0:
        symbols = symbols[:max_symbols]
    return symbols
