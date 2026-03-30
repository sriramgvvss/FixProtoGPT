#!/usr/bin/env python3
"""
Script: scripts/sync_symbols.py
================================

Weekly batch job that fetches all available symbols (equities, crypto,
forex) from the Twelve Data API and persists them to disk.

The fetched data is used by :class:`src.data.symbol_resolver.SymbolResolver`
for runtime symbol resolution and by the training pipeline for
generating NL→FIX training pairs.

Usage::

    # Full sync (stocks + forex + crypto)
    python3 -m scripts.sync_symbols

    # Sync only specific asset classes
    python3 -m scripts.sync_symbols --stocks --crypto

    # Include ETFs
    python3 -m scripts.sync_symbols --etfs

    # Rebuild the resolver cache after syncing
    python3 -m scripts.sync_symbols --rebuild-cache

    # Custom API key
    python3 -m scripts.sync_symbols --api-key YOUR_KEY

Scheduling (cron)::

    # Run every Sunday at 2 AM
    0 2 * * 0 cd /path/to/FixProtoGPT && python3 -m scripts.sync_symbols >> logs/sync.log 2>&1

Author : FixProtoGPT Team
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.twelve_data import SymbolStore, get_symbol_store
from src.data.symbol_resolver import SymbolResolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sync_symbols")


def sync(
    api_key: str | None = None,
    stocks: bool = True,
    forex: bool = True,
    crypto: bool = True,
    etfs: bool = False,
    rebuild_cache: bool = True,
) -> dict[str, int]:
    """Run a full symbol sync.

    Args:
        api_key:       Override Twelve Data API key.
        stocks:        Sync equities.
        forex:         Sync forex pairs.
        crypto:        Sync crypto pairs.
        etfs:          Sync ETFs.
        rebuild_cache: Rebuild the resolver cache from synced data.

    Returns:
        Dict with counts per asset class.
    """
    store = get_symbol_store(api_key=api_key)

    logger.info("Starting symbol sync...")
    t0 = time.time()

    counts = store.sync(
        stocks=stocks,
        forex=forex,
        crypto=crypto,
        etfs=etfs,
    )

    elapsed = time.time() - t0
    logger.info(
        "Sync complete in %.1fs — stocks=%s, forex=%s, crypto=%s",
        elapsed,
        counts.get("stocks", "skipped"),
        counts.get("forex", "skipped"),
        counts.get("crypto", "skipped"),
    )

    if rebuild_cache:
        logger.info("Rebuilding resolver cache from synced data...")
        _rebuild_resolver_cache(store)

    return counts


def _rebuild_resolver_cache(store: SymbolStore) -> None:
    """Rebuild the SymbolResolver cache from synced Twelve Data.

    This replaces the old hardcoded fallback data with the fresh
    API data.
    """
    combined = store.build_combined_map()
    if not combined:
        logger.warning("No symbol data available — cache not rebuilt")
        return

    resolver = SymbolResolver(use_api=False)
    # Bulk-load all mappings into the resolver cache
    resolver._cache.put_many(combined)
    logger.info(
        "Resolver cache rebuilt with %d entries from Twelve Data",
        len(combined),
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sync symbols from Twelve Data API",
    )
    parser.add_argument(
        "--api-key",
        help="Twelve Data API key (default: built-in key)",
    )
    parser.add_argument(
        "--stocks",
        action="store_true",
        default=False,
        help="Sync equities (default: included in --all)",
    )
    parser.add_argument(
        "--forex",
        action="store_true",
        default=False,
        help="Sync forex pairs (default: included in --all)",
    )
    parser.add_argument(
        "--crypto",
        action="store_true",
        default=False,
        help="Sync crypto pairs (default: included in --all)",
    )
    parser.add_argument(
        "--etfs",
        action="store_true",
        default=False,
        help="Sync ETFs (not included in --all by default)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Sync all asset classes (stocks + forex + crypto)",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        default=True,
        help="Rebuild resolver cache after sync (default: True)",
    )
    parser.add_argument(
        "--no-rebuild-cache",
        action="store_true",
        default=False,
        help="Skip rebuilding the resolver cache",
    )

    args = parser.parse_args()

    # If none specified, default to all
    do_stocks = args.stocks or args.all
    do_forex = args.forex or args.all
    do_crypto = args.crypto or args.all
    do_etfs = args.etfs

    # If no specific flags given, do all
    if not (args.stocks or args.forex or args.crypto or args.all or args.etfs):
        do_stocks = True
        do_forex = True
        do_crypto = True

    rebuild = args.rebuild_cache and not args.no_rebuild_cache

    try:
        counts = sync(
            api_key=args.api_key,
            stocks=do_stocks,
            forex=do_forex,
            crypto=do_crypto,
            etfs=do_etfs,
            rebuild_cache=rebuild,
        )
        total = sum(counts.values())
        logger.info("✓ Sync successful — %d total symbols", total)
    except Exception as exc:
        logger.error("Sync failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
