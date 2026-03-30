"""
Module: src.data.prepare_data
==============================

Data preparation and preprocessing for FIX protocol training data.

Tokenises raw text, splits into train/val sets, and creates binary
format files for efficient training.

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
import json
import random
from tqdm import tqdm

from src.core.tokenizer import FixProtocolTokenizer
from src.utils import paths
from src.data.symbol_resolver import get_resolver, get_training_symbols


class FIXDataGenerator:
    """Generate synthetic FIX 5.0 SP2 protocol messages for training"""

    FIX_VERSION = "FIXT.1.1"  # Session layer — overridden at runtime from config
    APPL_VER_ID = "9"  # FIX 5.0 SP2

    def __init__(self):
        """Initialise generator with symbols, message types, and config."""
        self.FIX_VERSION = paths.session_protocol()

        # Pull symbols from the SymbolResolver cache (includes Twelve Data)
        self.symbols = get_training_symbols()
        try:
            resolver = get_resolver(use_api=False)
            self._company_tickers = resolver.cached_entries
        except Exception:
            self._company_tickers = {}

        self.msg_types = {
            'D': 'NewOrderSingle',
            '8': 'ExecutionReport',
            'F': 'OrderCancelRequest',
            'G': 'OrderCancelReplaceRequest',
            'V': 'MarketDataRequest',
            'A': 'Logon',
            '0': 'Heartbeat',
        }
    
    def generate_new_order(self) -> str:
        """Generate a New Order Single (D) message — FIX 5.0 SP2.

        Returns:
            Pipe-delimited FIX message string.
        """
        symbol = random.choice(self.symbols)
        side = random.choice(['1', '2'])
        qty = random.randint(1, 1000) * 100
        price = round(random.uniform(50, 500), 2)
        ord_type = random.choice(['1', '2'])
        tif = random.choice(['0', '1', '3', '4'])

        fields = [
            f'8={self.FIX_VERSION}', f'9=200', '35=D',
            f'49=SENDER{random.randint(1,10)}',
            f'56=TARGET{random.randint(1,10)}',
            f'34={random.randint(1, 9999)}',
            f'52={self._generate_timestamp()}',
            f'1128={self.APPL_VER_ID}',
            f'11=ORD{random.randint(10000, 99999)}',
            '21=1', f'55={symbol}', f'54={side}',
            f'38={qty}', f'40={ord_type}',
        ]

        if ord_type == '2':
            fields.append(f'44={price}')

        fields.extend([
            f'59={tif}',
            f'60={self._generate_timestamp()}',
            f'10=000',
        ])

        return '|'.join(fields) + '|'
    
    def generate_execution_report(self) -> str:
        """Generate an Execution Report (8) message — FIX 5.0 SP2.

        Returns:
            Pipe-delimited FIX message string.
        """
        symbol = random.choice(self.symbols)
        side = random.choice(['1', '2'])
        qty = random.randint(1, 1000) * 100
        filled = random.randint(0, qty)
        price = round(random.uniform(50, 500), 2)
        exec_type = random.choice(['0', '4', '8', 'F'])
        ord_status_map = {'0': '0', '4': '4', '8': '8', 'F': '2' if filled == qty else '1'}
        ord_status = ord_status_map[exec_type]

        fields = [
            f'8={self.FIX_VERSION}', f'9=280', '35=8',
            f'49=TARGET{random.randint(1,10)}',
            f'56=SENDER{random.randint(1,10)}',
            f'34={random.randint(1, 9999)}',
            f'52={self._generate_timestamp()}',
            f'1128={self.APPL_VER_ID}',
            f'37=EXEC{random.randint(10000, 99999)}',
            f'11=ORD{random.randint(10000, 99999)}',
            f'17=EXEC{random.randint(10000, 99999)}',
            f'150={exec_type}', f'39={ord_status}',
            f'55={symbol}', f'54={side}',
            f'38={qty}', f'14={filled}', f'151={qty - filled}',
            f'6={price}',
            f'60={self._generate_timestamp()}',
            f'10=000',
        ]

        return '|'.join(fields) + '|'
    
    def generate_market_data_request(self) -> str:
        """Generate a Market Data Request (V) message — FIX 5.0 SP2.

        Returns:
            Pipe-delimited FIX message string.
        """
        symbol = random.choice(self.symbols)

        fields = [
            f'8={self.FIX_VERSION}', f'9=160', '35=V',
            f'49=SENDER{random.randint(1,10)}',
            f'56=TARGET{random.randint(1,10)}',
            f'34={random.randint(1, 9999)}',
            f'52={self._generate_timestamp()}',
            f'1128={self.APPL_VER_ID}',
            f'262=MDREQ{random.randint(10000, 99999)}',
            '263=1', '264=0', '265=1',
            '267=2', '269=0', '269=1',
            '146=1', f'55={symbol}',
            f'10=000',
        ]

        return '|'.join(fields) + '|'

    def generate_logon(self) -> str:
        """Generate a Logon (A) message — FIXT 1.1.

        Returns:
            Pipe-delimited FIX message string.
        """
        fields = [
            f'8={self.FIX_VERSION}', f'9=100', '35=A',
            f'49=SENDER{random.randint(1,10)}',
            f'56=TARGET{random.randint(1,10)}',
            f'34=1',
            f'52={self._generate_timestamp()}',
            '98=0', f'108={random.choice([10, 15, 30, 60])}',
            f'1137={self.APPL_VER_ID}',
            f'553=user{random.randint(1,100)}',
            f'554=pass{random.randint(1000,9999)}',
            f'10=000',
        ]
        return '|'.join(fields) + '|'

    def generate_cancel_request(self) -> str:
        """Generate an Order Cancel Request (F) — FIX 5.0 SP2.

        Returns:
            Pipe-delimited FIX message string.
        """
        symbol = random.choice(self.symbols)
        fields = [
            f'8={self.FIX_VERSION}', f'9=150', '35=F',
            f'49=SENDER{random.randint(1,10)}',
            f'56=TARGET{random.randint(1,10)}',
            f'34={random.randint(1, 9999)}',
            f'52={self._generate_timestamp()}',
            f'1128={self.APPL_VER_ID}',
            f'11=CANCEL{random.randint(10000, 99999)}',
            f'41=ORD{random.randint(10000, 99999)}',
            f'37=EXEC{random.randint(10000, 99999)}',
            f'55={symbol}', f'54={random.choice(["1","2"])}',
            f'38={random.randint(1,1000)*100}',
            f'60={self._generate_timestamp()}',
            f'10=000',
        ]
        return '|'.join(fields) + '|'

    def _generate_timestamp(self) -> str:
        """Generate a random FIX-format UTC timestamp.

        Returns:
            ``YYYYMMDD-HH:MM:SS.000`` string.
        """
        return f'20260101-{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}.000'
    
    def generate_dataset(self, num_samples: int = 10000) -> List[str]:
        """Generate a dataset of FIX 5.0 SP2 messages.

        Args:
            num_samples: Total number of messages to generate.

        Returns:
            List of pipe-delimited FIX message strings.
        """
        messages = []
        generators = [
            (self.generate_new_order, 40),
            (self.generate_execution_report, 30),
            (self.generate_market_data_request, 15),
            (self.generate_logon, 5),
            (self.generate_cancel_request, 10),
        ]
        weights = [w for _, w in generators]
        funcs = [f for f, _ in generators]

        for _ in tqdm(range(num_samples), desc="Generating FIX 5.0 SP2 messages"):
            gen_func = random.choices(funcs, weights=weights, k=1)[0]
            messages.append(gen_func())

        return messages
    
    def generate_natural_language_pairs(self, num_samples: int = 5000) -> List[Tuple[str, str]]:
        """Generate (natural language, FIX message) training pairs.

        Uses template-based generation with parameter slots that are
        filled from the *same* random context used to build the FIX
        message, so the NL description and the FIX output are
        semantically aligned (e.g. "buy 200 AAPL" actually maps to a
        FIX message containing ``54=1|55=AAPL|38=200``).

        Covers all five message types with diverse phrasing styles
        (formal, casual, terse, verbose) to improve generalisation.

        Args:
            num_samples: Number of pairs to produce.

        Returns:
            List of ``(nl_text, fix_message)`` tuples.
        """
        pairs: List[Tuple[str, str]] = []

        # Build a reverse ticker→company list for NL generation
        ticker_to_companies: dict[str, list[str]] = {}
        for company, ticker in self._company_tickers.items():
            if company == ticker.lower():
                continue
            ticker_to_companies.setdefault(ticker, []).append(company)

        # ── Template bank (keyed by message type) ─────────────────
        # Each template is a format string with named placeholders.
        # {name}  = company name or ticker
        # {side}  = "buy" / "sell"
        # {SIDE}  = "Buy" / "Sell"
        # {qty}   = integer quantity
        # {price} = dollar price
        # {tif}   = time-in-force phrase
        # {hb}    = heartbeat interval

        _ORDER_TEMPLATES = [
            # Formal / full sentence
            "{SIDE} {qty} shares of {name}",
            "Create a {side} order for {qty} shares of {name}",
            "Place a {side} order for {name}, quantity {qty}",
            "I want to {side} {qty} {name} at limit price ${price}",
            "Submit a {side} order: {qty} {name} at ${price}",
            "Please {side} {qty} shares of {name} at market price",
            "Execute a {side} for {qty} shares of {name}",
            "I'd like to {side} {qty} {name}",
            "Can you {side} {qty} shares of {name} for me?",
            "Send a new order to {side} {qty} {name}",
            # Terse / command-style
            "{side} {qty} {name}",
            "{side} {name} {qty}",
            "{side} {name} {qty} shares",
            "{side} {qty} {name} at ${price}",
            "{side} {name} x{qty}",
            "new order {side} {qty} {name}",
            "order: {side} {qty} {name} @ ${price}",
            # Casual / conversational
            "get me {qty} shares of {name}",
            "i want {qty} {name}",
            "pick up {qty} shares of {name}",
            "grab {qty} {name} at ${price}",
            "let's {side} {qty} {name}",
            "go long {qty} {name}" if True else "",  # buy-only handled below
            "dump {qty} shares of {name}",
            # With time-in-force
            "{SIDE} {qty} {name}, {tif}",
            "{side} {qty} {name} {tif}",
            "Place a {tif} {side} order for {qty} {name}",
            "{side} {qty} shares of {name}, valid {tif}",
            # With order type detail
            "market order to {side} {qty} {name}",
            "limit order: {side} {qty} {name} at ${price}",
            "place a market {side} for {qty} {name}",
            "{side} {qty} {name} with a limit of ${price}",
            # Verbose / explanatory
            "I need to {side} {qty} shares of {name} at ${price} per share",
            "Please submit a limit order to {side} {qty} shares of {name} at ${price}",
            "Would you create a {side} order for {qty} units of {name}?",
            "Generate a FIX NewOrderSingle to {side} {qty} {name}",
            "Send a FIX 35=D message to {side} {qty} {name} at ${price}",
            "Create a NewOrderSingle: {side} {qty} {name}",
        ]

        _CANCEL_TEMPLATES = [
            "Cancel my order for {name}",
            "Cancel the order for {name}",
            "Please cancel my {name} order",
            "I want to cancel my order for {name}",
            "Revoke the open order for {name}",
            "Withdraw order for {name}",
            "Remove my pending order for {name}",
            "Can you cancel the {name} order?",
            "cancel {name}",
            "cancel order {name}",
            "kill the {name} order",
            "pull my order for {name}",
            "scratch the {name} trade",
            "abort the {SIDE} order for {qty} {name}",
            "Cancel my {side} order for {qty} {name}",
            "Generate a FIX OrderCancelRequest for {name}",
            "Send a 35=F to cancel {name}",
        ]

        _MARKET_DATA_TEMPLATES = [
            "Subscribe to market data for {name}",
            "Get market data for {name}",
            "Show me the market data for {name}",
            "I want real-time quotes for {name}",
            "Start streaming prices for {name}",
            "Request best bid and offer for {name}",
            "Give me live data on {name}",
            "market data {name}",
            "stream {name} quotes",
            "subscribe {name}",
            "md request for {name}",
            "get bbo for {name}",
            "Request a market data snapshot for {name}",
            "Subscribe to bid/offer updates for {name}",
            "I need a live price feed for {name}",
            "Generate a FIX MarketDataRequest for {name}",
            "Send a 35=V for {name}",
        ]

        _LOGON_TEMPLATES = [
            "Connect to FIX session with heartbeat {hb} seconds",
            "Initiate a FIX session, heartbeat {hb}s",
            "Start a FIX connection with {hb}-second heartbeat",
            "Log on to the FIX session",
            "Begin FIX session, heartbeat interval {hb}",
            "logon heartbeat {hb}",
            "connect to fix",
            "start session",
            "establish fix connection",
            "initiate logon",
            "Send a Logon message with heartbeat {hb}",
            "Create a FIX Logon with {hb}s heartbeat interval",
            "Generate a 35=A Logon message",
            "FIX logon, heartbeat={hb}",
        ]

        _EXEC_REPORT_TEMPLATES = [
            "Show an execution report for {qty} {name} filled at ${price}",
            "Generate an execution report: {qty} {name} filled at ${price}",
            "Execution report for order {name}, filled {qty} shares at ${price}",
            "Confirm fill: {qty} {name} at ${price}",
            "Trade confirmation for {qty} shares of {name}",
            "Report a fill of {qty} {name} at ${price} per share",
            "Create a FIX ExecutionReport for {name}",
            "Generate a 35=8 for {qty} {name}",
            "exec report {qty} {name} @ ${price}",
            "fill report for {name} {qty} shares",
        ]

        # ── Time-in-force phrases ─────────────────────────────────
        _TIF_PHRASES = {
            '0': "good for today",
            '1': "good till cancelled",
            '3': "immediate or cancel",
            '4': "fill or kill",
        }

        # ── Weighted distribution across message types ────────────
        _type_weights = [
            ("order", 40),
            ("cancel", 15),
            ("market_data", 15),
            ("logon", 10),
            ("exec_report", 20),
        ]
        _type_names = [t for t, _ in _type_weights]
        _type_w = [w for _, w in _type_weights]

        for _ in tqdm(range(num_samples), desc="Generating NL-FIX pairs"):
            symbol = random.choice(self.symbols)
            side_code = random.choice(['1', '2'])
            side = "buy" if side_code == '1' else "sell"
            qty = random.randint(1, 1000) * 100
            price = round(random.uniform(50, 500), 2)
            tif_code = random.choice(['0', '1', '3', '4'])
            tif = _TIF_PHRASES[tif_code]
            hb = random.choice([10, 15, 30, 60])

            # Pick a company name or ticker
            company_names = ticker_to_companies.get(symbol, [])
            use_company = company_names and random.random() < 0.5
            name = random.choice(company_names).title() if use_company else symbol

            fmt_ctx = {
                "name": name, "side": side, "SIDE": side.capitalize(),
                "qty": qty, "price": price, "tif": tif, "hb": hb,
            }

            msg_type = random.choices(_type_names, weights=_type_w, k=1)[0]

            if msg_type == "order":
                nl = random.choice(_ORDER_TEMPLATES).format(**fmt_ctx)
                fix_msg = self._generate_aligned_order(
                    symbol, side_code, qty, price, tif_code,
                )
            elif msg_type == "cancel":
                nl = random.choice(_CANCEL_TEMPLATES).format(**fmt_ctx)
                fix_msg = self._generate_aligned_cancel(symbol, side_code, qty)
            elif msg_type == "market_data":
                nl = random.choice(_MARKET_DATA_TEMPLATES).format(**fmt_ctx)
                fix_msg = self._generate_aligned_market_data(symbol)
            elif msg_type == "logon":
                nl = random.choice(_LOGON_TEMPLATES).format(**fmt_ctx)
                fix_msg = self._generate_aligned_logon(hb)
            else:  # exec_report
                nl = random.choice(_EXEC_REPORT_TEMPLATES).format(**fmt_ctx)
                fix_msg = self._generate_aligned_exec_report(
                    symbol, side_code, qty, price,
                )

            # Random casing variation (10% lowercase, 5% uppercase)
            r = random.random()
            if r < 0.10:
                nl = nl.lower()
            elif r < 0.15:
                nl = nl.upper()

            pairs.append((nl, fix_msg))

        return pairs

    # ── Aligned message generators (NL params → FIX fields) ───────

    def _generate_aligned_order(
        self, symbol: str, side: str, qty: int, price: float, tif: str,
    ) -> str:
        """Build a NewOrderSingle whose fields match the NL context."""
        ord_type = random.choice(['1', '2'])
        fields = [
            f'8={self.FIX_VERSION}', '9=200', '35=D',
            f'49=SENDER{random.randint(1, 10)}',
            f'56=TARGET{random.randint(1, 10)}',
            f'34={random.randint(1, 9999)}',
            f'52={self._generate_timestamp()}',
            f'1128={self.APPL_VER_ID}',
            f'11=ORD{random.randint(10000, 99999)}',
            '21=1', f'55={symbol}', f'54={side}',
            f'38={qty}', f'40={ord_type}',
        ]
        if ord_type == '2':
            fields.append(f'44={price}')
        fields.extend([
            f'59={tif}',
            f'60={self._generate_timestamp()}',
            '10=000',
        ])
        return '|'.join(fields) + '|'

    def _generate_aligned_cancel(
        self, symbol: str, side: str, qty: int,
    ) -> str:
        """Build an OrderCancelRequest whose fields match the NL context."""
        fields = [
            f'8={self.FIX_VERSION}', '9=150', '35=F',
            f'49=SENDER{random.randint(1, 10)}',
            f'56=TARGET{random.randint(1, 10)}',
            f'34={random.randint(1, 9999)}',
            f'52={self._generate_timestamp()}',
            f'1128={self.APPL_VER_ID}',
            f'11=CANCEL{random.randint(10000, 99999)}',
            f'41=ORD{random.randint(10000, 99999)}',
            f'37=EXEC{random.randint(10000, 99999)}',
            f'55={symbol}', f'54={side}',
            f'38={qty}',
            f'60={self._generate_timestamp()}',
            '10=000',
        ]
        return '|'.join(fields) + '|'

    def _generate_aligned_market_data(self, symbol: str) -> str:
        """Build a MarketDataRequest whose symbol matches the NL context."""
        fields = [
            f'8={self.FIX_VERSION}', '9=160', '35=V',
            f'49=SENDER{random.randint(1, 10)}',
            f'56=TARGET{random.randint(1, 10)}',
            f'34={random.randint(1, 9999)}',
            f'52={self._generate_timestamp()}',
            f'1128={self.APPL_VER_ID}',
            f'262=MDREQ{random.randint(10000, 99999)}',
            '263=1', '264=0', '265=1',
            '267=2', '269=0', '269=1',
            '146=1', f'55={symbol}',
            '10=000',
        ]
        return '|'.join(fields) + '|'

    def _generate_aligned_logon(self, heartbeat: int) -> str:
        """Build a Logon whose heartbeat matches the NL context."""
        fields = [
            f'8={self.FIX_VERSION}', '9=100', '35=A',
            f'49=SENDER{random.randint(1, 10)}',
            f'56=TARGET{random.randint(1, 10)}',
            '34=1',
            f'52={self._generate_timestamp()}',
            '98=0', f'108={heartbeat}',
            f'1137={self.APPL_VER_ID}',
            f'553=user{random.randint(1, 100)}',
            f'554=pass{random.randint(1000, 9999)}',
            '10=000',
        ]
        return '|'.join(fields) + '|'

    def _generate_aligned_exec_report(
        self, symbol: str, side: str, qty: int, price: float,
    ) -> str:
        """Build an ExecutionReport whose fields match the NL context."""
        fields = [
            f'8={self.FIX_VERSION}', '9=280', '35=8',
            f'49=TARGET{random.randint(1, 10)}',
            f'56=SENDER{random.randint(1, 10)}',
            f'34={random.randint(1, 9999)}',
            f'52={self._generate_timestamp()}',
            f'1128={self.APPL_VER_ID}',
            f'37=EXEC{random.randint(10000, 99999)}',
            f'11=ORD{random.randint(10000, 99999)}',
            f'17=EXEC{random.randint(10000, 99999)}',
            '150=F', '39=2',
            f'55={symbol}', f'54={side}',
            f'38={qty}', f'14={qty}', '151=0',
            f'6={price}', f'31={price}', f'32={qty}',
            f'60={self._generate_timestamp()}',
            '10=000',
        ]
        return '|'.join(fields) + '|'


def _classify_message_type(text: str) -> str:
    """Classify a training line by its FIX MsgType or source.

    Returns a bucket key such as ``"D"`` (NewOrderSingle), ``"8"``
    (ExecutionReport), ``"nl_pair"``, ``"scraped"``, or ``"other"``.

    Args:
        text: A single training text line.

    Returns:
        String key identifying the message category.
    """
    import re as _re

    # NL+FIX pair (has newline separator)
    if "\n" in text and "35=" in text:
        return "nl_pair"

    # Extract MsgType tag
    match = _re.search(r'(?:^|\|)35=([A-Za-z0-9]+)', text)
    if match:
        return f"fix_{match.group(1)}"

    # Scraped / spec text
    if text.startswith("FIX") or "field" in text.lower():
        return "spec_text"

    return "other"


def _stratified_split(
    texts: List[str],
    train_ratio: float = 0.9,
) -> Tuple[List[str], List[str]]:
    """Split *texts* into train/val sets stratified by message type.

    Each message-type bucket is shuffled independently and then split
    at the *train_ratio* boundary.  This ensures that every message
    category appears proportionally in both train and val sets.

    Args:
        texts: All training examples.
        train_ratio: Proportion allocated to training.

    Returns:
        ``(train_texts, val_texts)`` tuple.
    """
    from collections import defaultdict

    buckets: dict[str, List[str]] = defaultdict(list)
    for t in texts:
        key = _classify_message_type(t)
        buckets[key].append(t)

    train_texts: List[str] = []
    val_texts: List[str] = []

    for key, items in buckets.items():
        random.shuffle(items)
        split_idx = max(1, int(len(items) * train_ratio))
        train_texts.extend(items[:split_idx])
        val_texts.extend(items[split_idx:])

    # Final shuffle within each set
    random.shuffle(train_texts)
    random.shuffle(val_texts)

    print(f"Stratified split: {len(buckets)} buckets — "
          + ", ".join(f"{k}={len(v)}" for k, v in sorted(buckets.items())))

    return train_texts, val_texts


def prepare_training_data(
    output_dir: str | None = None,
    num_fix_messages: int = 50000,
    num_nl_pairs: int = 10000,
    train_split: float = 0.9,
    stratify: bool = True,
):
    """
    Prepare training data for FixProtoGPT
    
    Args:
        output_dir: Directory to save processed data
        num_fix_messages: Number of FIX messages to generate
        num_nl_pairs: Number of natural language pairs to generate
        train_split: Train/validation split ratio
        stratify: If True, stratify train/val split by message type so
            every message category is represented proportionally.
    """
    if output_dir is None:
        output_dir = str(paths.processed_data_dir())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating training data...")
    
    # Load scraped FIX specifications if available
    scraped_data = []
    scraped_file = paths.raw_data_dir() / 'scraped_training.txt'
    if scraped_file.exists():
        print("Loading scraped FIX specifications...")
        with open(scraped_file, 'r') as f:
            scraped_data = [line.strip() for line in f if line.strip()]
        print(f"  ✓ Loaded {len(scraped_data)} scraped examples")

    # Load ingested canonical specs (PDF / DOCX / XML / CSV uploads)
    ingested_lines: list[str] = []
    try:
        from src.data.ingest.normalizer import load_canonical, specs_to_training_lines
        canonical = load_canonical()
        if canonical:
            ingested_lines = specs_to_training_lines(canonical)
            print(f"  ✓ Loaded {len(ingested_lines)} ingested spec lines from canonical.json")
    except Exception as exc:
        print(f"  ⚠ Could not load ingested specs: {exc}")
    
    # Generate data
    generator = FIXDataGenerator()
    
    # Generate FIX messages
    fix_messages = generator.generate_dataset(num_fix_messages)
    
    # Generate NL-FIX pairs
    nl_pairs = generator.generate_natural_language_pairs(num_nl_pairs)
    
    # Combine all texts (scraped + ingested + synthetic + NL pairs)
    all_texts = scraped_data + ingested_lines + fix_messages
    for nl, fix in nl_pairs:
        all_texts.append(f"{nl}\n{fix}")

    # ── Stratified split by message type ─────────────────────────
    if stratify:
        train_texts, val_texts = _stratified_split(all_texts, train_split)
    else:
        # Legacy random split
        random.shuffle(all_texts)
        split_idx = int(len(all_texts) * train_split)
        train_texts = all_texts[:split_idx]
        val_texts = all_texts[split_idx:]
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Val samples: {len(val_texts)}")
    
    # Save raw texts
    with open(output_path / 'train.txt', 'w') as f:
        f.write('\n'.join(train_texts))
    
    with open(output_path / 'val.txt', 'w') as f:
        f.write('\n'.join(val_texts))
    
    print("\nBuilding tokenizer...")
    
    # Build tokenizer
    tokenizer = FixProtocolTokenizer(vocab_size=1024)
    tokenizer.build_vocab(train_texts[:10000])  # Use subset for vocab building
    tokenizer.save(output_path / 'tokenizer')
    
    print("\nTokenizing data...")
    
    # Tokenize and save as binary
    def tokenize_and_save(texts: List[str], filename: str):
        """Encode a list of texts into token IDs and write them as a uint16 binary file."""
        all_tokens = []
        for text in tqdm(texts, desc=f"Tokenizing {filename}"):
            tokens = tokenizer.encode(text, add_special_tokens=True)
            all_tokens.extend(tokens)
        
        # Save as binary
        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(output_path / filename)
        print(f"Saved {len(arr)} tokens to {filename}")
    
    tokenize_and_save(train_texts, 'train.bin')
    tokenize_and_save(val_texts, 'val.bin')
    
    # Save metadata
    metadata = {
        'num_fix_messages': num_fix_messages,
        'num_nl_pairs': num_nl_pairs,
        'train_samples': len(train_texts),
        'val_samples': len(val_texts),
        'vocab_size': len(tokenizer.token_to_id),
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nData preparation complete!")
    print(f"Files saved to: {output_path}")


if __name__ == "__main__":
    prepare_training_data()
