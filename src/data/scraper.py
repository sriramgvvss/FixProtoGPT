"""
Module: src.data.scraper
=========================

FIX Protocol specification scraper and training-data generator.

This module provides :class:`FIXProtocolScraper`, which combines web
scraping of fixtrading.org with the built-in knowledge base
(:mod:`src.core.fix_reference`) to build complete specification
bundles and generate synthetic training data for FixProtoGPT.

Coding Standards
----------------
- PEP 8  : Python Style Guide — naming, spacing, line length ≤ 120
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_WEB = True
except ImportError:
    HAS_WEB = False

from src.utils import paths
from src.core.fix_reference import (
    MESSAGE_TYPES,
    FIELDS,
    ENUMERATIONS,
    DATA_TYPES,
    COMPONENTS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FIXProtocolScraper:
    """Comprehensive FIX Protocol scraper focused on the latest specification.

    Combines web scraping with the built-in knowledge base for FIX 5.0 SP2
    (the latest stable FIX version).

    Class-level references to the knowledge-base dicts are kept for backward
    compatibility — legacy callers that access ``FIXProtocolScraper.FIELDS``
    (e.g. the inference explainer) still work without changes.

    Attributes:
        FIX_VERSION:  Active protocol string (e.g. ``"FIX.5.0SP2"``).
        FIXT_VERSION: Session-layer protocol (e.g. ``"FIXT.1.1"``).
        output_dir:   Directory for saving generated artifacts.
    """

    # Backward-compatible class-level aliases to fix_reference constants
    MESSAGE_TYPES = MESSAGE_TYPES
    FIELDS = FIELDS
    ENUMERATIONS = ENUMERATIONS
    DATA_TYPES = DATA_TYPES
    COMPONENTS = COMPONENTS

    FIX_VERSION: Optional[str] = None
    FIXT_VERSION: Optional[str] = None

    # FIX Trading Community URLs
    BASE_URLS = [
        "https://www.fixtrading.org/standards/fix-5-0-sp2/",
        "https://www.fixtrading.org/online-specification/fix50sp2/",
        "https://fiximate.fixtrading.org/",
    ]

    # ── Lifecycle ─────────────────────────────────────────────────────

    def __init__(self, output_dir: str | None = None) -> None:
        """Initialise the scraper.

        Args:
            output_dir: Override directory for output artifacts.
                        Defaults to the project's raw-data directory.
        """
        self.FIX_VERSION = paths.active_protocol()
        self.FIXT_VERSION = paths.session_protocol()
        self.output_dir = Path(output_dir) if output_dir else paths.raw_data_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scraped_web_data: Dict[str, Any] = {}
        self.session = None

    # ── Web scraping ──────────────────────────────────────────────────

    def _init_session(self) -> bool:
        """Initialise an HTTP session for web scraping.

        Returns:
            ``True`` if *requests* / *beautifulsoup4* are available.
        """
        if not HAS_WEB:
            logger.warning("requests/beautifulsoup4 not installed. Using built-in data only.")
            return False
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "FixProtoGPT-Scraper/2.0 (Research/Training)",
            "Accept": "text/html,application/xhtml+xml,application/json",
        })
        return True

    def fetch_url(self, url: str, retries: int = 3) -> Optional[str]:
        """Fetch URL content with exponential-backoff retries.

        Args:
            url:     Target URL.
            retries: Maximum number of attempts.

        Returns:
            Response body text, or ``None`` on failure.
        """
        if not self.session:
            return None
        for attempt in range(retries):
            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                time.sleep(1)  # rate limiting
                return resp.text
            except Exception as exc:
                logger.warning("Attempt %d/%d failed for %s: %s", attempt + 1, retries, url, exc)
                time.sleep(2 ** attempt)
        return None

    def scrape_web_specs(self) -> Dict[str, Any]:
        """Attempt to scrape the latest specs from fixtrading.org.

        Returns:
            Dict with ``messages``, ``fields``, and ``source_urls`` keys.
        """
        if not self._init_session():
            logger.info("Web scraping unavailable, using built-in specification data")
            return {}

        web_data: Dict[str, Any] = {"messages": [], "fields": [], "source_urls": []}

        for url in self.BASE_URLS:
            logger.info("Trying %s...", url)
            html = self.fetch_url(url)
            if not html:
                continue

            web_data["source_urls"].append(url)
            soup = BeautifulSoup(html, "lxml")

            # Extract table data
            for table in soup.find_all("table"):
                rows = table.find_all("tr")
                if len(rows) < 2:
                    continue
                headers = [th.get_text(strip=True).lower() for th in rows[0].find_all(["th", "td"])]
                for row in rows[1:]:
                    cells = [td.get_text(strip=True) for td in row.find_all("td")]
                    if len(cells) >= 2:
                        entry = dict(zip(headers, cells))
                        if any(k in headers for k in ["msgtype", "message", "msg type"]):
                            web_data["messages"].append(entry)
                        elif any(k in headers for k in ["tag", "field", "name"]):
                            web_data["fields"].append(entry)

            # Follow sub-links for more spec pages
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if any(kw in href.lower() for kw in ["message", "field", "component", "datatype"]):
                    full_url = href if href.startswith("http") else f"https://www.fixtrading.org{href}"
                    sub_html = self.fetch_url(full_url)
                    if sub_html:
                        sub_soup = BeautifulSoup(sub_html, "lxml")
                        for sub_table in sub_soup.find_all("table"):
                            sub_rows = sub_table.find_all("tr")
                            if len(sub_rows) >= 2:
                                sub_headers = [
                                    th.get_text(strip=True).lower()
                                    for th in sub_rows[0].find_all(["th", "td"])
                                ]
                                for sub_row in sub_rows[1:]:
                                    sub_cells = [td.get_text(strip=True) for td in sub_row.find_all("td")]
                                    if len(sub_cells) >= 2:
                                        web_data["fields"].append(dict(zip(sub_headers, sub_cells)))

        if web_data["messages"] or web_data["fields"]:
            logger.info(
                "Scraped %d messages, %d fields from web",
                len(web_data["messages"]),
                len(web_data["fields"]),
            )
        else:
            logger.info("No additional data scraped from web. Built-in data covers FIX 5.0 SP2 completely.")

        self.scraped_web_data = web_data
        return web_data

    # ── Specification building ────────────────────────────────────────

    def build_specification(self) -> Dict[str, Any]:
        """Build the complete FIX 5.0 SP2 specification dictionary.

        Returns:
            Nested dict suitable for JSON serialisation.
        """
        spec: Dict[str, Any] = {
            "version": self.FIX_VERSION,
            "transport_version": self.FIXT_VERSION,
            "scraped_at": datetime.now().isoformat(),
            "message_types": MESSAGE_TYPES,
            "fields": {str(k): v for k, v in FIELDS.items()},
            "enumerations": ENUMERATIONS,
            "data_types": DATA_TYPES,
            "components": COMPONENTS,
            "statistics": {
                "total_message_types": len(MESSAGE_TYPES),
                "total_fields": len(FIELDS),
                "total_enumerations": sum(len(v) for v in ENUMERATIONS.values()),
                "total_data_types": len(DATA_TYPES),
                "total_components": len(COMPONENTS),
            },
        }
        if self.scraped_web_data:
            spec["web_scraped_data"] = self.scraped_web_data
        return spec

    # ── Training data generation ──────────────────────────────────────

    def generate_training_data(self) -> List[str]:
        """Generate comprehensive training text from the specification.

        Returns:
            List of training lines (NL descriptions + FIX messages).
        """
        lines: List[str] = []

        # Header
        lines.append(f"FIX Protocol Version: {self.FIX_VERSION} with transport {self.FIXT_VERSION}")
        lines.append("FIX 5.0 SP2 is the latest stable version of the FIX protocol specification.")
        lines.append("")

        # Message types
        for code, info in MESSAGE_TYPES.items():
            lines.append(
                f"FIX Message Type {code}: {info['name']} ({info['category']}). {info['description']}"
            )
        lines.append("")

        # Fields
        for tag, info in sorted(FIELDS.items()):
            lines.append(f"FIX Field {tag}: {info['name']} ({info['type']}). {info['description']}")
        lines.append("")

        # Enumerations
        for field_label, values in ENUMERATIONS.items():
            for val, meaning in values.items():
                lines.append(f"{field_label} value {val}: {meaning}")
        lines.append("")

        # Data types
        for dtype, info in DATA_TYPES.items():
            lines.append(f"FIX Data Type {dtype}: {info['description']}")
        lines.append("")

        # Components
        for comp_name, comp_info in COMPONENTS.items():
            field_names = [FIELDS.get(fid, {}).get("name", str(fid)) for fid in comp_info["fields"]]
            lines.append(
                f"FIX Component {comp_name}: {comp_info['description']}. Fields: {', '.join(field_names)}"
            )
        lines.append("")

        # Example messages
        examples = self._generate_example_messages()
        for desc, msg in examples:
            lines.append(desc)
            lines.append(msg)

        # Relationships & context
        lines.extend(self._generate_relationship_text())
        return lines

    def _generate_example_messages(self) -> List[tuple]:
        """Generate example FIX 5.0 SP2 messages with NL descriptions.

        Returns:
            List of ``(description, fix_message)`` tuples.
        """
        import random
        random.seed(42)

        # Pull symbols from the shared training-symbols helper
        from src.data.symbol_resolver import get_training_symbols
        symbols = get_training_symbols(max_symbols=40)
        examples: List[tuple] = []

        for symbol in symbols:
            qty = random.randint(1, 1000) * 100
            price = round(random.uniform(50, 500), 2)
            for side_code, side_name in [("1", "Buy"), ("2", "Sell")]:
                for ord_type_code, ord_name in [("1", "market"), ("2", "limit")]:
                    desc = (
                        f"{side_name} {qty} shares of {symbol} at {ord_name} price"
                        + (f" ${price}" if ord_type_code == "2" else "")
                    )
                    fields = [
                        "8=FIXT.1.1", "9=200", "35=D",
                        "49=SENDER01", "56=TARGET01",
                        f"34={random.randint(1, 9999)}",
                        f"52={datetime.now().strftime('%Y%m%d-%H:%M:%S.000')}",
                        "1128=9",
                        f"11=ORD{random.randint(10000, 99999)}",
                        "21=1", f"55={symbol}", f"54={side_code}",
                        f"38={qty}", f"40={ord_type_code}",
                    ]
                    if ord_type_code == "2":
                        fields.append(f"44={price}")
                    fields.extend([
                        "59=0",
                        f"60={datetime.now().strftime('%Y%m%d-%H:%M:%S.000')}",
                        "10=000",
                    ])
                    examples.append((desc, "|".join(fields) + "|"))

            # ExecutionReport
            qty2 = random.randint(1, 500) * 100
            price2 = round(random.uniform(50, 500), 2)
            desc = f"Execution report for filled order of {qty2} {symbol} shares at ${price2}"
            fields = [
                "8=FIXT.1.1", "9=280", "35=8",
                "49=TARGET01", "56=SENDER01",
                f"34={random.randint(1, 9999)}",
                f"52={datetime.now().strftime('%Y%m%d-%H:%M:%S.000')}",
                "1128=9",
                f"37=EXEC{random.randint(10000, 99999)}",
                f"11=ORD{random.randint(10000, 99999)}",
                f"17=EXEC{random.randint(10000, 99999)}",
                "150=F", "39=2", f"55={symbol}", "54=1",
                f"38={qty2}", f"14={qty2}", "151=0",
                f"6={price2}", f"31={price2}", f"32={qty2}",
                f"60={datetime.now().strftime('%Y%m%d-%H:%M:%S.000')}",
                "10=000",
            ]
            examples.append((desc, "|".join(fields) + "|"))

        # MarketDataRequest
        for symbol in symbols[:5]:
            desc = f"Market data subscription request for {symbol}"
            fields = [
                "8=FIXT.1.1", "9=160", "35=V",
                "49=SENDER01", "56=TARGET01",
                f"34={random.randint(1, 9999)}",
                f"52={datetime.now().strftime('%Y%m%d-%H:%M:%S.000')}",
                "1128=9",
                f"262=MDREQ{random.randint(10000, 99999)}",
                "263=1", "264=0", "265=1",
                "267=2", "269=0", "269=1",
                f"146=1", f"55={symbol}",
                "10=000",
            ]
            examples.append((desc, "|".join(fields) + "|"))

        # Logon
        examples.append((
            "Logon message to initiate FIX session with heartbeat interval 30 seconds",
            "|".join([
                "8=FIXT.1.1", "9=100", "35=A",
                "49=SENDER01", "56=TARGET01", "34=1",
                f"52={datetime.now().strftime('%Y%m%d-%H:%M:%S.000')}",
                "98=0", "108=30", "1137=9",
                "553=user1", "554=pass123", "10=000",
            ]) + "|",
        ))

        # Heartbeat
        examples.append((
            "Heartbeat message to confirm connection is alive",
            "|".join([
                "8=FIXT.1.1", "9=60", "35=0",
                "49=SENDER01", "56=TARGET01",
                f"34={random.randint(1, 9999)}",
                f"52={datetime.now().strftime('%Y%m%d-%H:%M:%S.000')}",
                "10=000",
            ]) + "|",
        ))

        # OrderCancelRequest
        examples.append((
            "Cancel an existing order for AAPL",
            "|".join([
                "8=FIXT.1.1", "9=150", "35=F",
                "49=SENDER01", "56=TARGET01",
                f"34={random.randint(1, 9999)}",
                f"52={datetime.now().strftime('%Y%m%d-%H:%M:%S.000')}",
                "1128=9",
                f"11=CANCEL{random.randint(10000, 99999)}",
                f"41=ORD{random.randint(10000, 99999)}",
                f"37=EXEC{random.randint(10000, 99999)}",
                "55=AAPL", "54=1", "38=100",
                f"60={datetime.now().strftime('%Y%m%d-%H:%M:%S.000')}",
                "10=000",
            ]) + "|",
        ))

        return examples

    @staticmethod
    def _generate_relationship_text() -> List[str]:
        """Generate contextual / relational training text.

        Returns:
            List of lines covering FIX concepts, structure, lifecycle, etc.
        """
        return [
            "",
            "# FIX Protocol Concepts",
            "The FIX (Financial Information eXchange) protocol is an electronic communications "
            "protocol for international real-time exchange of securities transaction information.",
            "FIX 5.0 SP2 is the latest stable version. It uses FIXT 1.1 as the session layer "
            "protocol, separating application and session concerns.",
            "In FIX 5.0, the session protocol (FIXT.1.1) is decoupled from the application "
            "protocol (FIX 5.0 SP2). BeginString(8) is always FIXT.1.1, and ApplVerID(1128) "
            "indicates the application version.",
            "",
            "# Message Structure",
            "Every FIX message consists of: StandardHeader + Body + StandardTrailer.",
            "The StandardHeader always begins with BeginString(8), BodyLength(9), MsgType(35).",
            "The StandardTrailer always ends with CheckSum(10).",
            "Fields are delimited by SOH (ASCII 01) in production, often shown as | in documentation.",
            "",
            "# Session Layer (FIXT 1.1)",
            "Session messages handle connection management: Logon(A), Logout(5), Heartbeat(0), "
            "TestRequest(1), ResendRequest(2), SequenceReset(4), Reject(3).",
            "Each side maintains an independent outgoing sequence number, starting at 1.",
            "The Logon message must be the first message sent by each side after TCP connection.",
            "",
            "# Order Lifecycle",
            "New orders are submitted via NewOrderSingle(D). The sell-side responds with ExecutionReport(8).",
            "Orders can be modified via OrderCancelReplaceRequest(G) or canceled via OrderCancelRequest(F).",
            "If a cancel or modify is rejected, OrderCancelReject(9) is returned.",
            "The ExecType(150) field in ExecutionReport indicates what happened: "
            "F=Trade, 0=New, 4=Canceled, 8=Rejected.",
            "OrdStatus(39) tracks the cumulative state: "
            "0=New, 1=PartiallyFilled, 2=Filled, 4=Canceled, 8=Rejected.",
            "",
            "# Market Data",
            "MarketDataRequest(V) subscribes to market data. "
            "SubscriptionRequestType(263): 0=Snapshot, 1=SnapshotAndUpdates.",
            "MarketDataSnapshotFullRefresh(W) provides a complete book. "
            "MarketDataIncrementalRefresh(X) provides updates.",
            "MDEntryType(269): 0=Bid, 1=Offer, 2=Trade, 4=OpeningPrice, 5=ClosingPrice.",
            "",
            "# Key Differences: FIX 4.x vs FIX 5.0 SP2",
            "FIX 4.x uses BeginString like 'FIX.4.2'. FIX 5.0 SP2 uses BeginString 'FIXT.1.1' "
            "with ApplVerID(1128)=9.",
            "FIX 5.0 introduced the Parties component block for identifying counterparties.",
            "ExecTransType(20) was deprecated in FIX 5.0. ExecType(150) values were rationalized.",
            "FIX 5.0 SP2 added enhanced derivatives support, strategy parameters, and improved market data.",
            "",
        ]

    # ── Persistence helpers ───────────────────────────────────────────

    def save_specification(self, spec: Dict[str, Any]) -> None:
        """Save the specification as a JSON file.

        Args:
            spec: Specification dict produced by :meth:`build_specification`.
        """
        spec_file = self.output_dir / "fix_latest_specification.json"
        with open(spec_file, "w") as fh:
            json.dump(spec, fh, indent=2)
        logger.info("Specification saved to %s", spec_file)

    def save_training_data(self, lines: List[str]) -> None:
        """Save training data as a plain-text file.

        Args:
            lines: Training lines produced by :meth:`generate_training_data`.
        """
        training_file = self.output_dir / "scraped_training.txt"
        with open(training_file, "w") as fh:
            fh.write("\n".join(lines))
        logger.info("Training data saved to %s (%d lines)", training_file, len(lines))

    def save_markdown_reference(self, spec: Dict[str, Any]) -> None:
        """Save a comprehensive Markdown reference document.

        Args:
            spec: Specification dict (metadata only; data from knowledge base).
        """
        docs_dir = Path("docs")
        docs_dir.mkdir(parents=True, exist_ok=True)

        md: List[str] = [
            f"# FIX {self.FIX_VERSION} Protocol Reference",
            "",
            f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            f"Transport Protocol: **{self.FIXT_VERSION}**",
            "", "---", "",
            f"## Message Types ({len(MESSAGE_TYPES)} total)",
            "",
            "| Code | Name | Category | Description |",
            "|------|------|----------|-------------|",
        ]
        for code, info in sorted(MESSAGE_TYPES.items(), key=lambda x: x[0]):
            md.append(f"| {code} | {info['name']} | {info['category']} | {info['description']} |")

        md.extend(["", "---", "", f"## Fields ({len(FIELDS)} total)", "",
                    "| Tag | Name | Type | Description |",
                    "|-----|------|------|-------------|"])
        for tag, info in sorted(FIELDS.items()):
            md.append(f"| {tag} | {info['name']} | {info['type']} | {info['description']} |")

        md.extend(["", "---", "", "## Enumerations", ""])
        for field_label, values in sorted(ENUMERATIONS.items()):
            md.append(f"### {field_label}")
            md.extend(["", "| Value | Meaning |", "|-------|---------|"])
            for val, meaning in sorted(values.items()):
                md.append(f"| {val} | {meaning} |")
            md.append("")

        md.extend(["---", "", f"## Data Types ({len(DATA_TYPES)} total)", "",
                    "| Type | Description |", "|------|-------------|"])
        for dtype, info in sorted(DATA_TYPES.items()):
            md.append(f"| {dtype} | {info['description']} |")

        md.extend(["", "---", "", f"## Components ({len(COMPONENTS)} total)", ""])
        for comp_name, comp_info in sorted(COMPONENTS.items()):
            field_names = [FIELDS.get(fid, {}).get("name", str(fid)) for fid in comp_info["fields"]]
            md.extend([
                f"### {comp_name}", "",
                comp_info["description"], "",
                f"**Fields:** {', '.join(field_names)}", "",
            ])

        with open(docs_dir / "FIX_SPECIFICATION_REFERENCE.md", "w") as fh:
            fh.write("\n".join(md))
        logger.info("Markdown reference saved to docs/FIX_SPECIFICATION_REFERENCE.md")

    def save_version_info(self) -> None:
        """Update the version tracking JSON file."""
        version_file = self.output_dir / "fix_versions.json"
        version_data = {
            "detected_versions": [self.FIX_VERSION],
            "primary_version": self.FIX_VERSION,
            "transport_version": self.FIXT_VERSION,
            "updated_at": datetime.now().isoformat(),
            "version_features": {
                self.FIX_VERSION: {
                    "message_types": len(MESSAGE_TYPES),
                    "fields": len(FIELDS),
                    "enumerations": sum(len(v) for v in ENUMERATIONS.values()),
                    "components": len(COMPONENTS),
                },
            },
        }
        with open(version_file, "w") as fh:
            json.dump(version_data, fh, indent=2)
        logger.info("Version info saved to %s", version_file)

    # ── Pipeline entrypoint ───────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Run the complete scraping and data-generation pipeline.

        Steps:
            1. Web-scrape for supplementary data.
            2. Build complete specification dict.
            3. Generate training data.
            4. Generate Markdown docs.
            5. Update version metadata.

        Returns:
            The compiled specification dict.
        """
        print("=" * 70)
        print(f"FIX Protocol Scraper — {self.FIX_VERSION}")
        print("=" * 70)

        print("\n[1/5] Scraping web for latest specifications...")
        self.scrape_web_specs()

        print("\n[2/5] Building complete specification...")
        spec = self.build_specification()
        self.save_specification(spec)

        print("\n[3/5] Generating training data...")
        training_lines = self.generate_training_data()
        self.save_training_data(training_lines)

        print("\n[4/5] Generating documentation...")
        self.save_markdown_reference(spec)

        print("\n[5/5] Updating version metadata...")
        self.save_version_info()

        # Summary
        print("\n" + "=" * 70)
        print("SCRAPING COMPLETE")
        print("=" * 70)
        print(f"  Version:       {self.FIX_VERSION} ({self.FIXT_VERSION})")
        print(f"  Messages:      {len(MESSAGE_TYPES)}")
        print(f"  Fields:        {len(FIELDS)}")
        print(f"  Enumerations:  {sum(len(v) for v in ENUMERATIONS.values())}")
        print(f"  Data Types:    {len(DATA_TYPES)}")
        print(f"  Components:    {len(COMPONENTS)}")
        print(f"  Training lines: {len(training_lines)}")
        print(f"  Output dir:    {self.output_dir}")
        print("=" * 70)

        return spec


if __name__ == "__main__":
    scraper = FIXProtocolScraper()
    scraper.run()
