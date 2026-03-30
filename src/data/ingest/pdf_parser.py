"""
Module: src.data.ingest.pdf_parser
====================================

Extract FIX protocol specification data from PDF documents.

Uses ``pdfplumber`` for text extraction and applies regex-based
heuristics to identify FIX field definitions, message layouts,
and enumeration tables.

If ``pdfplumber`` is not installed the parser is still registered but
will raise a clear ``ImportError`` at parse-time.

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from src.data.ingest.base import (
    CanonicalSpec,
    SpecKind,
    SpecParser,
    register_parser,
)

logger = logging.getLogger(__name__)

# ── Regex patterns for FIX spec elements ──────────────────────────

# Tag=Name (e.g. "35=MsgType", "55=Symbol") — common in overview docs
_TAG_DEF_RE = re.compile(
    r"(?:Tag|Field)\s*[:=]?\s*(\d{1,5})\s*[-–—=:]\s*([A-Za-z]\w+)",
    re.IGNORECASE,
)

# Free-text field definition as found in Vol-6 data dictionaries:
# "<tag_number> <FieldName> <DataType> <description...>"
# e.g. "7 BeginSeqNo SeqNum Message sequence number of ..."
_FIELD_LINE_RE = re.compile(
    r"^(\d{1,5})\s+"                        # tag number
    r"([A-Z][A-Za-z0-9]{2,40})\s+"          # FieldName (PascalCase)
    r"(String|Int|Qty|Price|Amt|Float|char"  # DataType (known FIX types)
    r"|Boolean|Length|SeqNum|NumInGroup"
    r"|MultipleCharValue|MultipleStringValue"
    r"|Country|Currency|Exchange|LocalMktDate"
    r"|MonthYear|UTCTimestamp|UTCTimeOnly"
    r"|UTCDateOnly|TZTimestamp|TZTimeOnly"
    r"|data|Percentage|PriceOffset"
    r"|TagNum|XMLData|Language|Pattern"
    r"|Tenor|Reserved\d+|LocalMkt\s*Date"
    r"|[A-Z][a-zA-Z]+)"                     # fallback: any PascalCase word
    r"\s+(.+)",                              # description
    re.MULTILINE,
)

# MsgType line (e.g. "MsgType = D (NewOrderSingle)")
_MSG_TYPE_RE = re.compile(
    r"MsgType\s*=\s*([A-Za-z0-9]+)\s*\(?\s*([A-Za-z]\w+)\s*\)?",
    re.IGNORECASE,
)

# Message definition headers (e.g. "New Order – Single (MsgType = D)")
_MSG_HEADER_RE = re.compile(
    r"([A-Z][\w\s\-–—]+?)\s*\(\s*MsgType\s*=\s*['\"]?([A-Za-z0-9]+)['\"]?\s*\)",
    re.IGNORECASE,
)

# Enum value line (e.g. "1 = Buy, 2 = Sell")
_ENUM_VALUE_RE = re.compile(
    r"(\w{1,10})\s*[=–—:]\s*([A-Za-z][\w\s]{0,60})",
)

# Data type line (e.g. "Type: STRING", "Data Type: INT")
_DATA_TYPE_RE = re.compile(
    r"(?:Data\s*)?Type\s*[:=]\s*([A-Z][A-Z_0-9]+)",
    re.IGNORECASE,
)


@register_parser
class PDFParser(SpecParser):
    """Extract FIX specs from PDF documents using ``pdfplumber``."""

    EXTENSIONS = {".pdf"}

    def can_handle(self, path: Path) -> bool:
        """Return ``True`` for ``.pdf`` files."""
        return path.suffix.lower() in self.EXTENSIONS

    def parse(self, path: Path) -> List[CanonicalSpec]:
        """Parse a PDF and extract FIX specification records.

        Args:
            path: Path to the PDF file.

        Returns:
            List of :class:`CanonicalSpec` records.

        Raises:
            ImportError: If ``pdfplumber`` is not installed.
            FileNotFoundError: If *path* does not exist.
        """
        try:
            import pdfplumber
        except ImportError as exc:
            raise ImportError(
                "pdfplumber is required for PDF parsing. "
                "Install it with: pip install pdfplumber"
            ) from exc

        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        specs: List[CanonicalSpec] = []
        logger.info("Parsing PDF spec: %s", path.name)

        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue

                # --- Tables first (richer structure) ---
                tables = page.extract_tables() or []
                for table in tables:
                    specs.extend(
                        self._parse_table(table, page_num, path.name)
                    )

                # --- Free-text fallback ---
                specs.extend(
                    self._parse_free_text(text, page_num, path.name)
                )

        logger.info(
            "PDF %s → %d canonical records from %d pages",
            path.name, len(specs),
            len(specs) and page_num,  # type: ignore[possibly-undefined]
        )
        return specs

    # ── Private helpers ───────────────────────────────────────────

    def _parse_table(
        self,
        table: List[List[Optional[str]]],
        page: int,
        source: str,
    ) -> List[CanonicalSpec]:
        """Parse a single tabular region for field definitions.

        Heuristic: look for columns named Tag / Name / Type / Required
        and interpret each row as a field definition.  Also handles
        continuation tables whose first row starts with a numeric tag
        (assuming standard 4-column layout).
        """
        if not table or len(table) < 2:
            return []

        header = [str(c).strip().lower() if c else "" for c in table[0]]
        col_map = self._map_columns(header)

        # If no "tag" column detected, try guessing from the data:
        # continuation tables often have the same structure but no
        # recognisable header.  If the first cell of row 0 or 1
        # is a small integer, assume Tag/Name/Type/Req'd layout.
        if "tag" not in col_map:
            first_data = table[0] if table else []
            if first_data and self._safe_int(
                str(first_data[0]).strip() if first_data[0] else ""
            ) is not None:
                # Looks like a headerless continuation — assign columns
                n = len(first_data)
                col_map = {"tag": 0}
                if n > 1:
                    col_map["name"] = 1
                if n > 2:
                    col_map["type"] = 2
                if n > 3:
                    col_map["required"] = 3
                if n > 4:
                    col_map["description"] = 4
                # Include row 0 as data (it's not a header)
                data_rows = table
            else:
                return []
        else:
            data_rows = table[1:]

        results: List[CanonicalSpec] = []
        for row in data_rows:
            if not row:
                continue
            tag_str = self._cell(row, col_map.get("tag"))
            name_str = self._cell(row, col_map.get("name"))
            type_str = self._cell(row, col_map.get("type"))
            req_str = self._cell(row, col_map.get("required"))
            desc_str = self._cell(row, col_map.get("description"))

            tag = self._safe_int(tag_str)
            if tag is None:
                continue

            results.append(
                CanonicalSpec(
                    kind=SpecKind.FIELD,
                    tag=tag,
                    name=name_str or "",
                    data_type=type_str.upper() if type_str else None,
                    required=self._parse_bool(req_str),
                    description=desc_str or "",
                    source=f"pdf:{source}",
                    meta={"page": page},
                )
            )
        return results

    def _parse_free_text(
        self,
        text: str,
        page: int,
        source: str,
    ) -> List[CanonicalSpec]:
        """Parse free-form text for FIX definitions via regex."""
        results: List[CanonicalSpec] = []

        # Message type definitions  (e.g. "MsgType = D (NewOrderSingle)")
        for m in _MSG_TYPE_RE.finditer(text):
            results.append(
                CanonicalSpec(
                    kind=SpecKind.MESSAGE,
                    msg_type=m.group(1),
                    name=m.group(2),
                    source=f"pdf:{source}",
                    meta={"page": page},
                )
            )

        # Message section headers (e.g. "New Order – Single (MsgType = D)")
        for m in _MSG_HEADER_RE.finditer(text):
            msg_name = m.group(1).strip().replace("\n", " ")
            results.append(
                CanonicalSpec(
                    kind=SpecKind.MESSAGE,
                    msg_type=m.group(2),
                    name=msg_name,
                    source=f"pdf:{source}",
                    meta={"page": page},
                )
            )

        # Structured field defs: "<tag> <Name> <Type> <description>"
        # (common in Vol-6 / data-dictionary pages)
        for m in _FIELD_LINE_RE.finditer(text):
            tag = self._safe_int(m.group(1))
            if tag is None:
                continue
            results.append(
                CanonicalSpec(
                    kind=SpecKind.FIELD,
                    tag=tag,
                    name=m.group(2),
                    data_type=m.group(3).upper(),
                    description=m.group(4).strip()[:2000],
                    source=f"pdf:{source}",
                    meta={"page": page},
                )
            )

        # Tag definitions  (e.g. "Tag: 35 - MsgType")
        for m in _TAG_DEF_RE.finditer(text):
            tag = self._safe_int(m.group(1))
            if tag is None:
                continue
            results.append(
                CanonicalSpec(
                    kind=SpecKind.FIELD,
                    tag=tag,
                    name=m.group(2),
                    source=f"pdf:{source}",
                    meta={"page": page},
                )
            )

        # If nothing structured was extracted, store the raw text
        if not results and len(text.strip()) > 80:
            results.append(
                CanonicalSpec(
                    kind=SpecKind.RAW_TEXT,
                    name="pdf_page",
                    description=text.strip()[:4000],
                    source=f"pdf:{source}",
                    meta={"page": page},
                )
            )

        return results

    # ── Utility ───────────────────────────────────────────────────

    @staticmethod
    def _map_columns(header: List[str]) -> dict:
        """Map standard column intent to column index.

        Uses first-match semantics so that e.g. ``FieldName`` at idx 1
        is not overwritten by ``FIXMLName`` at idx 4 (both contain
        the substring *name*).
        """
        mapping: dict = {}
        for idx, col in enumerate(header):
            if "tag" in col and "tag" not in mapping:
                mapping["tag"] = idx
            elif ("name" in col or "field" in col) and "name" not in mapping:
                mapping["name"] = idx
            elif "type" in col and "type" not in mapping:
                mapping["type"] = idx
            elif "req" in col and "required" not in mapping:
                mapping["required"] = idx
            elif ("desc" in col or "comment" in col) and "description" not in mapping:
                mapping["description"] = idx
        return mapping

    @staticmethod
    def _cell(row: list, idx: Optional[int]) -> str:
        """Safely extract a cell value."""
        if idx is None or idx >= len(row):
            return ""
        val = row[idx]
        return str(val).strip() if val else ""

    @staticmethod
    def _safe_int(s: str) -> Optional[int]:
        """Convert *s* to int, or ``None``."""
        try:
            return int(s)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_bool(s: str) -> Optional[bool]:
        """Interpret ``Y/N/Yes/No/Required/Optional`` as bool."""
        if not s:
            return None
        return s.strip().lower() in {"y", "yes", "required", "true", "1"}
