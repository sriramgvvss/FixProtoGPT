"""
Module: src.inference.enrichment
=================================

FIX message enrichment — fills in missing header, trailer, and
message-type-specific required fields so that model completions
are production-ready.

Coding Standards
----------------
- PEP 8  : Python Style Guide — naming, spacing, line length ≤ 120
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from __future__ import annotations

import random
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional


# ── Constants ─────────────────────────────────────────────────────────

HEADER_TAGS_ORDER: List[str] = ["8", "9", "35", "49", "56", "34", "52", "1128"]
"""Canonical ordering of FIX header tags."""

MSG_TYPE_REQUIRED: Dict[str, Dict[str, Optional[str]]] = {
    "D": {
        "11": "ORD{seq}",
        "21": "1",
        "55": None,
        "54": None,
        "38": None,
        "40": "1",
        "60": "{ts}",
    },
    "8": {
        "37": "EXEC{seq}",
        "17": "EXEC{seq}",
        "150": "0",
        "39": "0",
        "55": None,
        "54": None,
    },
    "F": {
        "11": "CANCEL{seq}",
        "41": "ORD{seq}",
        "55": None,
        "54": "1",
        "60": "{ts}",
    },
    "G": {
        "11": "ORD{seq}",
        "41": "ORD{seq}",
        "55": None,
        "54": None,
        "38": None,
        "40": "1",
        "60": "{ts}",
    },
    "V": {
        "262": "MDREQ{seq}",
        "263": "1",
        "264": "0",
    },
    "A": {
        "98": "0",
        "108": "30",
    },
}
"""Per-message-type required tags with default-value templates.

Template placeholders:
    ``{seq}``  — replaced with a random 5-digit integer.
    ``{ts}``   — replaced with the current UTC timestamp.
    ``None``   — tag is required but has no sensible default.
"""


# ── Public API ────────────────────────────────────────────────────────

def enrich_fix_message(
    raw_msg: str,
    begin_string: str = "FIXT.1.1",
    appl_ver_id: Optional[str] = "9",
) -> str:
    """Fill in missing standard FIX header, trailer, and required fields.

    Parses *raw_msg*, inserts any missing header tags, message-type-
    specific required fields, and a correct ``CheckSum(10)`` trailer,
    then reassembles the message in canonical tag order.

    Args:
        raw_msg:       Raw FIX message string (``|`` or SOH delimited).
        begin_string:  Value for ``BeginString(8)`` (default ``"FIXT.1.1"``).
        appl_ver_id:   Value for ``ApplVerID(1128)``, or ``None`` to omit.

    Returns:
        Enriched, correctly ordered FIX message string.
    """
    # Strip version markers like [FIX-4.2], [FIX-4.4], [FIX-5.0SP2], etc.
    raw_msg = re.sub(r'\[FIX[^\]]*\]\s*', '', raw_msg)

    delim = "\x01" if "\x01" in raw_msg else "|"

    # ── Parse existing fields (first occurrence wins) ─────────────
    parts = [p for p in raw_msg.split(delim) if "=" in p]
    field_map: Dict[str, str] = {}
    field_order: List[str] = []
    seen: set[str] = set()
    for p in parts:
        tag, val = p.split("=", 1)
        tag = tag.strip()
        val = val.strip()
        if tag not in seen:
            field_map[tag] = val
            field_order.append(tag)
            seen.add(tag)

    seq = str(random.randint(10_000, 99_999))
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.000")

    def _default(template: Optional[str]) -> Optional[str]:
        """Substitute sequence-number and timestamp placeholders in a template."""
        if template is None:
            return None
        return template.replace("{seq}", seq).replace("{ts}", ts)

    # ── Ensure header fields (version-aware) ──────────────────────
    header_defaults: Dict[str, str] = {
        "8": begin_string,
        "9": "200",
        "35": "D",
        "49": f"SENDER{random.randint(1, 10)}",
        "56": f"TARGET{random.randint(1, 10)}",
        "34": str(random.randint(1, 9999)),
        "52": ts,
    }
    if appl_ver_id:
        header_defaults["1128"] = appl_ver_id

    header_order = [t for t in HEADER_TAGS_ORDER if t in header_defaults]
    for tag in header_order:
        if tag not in field_map:
            field_map[tag] = header_defaults[tag]

    # ── Ensure message-type fields ────────────────────────────────
    msg_type = field_map.get("35", "D")
    required = MSG_TYPE_REQUIRED.get(msg_type, {})
    for tag, default_tmpl in required.items():
        if tag not in field_map:
            default_val = _default(default_tmpl)
            if default_val is not None:
                field_map[tag] = default_val

    # ── Ensure checksum trailer ───────────────────────────────────
    if "10" not in field_map:
        field_map["10"] = "000"

    # ── Compute body length (tag 9) ───────────────────────────────
    body_tags = {t for t in field_map if t not in ("8", "9", "10")}

    body_str = delim.join(
        f"{t}={field_map[t]}"
        for t in HEADER_TAGS_ORDER
        if t in body_tags and t in field_map
    )

    remaining = [
        t for t in field_order
        if t in body_tags and t not in HEADER_TAGS_ORDER
    ]
    for t in required:
        if t not in remaining and t in field_map and t in body_tags:
            remaining.append(t)
    if remaining:
        body_str += delim + delim.join(f"{t}={field_map[t]}" for t in remaining)

    field_map["9"] = str(len(body_str) + 1)  # +1 for trailing delim

    # ── Reassemble in correct order ───────────────────────────────
    ordered_tags: List[str] = []
    for t in HEADER_TAGS_ORDER:
        if t in field_map:
            ordered_tags.append(t)
    for t in field_order:
        if t not in ordered_tags and t != "10" and t in field_map:
            ordered_tags.append(t)
    for t in required:
        if t not in ordered_tags and t in field_map:
            ordered_tags.append(t)
    ordered_tags.append("10")

    # ── Compute real checksum ─────────────────────────────────────
    msg_body = (
        delim.join(f"{t}={field_map[t]}" for t in ordered_tags if t != "10")
        + delim
    )
    checksum = sum(ord(c) for c in msg_body) % 256
    field_map["10"] = f"{checksum:03d}"

    return delim.join(f"{t}={field_map[t]}" for t in ordered_tags) + delim
