"""
Module: src.inference.explainer
================================

Field-level and message-level explanation builders for FIX messages.

Provides deterministic, template-based human-readable explanations
used by :meth:`FixProtoGPTInference.explain_fix_message`.

Coding Standards
----------------
- PEP 8  : Python Style Guide — naming, spacing, line length ≤ 120
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _symbol_display(ticker: str) -> str:
    """Return ``'TICKER (Company Name)'`` if a name is available."""
    try:
        from src.data.symbol_resolver import lookup_symbol_name

        name = lookup_symbol_name(ticker)
        if name:
            return f"{ticker} ({name})"
    except Exception:
        pass
    return ticker


# ── Field-level explanation ───────────────────────────────────────────

def build_field_explanation(
    tag: str,
    name: str,
    value: str,
    value_meaning: str,
    description: str,
    field_type: str,
) -> str:
    """Return a one-sentence human-readable explanation for a single FIX field.

    Uses tag-specific templates for well-known tags and falls back to a
    generic pattern for others.

    Args:
        tag:           FIX field tag number as a string (e.g. ``"55"``).
        name:          Canonical field name (e.g. ``"Symbol"``).
        value:         Raw field value.
        value_meaning: Resolved enumeration meaning, or ``""``.
        description:   Short field description from the spec.
        field_type:    FIX data-type name (e.g. ``"String"``).

    Returns:
        A single explanatory sentence.
    """
    vm = value_meaning

    # ── Structural / header / trailer fields ──────────────────────
    if tag in ("8", "9", "10"):
        return description or ""

    # ── Tag-specific templates ────────────────────────────────────
    if tag == "35":
        return f"This message is a {vm or value} message." if vm else (description or "")
    if tag == "49":
        return f'The message originates from the firm identified as "{value}".'
    if tag == "56":
        return f'The message is addressed to the firm identified as "{value}".'
    if tag == "55":
        display = _symbol_display(value)
        return f"The financial instrument being traded is {display}."
    if tag == "54":
        if vm:
            return (
                f"The order is on the {vm} side — this indicates the client "
                f"wants to {vm.lower()}."
            )
        return f"Side = {value}."
    if tag == "38":
        return f"The requested quantity is {value} shares/units."
    if tag == "44":
        return f"The price is set at {value} per unit."
    if tag == "40":
        if vm:
            detail = (
                "means the trade executes at the specified price or better."
                if vm.lower() == "limit"
                else "determines how the order is executed on the exchange."
            )
            return f"The order type is {vm} — a {vm.lower()} order {detail}"
        return f"OrdType = {value}."
    if tag == "59":
        tif_desc = {
            "Day": "the order expires at the end of the trading day.",
            "GTC": "the order remains active until explicitly cancelled (Good Till Cancel).",
            "IOC": "any portion not immediately filled is cancelled (Immediate or Cancel).",
            "FOK": "the order must be filled entirely or not at all (Fill or Kill).",
            "AtTheOpening": "the order should execute at the market open.",
            "AtTheClose": "the order should execute at the market close.",
        }
        detail = tif_desc.get(vm, f"the order validity is {vm or value}.")
        return f"Time-in-force is {vm or value} — {detail}"
    if tag == "39":
        return (
            f"The current order status is {vm or value} — this reflects the "
            "most recent state of the order on the exchange."
        )
    if tag == "150":
        return (
            f"The execution type is {vm or value} — this indicates the "
            "specific action that triggered this execution report."
        )
    if tag == "11":
        return (
            f'The client order ID is "{value}" — this is the unique '
            "identifier assigned by the submitting institution."
        )
    if tag == "37":
        return f'The broker-assigned order ID is "{value}".'
    if tag == "17":
        return (
            f'The execution ID is "{value}" — uniquely identifies '
            "this execution report."
        )
    if tag == "14":
        return f"Cumulative quantity filled so far is {value} shares/units."
    if tag == "6":
        return f"The average fill price across all executions is {value}."
    if tag == "31":
        return f"The price of the last fill was {value}."
    if tag == "32":
        return f"The quantity of the last fill was {value} shares/units."
    if tag == "60":
        return f"The transaction occurred at {value} (UTC)."
    if tag == "52":
        return f"The message was sent at {value} (UTC)."
    if tag == "1":
        return f'The trading account is "{value}".'
    if tag == "21":
        return (
            f"Handling instructions: {vm or value} — indicates how the "
            "broker should handle the order."
        )
    if tag == "100":
        return f'The execution destination / exchange is "{value}".'
    if tag == "99":
        return (
            f"The stop price is {value} — the order activates when the "
            "market reaches this level."
        )
    if tag == "15":
        return f"The currency for pricing is {value}."
    if tag == "58":
        return f'Free-text note: "{value}".'
    if tag == "48":
        return (
            f'The security ID is "{value}" '
            "(see SecurityIDSource for the ID scheme used)."
        )
    if tag == "22":
        return f"The security ID source/scheme is {vm or value}."

    # ── Generic fallback ──────────────────────────────────────────
    if vm:
        if description:
            return f"{name} is set to {vm} ({value}) — {description.lower()}."
        return f"{name} is set to {vm} ({value})."
    if description:
        return f"{name} is \"{value}\" — {description[0].lower()}{description[1:]}"
    return f"{name} = {value}"


# ── Message-level summary ────────────────────────────────────────────

def build_explain_summary(fields: List[Dict], msg_info: Dict) -> str:
    """Build a presentable, multi-section natural-language summary.

    Composes separate paragraphs for routing, instrument/order details,
    execution status, fills, timestamps, and protocol metadata.

    Args:
        fields:   Enriched field dicts (with ``tag``, ``value``,
                  ``value_meaning``, etc.).
        msg_info: Message-type metadata (``code``, ``name``,
                  ``category``, ``description``).

    Returns:
        Markdown-formatted multi-paragraph summary string.
    """

    def _f(tag: str) -> Optional[Dict]:
        """Find enriched field dict by tag number."""
        return next((x for x in fields if x["tag"] == tag), None)

    def _val(tag: str) -> Optional[str]:
        """Return ``'meaning (raw)'`` or ``'raw'`` for *tag*."""
        f = _f(tag)
        if not f:
            return None
        m = f.get("value_meaning")
        return f"{m} ({f['value']})" if m else f["value"]

    def _raw(tag: str) -> Optional[str]:
        """Return the raw value for *tag*."""
        f = _f(tag)
        return f["value"] if f else None

    sections: List[str] = []

    # 1. Overview
    if msg_info:
        cat = msg_info.get("category", "").replace("-", " ").strip()
        cat_label = f" in the {cat} category" if cat else ""
        sections.append(
            f"This is a **{msg_info['name']}** message "
            f"(MsgType = {msg_info['code']}){cat_label}. "
            f"{msg_info.get('description', 'It performs a FIX protocol operation')}."
        )

    # 2. Routing
    sender = _raw("49")
    target = _raw("56")
    if sender and target:
        sections.append(
            f"The message is sent from **{sender}** to **{target}**, "
            "establishing the routing between the originating and receiving firms."
        )

    # 3. Instrument & Order details
    symbol = _raw("55")
    side = _val("54")
    qty = _raw("38")
    price = _raw("44")
    ord_type = _val("40")
    tif = _val("59")
    stop_px = _raw("99")
    account = _raw("1")

    if symbol:
        sym_label = _symbol_display(symbol)
        detail_parts = [f"for instrument **{sym_label}**"]
        if side:
            detail_parts.append(f"on the **{side}** side")
        if qty:
            detail_parts.append(f"with a quantity of **{qty}** shares/units")
        if price:
            detail_parts.append(f"at a price of **{price}**")
        if stop_px:
            detail_parts.append(f"with a stop price of **{stop_px}**")
        sections.append(f"The order is {', '.join(detail_parts)}.")

        specifics: List[str] = []
        if ord_type:
            specifics.append(f"The order type is **{ord_type}**")
        if tif:
            specifics.append(f"time-in-force is **{tif}**")
        if account:
            specifics.append(f"trading on account **{account}**")
        if specifics:
            sections.append(f"{', '.join(specifics)}.")

    # 4. Execution / Status details
    status = _val("39")
    exec_type = _val("150")
    cum_qty = _raw("14")
    avg_px = _raw("6")
    last_qty = _raw("32")
    last_px = _raw("31")

    exec_parts: List[str] = []
    if status:
        exec_parts.append(f"current order status is **{status}**")
    if exec_type:
        exec_parts.append(f"execution type is **{exec_type}**")
    if exec_parts:
        sections.append(f"The {', '.join(exec_parts)}.")

    fill_parts: List[str] = []
    if last_qty and last_px:
        fill_parts.append(f"last fill was **{last_qty}** @ **{last_px}**")
    if cum_qty:
        fill_parts.append(f"cumulative filled quantity is **{cum_qty}**")
    if avg_px:
        fill_parts.append(f"average fill price is **{avg_px}**")
    if fill_parts:
        sections.append(f"Fill details: {'; '.join(fill_parts)}.")

    # 5. Timestamps
    transact = _raw("60")
    sending = _raw("52")
    ts_parts: List[str] = []
    if transact:
        ts_parts.append(f"transaction time **{transact}**")
    if sending:
        ts_parts.append(f"sent at **{sending}**")
    if ts_parts:
        sections.append(f"Timing: {', '.join(ts_parts)}.")

    # 6. Protocol info
    begin_str = _raw("8")
    appl_ver = _val("1128")
    if begin_str or appl_ver:
        proto = f"Protocol: **{begin_str or 'FIX'}**"
        if appl_ver:
            proto += f", application version **{appl_ver}**"
        sections.append(f"{proto}.")

    if not sections:
        sections.append("This is a FIX protocol message.")

    return "\n\n".join(sections)
