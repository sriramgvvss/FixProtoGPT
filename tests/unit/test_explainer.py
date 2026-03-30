"""
Tests for src.inference.explainer
===================================

Validates field-level explanations, message-level summaries, and the
symbol display enrichment helper.

Coding Standards: PEP 8, PEP 257, PEP 484, Google Python Style Guide.
"""

from __future__ import annotations

from typing import Dict, List
from unittest.mock import patch

import pytest

from src.inference.explainer import (
    _symbol_display,
    build_field_explanation,
    build_explain_summary,
)


_PATCH_TARGET = "src.data.symbol_resolver.lookup_symbol_name"


# ══════════════════════════════════════════════════════════════════════
#  _symbol_display helper
# ══════════════════════════════════════════════════════════════════════

class TestSymbolDisplay:
    """Tests for the _symbol_display enrichment helper."""

    def test_returns_enriched_when_name_found(self) -> None:
        with patch(_PATCH_TARGET, return_value="Apple Inc."):
            assert _symbol_display("AAPL") == "AAPL (Apple Inc.)"

    def test_returns_ticker_only_when_no_name(self) -> None:
        with patch(_PATCH_TARGET, return_value=None):
            assert _symbol_display("XYZQQQ") == "XYZQQQ"

    def test_returns_ticker_on_import_error(self) -> None:
        with patch(_PATCH_TARGET, side_effect=ImportError("no module")):
            assert _symbol_display("AAPL") == "AAPL"

    def test_returns_ticker_on_runtime_error(self) -> None:
        with patch(_PATCH_TARGET, side_effect=RuntimeError("boom")):
            assert _symbol_display("AAPL") == "AAPL"

    def test_forex_pair(self) -> None:
        with patch(_PATCH_TARGET, return_value="Euro / US Dollar"):
            assert _symbol_display("EUR/USD") == "EUR/USD (Euro / US Dollar)"

    def test_crypto_ticker(self) -> None:
        with patch(_PATCH_TARGET, return_value="Bitcoin"):
            assert _symbol_display("BTC") == "BTC (Bitcoin)"


# ══════════════════════════════════════════════════════════════════════
#  build_field_explanation — tag 55 enrichment
# ══════════════════════════════════════════════════════════════════════

class TestFieldExplanationTag55:
    """Tag 55 (Symbol) should include enriched display name."""

    def test_tag_55_with_name(self) -> None:
        with patch(_PATCH_TARGET, return_value="Tesla, Inc."):
            result = build_field_explanation(
                tag="55", name="Symbol", value="TSLA",
                value_meaning="", description="Ticker symbol",
                field_type="String",
            )
            assert "TSLA (Tesla, Inc.)" in result
            assert result.startswith("The financial instrument")

    def test_tag_55_without_name(self) -> None:
        with patch(_PATCH_TARGET, return_value=None):
            result = build_field_explanation(
                tag="55", name="Symbol", value="ZZZZ",
                value_meaning="", description="Ticker symbol",
                field_type="String",
            )
            assert "ZZZZ" in result
            assert "()" not in result  # no empty parens


# ══════════════════════════════════════════════════════════════════════
#  build_explain_summary — instrument section enrichment
# ══════════════════════════════════════════════════════════════════════

def _make_field(tag: str, value: str, meaning: str = "") -> Dict:
    return {
        "tag": tag,
        "value": value,
        "value_meaning": meaning,
    }


class TestExplainSummaryInstrument:
    """The instrument section of the summary should use enriched names."""

    def test_summary_instrument_enriched(self) -> None:
        fields: List[Dict] = [
            _make_field("55", "AAPL"),
            _make_field("54", "Buy", "Buy"),
            _make_field("38", "100"),
        ]
        msg_info = {
            "code": "D",
            "name": "NewOrderSingle",
            "category": "order-handling",
            "description": "Submit a new order",
        }

        with patch(_PATCH_TARGET, return_value="Apple Inc."):
            result = build_explain_summary(fields, msg_info)
            assert "AAPL (Apple Inc.)" in result

    def test_summary_instrument_no_name(self) -> None:
        fields: List[Dict] = [
            _make_field("55", "XYZZ"),
            _make_field("38", "50"),
        ]
        msg_info = {
            "code": "D",
            "name": "NewOrderSingle",
            "category": "",
            "description": "Submit a new order",
        }

        with patch(_PATCH_TARGET, return_value=None):
            result = build_explain_summary(fields, msg_info)
            assert "**XYZZ**" in result
            assert "()" not in result


# ══════════════════════════════════════════════════════════════════════
#  Other tag templates (smoke tests)
# ══════════════════════════════════════════════════════════════════════

class TestFieldExplanationOtherTags:
    """Smoke tests for non-symbol tags — no regressions."""

    def test_tag_35(self) -> None:
        result = build_field_explanation(
            tag="35", name="MsgType", value="D",
            value_meaning="NewOrderSingle",
            description="", field_type="String",
        )
        assert "NewOrderSingle" in result

    def test_tag_49(self) -> None:
        result = build_field_explanation(
            tag="49", name="SenderCompID", value="BROKER1",
            value_meaning="", description="", field_type="String",
        )
        assert "BROKER1" in result

    def test_tag_38(self) -> None:
        result = build_field_explanation(
            tag="38", name="OrderQty", value="100",
            value_meaning="", description="", field_type="Qty",
        )
        assert "100" in result

    def test_tag_44(self) -> None:
        result = build_field_explanation(
            tag="44", name="Price", value="150.25",
            value_meaning="", description="", field_type="Price",
        )
        assert "150.25" in result

    def test_generic_fallback(self) -> None:
        result = build_field_explanation(
            tag="999", name="CustomTag", value="XYZ",
            value_meaning="", description="A custom field",
            field_type="String",
        )
        assert "CustomTag" in result or "XYZ" in result
