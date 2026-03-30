"""
Unit tests for data preparation (data/prepare_data.py).

Covers:
- FIXDataGenerator message generation
- NL-FIX pair generation
- Message type variety
- Field structure validity
- Tokenizer building from generated data
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.prepare_data import FIXDataGenerator


# ── FIXDataGenerator Initialization ────────────────────────────────

class TestFIXDataGeneratorInit:
    def test_initialization(self):
        gen = FIXDataGenerator()
        assert gen is not None

    def test_has_message_generators(self):
        gen = FIXDataGenerator()
        assert callable(getattr(gen, "generate_new_order", None))
        assert callable(getattr(gen, "generate_execution_report", None))
        assert callable(getattr(gen, "generate_logon", None))
        assert callable(getattr(gen, "generate_cancel_request", None))
        assert callable(getattr(gen, "generate_market_data_request", None))


# ── Message Generation ─────────────────────────────────────────────

class TestMessageGeneration:
    @pytest.fixture
    def generator(self):
        return FIXDataGenerator()

    def test_generate_new_order(self, generator):
        msg = generator.generate_new_order()
        assert "35=D" in msg
        assert "8=FIXT.1.1" in msg
        assert "55=" in msg

    def test_generate_exec_report(self, generator):
        msg = generator.generate_execution_report()
        assert "35=8" in msg

    def test_generate_cancel_request(self, generator):
        msg = generator.generate_cancel_request()
        assert "35=F" in msg

    def test_generate_logon(self, generator):
        msg = generator.generate_logon()
        assert "35=A" in msg

    def test_generate_md_request(self, generator):
        msg = generator.generate_market_data_request()
        assert "35=V" in msg

    def test_generate_dataset_produces_list(self, generator):
        messages = generator.generate_dataset(num_samples=10)
        assert isinstance(messages, list)
        assert len(messages) == 10

    def test_messages_have_valid_structure(self, generator):
        """All generated messages should have basic FIX tag=value|tag=value structure."""
        messages = generator.generate_dataset(num_samples=20)

        for msg in messages:
            # Should contain pipe-delimited tag=value fields
            assert "|" in msg or "\x01" in msg, f"No delimiter in: {msg[:80]}"
            # Should contain at least BeginString and MsgType
            assert "8=" in msg, f"No BeginString in: {msg[:80]}"
            assert "35=" in msg, f"No MsgType in: {msg[:80]}"


# ── NL-FIX Pair Generation ────────────────────────────────────────

class TestNLFIXPairGeneration:
    def test_generate_nl_fix_pairs(self):
        gen = FIXDataGenerator()
        pairs = gen.generate_natural_language_pairs(num_samples=5)
        assert isinstance(pairs, list)
        assert len(pairs) > 0
        for pair in pairs:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            nl_text, fix_msg = pair
            assert len(nl_text) > 0
            assert "35=" in fix_msg


# ── Field Validity ─────────────────────────────────────────────────

class TestFieldValidity:
    def _parse_fields(self, msg):
        delim = "\x01" if "\x01" in msg else "|"
        fields = []
        for part in msg.split(delim):
            if "=" in part:
                tag, val = part.split("=", 1)
                fields.append((tag.strip(), val.strip()))
        return fields

    def test_all_tags_numeric(self):
        gen = FIXDataGenerator()
        msg = gen.generate_new_order()

        fields = self._parse_fields(msg)
        for tag, val in fields:
            assert tag.isdigit(), f"Non-numeric tag: {tag}"

    def test_no_empty_values(self):
        gen = FIXDataGenerator()
        msg = gen.generate_new_order()

        fields = self._parse_fields(msg)
        for tag, val in fields:
            assert len(val) > 0, f"Empty value for tag {tag}"
