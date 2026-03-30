"""
Integration tests for the inference engine (inference/generate.py).

These tests create a real (small) model and tokenizer, wiring
them together to verify the full inference pipeline without
loading a trained checkpoint.

Covers:
- enrich_fix_message (standalone function in inference/enrichment.py)
- validate_fix_message
- explain_fix_message
- _load_model_config fallback logic
"""

import pytest
import torch
import yaml
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.inference.generate import FixProtoGPTInference
from src.inference.enrichment import enrich_fix_message
from src.core.transformer import ModelConfig


# ── enrich_fix_message (standalone function, no trained model needed) ─

class TestEnrichFixMessage:
    """Test the post-processor that fills missing header/trailer fields."""

    def _enrich(self, raw: str) -> str:
        """Helper that calls the standalone function with default params."""
        return enrich_fix_message(raw, begin_string="FIXT.1.1", appl_ver_id="9")

    def test_adds_missing_header_tags(self):
        raw = "35=D|55=AAPL|54=1|38=100|40=1|"
        enriched = self._enrich(raw)
        fields = self._parse(enriched)
        tag_set = {t for t, _ in fields}
        for tag in ["8", "9", "35", "49", "56", "34", "52", "1128"]:
            assert tag in tag_set, f"Header tag {tag} missing after enrichment"

    def test_adds_checksum(self):
        raw = "8=FIXT.1.1|35=D|55=AAPL|54=1|38=100|40=1|"
        enriched = self._enrich(raw)
        fields = self._parse(enriched)
        tags = [t for t, _ in fields]
        assert "10" in tags
        # Checksum should be last
        assert tags[-1] == "10"

    def test_checksum_is_3_digits(self):
        raw = "8=FIXT.1.1|35=D|55=AAPL|"
        enriched = self._enrich(raw)
        fields = self._parse(enriched)
        checksum = [v for t, v in fields if t == "10"][0]
        assert len(checksum) == 3
        assert checksum.isdigit()

    def test_begin_string_is_first(self):
        raw = "35=D|55=AAPL|8=FIXT.1.1|"
        enriched = self._enrich(raw)
        fields = self._parse(enriched)
        assert fields[0][0] == "8"

    def test_preserves_existing_values(self):
        raw = "8=FIXT.1.1|35=D|49=MYSENDER|56=MYTARGET|55=AAPL|54=1|38=100|40=1|10=999|"
        enriched = self._enrich(raw)
        fields = self._parse(enriched)
        field_map = {t: v for t, v in fields}
        assert field_map["49"] == "MYSENDER"
        assert field_map["56"] == "MYTARGET"

    def test_body_length_computed(self):
        raw = "8=FIXT.1.1|35=D|55=AAPL|54=1|38=100|40=1|"
        enriched = self._enrich(raw)
        fields = self._parse(enriched)
        body_length = [v for t, v in fields if t == "9"][0]
        assert int(body_length) > 0

    def test_msg_type_required_fields_added(self):
        """NewOrderSingle (D) should get required fields like 11, 21, 60."""
        raw = "8=FIXT.1.1|35=D|55=AAPL|54=1|38=100|"
        enriched = self._enrich(raw)
        fields = self._parse(enriched)
        tag_set = {t for t, _ in fields}
        for tag in ["11", "21", "60"]:
            assert tag in tag_set, f"Required tag {tag} for MsgType D missing"

    def test_logon_required_fields(self):
        raw = "8=FIXT.1.1|35=A|"
        enriched = self._enrich(raw)
        fields = self._parse(enriched)
        tag_set = {t for t, _ in fields}
        assert "98" in tag_set  # EncryptMethod
        assert "108" in tag_set  # HeartBtInt

    def test_exec_report_required_fields(self):
        raw = "8=FIXT.1.1|35=8|55=AAPL|54=1|"
        enriched = self._enrich(raw)
        fields = self._parse(enriched)
        tag_set = {t for t, _ in fields}
        assert "37" in tag_set  # OrderID
        assert "17" in tag_set  # ExecID

    def test_soh_delimiter_handled(self):
        raw = "8=FIXT.1.1\x0135=D\x0155=AAPL\x01"
        enriched = self._enrich(raw)
        # SOH delimiter should be used in output
        assert "\x01" in enriched

    def _parse(self, msg):
        delim = "\x01" if "\x01" in msg else "|"
        fields = []
        for p in msg.split(delim):
            if "=" in p:
                t, v = p.split("=", 1)
                fields.append((t.strip(), v.strip()))
        return fields


# ── validate_fix_message (uses tokenizer only) ─────────────────────

class TestValidateFixMessage:
    """Test validation logic using a mock inference engine."""

    @pytest.fixture
    def mock_engine(self, tokenizer):
        engine = object.__new__(FixProtoGPTInference)
        engine.tokenizer = tokenizer
        return engine

    def test_valid_message(self, mock_engine, sample_messages):
        result = mock_engine.validate_fix_message(sample_messages["new_order"])
        assert result["valid"] is True
        assert result["num_fields"] > 0
        assert result["missing_required_fields"] == []

    def test_missing_required_fields(self, mock_engine):
        msg = "55=AAPL|54=1|38=100|"
        result = mock_engine.validate_fix_message(msg)
        assert result["valid"] is False
        assert len(result["missing_required_fields"]) > 0

    def test_returns_field_list(self, mock_engine, sample_messages):
        result = mock_engine.validate_fix_message(sample_messages["logon"])
        assert isinstance(result["fields"], list)
        assert all("tag" in f for f in result["fields"])


# ── explain_fix_message ────────────────────────────────────────────

class TestExplainFixMessage:
    @pytest.fixture
    def mock_engine(self, tokenizer):
        engine = object.__new__(FixProtoGPTInference)
        engine.tokenizer = tokenizer
        return engine

    def test_explains_new_order(self, mock_engine, sample_messages):
        explanation = mock_engine.explain_fix_message(sample_messages["new_order"])
        # Now returns a dict with summary, fields, message_type
        assert isinstance(explanation, dict)
        assert "summary" in explanation
        assert "fields" in explanation
        assert isinstance(explanation["fields"], list)
        assert any("NewOrderSingle" in str(f.get("value_meaning", "")) or "NewOrderSingle" in str(explanation.get("message_type", {}).get("name", "")) for f in explanation["fields"])

    def test_explains_exec_report(self, mock_engine, sample_messages):
        explanation = mock_engine.explain_fix_message(sample_messages["exec_report"])
        assert isinstance(explanation, dict)
        mt = explanation.get("message_type", {})
        assert "ExecutionReport" in mt.get("name", "") or "Execution" in explanation.get("summary", "")

    def test_field_tags_listed(self, mock_engine, sample_messages):
        explanation = mock_engine.explain_fix_message(sample_messages["new_order"])
        tags = [str(f["tag"]) for f in explanation["fields"]]
        assert "55" in tags  # Symbol
        assert "54" in tags  # Side


# ── _load_model_config ─────────────────────────────────────────────

class TestLoadModelConfig:
    def test_loads_from_yaml(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        yaml_data = {
            "model": {
                "n_layers": 4, "n_heads": 4, "d_model": 256, "d_ff": 512,
                "vocab_size": 512, "max_seq_len": 128,
                "dropout": 0.1, "attention_dropout": 0.05, "use_rotary": True,
            }
        }
        with open(config_dir / "model_config.yaml", "w") as f:
            yaml.dump(yaml_data, f)

        model_path = str(tmp_path / "model.pt")
        cfg = FixProtoGPTInference._load_model_config(model_path)
        assert isinstance(cfg, ModelConfig)
        assert cfg.n_layers == 4
        assert cfg.d_model == 256

    def test_fallback_to_defaults(self, tmp_path):
        # No YAML file exists → should return defaults
        model_path = str(tmp_path / "nonexistent" / "model.pt")
        cfg = FixProtoGPTInference._load_model_config(model_path)
        assert isinstance(cfg, ModelConfig)
        assert cfg.n_layers == 6  # default
