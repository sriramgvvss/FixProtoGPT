"""
Unit tests for FixProtocolTokenizer (tokenizer/fix_tokenizer.py).

Covers:
- Initialization and special tokens
- Vocabulary building
- FIX message encoding / decoding round-trip
- Natural-language encoding / decoding
- for_generation flag behaviour
- Field parsing
- Save / load persistence
- Edge cases (empty input, unknown chars, malformed FIX)
"""

import json
import pytest
import tempfile
from pathlib import Path

from src.core.tokenizer import FixProtocolTokenizer


# ── Initialization ──────────────────────────────────────────────────

class TestTokenizerInit:
    def test_default_vocab_size(self):
        tok = FixProtocolTokenizer()
        assert tok.vocab_size == 1024

    def test_custom_vocab_size(self):
        tok = FixProtocolTokenizer(vocab_size=512)
        assert tok.vocab_size == 512

    def test_special_tokens_present(self):
        tok = FixProtocolTokenizer()
        expected = {"<|pad|>", "<|bos|>", "<|eos|>", "<|fix|>",
                    "<|field|>", "<|eom|>", "<|unk|>"}
        assert expected == set(tok.special_tokens.keys())

    def test_special_token_ids_unique(self):
        tok = FixProtocolTokenizer()
        ids = list(tok.special_tokens.values())
        assert len(ids) == len(set(ids))

    def test_special_token_id_values(self):
        tok = FixProtocolTokenizer()
        assert tok.special_tokens["<|pad|>"] == 0
        assert tok.special_tokens["<|bos|>"] == 1
        assert tok.special_tokens["<|eos|>"] == 2
        assert tok.special_tokens["<|fix|>"] == 3
        assert tok.special_tokens["<|field|>"] == 4
        assert tok.special_tokens["<|eom|>"] == 5
        assert tok.special_tokens["<|unk|>"] == 6

    def test_token_properties(self):
        tok = FixProtocolTokenizer()
        assert tok.pad_token_id == 0
        assert tok.bos_token_id == 1
        assert tok.eos_token_id == 2

    def test_common_fix_tags_loaded(self):
        tok = FixProtocolTokenizer()
        assert "35" in tok.fix_tags
        assert tok.fix_tags["35"] == "MsgType"
        assert "8" in tok.fix_tags
        assert tok.fix_tags["8"] == "BeginString"


# ── Vocabulary Building ────────────────────────────────────────────

class TestBuildVocab:
    def test_build_vocab_adds_characters(self, tokenizer):
        # After building vocab, individual chars should be in token_to_id
        for ch in "0123456789=|.":
            assert ch in tokenizer.token_to_id, f"char '{ch}' missing from vocab"

    def test_build_vocab_adds_fix_tag_patterns(self, tokenizer):
        # Common FIX tag patterns like "35=" should exist
        for tag in ["8=", "35=", "55=", "10="]:
            assert tag in tokenizer.token_to_id, f"tag pattern '{tag}' missing"

    def test_vocab_size_within_limit(self, tokenizer):
        assert len(tokenizer.token_to_id) <= tokenizer.vocab_size

    def test_id_to_token_inverse(self, tokenizer):
        for tok, tid in tokenizer.token_to_id.items():
            assert tokenizer.id_to_token[tid] == tok


# ── FIX Message Parsing ────────────────────────────────────────────

class TestParseFixMessage:
    def test_pipe_delimited_message(self, tokenizer):
        msg = "8=FIXT.1.1|35=D|55=AAPL|10=123|"
        fields = tokenizer.parse_fix_message(msg)
        tags = [f["tag"] for f in fields]
        assert tags == ["8", "35", "55", "10"]

    def test_soh_delimited_message(self, tokenizer):
        msg = "8=FIXT.1.1\x0135=D\x0155=AAPL\x0110=123\x01"
        fields = tokenizer.parse_fix_message(msg)
        assert len(fields) == 4

    def test_field_values(self, tokenizer):
        msg = "8=FIXT.1.1|35=D|55=AAPL|"
        fields = tokenizer.parse_fix_message(msg)
        assert fields[0]["value"] == "FIXT.1.1"
        assert fields[1]["value"] == "D"
        assert fields[2]["value"] == "AAPL"

    def test_field_names_resolved(self, tokenizer):
        msg = "8=FIXT.1.1|35=D|55=AAPL|"
        fields = tokenizer.parse_fix_message(msg)
        assert fields[0]["name"] == "BeginString"
        assert fields[1]["name"] == "MsgType"
        assert fields[2]["name"] == "Symbol"

    def test_unknown_tag_name(self, tokenizer):
        msg = "9999=foo|"
        fields = tokenizer.parse_fix_message(msg)
        assert fields[0]["name"] == "Unknown"

    def test_empty_message(self, tokenizer):
        fields = tokenizer.parse_fix_message("")
        assert fields == []

    def test_no_equals_sign(self, tokenizer):
        fields = tokenizer.parse_fix_message("no-equals-here")
        assert fields == []


# ── Encoding ────────────────────────────────────────────────────────

class TestEncode:
    def test_fix_message_starts_with_bos(self, tokenizer):
        msg = "8=FIXT.1.1|35=D|55=AAPL|10=123|"
        ids = tokenizer.encode(msg, add_special_tokens=True)
        assert ids[0] == tokenizer.bos_token_id

    def test_fix_message_ends_with_eos(self, tokenizer):
        msg = "8=FIXT.1.1|35=D|55=AAPL|10=123|"
        ids = tokenizer.encode(msg, add_special_tokens=True, for_generation=False)
        assert ids[-1] == tokenizer.eos_token_id

    def test_fix_message_has_fix_token(self, tokenizer):
        msg = "8=FIXT.1.1|35=D|55=AAPL|10=123|"
        ids = tokenizer.encode(msg)
        assert tokenizer.special_tokens["<|fix|>"] in ids

    def test_fix_message_has_field_tokens(self, tokenizer):
        msg = "8=FIXT.1.1|35=D|55=AAPL|10=123|"
        ids = tokenizer.encode(msg)
        field_id = tokenizer.special_tokens["<|field|>"]
        # Should have one <|field|> per field
        assert ids.count(field_id) == 4

    def test_fix_message_has_eom_token(self, tokenizer):
        msg = "8=FIXT.1.1|35=D|55=AAPL|10=123|"
        ids = tokenizer.encode(msg, for_generation=False)
        eom_id = tokenizer.special_tokens["<|eom|>"]
        assert eom_id in ids

    def test_for_generation_omits_eom(self, tokenizer):
        msg = "8=FIXT.1.1|35=D|55=AAPL|10=123|"
        ids = tokenizer.encode(msg, for_generation=True)
        eom_id = tokenizer.special_tokens["<|eom|>"]
        eos_id = tokenizer.eos_token_id
        assert eom_id not in ids
        assert eos_id not in ids

    def test_for_generation_still_has_bos(self, tokenizer):
        msg = "8=FIXT.1.1|35=D|"
        ids = tokenizer.encode(msg, add_special_tokens=True, for_generation=True)
        assert ids[0] == tokenizer.bos_token_id

    def test_nl_text_encoding(self, tokenizer):
        text = "Buy 100 shares"
        ids = tokenizer.encode(text)
        assert ids[0] == tokenizer.bos_token_id
        assert ids[-1] == tokenizer.eos_token_id

    def test_no_special_tokens(self, tokenizer):
        text = "Buy 100 shares"
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert ids[0] != tokenizer.bos_token_id

    def test_unknown_char_gets_unk_id(self):
        tok = FixProtocolTokenizer()
        # Build with minimal data so most chars are unknown
        tok.build_vocab(["abc"])
        ids = tok.encode("xyz abc", add_special_tokens=False)
        unk_id = tok.special_tokens["<|unk|>"]
        # x, y, z not in trained vocab → should be unk
        assert unk_id in ids


# ── Decoding ────────────────────────────────────────────────────────

class TestDecode:
    def test_fix_roundtrip_preserves_structure(self, tokenizer):
        msg = "8=FIXT.1.1|35=D|55=AAPL|10=123|"
        ids = tokenizer.encode(msg)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        # Should contain pipes and tag=value pairs
        assert "8=FIXT.1.1" in decoded
        assert "35=D" in decoded
        assert "55=AAPL" in decoded
        assert "10=123" in decoded

    def test_field_token_decoded_as_pipe(self, tokenizer):
        field_id = tokenizer.special_tokens["<|field|>"]
        decoded = tokenizer.decode([field_id])
        assert "|" in decoded

    def test_eom_token_stops_decoding(self, tokenizer):
        bos = tokenizer.bos_token_id
        eom = tokenizer.special_tokens["<|eom|>"]
        char_a = tokenizer.token_to_id.get("A", tokenizer.special_tokens["<|unk|>"])
        char_b = tokenizer.token_to_id.get("B", tokenizer.special_tokens["<|unk|>"])
        ids = [bos, char_a, eom, char_b]
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert "A" in decoded
        # B comes after EOM, should be dropped
        assert "B" not in decoded

    def test_skip_special_tokens_removes_bos_eos(self, tokenizer):
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        char_a = tokenizer.token_to_id.get("x", tokenizer.special_tokens["<|unk|>"])
        ids = [bos, char_a, eos]
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert "<|bos|>" not in decoded
        assert "<|eos|>" not in decoded

    def test_nl_text_roundtrip(self, tokenizer):
        text = "Buy 100 shares of AAPL"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        # Word content should be present
        for word in ["Buy", "100", "shares", "of", "AAPL"]:
            assert word in decoded

    def test_empty_ids_returns_empty(self, tokenizer):
        decoded = tokenizer.decode([])
        assert decoded == ""


# ── Save / Load ─────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_creates_files(self, tokenizer, tmp_path):
        tokenizer.save(str(tmp_path / "tok"))
        assert (tmp_path / "tok" / "vocab.json").exists()
        assert (tmp_path / "tok" / "fix_tags.json").exists()
        assert (tmp_path / "tok" / "merges.pkl").exists()

    def test_load_restores_vocab(self, tokenizer, tmp_path):
        tokenizer.save(str(tmp_path / "tok"))

        loaded = FixProtocolTokenizer()
        loaded.load(str(tmp_path / "tok"))

        assert loaded.token_to_id == tokenizer.token_to_id
        assert loaded.fix_tags == tokenizer.fix_tags

    def test_roundtrip_encode_after_load(self, tokenizer, tmp_path):
        msg = "8=FIXT.1.1|35=D|55=AAPL|10=123|"
        original_ids = tokenizer.encode(msg)

        tokenizer.save(str(tmp_path / "tok"))
        loaded = FixProtocolTokenizer()
        loaded.load(str(tmp_path / "tok"))
        loaded_ids = loaded.encode(msg)

        assert original_ids == loaded_ids

    def test_vocab_json_is_valid_json(self, tokenizer, tmp_path):
        tokenizer.save(str(tmp_path / "tok"))
        with open(tmp_path / "tok" / "vocab.json") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert len(data) > 0
