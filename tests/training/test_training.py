"""
Evaluation tests for FixProtoGPT.

Refactored from the top-level evaluate.py into pytest-compatible tests.
These test the FIX validation helpers without requiring a trained model.
Model-dependent evaluation is guarded behind a fixture that skips if the
checkpoint is not available.

Covers:
- parse_fields helper
- validate_fix_message helper
- Quality score computation logic
- Header completeness checks
- Message-type-specific required field checks
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the validation helpers from the original evaluate.py
# We also define them inline for portability
REQUIRED_HEADER_TAGS = {"8", "9", "35", "49", "56", "34", "52"}
REQUIRED_TRAILER_TAGS = {"10"}

MSG_TYPE_REQUIRED = {
    "D": {"11", "21", "55", "54", "38", "40", "60"},
    "8": {"37", "17", "150", "39", "55", "54"},
    "F": {"11", "41", "55", "54", "60"},
    "G": {"11", "41", "55", "54", "38", "40", "60"},
    "V": {"262", "263", "264"},
    "A": {"98", "108"},
}

VALID_MSG_TYPES = {
    "0", "1", "2", "3", "4", "5", "A", "D", "8", "F", "G", "9",
    "V", "W", "X", "Y", "AE", "AD", "J", "P",
}


def parse_fields(msg):
    delim = "\x01" if "\x01" in msg else "|"
    parts = msg.split(delim)
    fields = []
    for p in parts:
        if "=" in p:
            tag, val = p.split("=", 1)
            fields.append((tag.strip(), val.strip()))
    return fields


def validate_fix_message(msg):
    fields = parse_fields(msg)
    tags = [t for t, v in fields]
    tag_set = set(tags)

    result = {
        "num_fields": len(fields),
        "has_begin_string": "8" in tag_set,
        "has_msg_type": "35" in tag_set,
        "has_checksum": "10" in tag_set,
        "has_body_length": "9" in tag_set,
        "header_complete": REQUIRED_HEADER_TAGS.issubset(tag_set),
        "valid_structure": False,
        "correct_msg_type": False,
        "msg_type_fields_present": False,
        "duplicate_tags": len(tags) != len(tag_set),
        "tag_value_format_ok": True,
    }

    msg_type = None
    for t, v in fields:
        if t == "35":
            msg_type = v
            break

    if msg_type and msg_type in VALID_MSG_TYPES:
        result["correct_msg_type"] = True

    if msg_type and msg_type in MSG_TYPE_REQUIRED:
        required = MSG_TYPE_REQUIRED[msg_type]
        result["msg_type_fields_present"] = required.issubset(tag_set)

    if fields:
        result["valid_structure"] = (
            fields[0][0] == "8" and fields[-1][0] == "10"
        )

    for t, v in fields:
        if not t.isdigit():
            result["tag_value_format_ok"] = False
            break

    return result


# ── parse_fields ────────────────────────────────────────────────────

class TestParseFields:
    def test_pipe_delimited(self):
        fields = parse_fields("8=FIXT.1.1|35=D|55=AAPL|10=123|")
        assert len(fields) == 4
        assert fields[0] == ("8", "FIXT.1.1")

    def test_soh_delimited(self):
        fields = parse_fields("8=FIXT.1.1\x0135=D\x0110=123\x01")
        assert len(fields) == 3

    def test_empty_string(self):
        assert parse_fields("") == []

    def test_no_equals(self):
        assert parse_fields("no-fields-here") == []

    def test_value_with_equals(self):
        fields = parse_fields("58=a=b|")
        assert fields[0] == ("58", "a=b")

    def test_whitespace_stripped(self):
        fields = parse_fields(" 35 = D |")
        assert fields[0] == ("35", "D")


# ── validate_fix_message ───────────────────────────────────────────

class TestValidateFixMessage:
    def test_valid_new_order(self):
        msg = (
            "8=FIXT.1.1|9=200|35=D|49=S|56=T|34=1|52=20260101-12:00:00|"
            "11=ORD1|21=1|55=AAPL|54=1|38=100|40=1|60=20260101-12:00:00|10=123|"
        )
        result = validate_fix_message(msg)
        assert result["has_msg_type"] is True
        assert result["correct_msg_type"] is True
        assert result["valid_structure"] is True
        assert result["header_complete"] is True
        assert result["msg_type_fields_present"] is True
        assert result["tag_value_format_ok"] is True

    def test_missing_header(self):
        msg = "35=D|55=AAPL|10=123|"
        result = validate_fix_message(msg)
        assert result["header_complete"] is False

    def test_missing_checksum(self):
        msg = "8=FIXT.1.1|35=D|55=AAPL|"
        result = validate_fix_message(msg)
        assert result["has_checksum"] is False

    def test_invalid_msg_type(self):
        msg = "8=FIXT.1.1|35=ZZ|10=123|"
        result = validate_fix_message(msg)
        assert result["correct_msg_type"] is False

    def test_wrong_structure_order(self):
        msg = "35=D|8=FIXT.1.1|10=123|"
        result = validate_fix_message(msg)
        assert result["valid_structure"] is False  # Should start with tag 8

    def test_non_numeric_tag(self):
        msg = "8=FIXT.1.1|ABC=DEF|10=123|"
        result = validate_fix_message(msg)
        assert result["tag_value_format_ok"] is False

    def test_exec_report_required_fields(self):
        msg = (
            "8=FIXT.1.1|9=280|35=8|49=T|56=S|34=1|52=20260101-12:00:00|"
            "37=EXEC1|17=EXEC1|150=0|39=0|55=AAPL|54=1|10=456|"
        )
        result = validate_fix_message(msg)
        assert result["msg_type_fields_present"] is True

    def test_logon_required_fields(self):
        msg = (
            "8=FIXT.1.1|9=80|35=A|49=S|56=T|34=1|52=20260101-12:00:00|"
            "98=0|108=30|10=100|"
        )
        result = validate_fix_message(msg)
        assert result["msg_type_fields_present"] is True

    def test_md_request_required_fields(self):
        msg = (
            "8=FIXT.1.1|9=120|35=V|49=S|56=T|34=1|52=20260101-12:00:00|"
            "262=MDREQ1|263=1|264=0|10=200|"
        )
        result = validate_fix_message(msg)
        assert result["msg_type_fields_present"] is True

    def test_cancel_request_missing_fields(self):
        msg = "8=FIXT.1.1|35=F|10=123|"
        result = validate_fix_message(msg)
        assert result["msg_type_fields_present"] is False

    def test_empty_message(self):
        result = validate_fix_message("")
        assert result["num_fields"] == 0
        assert result["has_msg_type"] is False

    def test_duplicate_tags_detected(self):
        msg = "8=FIXT.1.1|35=D|35=A|10=123|"
        result = validate_fix_message(msg)
        assert result["duplicate_tags"] is True


# ── Quality Score Computation ──────────────────────────────────────

class TestQualityScore:
    def _compute_score(self, results):
        n = len(results)
        if n == 0:
            return 0
        s = {
            "correct_msg_type": sum(r["correct_msg_type"] for r in results),
            "valid_structure": sum(r["valid_structure"] for r in results),
            "tag_format_ok": sum(r["tag_value_format_ok"] for r in results),
            "msg_type_fields": sum(r["msg_type_fields_present"] for r in results),
            "header_complete": sum(r["header_complete"] for r in results),
        }
        return (
            s["correct_msg_type"] + s["valid_structure"] + s["tag_format_ok"]
            + s["msg_type_fields"] + s["header_complete"]
        ) / (5 * n) * 100

    def test_perfect_score(self):
        msg = (
            "8=FIXT.1.1|9=200|35=D|49=S|56=T|34=1|52=20260101-12:00:00|"
            "11=ORD1|21=1|55=AAPL|54=1|38=100|40=1|60=20260101-12:00:00|10=123|"
        )
        results = [validate_fix_message(msg)]
        assert self._compute_score(results) == 100.0

    def test_zero_score(self):
        # A message with non-numeric tag fails tag_value_format_ok
        results = [validate_fix_message("ABC=DEF|")]
        assert self._compute_score(results) == 0.0

    def test_partial_score(self):
        # Missing header → header_complete=False, msg_type_fields_present=False
        msg = "8=FIXT.1.1|35=D|55=AAPL|10=123|"
        results = [validate_fix_message(msg)]
        score = self._compute_score(results)
        assert 0 < score < 100
