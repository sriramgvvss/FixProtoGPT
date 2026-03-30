"""
Unit tests for FixMessageConverter (utils/fix_converter.py).

Covers:
- FIX message parsing (pipe and SOH delimiters)
- FIX → JSON conversion and back
- FIX → XML conversion and back
- JSON / XML validation helpers
- Field name resolution
- Message type name resolution
- Edge cases
"""

import json
import pytest
import xml.etree.ElementTree as ET

from src.utils.fix_converter import FixMessageConverter


# ── Parsing ─────────────────────────────────────────────────────────

class TestParseFixMessage:
    def test_pipe_delimiter(self, converter):
        msg = "8=FIXT.1.1|35=D|55=AAPL|10=123|"
        fields = converter.parse_fix_message(msg)
        assert len(fields) == 4
        assert fields[0]["tag"] == "8"
        assert fields[0]["value"] == "FIXT.1.1"

    def test_soh_delimiter(self, converter):
        msg = "8=FIXT.1.1\x0135=D\x0155=AAPL\x0110=123\x01"
        fields = converter.parse_fix_message(msg)
        assert len(fields) == 4

    def test_field_names(self, converter):
        msg = "8=FIXT.1.1|35=D|55=AAPL|"
        fields = converter.parse_fix_message(msg)
        assert fields[0]["name"] == "BeginString"
        assert fields[1]["name"] == "MsgType"
        assert fields[2]["name"] == "Symbol"

    def test_unknown_tag(self, converter):
        msg = "99999=foo|"
        fields = converter.parse_fix_message(msg)
        assert fields[0]["name"] == "Tag99999"

    def test_empty_string(self, converter):
        fields = converter.parse_fix_message("")
        assert fields == []

    def test_value_with_equals(self, converter):
        msg = "58=price=100|"
        fields = converter.parse_fix_message(msg)
        assert fields[0]["tag"] == "58"
        assert fields[0]["value"] == "price=100"


# ── FIX → JSON ─────────────────────────────────────────────────────

class TestFixToJson:
    def test_valid_json(self, converter, sample_messages):
        result = converter.fix_to_json(sample_messages["new_order"])
        data = json.loads(result)
        assert data["protocol"] == "FIX"
        assert "fields" in data

    def test_message_type_extracted(self, converter, sample_messages):
        result = converter.fix_to_json(sample_messages["new_order"])
        data = json.loads(result)
        assert data["messageType"] == "NewOrderSingle"
        assert data["messageTypeCode"] == "D"

    def test_all_fields_present(self, converter, sample_messages):
        msg = "8=FIXT.1.1|35=D|55=AAPL|10=123|"
        result = converter.fix_to_json(msg)
        data = json.loads(result)
        assert "BeginString" in data["fields"]
        assert "MsgType" in data["fields"]
        assert "Symbol" in data["fields"]

    def test_metadata(self, converter):
        msg = "8=FIXT.1.1|35=D|55=AAPL|10=123|"
        result = converter.fix_to_json(msg)
        data = json.loads(result)
        assert data["metadata"]["fieldCount"] == 4

    def test_compact_json(self, converter):
        msg = "8=FIXT.1.1|35=D|"
        result = converter.fix_to_json(msg, pretty=False)
        assert "\n" not in result

    def test_exec_report_type(self, converter, sample_messages):
        result = converter.fix_to_json(sample_messages["exec_report"])
        data = json.loads(result)
        assert data["messageType"] == "ExecutionReport"


# ── JSON → FIX ─────────────────────────────────────────────────────

class TestJsonToFix:
    def test_round_trip_fields(self, converter):
        original = "8=FIXT.1.1|35=D|55=AAPL|10=123|"
        json_str = converter.fix_to_json(original)
        reconstructed = converter.json_to_fix(json_str)
        # Should contain the same tag=value pairs
        assert "8=FIXT.1.1" in reconstructed
        assert "35=D" in reconstructed
        assert "55=AAPL" in reconstructed

    def test_empty_json(self, converter):
        result = converter.json_to_fix('{"fields": {}}')
        assert result == ""


# ── FIX → XML ──────────────────────────────────────────────────────

class TestFixToXml:
    def test_valid_xml(self, converter, sample_messages):
        result = converter.fix_to_xml(sample_messages["new_order"])
        root = ET.fromstring(result)
        assert root.tag == "FIXMessage"

    def test_header_section(self, converter, sample_messages):
        result = converter.fix_to_xml(sample_messages["new_order"])
        root = ET.fromstring(result)
        header = root.find("Header")
        assert header is not None
        tags = [f.get("tag") for f in header.findall("Field")]
        assert "8" in tags
        assert "35" in tags

    def test_body_section(self, converter, sample_messages):
        result = converter.fix_to_xml(sample_messages["new_order"])
        root = ET.fromstring(result)
        body = root.find("Body")
        assert body is not None
        tags = [f.get("tag") for f in body.findall("Field")]
        assert "55" in tags

    def test_trailer_section(self, converter, sample_messages):
        result = converter.fix_to_xml(sample_messages["new_order"])
        root = ET.fromstring(result)
        trailer = root.find("Trailer")
        assert trailer is not None
        tags = [f.get("tag") for f in trailer.findall("Field")]
        assert "10" in tags

    def test_message_type_attribute(self, converter, sample_messages):
        result = converter.fix_to_xml(sample_messages["new_order"])
        root = ET.fromstring(result)
        assert root.get("type") == "NewOrderSingle"
        assert root.get("typeCode") == "D"

    def test_compact_xml(self, converter):
        msg = "8=FIXT.1.1|35=D|10=123|"
        result = converter.fix_to_xml(msg, pretty=False)
        # Compact should not have extra whitespace indentation
        assert "<?xml" not in result  # pretty prints with XML declaration


# ── XML → FIX ──────────────────────────────────────────────────────

class TestXmlToFix:
    def test_round_trip(self, converter, sample_messages):
        xml_str = converter.fix_to_xml(sample_messages["new_order"])
        reconstructed = converter.xml_to_fix(xml_str)
        assert "8=FIXT.1.1" in reconstructed
        assert "35=D" in reconstructed
        assert "55=AAPL" in reconstructed
        assert "10=" in reconstructed

    def test_empty_sections(self, converter):
        xml = """<FIXMessage><Header/><Body/><Trailer/></FIXMessage>"""
        result = converter.xml_to_fix(xml)
        assert result == ""


# ── MSG_TYPE_NAMES ─────────────────────────────────────────────────

class TestMsgTypeNames:
    def test_common_types(self):
        c = FixMessageConverter()
        assert c.MSG_TYPE_NAMES["D"] == "NewOrderSingle"
        assert c.MSG_TYPE_NAMES["8"] == "ExecutionReport"
        assert c.MSG_TYPE_NAMES["F"] == "OrderCancelRequest"
        assert c.MSG_TYPE_NAMES["A"] == "Logon"
        assert c.MSG_TYPE_NAMES["V"] == "MarketDataRequest"

    def test_all_types_are_strings(self):
        c = FixMessageConverter()
        for k, v in c.MSG_TYPE_NAMES.items():
            assert isinstance(k, str)
            assert isinstance(v, str)


# ── FIX_FIELD_NAMES ────────────────────────────────────────────────

class TestFieldNames:
    def test_contains_standard_fields(self):
        c = FixMessageConverter()
        assert "8" in c.FIX_FIELD_NAMES   # BeginString
        assert "9" in c.FIX_FIELD_NAMES   # BodyLength
        assert "35" in c.FIX_FIELD_NAMES  # MsgType
        assert "55" in c.FIX_FIELD_NAMES  # Symbol
        assert "10" in c.FIX_FIELD_NAMES  # CheckSum
