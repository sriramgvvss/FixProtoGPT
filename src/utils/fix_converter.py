"""
Module: src.utils.fix_converter
================================

FIX Protocol message format converter.

Provides :class:`FixMessageConverter` for bi-directional conversion
between raw FIX tag=value messages and structured formats (JSON, XML).
Includes structural validation helpers for both output formats.

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
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Any, Dict, List, Optional
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from src.core.tokenizer import FixProtocolTokenizer


class FixMessageConverter:
    """Bi-directional converter between FIX, JSON, and XML formats.

    Maintains dictionaries of well-known FIX 5.0 SP2 field names and
    message types for human-readable output.

    Args:
        tokenizer: Optional :class:`FixProtocolTokenizer`. A default
                   instance is created if omitted.
    """
    
    # Common FIX field names (FIX 5.0 SP2)
    FIX_FIELD_NAMES = {
        '8': 'BeginString',
        '9': 'BodyLength',
        '35': 'MsgType',
        '49': 'SenderCompID',
        '56': 'TargetCompID',
        '34': 'MsgSeqNum',
        '52': 'SendingTime',
        '11': 'ClOrdID',
        '21': 'HandlInst',
        '55': 'Symbol',
        '54': 'Side',
        '38': 'OrderQty',
        '40': 'OrdType',
        '44': 'Price',
        '59': 'TimeInForce',
        '150': 'ExecType',
        '151': 'LeavesQty',
        '14': 'CumQty',
        '6': 'AvgPx',
        '10': 'CheckSum',
        '37': 'OrderID',
        '17': 'ExecID',
        '39': 'OrdStatus',
        '32': 'LastQty',
        '31': 'LastPx',
        '58': 'Text',
        '60': 'TransactTime',
        '41': 'OrigClOrdID',
        '1': 'Account',
        '15': 'Currency',
        '48': 'SecurityID',
        '22': 'SecurityIDSource',
        '167': 'SecurityType',
        '99': 'StopPx',
        '100': 'ExDestination',
        '98': 'EncryptMethod',
        '108': 'HeartBtInt',
        '262': 'MDReqID',
        '263': 'SubscriptionRequestType',
        '264': 'MarketDepth',
        '265': 'MDUpdateType',
        '266': 'AggregatedBook',
        '267': 'NoMDEntryTypes',
        '268': 'NoMDEntries',
        '269': 'MDEntryType',
        '270': 'MDEntryPx',
        '271': 'MDEntrySize',
        '1128': 'ApplVerID',
        '1137': 'DefaultApplVerID',
        '553': 'Username',
        '554': 'Password',
        '453': 'NoPartyIDs',
        '448': 'PartyID',
        '447': 'PartyIDSource',
        '452': 'PartyRole',
    }
    
    MSG_TYPE_NAMES = {
        '0': 'Heartbeat',
        '1': 'TestRequest',
        '2': 'ResendRequest',
        '3': 'Reject',
        '4': 'SequenceReset',
        '5': 'Logout',
        'A': 'Logon',
        'D': 'NewOrderSingle',
        '8': 'ExecutionReport',
        'F': 'OrderCancelRequest',
        'G': 'OrderCancelReplaceRequest',
        '9': 'OrderCancelReject',
        'V': 'MarketDataRequest',
        'W': 'MarketDataSnapshotFullRefresh',
        'X': 'MarketDataIncrementalRefresh',
        'Y': 'MarketDataRequestReject',
        'AE': 'TradeCaptureReport',
        'AD': 'TradeCaptureReportRequest',
        'J': 'Allocation',
        'P': 'AllocationAck',
        'BN': 'PositionReport',
        'S': 'Quote',
        'R': 'QuoteRequest',
        'j': 'BusinessMessageReject',
        'B': 'News',
        'q': 'OrderMassCancelRequest',
        'r': 'OrderMassCancelReport',
    }
    
    def __init__(self, tokenizer: Optional[FixProtocolTokenizer] = None) -> None:
        """Initialise the converter.

        Args:
            tokenizer: Optional FIX protocol tokenizer instance.
        """
        self.tokenizer = tokenizer or FixProtocolTokenizer()
        
    def parse_fix_message(self, fix_message: str) -> List[Dict[str, str]]:
        """Parse a raw FIX message into structured field dicts.

        Args:
            fix_message: Raw FIX string (SOH or ``|`` delimited).

        Returns:
            List of dicts with ``tag``, ``name``, ``value`` keys.
        """
        # Split by SOH or pipe
        delimiter = '\x01' if '\x01' in fix_message else '|'
        fields = fix_message.split(delimiter)
        
        parsed_fields = []
        for field in fields:
            if '=' in field:
                tag, value = field.split('=', 1)
                tag = tag.strip()
                value = value.strip()
                
                parsed_fields.append({
                    'tag': tag,
                    'name': self.FIX_FIELD_NAMES.get(tag, f'Tag{tag}'),
                    'value': value
                })
        
        return parsed_fields
    
    def fix_to_json(self, fix_message: str, pretty: bool = True) -> str:
        """Convert a FIX message to JSON.

        Args:
            fix_message: Raw FIX message string.
            pretty:      Pretty-print the output.

        Returns:
            JSON string.
        """
        fields = self.parse_fix_message(fix_message)
        
        # Build JSON structure
        json_data = {
            'protocol': 'FIX',
            'fields': {}
        }
        
        # Add message type info
        msg_type_field = next((f for f in fields if f['tag'] == '35'), None)
        if msg_type_field:
            msg_type = msg_type_field['value']
            json_data['messageType'] = self.MSG_TYPE_NAMES.get(msg_type, msg_type)
            json_data['messageTypeCode'] = msg_type
        
        # Add all fields
        for field in fields:
            json_data['fields'][field['name']] = {
                'tag': field['tag'],
                'value': field['value']
            }
        
        # Add metadata
        json_data['metadata'] = {
            'fieldCount': len(fields),
            'rawMessage': fix_message
        }
        
        if pretty:
            return json.dumps(json_data, indent=2, ensure_ascii=False)
        return json.dumps(json_data, ensure_ascii=False)
    
    def json_to_fix(self, json_str: str) -> str:
        """Convert a JSON representation back to FIX format.

        Args:
            json_str: JSON string (as produced by :meth:`fix_to_json`).

        Returns:
            Pipe-delimited FIX message string.
        """
        data = json.loads(json_str)
        
        # Extract fields from JSON
        fields = []
        
        if 'fields' in data:
            for field_name, field_data in data['fields'].items():
                if isinstance(field_data, dict):
                    tag = field_data.get('tag', '')
                    value = field_data.get('value', '')
                else:
                    # Simple tag=value format
                    tag = field_name
                    value = field_data
                
                if tag and value:
                    fields.append(f"{tag}={value}")
        
        # Join with pipe delimiter
        return '|'.join(fields) + '|' if fields else ''
    
    def fix_to_xml(self, fix_message: str, pretty: bool = True) -> str:
        """Convert a FIX message to XML.

        Produces a ``<FIXMessage>`` root with ``<Header>``, ``<Body>``,
        and ``<Trailer>`` sub-elements.

        Args:
            fix_message: Raw FIX message string.
            pretty:      Pretty-print the output.

        Returns:
            XML string.
        """
        fields = self.parse_fix_message(fix_message)
        
        # Create XML root
        root = ET.Element('FIXMessage')
        
        # Add message type info
        msg_type_field = next((f for f in fields if f['tag'] == '35'), None)
        if msg_type_field:
            msg_type = msg_type_field['value']
            root.set('type', self.MSG_TYPE_NAMES.get(msg_type, msg_type))
            root.set('typeCode', msg_type)
        
        root.set('protocol', 'FIX')
        
        # Add header
        header = ET.SubElement(root, 'Header')
        header_tags = {'8', '9', '35', '49', '56', '34', '52'}
        for field in fields:
            if field['tag'] in header_tags:
                field_elem = ET.SubElement(header, 'Field')
                field_elem.set('tag', field['tag'])
                field_elem.set('name', field['name'])
                field_elem.text = field['value']
        
        # Add body
        body = ET.SubElement(root, 'Body')
        for field in fields:
            if field['tag'] not in header_tags and field['tag'] != '10':
                field_elem = ET.SubElement(body, 'Field')
                field_elem.set('tag', field['tag'])
                field_elem.set('name', field['name'])
                field_elem.text = field['value']
        
        # Add trailer
        trailer = ET.SubElement(root, 'Trailer')
        checksum_field = next((f for f in fields if f['tag'] == '10'), None)
        if checksum_field:
            field_elem = ET.SubElement(trailer, 'Field')
            field_elem.set('tag', '10')
            field_elem.set('name', 'CheckSum')
            field_elem.text = checksum_field['value']
        
        # Convert to string
        xml_str = ET.tostring(root, encoding='unicode')
        
        if pretty:
            dom = minidom.parseString(xml_str)
            return dom.toprettyxml(indent='  ')
        
        return xml_str
    
    def xml_to_fix(self, xml_str: str) -> str:
        """Convert an XML representation back to FIX format.

        Args:
            xml_str: XML string (as produced by :meth:`fix_to_xml`).

        Returns:
            Pipe-delimited FIX message string.
        """
        root = ET.fromstring(xml_str)
        
        fields = []
        
        # Extract fields from all sections
        for section in ['Header', 'Body', 'Trailer']:
            section_elem = root.find(section)
            if section_elem is not None:
                for field_elem in section_elem.findall('Field'):
                    tag = field_elem.get('tag', '')
                    value = field_elem.text or ''
                    if tag:
                        fields.append(f"{tag}={value}")
        
        # Join with pipe delimiter
        return '|'.join(fields) + '|' if fields else ''
    
    def validate_json_structure(self, json_str: str) -> Dict[str, Any]:
        """Validate a JSON string as a well-formed FIX message.

        Args:
            json_str: JSON string to validate.

        Returns:
            Dict with ``valid`` (bool), ``errors``, ``warnings`` keys.
        """
        try:
            data = json.loads(json_str)
            
            errors = []
            warnings = []
            
            # Check required keys
            if 'fields' not in data:
                errors.append("Missing 'fields' key in JSON")
            
            # Check field structure
            if 'fields' in data:
                for field_name, field_data in data['fields'].items():
                    if isinstance(field_data, dict):
                        if 'tag' not in field_data:
                            warnings.append(f"Field '{field_name}' missing 'tag'")
                        if 'value' not in field_data:
                            warnings.append(f"Field '{field_name}' missing 'value'")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings
            }
            
        except json.JSONDecodeError as e:
            return {
                'valid': False,
                'errors': [f"Invalid JSON: {str(e)}"],
                'warnings': []
            }
    
    def validate_xml_structure(self, xml_str: str) -> Dict[str, Any]:
        """Validate an XML string as a well-formed FIX message.

        Args:
            xml_str: XML string to validate.

        Returns:
            Dict with ``valid`` (bool), ``errors``, ``warnings`` keys.
        """
        try:
            root = ET.fromstring(xml_str)
            
            errors = []
            warnings = []
            
            # Check root element
            if root.tag != 'FIXMessage':
                errors.append("Root element must be 'FIXMessage'")
            
            # Check for at least one field
            has_fields = False
            for section in ['Header', 'Body', 'Trailer']:
                section_elem = root.find(section)
                if section_elem is not None and len(section_elem.findall('Field')) > 0:
                    has_fields = True
                    break
            
            if not has_fields:
                warnings.append("No fields found in message")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings
            }
            
        except ET.ParseError as e:
            return {
                'valid': False,
                'errors': [f"Invalid XML: {str(e)}"],
                'warnings': []
            }


def demo() -> None:
    """Demonstrate FIX ↔ JSON ↔ XML round-trip conversion."""
    converter = FixMessageConverter()
    
    # Sample FIX message
    fix_msg = "8=FIXT.1.1|9=178|35=D|49=SENDER|56=TARGET|34=1|52=20240101-12:30:00|1128=9|11=ORD123|55=AAPL|54=1|38=100|40=2|44=150.50|59=0|10=123|"
    
    print("=" * 70)
    print("FIX Message Converter Demo")
    print("=" * 70)
    
    print("\n[1] Original FIX Message:")
    print(fix_msg)
    
    print("\n[2] Convert to JSON:")
    json_output = converter.fix_to_json(fix_msg)
    print(json_output)
    
    print("\n[3] Convert to XML:")
    xml_output = converter.fix_to_xml(fix_msg)
    print(xml_output)
    
    print("\n[4] Convert JSON back to FIX:")
    fix_from_json = converter.json_to_fix(json_output)
    print(fix_from_json)
    
    print("\n[5] Convert XML back to FIX:")
    fix_from_xml = converter.xml_to_fix(xml_output)
    print(fix_from_xml)


if __name__ == "__main__":
    demo()
