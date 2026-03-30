"""
Module: src.utils.fix_enrichment
=================================

Shared FIX field enrichment and message-type extraction helpers.

Eliminates triplicated enrichment loops that were previously inlined
in ``explain_fix_message``, ``get_model_insight``, and ``_demo_explain``.

Author : FixProtoGPT Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def enrich_parsed_fields(
    parsed: List[Dict[str, str]],
    fields_db: Dict,
    enum_db: Dict,
    *,
    full: bool = True,
) -> List[Dict[str, Any]]:
    """Build enriched field dicts from raw parsed FIX fields.

    Args:
        parsed:    Output of ``FixProtocolTokenizer.parse_fix_message()``.
        fields_db: ``FIXProtocolScraper.FIELDS`` mapping.
        enum_db:   ``FIXProtocolScraper.ENUMERATIONS`` mapping.
        full:      If ``True``, include ``description``, ``type``,
                   ``possible_values``, and ``explanation`` (used by
                   ``explain_fix_message`` / ``_demo_explain``).
                   If ``False``, return a lightweight dict (used by
                   ``get_model_insight``).

    Returns:
        List of enriched field dicts.
    """
    enriched: List[Dict[str, Any]] = []
    for f in parsed:
        tag_num = f["tag"]
        value = f["value"]
        tag_int = int(tag_num) if tag_num.isdigit() else None
        meta = fields_db.get(tag_int, {}) if tag_int else {}

        field_name = meta.get("name", f.get("name", "Unknown"))
        enum_key = f"{field_name}({tag_num})"
        enum_map = enum_db.get(enum_key, {})
        value_meaning = enum_map.get(value, "")

        entry: Dict[str, Any] = {
            "tag": tag_num,
            "name": field_name,
            "value": value,
            "value_meaning": value_meaning,
        }

        if full:
            from src.inference.explainer import build_field_explanation

            description = meta.get("description", "")
            field_type = meta.get("type", "")
            entry.update({
                "description": description,
                "type": field_type,
                "possible_values": dict(enum_map) if enum_map else {},
                "explanation": build_field_explanation(
                    tag_num, field_name, value, value_meaning, description, field_type,
                ),
            })

        enriched.append(entry)
    return enriched


def extract_msg_type_info(
    enriched: List[Dict[str, Any]],
    msg_db: Dict,
) -> Dict[str, Any]:
    """Extract message-type metadata from an enriched field list.

    Locates the ``tag == "35"`` entry, looks up ``msg_db``, and returns
    a ``{code, name, category, description}`` dict.  Also patches the
    ``value_meaning`` on the tag-35 field in *enriched* in-place.

    Args:
        enriched: List produced by :func:`enrich_parsed_fields`.
        msg_db:   ``FIXProtocolScraper.MESSAGE_TYPES`` mapping.

    Returns:
        Message-type info dict (empty if tag 35 is absent).
    """
    msg_type_field = next((f for f in enriched if f["tag"] == "35"), None)
    if not msg_type_field:
        return {}

    code = msg_type_field["value"]
    mt = msg_db.get(code, {})
    msg_info: Dict[str, Any] = {
        "code": code,
        "name": mt.get("name", "Unknown"),
        "category": mt.get("category", ""),
        "description": mt.get("description", ""),
    }

    # Patch value_meaning on the tag-35 field
    msg_type_field["value_meaning"] = mt.get("name", "")

    return msg_info
