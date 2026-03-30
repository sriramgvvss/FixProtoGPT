"""
Module: src.data.ingest.xml_parser
====================================

Extract FIX protocol specification data from XML files.

Supports:

* **FIX Orchestra** (``fixr:`` namespace) — the official machine-readable
  specification format from FIX Trading Community.
* **FIXML Schema** (``.xsd`` files with ``FIXML`` in root tag).
* **FIX Repository XML** (``<fix>`` root with ``<messages>``,
  ``<fields>``, ``<components>``).

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET

from src.data.ingest.base import (
    CanonicalSpec,
    SpecKind,
    SpecParser,
    register_parser,
)

logger = logging.getLogger(__name__)

# Common Orchestra / Repository namespaces
_NS: Dict[str, str] = {
    "fixr": "http://fixprotocol.io/2020/orchestra/repository",
    "dcterms": "http://purl.org/dc/terms/",
    "xs": "http://www.w3.org/2001/XMLSchema",
}


@register_parser
class XMLParser(SpecParser):
    """Extract FIX specs from XML repositories and Orchestra files."""

    EXTENSIONS = {".xml", ".xsd"}

    def can_handle(self, path: Path) -> bool:
        """Return ``True`` for XML/XSD files."""
        return path.suffix.lower() in self.EXTENSIONS

    def parse(self, path: Path) -> List[CanonicalSpec]:
        """Parse an XML file and return canonical records.

        Args:
            path: Path to the XML file.

        Returns:
            List of :class:`CanonicalSpec` records.
        """
        if not path.exists():
            raise FileNotFoundError(f"XML not found: {path}")

        logger.info("Parsing XML spec: %s", path.name)

        tree = ET.parse(path)
        root = tree.getroot()
        tag_lower = root.tag.lower()

        # Detect file type from root tag
        if "orchestra" in tag_lower or "repository" in tag_lower:
            return self._parse_orchestra(root, path.name)
        if "fix" in tag_lower:
            return self._parse_fix_repository(root, path.name)
        if "schema" in tag_lower:
            return self._parse_fixml_schema(root, path.name)

        # Generic fallback — try to find <fields>, <messages> anywhere
        specs = self._parse_fix_repository(root, path.name)
        if not specs:
            logger.warning("Unrecognised XML structure in %s", path.name)
        return specs

    # ── FIX Orchestra ─────────────────────────────────────────────

    def _parse_orchestra(
        self, root: ET.Element, source: str,
    ) -> List[CanonicalSpec]:
        """Parse FIX Orchestra (fixr: namespace)."""
        specs: List[CanonicalSpec] = []

        # Fields (with code sets for enums)
        for field_el in root.iter(self._ns("fixr", "field")):
            tag = self._attr_int(field_el, "id")
            name = field_el.attrib.get("name", "")
            dtype = field_el.attrib.get("type", "")

            spec = CanonicalSpec(
                kind=SpecKind.FIELD,
                tag=tag,
                name=name,
                data_type=dtype.upper() if dtype else None,
                source=f"orchestra:{source}",
            )

            # Description from annotation child
            annotation = field_el.find(self._ns("fixr", "annotation"))
            if annotation is not None:
                doc = annotation.find(self._ns("fixr", "documentation"))
                if doc is not None and doc.text:
                    spec.description = doc.text.strip()

            specs.append(spec)

        # Code sets → enum values, linked back to their field
        for codeset in root.iter(self._ns("fixr", "codeSet")):
            cs_name = codeset.attrib.get("name", "")
            for code in codeset.iter(self._ns("fixr", "code")):
                specs.append(
                    CanonicalSpec(
                        kind=SpecKind.ENUM_VALUE,
                        name=cs_name,
                        values={
                            code.attrib.get("value", ""): code.attrib.get(
                                "name", ""
                            )
                        },
                        source=f"orchestra:{source}",
                    )
                )

        # Messages
        for msg_el in root.iter(self._ns("fixr", "message")):
            msg_type = msg_el.attrib.get("msgType", "")
            name = msg_el.attrib.get("name", "")
            children: List[str] = []

            for member in msg_el.iter():
                child_name = member.attrib.get("name", "")
                if child_name and member is not msg_el:
                    children.append(child_name)

            spec = CanonicalSpec(
                kind=SpecKind.MESSAGE,
                msg_type=msg_type,
                name=name,
                children=children,
                source=f"orchestra:{source}",
            )

            annotation = msg_el.find(self._ns("fixr", "annotation"))
            if annotation is not None:
                doc = annotation.find(self._ns("fixr", "documentation"))
                if doc is not None and doc.text:
                    spec.description = doc.text.strip()

            specs.append(spec)

        # Components
        for comp in root.iter(self._ns("fixr", "component")):
            name = comp.attrib.get("name", "")
            children = [
                m.attrib.get("name", "")
                for m in comp.iter()
                if m is not comp and m.attrib.get("name")
            ]
            specs.append(
                CanonicalSpec(
                    kind=SpecKind.COMPONENT,
                    name=name,
                    children=children,
                    source=f"orchestra:{source}",
                )
            )

        logger.info("Orchestra %s → %d records", source, len(specs))
        return specs

    # ── FIX Repository XML ────────────────────────────────────────

    def _parse_fix_repository(
        self, root: ET.Element, source: str,
    ) -> List[CanonicalSpec]:
        """Parse a traditional FIX Repository XML file.

        Expects ``<fields>``, ``<messages>``, ``<components>`` sections.
        """
        specs: List[CanonicalSpec] = []

        # Fields
        fields_section = root.find("fields") or root.find("Fields")
        if fields_section is not None:
            for f in fields_section:
                tag = self._attr_int(f, "number") or self._attr_int(f, "tag")
                name = f.attrib.get("name", f.findtext("Name", ""))
                dtype = f.attrib.get("type", f.findtext("Type", ""))

                spec = CanonicalSpec(
                    kind=SpecKind.FIELD,
                    tag=tag,
                    name=name,
                    data_type=dtype.upper() if dtype else None,
                    source=f"repository:{source}",
                )

                # Enum values
                for val in f.findall("value") + f.findall("Value"):
                    enum_key = val.attrib.get("enum", val.findtext(".", ""))
                    desc = val.attrib.get("description", val.text or "")
                    if enum_key:
                        spec.values[enum_key] = desc

                specs.append(spec)

        # Messages
        messages_section = root.find("messages") or root.find("Messages")
        if messages_section is not None:
            for m in messages_section:
                msg_type = m.attrib.get("msgtype", m.attrib.get("MsgType", ""))
                name = m.attrib.get("name", m.findtext("Name", ""))
                children = [
                    c.attrib.get("name", "")
                    for c in m
                    if c.attrib.get("name")
                ]
                specs.append(
                    CanonicalSpec(
                        kind=SpecKind.MESSAGE,
                        msg_type=msg_type,
                        name=name,
                        children=children,
                        source=f"repository:{source}",
                    )
                )

        # Components
        components_section = root.find("components") or root.find("Components")
        if components_section is not None:
            for c in components_section:
                name = c.attrib.get("name", c.findtext("Name", ""))
                children = [
                    ch.attrib.get("name", "")
                    for ch in c
                    if ch.attrib.get("name")
                ]
                specs.append(
                    CanonicalSpec(
                        kind=SpecKind.COMPONENT,
                        name=name,
                        children=children,
                        source=f"repository:{source}",
                    )
                )

        logger.info("Repository %s → %d records", source, len(specs))
        return specs

    # ── FIXML Schema (.xsd) ──────────────────────────────────────

    def _parse_fixml_schema(
        self, root: ET.Element, source: str,
    ) -> List[CanonicalSpec]:
        """Parse a FIXML XSD schema for element/type definitions."""
        specs: List[CanonicalSpec] = []

        for element in root.iter(self._ns("xs", "element")):
            name = element.attrib.get("name", "")
            if not name:
                continue
            specs.append(
                CanonicalSpec(
                    kind=SpecKind.COMPONENT,
                    name=name,
                    source=f"schema:{source}",
                )
            )

        for simple in root.iter(self._ns("xs", "simpleType")):
            name = simple.attrib.get("name", "")
            values: Dict[str, str] = {}
            for enum in simple.iter(self._ns("xs", "enumeration")):
                val = enum.attrib.get("value", "")
                if val:
                    values[val] = val
            if name:
                specs.append(
                    CanonicalSpec(
                        kind=SpecKind.ENUM_VALUE if values else SpecKind.DATA_TYPE,
                        name=name,
                        values=values,
                        source=f"schema:{source}",
                    )
                )

        logger.info("FIXML schema %s → %d records", source, len(specs))
        return specs

    # ── Utility ───────────────────────────────────────────────────

    @staticmethod
    def _ns(prefix: str, local: str) -> str:
        """Build a Clark-notation tag ``{namespace}local``."""
        uri = _NS.get(prefix, "")
        return f"{{{uri}}}{local}" if uri else local

    @staticmethod
    def _attr_int(el: ET.Element, attr: str) -> Optional[int]:
        """Read an integer attribute, or ``None``."""
        val = el.attrib.get(attr)
        if val is None:
            return None
        try:
            return int(val)
        except ValueError:
            return None
