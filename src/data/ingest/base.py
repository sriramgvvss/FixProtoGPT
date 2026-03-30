"""
Module: src.data.ingest.base
==============================

Abstract base class and shared data model for all spec parsers.

Every format-specific parser (PDF, DOCX, XML, CSV) inherits from
:class:`SpecParser` and returns a list of :class:`CanonicalSpec`
records — the normalizer then merges and deduplicates them.

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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Canonical data model ──────────────────────────────────────────

class SpecKind(str, Enum):
    """Type of specification record."""

    MESSAGE = "message"
    FIELD = "field"
    COMPONENT = "component"
    ENUM_VALUE = "enum_value"
    DATA_TYPE = "data_type"
    HEADER = "header"
    TRAILER = "trailer"
    RAW_TEXT = "raw_text"


@dataclass
class CanonicalSpec:
    """A single normalised specification record.

    Every parser must produce a list of these.  The normalizer merges
    them by ``(kind, tag, name)`` as the dedup key.

    Attributes:
        kind: The type of spec entry (message, field, etc.).
        tag: FIX tag number (e.g. 35, 55). ``None`` for messages.
        name: Human-readable name (e.g. ``"NewOrderSingle"``).
        msg_type: MsgType value for messages (e.g. ``"D"``).
        data_type: FIX data type (e.g. ``"STRING"``, ``"INT"``).
        required: Whether this field/component is required.
        description: Free-text description or documentation.
        values: For enumerations — maps enum code to label.
        children: Ordered list of child field/component names.
        source: Origin of this record (e.g. ``"official_pdf"``,
            ``"client_overlay"``).
        meta: Arbitrary metadata (page number, line, etc.).
    """

    kind: SpecKind
    tag: Optional[int] = None
    name: str = ""
    msg_type: Optional[str] = None
    data_type: Optional[str] = None
    required: Optional[bool] = None
    description: str = ""
    values: Dict[str, str] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    source: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def dedup_key(self) -> tuple:
        """Return the natural key used for merge/dedup."""
        return (self.kind.value, self.tag, self.name)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        d = asdict(self)
        d["kind"] = self.kind.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CanonicalSpec":
        """Deserialise from a dict (e.g. loaded from JSON)."""
        d = dict(d)
        d["kind"] = SpecKind(d["kind"])
        return cls(**d)


# ── Abstract parser ───────────────────────────────────────────────

class SpecParser(ABC):
    """Interface that every format-specific parser must implement.

    Subclasses register themselves in :data:`PARSER_REGISTRY` via
    the :func:`register_parser` decorator.
    """

    @abstractmethod
    def can_handle(self, path: Path) -> bool:
        """Return ``True`` if this parser supports the given file.

        Args:
            path: Path to the spec file.
        """

    @abstractmethod
    def parse(self, path: Path) -> List[CanonicalSpec]:
        """Parse the file and return canonical records.

        Args:
            path: Path to the spec file.

        Returns:
            List of :class:`CanonicalSpec` records extracted.
        """


# ── Parser registry ───────────────────────────────────────────────

PARSER_REGISTRY: List[SpecParser] = []


def register_parser(parser_cls: type) -> type:
    """Class decorator — add an instance of *parser_cls* to the registry.

    Usage::

        @register_parser
        class MyParser(SpecParser):
            ...
    """
    instance = parser_cls()
    PARSER_REGISTRY.append(instance)
    logger.debug("Registered spec parser: %s", parser_cls.__name__)
    return parser_cls


def get_parser_for(path: Path) -> Optional[SpecParser]:
    """Find the first registered parser that can handle *path*.

    Args:
        path: File to look up.

    Returns:
        A :class:`SpecParser` instance, or ``None``.
    """
    for parser in PARSER_REGISTRY:
        if parser.can_handle(path):
            return parser
    return None
