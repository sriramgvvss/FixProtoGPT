"""
Package: src.data.ingest
=========================

Multi-format ingestion framework for FIX protocol specifications.

Supports PDF, DOCX, XML (FIX Orchestra / FIXML), and CSV tag
definition files.  Each parser implements the :class:`SpecParser`
interface and produces a list of :class:`CanonicalSpec` records that
the normalizer merges into a unified JSON representation.

Public API::

    from src.data.ingest import ingest_file, ingest_directory

    specs = ingest_file("/path/to/spec.pdf")
    all_specs = ingest_directory("/path/to/specs/")

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from src.data.ingest.base import CanonicalSpec, SpecParser
from src.data.ingest.normalizer import ingest_file, ingest_directory

# Import all parsers so that @register_parser decorators fire and
# populate PARSER_REGISTRY.  Without these imports, get_parser_for()
# would always return None.
import src.data.ingest.pdf_parser   # noqa: F401
import src.data.ingest.csv_parser   # noqa: F401
import src.data.ingest.xml_parser   # noqa: F401
import src.data.ingest.docx_parser  # noqa: F401

__all__ = [
    "CanonicalSpec",
    "SpecParser",
    "ingest_file",
    "ingest_directory",
]
