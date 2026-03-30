"""
Module: src.data.ingest.normalizer
====================================

Merge, deduplicate, and persist canonical spec records.

Public API
----------
* :func:`ingest_file`  — parse one file → merged canonical JSON.
* :func:`ingest_directory` — parse all supported files in a dir.
* :func:`merge_specs` — merge/dedup a list of :class:`CanonicalSpec`.
* :func:`specs_to_training_lines` — convert specs into training text.

The canonical JSON output is saved to
``model_store/data/<version>/specs/canonical.json``.

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data.ingest.base import (
    CanonicalSpec,
    SpecKind,
    get_parser_for,
)
from src.utils import paths

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────

def ingest_file(
    file_path: str | Path,
    version: Optional[str] = None,
    save: bool = True,
) -> List[CanonicalSpec]:
    """Parse a single spec file and optionally save canonical JSON.

    Args:
        file_path: Path to the specification file.
        version: FIX version (default from config).
        save: Whether to persist the merged result.

    Returns:
        List of canonical records extracted.

    Raises:
        ValueError: If no parser can handle the file format.
    """
    file_path = Path(file_path)
    parser = get_parser_for(file_path)
    if parser is None:
        raise ValueError(
            f"No parser registered for {file_path.suffix!r}. "
            f"Supported: .pdf, .docx, .xml, .xsd, .csv, .tsv, .xlsx"
        )

    specs = parser.parse(file_path)
    logger.info("Ingested %d records from %s", len(specs), file_path.name)

    if save:
        _save_canonical(specs, version)

    return specs


def ingest_directory(
    dir_path: str | Path,
    version: Optional[str] = None,
    recursive: bool = True,
    save: bool = True,
) -> List[CanonicalSpec]:
    """Parse all supported files in a directory.

    Args:
        dir_path: Directory containing spec files.
        version: FIX version (default from config).
        recursive: Whether to descend into subdirectories.
        save: Whether to persist the merged result.

    Returns:
        Combined list of all canonical records.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    pattern = "**/*" if recursive else "*"
    all_specs: List[CanonicalSpec] = []

    for child in sorted(dir_path.glob(pattern)):
        if not child.is_file():
            continue
        parser = get_parser_for(child)
        if parser is None:
            continue
        try:
            specs = parser.parse(child)
            all_specs.extend(specs)
            logger.info("  %s → %d records", child.name, len(specs))
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", child.name, exc)

    if save and all_specs:
        _save_canonical(all_specs, version)

    logger.info(
        "Directory %s → %d total records from %s",
        dir_path, len(all_specs), dir_path,
    )
    return all_specs


# ── Merge / dedup ─────────────────────────────────────────────────

def merge_specs(specs: List[CanonicalSpec]) -> List[CanonicalSpec]:
    """Merge and deduplicate canonical records.

    When two records share the same ``dedup_key()``, later records
    (higher index) override earlier ones for non-empty fields.

    Args:
        specs: Flat list of canonical records from any source.

    Returns:
        Deduplicated list, preserving insertion order of first occurrence.
    """
    index: Dict[tuple, CanonicalSpec] = {}

    for spec in specs:
        key = spec.dedup_key()
        if key in index:
            existing = index[key]
            # Merge: later overrides empty fields in earlier
            if spec.name:
                existing.name = spec.name
            if spec.data_type:
                existing.data_type = spec.data_type
            if spec.msg_type:
                existing.msg_type = spec.msg_type
            if spec.description:
                existing.description = spec.description
            if spec.required is not None:
                existing.required = spec.required
            if spec.values:
                existing.values.update(spec.values)
            if spec.children:
                existing.children = spec.children
            # Track all sources
            if spec.source and spec.source not in existing.source:
                existing.source += f",{spec.source}"
        else:
            index[key] = spec

    merged = list(index.values())
    logger.debug("Merged %d → %d canonical records", len(specs), len(merged))
    return merged


# ── Training-line generation ──────────────────────────────────────

def specs_to_training_lines(specs: List[CanonicalSpec]) -> List[str]:
    """Convert canonical specs into text lines suitable for training.

    Produces one or more training lines per record:

    * Fields → ``"Tag 55 (Symbol) : STRING — Ticker symbol for the security"``
    * Messages → ``"Message D (NewOrderSingle): ClOrdID, Side, OrderQty …"``
    * Enums → ``"Field OrdType values: 1=Market, 2=Limit …"``
    * Raw text → passed through verbatim.

    Args:
        specs: Canonical records.

    Returns:
        List of training-ready text lines.
    """
    lines: List[str] = []

    for spec in specs:
        if spec.kind == SpecKind.FIELD:
            parts = [f"Tag {spec.tag}"]
            if spec.name:
                parts[0] += f" ({spec.name})"
            if spec.data_type:
                parts.append(f": {spec.data_type}")
            if spec.required is not None:
                parts.append(f"[{'Required' if spec.required else 'Optional'}]")
            if spec.description:
                parts.append(f"— {spec.description}")
            lines.append(" ".join(parts))

            # Enum values as a separate training line
            if spec.values:
                vals = ", ".join(
                    f"{k}={v}" for k, v in spec.values.items()
                )
                lines.append(
                    f"Field {spec.name or spec.tag} values: {vals}"
                )

        elif spec.kind == SpecKind.MESSAGE:
            line = f"Message {spec.msg_type or '?'}"
            if spec.name:
                line += f" ({spec.name})"
            if spec.children:
                line += f": {', '.join(spec.children[:20])}"
            if spec.description:
                line += f" — {spec.description}"
            lines.append(line)

        elif spec.kind == SpecKind.COMPONENT:
            line = f"Component {spec.name}"
            if spec.children:
                line += f": {', '.join(spec.children[:20])}"
            lines.append(line)

        elif spec.kind == SpecKind.ENUM_VALUE:
            if spec.values:
                vals = ", ".join(
                    f"{k}={v}" for k, v in spec.values.items()
                )
                lines.append(f"Enum {spec.name}: {vals}")

        elif spec.kind == SpecKind.DATA_TYPE:
            lines.append(f"DataType {spec.name}")

        elif spec.kind == SpecKind.RAW_TEXT:
            if spec.description:
                lines.append(spec.description)

    return lines


# ── Persistence ───────────────────────────────────────────────────

def canonical_json_path(version: Optional[str] = None) -> Path:
    """Return the canonical JSON path for a FIX version.

    ``model_store/data/<version_slug>/specs/canonical.json``
    """
    return paths.data_dir(version) / "specs" / "canonical.json"


def load_canonical(version: Optional[str] = None) -> List[CanonicalSpec]:
    """Load previously saved canonical records from JSON.

    Args:
        version: FIX version (default from config).

    Returns:
        List of :class:`CanonicalSpec`, or empty list if file missing.
    """
    path = canonical_json_path(version)
    if not path.exists():
        return []

    with open(path) as fh:
        data = json.load(fh)

    return [CanonicalSpec.from_dict(d) for d in data]


def _save_canonical(
    new_specs: List[CanonicalSpec],
    version: Optional[str] = None,
) -> Path:
    """Merge *new_specs* with existing canonical and persist.

    Returns the path to the saved JSON file.
    """
    existing = load_canonical(version)
    merged = merge_specs(existing + new_specs)

    out = canonical_json_path(version)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as fh:
        json.dump([s.to_dict() for s in merged], fh, indent=2)

    logger.info("Saved %d canonical records → %s", len(merged), out)
    return out
