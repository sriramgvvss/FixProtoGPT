"""
Module: src.data.ingest.client_overlay
========================================

Per-client specification overlays.

A *client overlay* is a set of custom FIX specs (custom tags, modified
enumerations, bespoke message layouts) that sits on top of the official
base spec.  Each client gets an isolated directory::

    model_store/data/<version_slug>/overlays/<client_id>/
        ├── canonical.json   ← merged client-specific canonical specs
        ├── training.txt     ← client-specific training lines
        └── uploads/         ← raw uploaded spec files

Workflow
--------
1. Admin uploads spec files via API  → saved to ``uploads/``.
2. :func:`ingest_client_specs` parses them and saves ``canonical.json``.
3. :func:`build_client_training_data` merges base + overlay → ``training.txt``.
4. Fine-tuner reads ``training.txt`` to produce a client checkpoint.

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
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from src.data.ingest.base import CanonicalSpec
from src.data.ingest.normalizer import (
    ingest_directory,
    load_canonical,
    merge_specs,
    specs_to_training_lines,
)
from src.utils import paths

logger = logging.getLogger(__name__)


# ── Directory layout helpers ──────────────────────────────────────

def client_overlay_dir(
    client_id: str,
    version: Optional[str] = None,
) -> Path:
    """Return the overlay root for a client.

    ``model_store/data/<version_slug>/overlays/<client_id>/``
    """
    return paths.data_dir(version) / "overlays" / client_id


def client_uploads_dir(
    client_id: str,
    version: Optional[str] = None,
) -> Path:
    """``overlays/<client_id>/uploads/``"""
    return client_overlay_dir(client_id, version) / "uploads"


def client_canonical_path(
    client_id: str,
    version: Optional[str] = None,
) -> Path:
    """``overlays/<client_id>/canonical.json``"""
    return client_overlay_dir(client_id, version) / "canonical.json"


def client_training_path(
    client_id: str,
    version: Optional[str] = None,
) -> Path:
    """``overlays/<client_id>/training.txt``"""
    return client_overlay_dir(client_id, version) / "training.txt"


# ── Client management ────────────────────────────────────────────

def list_clients(version: Optional[str] = None) -> List[str]:
    """Return all client IDs that have overlay directories."""
    overlay_root = paths.data_dir(version) / "overlays"
    if not overlay_root.is_dir():
        return []
    return sorted(
        d.name for d in overlay_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def create_client(
    client_id: str,
    version: Optional[str] = None,
) -> Path:
    """Ensure the overlay directory tree exists for *client_id*.

    Returns the overlay root directory.
    """
    overlay = client_overlay_dir(client_id, version)
    uploads = client_uploads_dir(client_id, version)
    uploads.mkdir(parents=True, exist_ok=True)
    logger.info("Created client overlay dir: %s", overlay)
    return overlay


def delete_client(
    client_id: str,
    version: Optional[str] = None,
) -> bool:
    """Remove a client's overlay directory and all its data.

    Also removes the client's checkpoint if one exists.

    Returns ``True`` if something was deleted.
    """
    overlay = client_overlay_dir(client_id, version)
    ckpt = paths.data_dir(version).parent.parent / "checkpoints" / paths._version_to_slug(
        version or paths.active_version()
    ) / "clients" / client_id
    deleted = False

    if overlay.is_dir():
        shutil.rmtree(overlay)
        logger.info("Deleted overlay dir: %s", overlay)
        deleted = True

    if ckpt.is_dir():
        shutil.rmtree(ckpt)
        logger.info("Deleted client checkpoint dir: %s", ckpt)
        deleted = True

    return deleted


# ── Spec ingestion ────────────────────────────────────────────────

def save_uploaded_file(
    client_id: str,
    filename: str,
    data: bytes,
    version: Optional[str] = None,
) -> Path:
    """Persist a raw uploaded file to the client's ``uploads/`` dir.

    Args:
        client_id: Client identifier.
        filename: Original filename (e.g. ``"custom_tags.pdf"``).
        data: Raw file bytes.
        version: FIX version.

    Returns:
        Path to the saved file.
    """
    uploads = client_uploads_dir(client_id, version)
    uploads.mkdir(parents=True, exist_ok=True)
    dest = uploads / filename

    with open(dest, "wb") as fh:
        fh.write(data)

    logger.info("Saved upload %s for client %s (%d bytes)",
                filename, client_id, len(data))
    return dest


def ingest_client_specs(
    client_id: str,
    version: Optional[str] = None,
) -> List[CanonicalSpec]:
    """Parse all uploaded files for a client into canonical JSON.

    Returns the list of client-specific canonical records.
    """
    uploads = client_uploads_dir(client_id, version)
    if not uploads.is_dir():
        logger.warning("No uploads dir for client %s", client_id)
        return []

    # Parse everything in the uploads dir
    specs = ingest_directory(uploads, version=version, save=False)

    # Tag every record with the client source
    for spec in specs:
        spec.source = f"client:{client_id},{spec.source}"

    # Save client-specific canonical JSON
    merged = merge_specs(specs)
    out_path = client_canonical_path(client_id, version)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as fh:
        json.dump([s.to_dict() for s in merged], fh, indent=2)

    logger.info(
        "Client %s: ingested %d records → %s",
        client_id, len(merged), out_path,
    )
    return merged


def load_client_canonical(
    client_id: str,
    version: Optional[str] = None,
) -> List[CanonicalSpec]:
    """Load previously saved client canonical records."""
    path = client_canonical_path(client_id, version)
    if not path.exists():
        return []
    with open(path) as fh:
        data = json.load(fh)
    return [CanonicalSpec.from_dict(d) for d in data]


# ── Training data generation ─────────────────────────────────────

def build_client_training_data(
    client_id: str,
    version: Optional[str] = None,
) -> Path:
    """Build merged training text: base specs + client overlay.

    The result is written to ``overlays/<client_id>/training.txt``
    and is ready for the fine-tuner to consume.

    Args:
        client_id: Client identifier.
        version: FIX version.

    Returns:
        Path to the generated ``training.txt``.
    """
    # 1. Load base canonical specs
    base_specs = load_canonical(version)

    # 2. Load client-specific overlay
    client_specs = load_client_canonical(client_id, version)

    # 3. Merge: client overrides base for same dedup_key
    merged = merge_specs(base_specs + client_specs)

    # 4. Convert to training lines
    lines = specs_to_training_lines(merged)

    # 5. Also include existing base training text (scraped + synthetic)
    base_train = paths.processed_data_dir(version) / "train.txt"
    base_lines: List[str] = []
    if base_train.exists():
        with open(base_train) as fh:
            base_lines = [l.rstrip("\n") for l in fh if l.strip()]

    all_lines = base_lines + lines

    # 6. Write client training file
    out = client_training_path(client_id, version)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as fh:
        fh.write("\n".join(all_lines) + "\n")

    logger.info(
        "Client %s training data: %d lines (%d base + %d overlay) → %s",
        client_id, len(all_lines), len(base_lines), len(lines), out,
    )
    return out


def get_client_stats(
    client_id: str,
    version: Optional[str] = None,
) -> Dict:
    """Return summary statistics for a client overlay."""
    overlay = client_overlay_dir(client_id, version)
    uploads = client_uploads_dir(client_id, version)
    canonical = client_canonical_path(client_id, version)
    training = client_training_path(client_id, version)

    upload_files = list(uploads.glob("*")) if uploads.is_dir() else []
    spec_count = 0
    if canonical.exists():
        with open(canonical) as fh:
            spec_count = len(json.load(fh))

    training_lines = 0
    if training.exists():
        with open(training) as fh:
            training_lines = sum(1 for _ in fh)

    # Check if client checkpoint exists
    from src.utils.paths import _version_to_slug
    slug = _version_to_slug(version or paths.active_version())
    ckpt_dir = paths.MODEL_STORE / "checkpoints" / slug / "clients" / client_id
    has_checkpoint = (ckpt_dir / "best.pt").exists() if ckpt_dir.is_dir() else False

    return {
        "client_id": client_id,
        "overlay_dir": str(overlay),
        "upload_count": len(upload_files),
        "upload_files": [f.name for f in upload_files],
        "spec_count": spec_count,
        "training_lines": training_lines,
        "has_checkpoint": has_checkpoint,
    }
