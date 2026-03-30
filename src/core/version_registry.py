"""
Module: src.core.version_registry
==================================

FIX Protocol version registry.

Defines every supported FIX version with its metadata and provides
helpers to discover which versions are "installed" (have data and/or
a trained checkpoint on disk).

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team

Usage::

    from src.core.version_registry import VERSIONS, get_version_info, list_installed

    info = get_version_info("5.0SP2")  # → dict with begin_string, appl_ver_id, …
    installed = list_installed()        # → [{"version": "5.0SP2", "has_model": True, …}]
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from src.utils import paths


# ── Version metadata ──────────────────────────────────────────────

@dataclass(frozen=True)
class FIXVersionInfo:
    """Immutable descriptor for a FIX protocol version."""
    version: str          # e.g. "4.4", "5.0SP2"
    label: str            # human-friendly, e.g. "FIX 4.4"
    protocol: str         # full protocol id, e.g. "FIX.4.4"
    begin_string: str     # tag 8 value
    session: str          # session-layer protocol
    appl_ver_id: Optional[str] = None   # tag 1128 (FIX 5.x only)
    family: str = "4"     # "4" or "5"

    def to_dict(self) -> dict:
        """Serialise this info object to a plain dict."""
        return asdict(self)


# ── Registry of all known versions ────────────────────────────────

_VERSIONS: Dict[str, FIXVersionInfo] = {}


def _register(*versions: FIXVersionInfo) -> None:
    """Add one or more :class:`FIXVersionInfo` instances to the registry."""
    for v in versions:
        _VERSIONS[v.version] = v


_register(
    FIXVersionInfo("4.2", "FIX 4.2", "FIX.4.2", "FIX.4.2", "FIX.4.2", family="4"),
    FIXVersionInfo("4.4", "FIX 4.4", "FIX.4.4", "FIX.4.4", "FIX.4.4", family="4"),
    FIXVersionInfo("5.0SP2", "FIX 5.0 SP2",  "FIX.5.0SP2", "FIXT.1.1", "FIXT.1.1", appl_ver_id="9", family="5"),
    FIXVersionInfo("Latest", "FIX Latest", "FIX.Latest", "FIXT.1.1", "FIXT.1.1", appl_ver_id="11", family="5"),
)

# ── Public API ────────────────────────────────────────────────────

VERSIONS = _VERSIONS  # read-only reference


def get_version_info(version: str) -> Optional[FIXVersionInfo]:
    """Return metadata for a version string, or ``None``."""
    return _VERSIONS.get(version)


def all_version_keys() -> List[str]:
    """Return all registered version keys in order."""
    return list(_VERSIONS.keys())


def is_valid_version(version: str) -> bool:
    """Return ``True`` if *version* is in the registry."""
    return version in _VERSIONS


def list_installed() -> List[dict]:
    """Scan ``model_store/`` and return status of each known version.

    Each entry has::

        {
            "version": "5.0SP2",
            "label": "FIX 5.0 SP2",
            "has_data": True/False,
            "has_model": True/False,
            "active": True/False,      # matches YAML config active version
            ...FIXVersionInfo fields...
        }
    """
    active = paths.active_version()
    result: List[dict] = []

    for key, info in _VERSIONS.items():
        d = info.to_dict()
        d["has_data"] = paths.processed_data_dir(key).exists()
        d["has_model"] = paths.best_model(key).exists()
        d["active"] = (key == active)
        result.append(d)

    return result


def default_version() -> str:
    """Return the configured default version string from YAML.

    Returns:
        Active version key, e.g. ``"5.0SP2"``.
    """
    return paths.active_version()
