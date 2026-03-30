"""
Module: src.utils.paths
========================

Centralised path resolution for FixProtoGPT model store.

All version-specific paths are derived from the ``active`` version
in ``config/model_config.yaml → version.active``.  Every module that
needs a checkpoint, tokenizer, or data path should import helpers
from here instead of hard-coding the version slug.

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team

Usage::

    from src.utils.paths import paths

    paths.best_model()          # → <root>/model_store/checkpoints/fix-5-0sp2/best.pt
    paths.tokenizer_dir()       # → <root>/model_store/data/fix-5-0sp2/processed/tokenizer
    paths.active_version()      # → "5.0SP2"
    paths.active_protocol()     # → "FIX.5.0SP2"
    paths.session_protocol()    # → "FIXT.1.1"

    # Parameterized for any version:
    paths.best_model_for("4.4") # → <root>/model_store/checkpoints/fix-4-4/best.pt
"""

from __future__ import annotations

import yaml
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional

# ── Project root (repo top-level) ─────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.yaml"
MODEL_STORE = PROJECT_ROOT / "model_store"


# ── Config loading ────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """Load the full model_config.yaml as a dict (cached)."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _load_version_config() -> Dict[str, Any]:
    """Load the ``version`` section from model_config.yaml."""
    return load_config().get("version", {})


def _version_to_slug(version: str) -> str:
    """Convert a version string to a filesystem slug.

    ``"5.0SP2"`` → ``"fix-5-0sp2"``
    """
    return "fix-" + version.lower().replace(".", "-")


# ── Public helpers (default = active version from YAML) ───────────

def active_version() -> str:
    """Active FIX application version, e.g. ``"5.0SP2"``."""
    return _load_version_config().get("active", "5.0SP2")


def active_protocol() -> str:
    """Full protocol identifier, e.g. ``"FIX.5.0SP2"``."""
    return _load_version_config().get("protocol", f"FIX.{active_version()}")


def session_protocol() -> str:
    """FIXT session-layer version, e.g. ``"FIXT.1.1"``."""
    return _load_version_config().get("session", "FIXT.1.1")


def version_slug() -> str:
    """Filesystem-safe slug for the active version, e.g. ``"fix-5-0sp2"``."""
    return _version_to_slug(active_version())


# ── Checkpoint paths ──────────────────────────────────────────────

def checkpoint_dir(version: Optional[str] = None) -> Path:
    """``model_store/checkpoints/<version_slug>/``"""
    slug = _version_to_slug(version) if version else version_slug()
    return MODEL_STORE / "checkpoints" / slug


def best_model(version: Optional[str] = None) -> Path:
    """``model_store/checkpoints/<version_slug>/best.pt``"""
    return checkpoint_dir(version) / "best.pt"


# ── Data paths ────────────────────────────────────────────────────

def data_dir(version: Optional[str] = None) -> Path:
    """``model_store/data/<version_slug>/``"""
    slug = _version_to_slug(version) if version else version_slug()
    return MODEL_STORE / "data" / slug


def raw_data_dir(version: Optional[str] = None) -> Path:
    """``model_store/data/<version_slug>/raw/``"""
    return data_dir(version) / "raw"


def processed_data_dir(version: Optional[str] = None) -> Path:
    """``model_store/data/<version_slug>/processed/``"""
    return data_dir(version) / "processed"


def tokenizer_dir(version: Optional[str] = None) -> Path:
    """``model_store/data/<version_slug>/processed/tokenizer/``"""
    return processed_data_dir(version) / "tokenizer"


def train_data(version: Optional[str] = None) -> Path:
    """``model_store/data/<version_slug>/processed/train.bin``"""
    return processed_data_dir(version) / "train.bin"


def val_data(version: Optional[str] = None) -> Path:
    """``model_store/data/<version_slug>/processed/val.bin``"""
    return processed_data_dir(version) / "val.bin"


# ── Shared (version-independent) paths ────────────────────────────

def symbols_dir() -> Path:
    """``model_store/data/symbols/`` — shared across all FIX versions."""
    return MODEL_STORE / "data" / "symbols"


def symbol_cache_path() -> Path:
    """``model_store/data/symbol_cache.json`` — shared across all FIX versions."""
    return MODEL_STORE / "data" / "symbol_cache.json"


# ── Client-specific paths ────────────────────────────────────────

def client_checkpoint_dir(
    client_id: str,
    version: Optional[str] = None,
) -> Path:
    """``model_store/checkpoints/<version_slug>/clients/<client_id>/``"""
    return checkpoint_dir(version) / "clients" / client_id


def client_best_model(
    client_id: str,
    version: Optional[str] = None,
) -> Path:
    """``model_store/checkpoints/<ver>/clients/<client_id>/best.pt``"""
    return client_checkpoint_dir(client_id, version) / "best.pt"


def client_overlay_dir(
    client_id: str,
    version: Optional[str] = None,
) -> Path:
    """``model_store/data/<version_slug>/overlays/<client_id>/``"""
    return data_dir(version) / "overlays" / client_id


def client_training_data(
    client_id: str,
    version: Optional[str] = None,
) -> Path:
    """``model_store/data/<ver>/overlays/<client_id>/training.txt``"""
    return client_overlay_dir(client_id, version) / "training.txt"
