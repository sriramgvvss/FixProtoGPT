"""
Module: src.api.state
======================

Shared application state for FixProtoGPT.

Holds the global inference engine(s) and interaction logger so that
Flask Blueprints can import them without circular dependencies.

Engine Registry
---------------
Instead of a single ``inference_engine``, we keep a dict keyed by
version string (e.g. ``"5.0SP2"``).  ``load_model(version)`` lazily
loads the right checkpoint and caches it.  The legacy
``inference_engine`` global is kept as an alias to the *default*
version's engine for backward compatibility.

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from src.persistence.interaction_logger import InteractionLogger
from src.persistence.user_manager import UserManager
from src.core.version_registry import default_version
from src.utils import paths

logger = logging.getLogger(__name__)

# ── Engine registry (version → FixProtoGPTInference | None) ──────
_engines: Dict[str, object] = {}

# Legacy alias — points to the default version's engine
inference_engine = None

# Interaction logger — captures all conversations for future fine-tuning
interaction_log = InteractionLogger()

# User manager — authentication and account management
user_manager = UserManager()


def load_model(version: Optional[str] = None):
    """Load (or return cached) inference engine for *version*.

    Parameters
    ----------
    version : str or None
        FIX version key, e.g. ``"5.0SP2"`` or ``"4.4"``.
        ``None`` → the YAML default version.

    Returns
    -------
    FixProtoGPTInference or None
        The engine, or ``None`` if the checkpoint doesn't exist.
    """
    global inference_engine

    if version is None:
        version = default_version()

    # Return cached engine if already loaded
    if version in _engines:
        engine = _engines[version]
        # Keep legacy alias in sync with default version
        if version == default_version():
            inference_engine = engine
        return engine

    try:
        from src.inference.generate import FixProtoGPTInference

        model_path = str(paths.best_model(version))
        tokenizer_path = str(paths.tokenizer_dir(version))

        if not Path(model_path).exists():
            logger.warning(
                "Model not found for version %s at %s — demo mode.",
                version, model_path,
            )
            _engines[version] = None
            return None

        engine = FixProtoGPTInference(model_path, tokenizer_path,
                                       fix_version=version)
        _engines[version] = engine
        logger.info("Model loaded for FIX %s", version)

        # Keep legacy alias for the default version
        if version == default_version():
            inference_engine = engine

        return engine

    except Exception as e:
        logger.error("Error loading model for FIX %s: %s", version, e)
        _engines[version] = None
        return None


def unload_model(version: Optional[str] = None) -> None:
    """Remove a cached engine for *version*.

    The next call to :func:`load_model` will reload from disk.

    Args:
        version: FIX version key.  Defaults to the configured default.
    """
    global inference_engine

    if version is None:
        version = default_version()

    _engines.pop(version, None)
    if version == default_version():
        inference_engine = None


def get_engine(version: Optional[str] = None):
    """Get the inference engine for *version*, loading lazily.

    Args:
        version: FIX version key.  Defaults to the configured default.

    Returns:
        :class:`FixProtoGPTInference` instance or ``None``.
    """
    return load_model(version)


# ── Client-specific engine registry ──────────────────────────────
# Keys are "(version, client_id)" tuples.
_client_engines: Dict[str, object] = {}


def _client_key(client_id: str, version: Optional[str] = None) -> str:
    """Build a cache key for a client engine."""
    v = version or default_version()
    return f"{v}:{client_id}"


def load_client_model(client_id: str, version: Optional[str] = None):
    """Load (or return cached) inference engine for a client.

    Falls back to the base model if no client checkpoint exists.

    Parameters
    ----------
    client_id : str
        Client identifier.
    version : str or None
        FIX version key.

    Returns
    -------
    FixProtoGPTInference or None
    """
    key = _client_key(client_id, version)
    if key in _client_engines:
        return _client_engines[key]

    try:
        from src.inference.generate import FixProtoGPTInference

        client_ckpt = paths.client_best_model(client_id, version)
        tokenizer_path = str(paths.tokenizer_dir(version))

        if client_ckpt.exists():
            engine = FixProtoGPTInference(
                str(client_ckpt), tokenizer_path,
                fix_version=version or default_version(),
            )
            _client_engines[key] = engine
            logger.info(
                "Client %s model loaded for FIX %s",
                client_id, version or default_version(),
            )
            return engine

        # No client checkpoint → fall back to base model
        logger.info(
            "No checkpoint for client %s — using base model", client_id,
        )
        base_engine = load_model(version)
        _client_engines[key] = base_engine
        return base_engine

    except Exception as e:
        logger.error(
            "Error loading client %s model: %s", client_id, e,
        )
        _client_engines[key] = None
        return None


def unload_client_model(client_id: str, version: Optional[str] = None) -> None:
    """Remove a cached client engine.

    The next call to :func:`load_client_model` will reload from disk.
    """
    key = _client_key(client_id, version)
    _client_engines.pop(key, None)


def get_client_engine(client_id: str, version: Optional[str] = None):
    """Get the inference engine for a client, loading lazily.

    Falls back to the base model if the client has no checkpoint.
    """
    return load_client_model(client_id, version)


# ── Introspection ────────────────────────────────────────────────

def list_loaded_engines() -> Dict[str, dict]:
    """Return status of every known FIX version's engine.

    Returns a dict keyed by version string with fields:

    - ``loaded`` (bool): engine is cached and not ``None``
    - ``cached`` (bool): version key exists in the cache (even if ``None``)
    - ``has_checkpoint`` (bool): ``best.pt`` exists on disk
    - ``fix_version`` (str): version key

    This is the single source of truth for model status and should be
    used by the control panel, admin CLI, API, and health endpoint.
    """
    from src.core.version_registry import all_version_keys

    result: Dict[str, dict] = {}
    for ver in all_version_keys():
        engine = _engines.get(ver)
        result[ver] = {
            "fix_version": ver,
            "loaded": engine is not None,
            "cached": ver in _engines,
            "has_checkpoint": paths.best_model(ver).exists(),
        }
    return result


def unload_all_models() -> int:
    """Unload every cached engine. Returns the count of engines removed."""
    global inference_engine
    count = sum(1 for e in _engines.values() if e is not None)
    _engines.clear()
    _client_engines.clear()
    inference_engine = None
    return count

