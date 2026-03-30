"""
Module: src.api.routes.ops
===========================

Operational / sanity-check endpoints.

Available in **all** environments:

* ``GET /ops/health`` — liveness + service status (model, DB, env).

Available only in **dev / qa** (returns 404 in preprod / prod):

* ``GET /ops/config`` — current environment configuration (redacted).
* ``GET /ops/logs``   — tail recent log entries from each log stream.

These endpoints do **not** require authentication so that external
monitors (load-balancers, CI, k8s probes) can reach them.

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
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from flask import Blueprint, abort, jsonify, request

from src.config.env_config import env as env_config
from src.utils import paths

logger = logging.getLogger(__name__)

ops_bp = Blueprint("ops", __name__)

# ── Startup timestamp (for uptime calculation) ────────────────────
_STARTED_AT = time.time()
_STARTED_ISO = datetime.fromtimestamp(_STARTED_AT, tz=timezone.utc).isoformat()

# ── Log directory (env-aware) ─────────────────────────────────────
LOG_DIR = paths.PROJECT_ROOT / "logs" / env_config.ENV_NAME


# ── Helpers ───────────────────────────────────────────────────────

def _dev_qa_only(f):
    """Decorator: return 404 unless running in dev or qa."""
    from functools import wraps

    @wraps(f)
    def wrapper(*args, **kwargs):
        """Return 404 unless the current environment is dev or qa."""
        if env_config.ENV_NAME not in ("dev", "qa"):
            abort(404)
        return f(*args, **kwargs)
    return wrapper


def _check_model() -> Dict[str, Any]:
    """Return model/inference engine status — per-version breakdown."""
    try:
        import src.api.state as state
        engines = state.list_loaded_engines()
        any_loaded = any(e["loaded"] for e in engines.values())
        any_checkpoint = any(e["has_checkpoint"] for e in engines.values())

        versions = {}
        for ver, info in engines.items():
            if info["loaded"]:
                versions[ver] = "loaded"
            elif info["has_checkpoint"]:
                versions[ver] = "available"
            else:
                versions[ver] = "no_checkpoint"

        if any_loaded:
            return {"status": "loaded", "demo_mode": False, "versions": versions}
        if any_checkpoint:
            return {"status": "not_loaded", "demo_mode": True, "message": "Checkpoints available but no engine loaded — demo mode", "versions": versions}
        return {"status": "not_loaded", "demo_mode": True, "message": "No model checkpoint — demo mode active", "versions": versions}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def _check_db() -> Dict[str, Any]:
    """Return quick DB connectivity check for both user and interaction DBs."""
    try:
        import src.api.state as state
        result: Dict[str, Any] = {"status": "ok"}

        # User DB
        mgr = state.user_manager
        if mgr is None:
            result["users"] = {"status": "unavailable"}
        else:
            count = mgr.get_user_count()
            result["users"] = {
                "status": "ok",
                "user_count": count,
                "db_path": str(mgr.db_path),
            }

        # Interactions DB
        il = state.interaction_log
        if il is None:
            result["interactions"] = {"status": "unavailable"}
        else:
            result["interactions"] = {
                "status": "ok",
                "db_path": str(il.db_path),
            }

        # Overall status
        statuses = [
            result.get("users", {}).get("status"),
            result.get("interactions", {}).get("status"),
        ]
        if "error" in statuses:
            result["status"] = "error"
        elif "unavailable" in statuses:
            result["status"] = "degraded"

        return result
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def _tail_log(filepath: Path, lines: int = 50) -> List[str]:
    """Return the last *lines* from a log file (or empty if missing)."""
    if not filepath.is_file():
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            all_lines = fh.readlines()
        return [ln.rstrip() for ln in all_lines[-lines:]]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════
# PUBLIC ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@ops_bp.route("/ops/health", methods=["GET"])
def health():
    """Liveness / readiness probe.

    Returns environment info, uptime, model status and DB status.
    Always available in every environment (no auth required).

    Returns:
        JSON response with ``status``, ``env``, ``uptime_s``,
        ``model``, ``db``.

    Example::

        GET /ops/health
        {
            "status": "ok",
            "env": "dev",
            "started_at": "2026-02-18T10:00:00+00:00",
            "uptime_s": 3600,
            "model": {"status": "loaded", "version": "fix-5-0sp2"},
            "db": {"status": "ok", "user_count": 3}
        }
    """
    uptime = round(time.time() - _STARTED_AT, 1)
    model = _check_model()
    db = _check_db()

    overall = "ok" if model["status"] != "error" and db["status"] != "error" else "degraded"

    return jsonify({
        "status": overall,
        "env": env_config.ENV_NAME,
        "started_at": _STARTED_ISO,
        "uptime_s": uptime,
        "debug": env_config.DEBUG,
        "log_level": env_config.LOG_LEVEL,
        "model": model,
        "db": db,
    })


@ops_bp.route("/ops/config", methods=["GET"])
@_dev_qa_only
def config_view():
    """Show current (redacted) environment configuration.

    Available only in **dev** and **qa** — returns 404 in other envs.
    The ``SECRET_KEY`` is redacted to ``***``.

    Returns:
        JSON response with all config fields.
    """
    return jsonify({
        "env": env_config.ENV_NAME,
        "debug": env_config.DEBUG,
        "secret_key": "***",
        "session_cookie_secure": env_config.SESSION_COOKIE_SECURE,
        "session_cookie_httponly": env_config.SESSION_COOKIE_HTTPONLY,
        "session_cookie_samesite": env_config.SESSION_COOKIE_SAMESITE,
        "session_lifetime": env_config.SESSION_LIFETIME,
        "seed_demo_users": env_config.SEED_DEMO_USERS,
        "cors_origins": env_config.CORS_ORIGINS,
        "log_level": env_config.LOG_LEVEL,
        "host": env_config.HOST,
        "port": env_config.PORT,
        "csp_policy": env_config.CSP_POLICY,
        "hsts_enabled": env_config.HSTS_ENABLED,
        "hsts_max_age": env_config.HSTS_MAX_AGE,
        "log_dir": str(LOG_DIR),
    })


@ops_bp.route("/ops/logs", methods=["GET"])
@_dev_qa_only
def logs_view():
    """Tail recent log entries from each log stream.

    Available only in **dev** and **qa** — returns 404 in other envs.

    Query params:
        ``lines`` — number of lines to return per file (default 50,
        max 500).

    Returns:
        JSON response with ``server``, ``user_actions``, ``debug``
        arrays of recent log lines.
    """
    n = min(int(request.args.get("lines", "50")), 500)
    return jsonify({
        "env": env_config.ENV_NAME,
        "log_dir": str(LOG_DIR),
        "server": _tail_log(LOG_DIR / "server.log", n),
        "user_actions": _tail_log(LOG_DIR / "user_actions.log", n),
        "debug": _tail_log(LOG_DIR / "debug.log", n),
    })
