"""
Module: src.persistence.action_logger
======================================

Structured logging for FixProtoGPT — server requests and user actions.

Provides three log files under ``logs/<env_name>/``:

* **server.log** — every HTTP request/response (method, path, status,
  duration, IP, user).  Rotated daily, 30-day retention.
* **user_actions.log** — meaningful user actions (login, register,
  logout, generate, nl2fix, explain, validate, export, feedback,
  admin operations).  Rotated daily, 90-day retention.
* **debug.log** — verbose DEBUG-level output (SQL queries, resolver
  lookups, internal operations).  Only written when the log level
  is set to ``DEBUG``.  Rotated daily, 14-day retention.

Each environment (dev, qa, preprod, prod) writes to its own
subdirectory so logs never mix across environments.

All loggers use structured JSON lines for easy parsing by log
aggregators (ELK, Datadog, Loki, etc.).

Log Level Configuration
-----------------------
Set the ``FIXPROTOGPT_LOG_LEVEL`` environment variable to control
verbosity.  Accepted values: ``DEBUG``, ``INFO`` (default), ``WARNING``,
``ERROR``, ``CRITICAL``.

* ``INFO`` (default) — server.log + user_actions.log only.
* ``DEBUG`` — additionally writes debug.log with fine-grained output.

Usage::

    from src.persistence.action_logger import (
        setup_logging, log_user_action, log_debug,
    )

    # In app factory:
    setup_logging(app)

    # In a route handler:
    log_user_action("generate", user_id="U00001", username="trader",
                    detail={"prompt": "Buy 100 AAPL", "demo_mode": True})

    # Anywhere for debug output:
    log_debug("symbol_resolve", detail={"query": "apple", "result": "AAPL"})

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
import os
import time
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, request, session, g

from src.utils import paths

# ── Paths ─────────────────────────────────────────────────────────
# Logs are written to logs/<env_name>/ so each environment keeps
# its own isolated history.
try:
    from src.config.env_config import env as _env_cfg
    LOG_DIR = paths.PROJECT_ROOT / "logs" / _env_cfg.ENV_NAME
except Exception:
    # Fallback for early import / test scenarios
    LOG_DIR = paths.PROJECT_ROOT / "logs"

# ── Logger names (importable for testing) ─────────────────────────

SERVER_LOGGER_NAME = "fixprotogpt.server"
ACTION_LOGGER_NAME = "fixprotogpt.actions"
DEBUG_LOGGER_NAME = "fixprotogpt.debug"

# ── Log level from environment ────────────────────────────────────

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _get_log_level() -> int:
    """Return the effective log level.

    Checks ``FIXPROTOGPT_LOG_LEVEL`` env var first (allows per-run
    override), then falls back to ``env_config.LOG_LEVEL``, then ``INFO``.
    """
    raw = os.environ.get("FIXPROTOGPT_LOG_LEVEL", "").upper()
    if raw not in _VALID_LEVELS:
        try:
            from src.config.env_config import env as _env_cfg
            raw = _env_cfg.LOG_LEVEL.upper()
        except Exception:
            raw = "INFO"
    if raw not in _VALID_LEVELS:
        raw = "INFO"
    return getattr(logging, raw)


# ── JSON formatter ────────────────────────────────────────────────

class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record to a single-line JSON string."""
        entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }

        # Merge structured data attached via `extra={"data": {...}}`
        data = getattr(record, "data", None)
        if isinstance(data, dict):
            entry.update(data)
        else:
            entry["message"] = record.getMessage()

        return json.dumps(entry, default=str)


# ── Factory helpers ───────────────────────────────────────────────

def _make_file_handler(
    filepath: Path,
    when: str = "midnight",
    backup_count: int = 30,
    level: Optional[int] = None,
) -> TimedRotatingFileHandler:
    """Create a timed-rotating file handler with JSON formatting."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    handler = TimedRotatingFileHandler(
        filename=str(filepath),
        when=when,
        backupCount=backup_count,
        encoding="utf-8",
        utc=True,
    )
    handler.setFormatter(_JSONFormatter())
    handler.setLevel(level if level is not None else logging.INFO)
    return handler


def _get_server_logger() -> logging.Logger:
    """Return the server request logger (creates file handler once)."""
    logger = logging.getLogger(SERVER_LOGGER_NAME)
    if not logger.handlers:
        effective = _get_log_level()
        logger.setLevel(effective)
        logger.propagate = False
        logger.addHandler(
            _make_file_handler(LOG_DIR / "server.log", backup_count=30,
                               level=logging.INFO),
        )
        # When DEBUG is active, also feed server events into debug.log
        if effective <= logging.DEBUG:
            logger.addHandler(
                _make_file_handler(LOG_DIR / "debug.log", backup_count=14,
                                   level=logging.DEBUG),
            )
    return logger


def _get_action_logger() -> logging.Logger:
    """Return the user-action logger (creates file handler once)."""
    logger = logging.getLogger(ACTION_LOGGER_NAME)
    if not logger.handlers:
        effective = _get_log_level()
        logger.setLevel(effective)
        logger.propagate = False
        logger.addHandler(
            _make_file_handler(LOG_DIR / "user_actions.log", backup_count=90,
                               level=logging.INFO),
        )
        if effective <= logging.DEBUG:
            logger.addHandler(
                _make_file_handler(LOG_DIR / "debug.log", backup_count=14,
                                   level=logging.DEBUG),
            )
    return logger


def _get_debug_logger() -> logging.Logger:
    """Return the dedicated debug logger (creates file handler once).

    Only writes to ``debug.log`` when ``FIXPROTOGPT_LOG_LEVEL=DEBUG``.
    At higher levels the handler is set to INFO so debug calls are
    silently dropped.
    """
    logger = logging.getLogger(DEBUG_LOGGER_NAME)
    if not logger.handlers:
        effective = _get_log_level()
        logger.setLevel(effective)
        logger.propagate = False
        logger.addHandler(
            _make_file_handler(LOG_DIR / "debug.log", backup_count=14,
                               level=logging.DEBUG if effective <= logging.DEBUG
                               else logging.INFO),
        )
    return logger


# ── Public API ────────────────────────────────────────────────────

def setup_logging(app: Flask) -> None:
    """Wire request/response logging into the Flask application.

    Registers ``before_request`` and ``after_request`` hooks that
    emit structured JSON to ``logs/server.log``.

    Args:
        app: The Flask application instance.
    """
    server_log = _get_server_logger()

    # Ensure the action & debug loggers are initialised too
    _get_action_logger()
    _get_debug_logger()

    @app.before_request
    def _start_timer():
        """Record request start time."""
        g._request_start = time.perf_counter()

    @app.after_request
    def _log_request(response):
        """Log every request/response pair."""
        duration_ms = round(
            (time.perf_counter() - getattr(g, "_request_start", time.perf_counter())) * 1000, 2,
        )

        # Skip logging static asset requests to reduce noise
        if request.path.startswith("/static/"):
            return response

        entry = {
            "event": "http_request",
            "method": request.method,
            "path": request.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
            "ip": request.remote_addr,
            "user_agent": request.user_agent.string[:200],
            "user_id": session.get("user_id"),
            "username": session.get("username"),
            "content_length": response.content_length,
        }

        server_log.info("request", extra={"data": entry})
        return response

    @app.errorhandler(Exception)
    def _log_unhandled(error):
        """Log unhandled exceptions."""
        from werkzeug.exceptions import HTTPException

        # Let HTTP exceptions (404, 405, etc.) pass through normally
        if isinstance(error, HTTPException):
            # Skip static asset 404s from polluting the log
            if not request.path.startswith("/static/"):
                server_log.warning(
                    "http_error",
                    extra={"data": {
                        "event": "http_error",
                        "method": request.method,
                        "path": request.path,
                        "status": error.code,
                        "error": str(error),
                        "ip": request.remote_addr,
                        "user_id": session.get("user_id"),
                    }},
                )
            return error

        # Skip static asset errors to reduce noise
        if request.path.startswith("/static/"):
            return {"error": "Internal server error"}, 500

        server_log.error(
            "unhandled_error",
            extra={"data": {
                "event": "unhandled_error",
                "method": request.method,
                "path": request.path,
                "error": str(error),
                "error_type": type(error).__name__,
                "ip": request.remote_addr,
                "user_id": session.get("user_id"),
            }},
        )
        return {"error": "Internal server error"}, 500


def log_user_action(
    action: str,
    *,
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    detail: Optional[Dict[str, Any]] = None,
    status: str = "success",
) -> None:
    """Log a meaningful user action to ``logs/user_actions.log``.

    Args:
        action:   Short action name (e.g. ``"login"``, ``"generate"``).
        user_id:  User ID (falls back to ``session["user_id"]``).
        username: Username (falls back to ``session["username"]``).
        detail:   Extra key-value pairs to include in the log entry.
        status:   ``"success"`` or ``"failure"``.
    """
    action_log = _get_action_logger()

    # Fall back to session context if available
    try:
        user_id = user_id or session.get("user_id")
        username = username or session.get("username")
        ip = request.remote_addr
    except RuntimeError:
        # Outside request context (e.g. CLI, tests)
        ip = None

    entry: Dict[str, Any] = {
        "event": "user_action",
        "action": action,
        "user_id": user_id,
        "username": username,
        "ip": ip,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if detail:
        entry["detail"] = detail

    level = logging.INFO if status == "success" else logging.WARNING
    action_log.log(level, action, extra={"data": entry})


def log_debug(
    event: str,
    *,
    detail: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a DEBUG-level entry to ``logs/debug.log``.

    When ``FIXPROTOGPT_LOG_LEVEL`` is not ``DEBUG`` the call is
    effectively a no-op (the handler filters it out).

    Args:
        event:  Short event name (e.g. ``"db_query"``, ``"resolve"``).
        detail: Extra key-value pairs to include in the log entry.
    """
    debug_log = _get_debug_logger()

    entry: Dict[str, Any] = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if detail:
        entry["detail"] = detail

    debug_log.debug(event, extra={"data": entry})
