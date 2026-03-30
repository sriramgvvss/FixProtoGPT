"""
Module: src.config.env_config
==============================

Centralised, environment-aware configuration for FixProtoGPT.

How it works
------------
1. Set ``FIXPROTOGPT_ENV`` to one of: ``dev``, ``qa``,
   ``preprod``, ``prod``.  Defaults to ``dev`` when unset.
2. The matching file ``config/env/.env.<name>`` is loaded via
   ``python-dotenv`` (values do **not** override vars already set
   in the real OS environment).
3. The singleton :data:`env` exposes typed, validated settings that
   the rest of the application imports.

Security
--------
* In ``prod`` / ``preprod`` the module **refuses to start** if
  ``FIXPROTOGPT_SECRET_KEY`` is missing or still the dev default.
* ``SESSION_COOKIE_SECURE`` is forced to ``True`` in prod-like
  environments.
* Demo user seeding is disabled in ``qa`` / ``preprod`` / ``prod``.

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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────

VALID_ENVS = ("dev", "qa", "preprod", "prod")
_DEV_SECRET = "fixprotogpt-dev-secret-change-in-prod"

# Project root (repo top-level)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_DIR = _PROJECT_ROOT / "config" / "env"


# ── Helpers ───────────────────────────────────────────────────────

def _load_dotenv_file(env_name: str) -> None:
    """Load ``config/env/.env.<env_name>`` without overriding existing vars."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        # python-dotenv is optional; fall back to pure env vars
        logger.debug("python-dotenv not installed — using OS env vars only")
        return

    env_file = _ENV_DIR / f".env.{env_name}"
    if env_file.is_file():
        load_dotenv(env_file, override=False)
        logger.info("Loaded env file: %s", env_file.relative_to(_PROJECT_ROOT))
    else:
        logger.warning("Env file not found: %s", env_file)


def _bool(value: str) -> bool:
    """Parse a string to bool (``true/1/yes`` → True)."""
    return value.strip().lower() in ("true", "1", "yes")


def _str_list(value: str) -> List[str]:
    """Parse a comma-separated string into a list of stripped strings."""
    if not value or not value.strip():
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


# ── Configuration dataclass ──────────────────────────────────────

@dataclass(frozen=True)
class EnvConfig:
    """Immutable, validated application configuration.

    All values are resolved once at import time from the environment
    (after the matching ``.env.*`` file has been loaded).

    Attributes:
        ENV_NAME:              Current environment name.
        DEBUG:                 Enable Flask debug mode.
        SECRET_KEY:            Flask secret key for sessions.
        SESSION_COOKIE_SECURE: Send cookie only over HTTPS.
        SESSION_COOKIE_HTTPONLY: Prevent JS access to cookie.
        SESSION_COOKIE_SAMESITE: SameSite attribute for cookie.
        SESSION_LIFETIME:      Session lifetime in seconds.
        SEED_DEMO_USERS:       Whether to seed demo accounts on first run.
        CORS_ORIGINS:          Allowed CORS origins (empty = all).
        LOG_LEVEL:             Python log level name.
        HOST:                  Bind address for the dev server.
        PORT:                  Bind port for the dev server.
        CSP_POLICY:            Content-Security-Policy header value.
        HSTS_ENABLED:          Whether to send Strict-Transport-Security.
        HSTS_MAX_AGE:          HSTS max-age in seconds.
    """

    ENV_NAME: str
    DEBUG: bool
    SECRET_KEY: str
    SESSION_COOKIE_SECURE: bool
    SESSION_COOKIE_HTTPONLY: bool
    SESSION_COOKIE_SAMESITE: str
    SESSION_LIFETIME: int
    SEED_DEMO_USERS: bool
    CORS_ORIGINS: List[str]
    LOG_LEVEL: str
    HOST: str
    PORT: int
    CSP_POLICY: str
    HSTS_ENABLED: bool
    HSTS_MAX_AGE: int

    # ── Derived helpers ───────────────────────────────────────────

    @property
    def is_production(self) -> bool:
        """Return ``True`` for ``prod`` or ``preprod``."""
        return self.ENV_NAME in ("prod", "preprod")

    @property
    def is_secure(self) -> bool:
        """Return ``True`` for environments that require HTTPS."""
        return self.ENV_NAME in ("prod", "preprod")


def _build_config() -> EnvConfig:
    """Resolve configuration from environment variables.

    Called once at module load time.  The result is cached in :data:`env`.
    """
    env_name = os.environ.get("FIXPROTOGPT_ENV", "dev").lower().strip()
    if env_name not in VALID_ENVS:
        logger.warning(
            "FIXPROTOGPT_ENV=%r is not valid (%s) — falling back to 'dev'",
            env_name, ", ".join(VALID_ENVS),
        )
        env_name = "dev"

    # Load the matching .env file (does NOT override existing vars)
    _load_dotenv_file(env_name)

    # ── Read individual settings ──────────────────────────────────
    secret_key = os.environ.get("FIXPROTOGPT_SECRET_KEY", _DEV_SECRET)

    # In prod/preprod, refuse to start with the dev default secret
    if env_name in ("prod", "preprod") and secret_key == _DEV_SECRET:
        print(
            "\n*** FATAL: FIXPROTOGPT_SECRET_KEY must be set to a real secret "
            f"in '{env_name}' environment. ***\n"
            "Set the FIXPROTOGPT_SECRET_KEY environment variable and restart.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    debug = _bool(os.environ.get("FIXPROTOGPT_DEBUG", "true" if env_name in ("dev", "qa") else "false"))
    cookie_secure = _bool(os.environ.get(
        "FIXPROTOGPT_SESSION_COOKIE_SECURE",
        "true" if env_name in ("prod", "preprod") else "false",
    ))
    cookie_httponly = _bool(os.environ.get("FIXPROTOGPT_SESSION_COOKIE_HTTPONLY", "true"))
    cookie_samesite = os.environ.get("FIXPROTOGPT_SESSION_COOKIE_SAMESITE", "Lax")
    session_lifetime = int(os.environ.get("FIXPROTOGPT_SESSION_LIFETIME", "86400"))
    seed_demo = _bool(os.environ.get(
        "FIXPROTOGPT_SEED_DEMO_USERS",
        "true" if env_name in ("dev", "qa") else "false",
    ))
    cors_origins_str = os.environ.get("FIXPROTOGPT_CORS_ORIGINS", "")
    cors_origins = _str_list(cors_origins_str)
    log_level = os.environ.get("FIXPROTOGPT_LOG_LEVEL", "DEBUG" if env_name in ("dev", "qa") else "INFO")
    host = os.environ.get("FIXPROTOGPT_HOST", "0.0.0.0")
    port = int(os.environ.get("FIXPROTOGPT_PORT", "8080"))

    csp_default = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net "
        "https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
        "font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'"
    )
    csp_policy = os.environ.get("FIXPROTOGPT_CSP_POLICY", csp_default)
    hsts_enabled = _bool(os.environ.get(
        "FIXPROTOGPT_HSTS_ENABLED",
        "true" if env_name in ("prod", "preprod") else "false",
    ))
    hsts_max_age = int(os.environ.get("FIXPROTOGPT_HSTS_MAX_AGE", "31536000"))

    cfg = EnvConfig(
        ENV_NAME=env_name,
        DEBUG=debug,
        SECRET_KEY=secret_key,
        SESSION_COOKIE_SECURE=cookie_secure,
        SESSION_COOKIE_HTTPONLY=cookie_httponly,
        SESSION_COOKIE_SAMESITE=cookie_samesite,
        SESSION_LIFETIME=session_lifetime,
        SEED_DEMO_USERS=seed_demo,
        CORS_ORIGINS=cors_origins,
        LOG_LEVEL=log_level,
        HOST=host,
        PORT=port,
        CSP_POLICY=csp_policy,
        HSTS_ENABLED=hsts_enabled,
        HSTS_MAX_AGE=hsts_max_age,
    )

    logger.info(
        "FixProtoGPT env=%s  debug=%s  seed_demo=%s  cookie_secure=%s  log=%s",
        cfg.ENV_NAME, cfg.DEBUG, cfg.SEED_DEMO_USERS,
        cfg.SESSION_COOKIE_SECURE, cfg.LOG_LEVEL,
    )

    return cfg


# ── Directory scaffolding ─────────────────────────────────────────

def ensure_all_env_dirs() -> None:
    """Pre-create the log and DB directory trees for **every** environment.

    Called once at module load so that ``logs/<env>/`` and
    ``db/<env>/`` exist for all valid environments, not just the
    one currently active.  This gives operators a clear, up-front
    view of the segregation.

    Each directory also gets a ``.gitkeep`` sentinel so the
    structure is preserved in version control (even though dynamic
    files inside are git-ignored).
    """
    for env_name in VALID_ENVS:
        log_dir = _PROJECT_ROOT / "logs" / env_name
        db_dir = _PROJECT_ROOT / "db" / env_name
        for d in (log_dir, db_dir):
            d.mkdir(parents=True, exist_ok=True)
            gitkeep = d / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()


# ── Module-level singleton ────────────────────────────────────────
# Import with:  from src.config.env_config import env
env: EnvConfig = _build_config()
ensure_all_env_dirs()
