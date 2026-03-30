"""
Module: src.persistence.user_manager
==============================

User manager for FixProtoGPT.

Handles user registration, authentication, and session management
using SQLite for storage and Werkzeug for password hashing.

Schema
------
users table:
    id         TEXT PRIMARY KEY  (sequential, e.g. U00001)
    username   TEXT UNIQUE NOT NULL
    email      TEXT UNIQUE NOT NULL
    password   TEXT NOT NULL      (werkzeug scrypt hash)
    full_name  TEXT
    role       TEXT DEFAULT 'user'   ('admin' | 'user')
    created_at TEXT NOT NULL      (ISO-8601 UTC)
    last_login TEXT               (ISO-8601 UTC, updated on each login)
    is_active  INTEGER DEFAULT 1  (0 = disabled)

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

import os
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from werkzeug.security import generate_password_hash, check_password_hash

from src.persistence.action_logger import log_debug
from src.utils import paths

# DB directory — isolated per environment so dev/qa/prod never share data.
try:
    from src.config.env_config import env as _env_cfg
    _DB_DIR = paths.PROJECT_ROOT / "db" / _env_cfg.ENV_NAME
except Exception:
    _DB_DIR = paths.PROJECT_ROOT / "db"

_USER_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id         TEXT PRIMARY KEY,
    username   TEXT UNIQUE NOT NULL,
    email      TEXT UNIQUE NOT NULL,
    password   TEXT NOT NULL,
    full_name  TEXT DEFAULT '',
    role       TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL,
    last_login TEXT,
    is_active  INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email    ON users(email);
"""

_TOKEN_USAGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS token_usage (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    endpoint   TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    timestamp  TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS idx_token_usage_user   ON token_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_token_usage_ts     ON token_usage(timestamp);
"""


class UserManager:
    """Manages user accounts in SQLite with scrypt password hashing."""

    def __init__(self, db_dir: Optional[Path] = None):
        """Open (or create) the users SQLite database and seed default accounts."""
        self.db_dir = Path(db_dir) if db_dir else _DB_DIR
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir / "users.db"
        self._maybe_migrate_from_interactions_db()
        self._init_db()

    # ── DB helpers ────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Return a WAL-mode SQLite connection.

        Returns:
            A ``sqlite3.Connection`` configured with ``Row`` factory.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the users and token_usage tables if they don't exist, then seed defaults."""
        with self._get_conn() as conn:
            conn.executescript(_USER_SCHEMA)
            conn.executescript(_TOKEN_USAGE_SCHEMA)
        self._seed_defaults()

    # ── Default users ─────────────────────────────────────────────

    @staticmethod
    def _generate_default_users() -> List[Dict[str, str]]:
        """Build default demo users with random passwords.

        Passwords are generated at runtime via :func:`secrets.token_urlsafe`
        so that no credentials are hard-coded in source.

        Returns:
            List of user dicts suitable for :meth:`register`.
        """
        return [
            {
                "username": "admin",
                "email": "admin@fixprotogpt.local",
                "password": os.environ.get("FIXPROTOGPT_ADMIN_PASSWORD", secrets.token_urlsafe(16)),
                "full_name": "Administrator",
                "role": "admin",
            },
            {
                "username": "trader",
                "email": "trader@fixprotogpt.local",
                "password": os.environ.get("FIXPROTOGPT_TRADER_PASSWORD", secrets.token_urlsafe(16)),
                "full_name": "Demo Trader",
                "role": "user",
            },
            {
                "username": "developer",
                "email": "developer@fixprotogpt.local",
                "password": os.environ.get("FIXPROTOGPT_DEV_PASSWORD", secrets.token_urlsafe(16)),
                "full_name": "Demo Developer",
                "role": "user",
            },
        ]

    def _maybe_migrate_from_interactions_db(self) -> None:
        """One-time migration: copy user tables from legacy ``interactions.db``.

        Before the DB-per-service split, user data lived alongside
        interaction data inside ``interactions.db``.  This method
        checks two legacy locations:

        1. ``db/<env>/interactions.db`` — co-located with current DB dir.
        2. ``model_store/interactions/<env>/interactions.db`` — the old
           env-aware path before the DB directory was moved.

        If either contains a ``users`` table and ``users.db`` does not
        yet exist, copy the rows across so no data is lost.
        """
        if self.db_path.exists():
            return

        # Build list of candidate legacy DB paths (newest location first)
        env_name = self.db_dir.name
        candidates = [
            self.db_dir / "interactions.db",
            paths.PROJECT_ROOT / "model_store" / "db" / env_name / "interactions.db",
            paths.PROJECT_ROOT / "model_store" / "interactions" / env_name / "interactions.db",
        ]

        legacy_db: Optional[Path] = None
        for candidate in candidates:
            if candidate.exists():
                legacy_db = candidate
                break

        if legacy_db is None:
            return
        try:
            legacy_conn = sqlite3.connect(str(legacy_db), timeout=10)
            # Check if the legacy DB actually has a users table
            tables = [
                r[0] for r in legacy_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            if "users" not in tables:
                legacy_conn.close()
                return
            log_debug("db_migrate_start", detail={
                "source": str(legacy_db),
                "target": str(self.db_path),
            })
            # Create the new DB and copy data
            new_conn = sqlite3.connect(str(self.db_path), timeout=10)
            new_conn.execute("PRAGMA journal_mode=WAL")
            new_conn.executescript(_USER_SCHEMA)
            new_conn.executescript(_TOKEN_USAGE_SCHEMA)
            # Copy users
            rows = legacy_conn.execute("SELECT * FROM users").fetchall()
            cols = [d[0] for d in legacy_conn.execute("SELECT * FROM users LIMIT 0").description]
            if rows:
                placeholders = ", ".join(["?"] * len(cols))
                col_names = ", ".join(cols)
                new_conn.executemany(
                    f"INSERT OR IGNORE INTO users ({col_names}) VALUES ({placeholders})",
                    rows,
                )
            # Copy token_usage if it exists
            if "token_usage" in tables:
                tu_rows = legacy_conn.execute("SELECT * FROM token_usage").fetchall()
                tu_cols = [d[0] for d in legacy_conn.execute("SELECT * FROM token_usage LIMIT 0").description]
                if tu_rows:
                    tu_ph = ", ".join(["?"] * len(tu_cols))
                    tu_cn = ", ".join(tu_cols)
                    new_conn.executemany(
                        f"INSERT OR IGNORE INTO token_usage ({tu_cn}) VALUES ({tu_ph})",
                        tu_rows,
                    )
            new_conn.commit()
            new_conn.close()
            legacy_conn.close()
            log_debug("db_migrate_done", detail={
                "users_copied": len(rows),
                "target": str(self.db_path),
            })
        except Exception as exc:
            log_debug("db_migrate_error", detail={"error": str(exc)})

    def _seed_defaults(self) -> None:
        """Create default admin/trader/developer users if table is empty.

        Seeding is controlled by the ``FIXPROTOGPT_SEED_DEMO_USERS``
        environment variable (via :data:`src.config.env_config.env`).
        In ``qa``, ``preprod``, and ``prod`` this defaults to ``false``
        so that no demo credentials are shipped.
        """
        if self.get_user_count() > 0:
            return

        # Check env config — skip seeding when disabled
        try:
            from src.config.env_config import env as _env_cfg
            if not _env_cfg.SEED_DEMO_USERS:
                log_debug("seed_skip", detail={
                    "reason": "SEED_DEMO_USERS is false",
                    "env": _env_cfg.ENV_NAME,
                })
                return
        except Exception:
            # Fallback: seed if we can't load config (e.g. during early tests)
            pass

        default_users = self._generate_default_users()
        for u in default_users:
            result = self.register(**u)
            if result.get("success"):
                log_debug("seed_user_created", detail={
                    "username": u["username"],
                    "role": u["role"],
                    "password": u["password"],  # logged once at seed time only
                })

    @staticmethod
    def _next_user_id() -> str:
        """Generate a unique user ID using UUID4.

        Returns a collision-free identifier like ``U-<uuid4>`` that is
        safe under concurrent writes without database locking.

        Returns:
            A string like ``U-a1b2c3d4e5f6``.
        """
        return f"U-{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a ``sqlite3.Row`` to a safe dict (excludes password).

        Args:
            row: Database row from the ``users`` table.

        Returns:
            Dict with user profile fields.
        """
        return {
            "id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "full_name": row["full_name"],
            "role": row["role"],
            "created_at": row["created_at"],
            "last_login": row["last_login"],
            "is_active": bool(row["is_active"]),
        }

    # ── Registration ──────────────────────────────────────────────

    def register(
        self,
        username: str,
        email: str,
        password: str,
        full_name: str = "",
        role: str = "user",
    ) -> Dict[str, Any]:
        """
        Register a new user.

        Returns
        -------
        dict
            ``{"success": True, "user": {...}}`` or
            ``{"success": False, "error": "..."}``
        """
        username = username.strip().lower()
        email = email.strip().lower()

        if not username or not email or not password:
            return {"success": False, "error": "Username, email, and password are required"}

        if len(username) < 3:
            return {"success": False, "error": "Username must be at least 3 characters"}

        if len(password) < 6:
            return {"success": False, "error": "Password must be at least 6 characters"}

        if "@" not in email or "." not in email:
            return {"success": False, "error": "Invalid email address"}

        user_id = self._next_user_id()
        now = datetime.now(timezone.utc).isoformat()
        pw_hash = generate_password_hash(password)

        log_debug("db_user_register", detail={
            "user_id": user_id, "username": username, "role": role,
        })

        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO users (id, username, email, password, full_name, role, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (user_id, username, email, pw_hash, full_name, role, now),
                )
        except sqlite3.IntegrityError as e:
            err = str(e).lower()
            if "username" in err:
                return {"success": False, "error": "Username already taken"}
            if "email" in err:
                return {"success": False, "error": "Email already registered"}
            return {"success": False, "error": "Registration failed"}

        return {
            "success": True,
            "user": {
                "id": user_id,
                "username": username,
                "email": email,
                "full_name": full_name,
                "role": role,
                "created_at": now,
            },
        }

    # ── Authentication ────────────────────────────────────────────

    def authenticate(self, username: str, password: str) -> Dict[str, Any]:
        """
        Verify credentials and update last_login.

        Returns
        -------
        dict
            ``{"success": True, "user": {...}}`` or
            ``{"success": False, "error": "..."}``
        """
        username = username.strip().lower()

        log_debug("db_user_auth", detail={"username": username})

        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()

        if row is None:
            return {"success": False, "error": "Invalid username or password"}

        if not bool(row["is_active"]):
            return {"success": False, "error": "Account is disabled"}

        if not check_password_hash(row["password"], password):
            return {"success": False, "error": "Invalid username or password"}

        # Update last_login
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE users SET last_login = ? WHERE id = ?", (now, row["id"])
            )

        user = self._row_to_dict(row)
        user["last_login"] = now
        return {"success": True, "user": user}

    # ── Queries ───────────────────────────────────────────────────

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Look up a user by primary-key ID.

        Args:
            user_id: Sequential ID of the user (e.g. ``U00001``).

        Returns:
            User dict or ``None``.
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Look up a user by username (case-insensitive).

        Args:
            username: Login name.

        Returns:
            User dict or ``None``.
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username.strip().lower(),)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_users(self) -> List[Dict[str, Any]]:
        """Return all users, most recently created first.

        Returns:
            List of user dicts (admin use).
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM users ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_user_count(self) -> int:
        """Return total number of registered users.

        Returns:
            Integer count.
        """
        with self._get_conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    # ── Updates ───────────────────────────────────────────────────

    def update_password(self, user_id: str, new_password: str) -> bool:
        """Change a user's password.

        Args:
            user_id: Sequential ID of the user (e.g. ``U00001``).
            new_password: New plain-text password (min 6 chars).

        Returns:
            ``True`` if the user was found and updated.
        """
        if len(new_password) < 6:
            return False
        pw_hash = generate_password_hash(new_password)
        with self._get_conn() as conn:
            cur = conn.execute(
                "UPDATE users SET password = ? WHERE id = ?", (pw_hash, user_id)
            )
        return cur.rowcount > 0

    def set_active(self, user_id: str, active: bool) -> bool:
        """Enable or disable a user account.

        Args:
            user_id: Sequential ID of the user (e.g. ``U00001``).
            active: ``True`` to activate, ``False`` to deactivate.

        Returns:
            ``True`` if the user was found and updated.
        """
        with self._get_conn() as conn:
            cur = conn.execute(
                "UPDATE users SET is_active = ? WHERE id = ?",
                (1 if active else 0, user_id),
            )
        return cur.rowcount > 0

    def delete_user(self, user_id: str) -> bool:
        """Permanently delete a user.

        Args:
            user_id: Sequential ID of the user (e.g. ``U00001``).

        Returns:
            ``True`` if the row was found and removed.
        """
        with self._get_conn() as conn:
            cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        return cur.rowcount > 0

    # ── Token Usage Tracking ──────────────────────────────────────

    # Default daily token quota per user (0 = unlimited).
    # Override via FIXPROTOGPT_DAILY_TOKEN_QUOTA env var.
    _DAILY_TOKEN_QUOTA: int = int(
        os.environ.get("FIXPROTOGPT_DAILY_TOKEN_QUOTA", "0")
    )

    def check_token_quota(self, user_id: str) -> Dict[str, Any]:
        """Check whether a user has exceeded their daily token quota.

        The daily quota is read from the
        ``FIXPROTOGPT_DAILY_TOKEN_QUOTA`` environment variable
        (default ``0`` = unlimited).

        Args:
            user_id: Sequential user ID (e.g. ``U00001``).

        Returns:
            Dict with ``allowed`` (bool), ``daily_used``,
            ``daily_limit``, and ``remaining``.
        """
        limit = self._DAILY_TOKEN_QUOTA
        if limit <= 0:
            return {
                "allowed": True,
                "daily_used": 0,
                "daily_limit": 0,
                "remaining": -1,  # unlimited
            }

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) AS total "
                "FROM token_usage "
                "WHERE user_id = ? AND timestamp >= ?",
                (user_id, today),
            ).fetchone()

        used = row["total"] if row else 0
        remaining = max(0, limit - used)
        return {
            "allowed": used < limit,
            "daily_used": used,
            "daily_limit": limit,
            "remaining": remaining,
        }

    def record_token_usage(
        self,
        user_id: str,
        endpoint: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record a token-usage event.

        Args:
            user_id: Sequential ID of the user (e.g. ``U00001``).
            endpoint: API endpoint that was called.
            input_tokens: Count of tokens in the request.
            output_tokens: Count of tokens in the response.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO token_usage (user_id, endpoint, input_tokens, output_tokens, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, endpoint, input_tokens, output_tokens, now),
            )

    def get_user_token_usage(self, user_id: str) -> Dict[str, Any]:
        """Get aggregated token-usage stats for a single user.

        Args:
            user_id: Sequential ID of the user (e.g. ``U00001``).

        Returns:
            Dict with ``total_tokens``, ``input_tokens``,
            ``output_tokens``, ``request_count``, and ``by_endpoint``.
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT "
                "  COALESCE(SUM(input_tokens), 0)  AS total_input, "
                "  COALESCE(SUM(output_tokens), 0) AS total_output, "
                "  COUNT(*)                         AS request_count "
                "FROM token_usage WHERE user_id = ?",
                (user_id,),
            ).fetchone()

            by_endpoint = {}
            rows = conn.execute(
                "SELECT endpoint, "
                "  SUM(input_tokens) AS input_tokens, "
                "  SUM(output_tokens) AS output_tokens, "
                "  COUNT(*) AS requests "
                "FROM token_usage WHERE user_id = ? GROUP BY endpoint",
                (user_id,),
            ).fetchall()
            for r in rows:
                by_endpoint[r["endpoint"]] = {
                    "input_tokens": r["input_tokens"],
                    "output_tokens": r["output_tokens"],
                    "requests": r["requests"],
                }

        total_input = row["total_input"]
        total_output = row["total_output"]
        return {
            "total_tokens": total_input + total_output,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "request_count": row["request_count"],
            "by_endpoint": by_endpoint,
        }

    def get_all_users_token_usage(self) -> List[Dict[str, Any]]:
        """Get aggregated token usage for every user (admin overview).

        Returns:
            List of dicts with ``user_id``, ``username``, ``role``,
            token counts, and ``request_count``.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT u.id, u.username, u.role, "
                "  COALESCE(SUM(t.input_tokens), 0)  AS input_tokens, "
                "  COALESCE(SUM(t.output_tokens), 0) AS output_tokens, "
                "  COUNT(t.id)                        AS request_count "
                "FROM users u "
                "LEFT JOIN token_usage t ON u.id = t.user_id "
                "GROUP BY u.id "
                "ORDER BY (COALESCE(SUM(t.input_tokens), 0) + COALESCE(SUM(t.output_tokens), 0)) DESC",
            ).fetchall()
        return [
            {
                "user_id": r["id"],
                "username": r["username"],
                "role": r["role"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "total_tokens": r["input_tokens"] + r["output_tokens"],
                "request_count": r["request_count"],
            }
            for r in rows
        ]
