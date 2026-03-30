"""
Module: src.persistence.interaction_logger
====================================

Interaction logger for FixProtoGPT.

Captures all web interactions (queries, model responses, user feedback)
to a SQLite database that can later be exported as fine-tuning data.

Each interaction row stores:
  - id:        unique sequential ID (e.g. IN00000001)
  - timestamp: ISO-8601 (UTC)
  - endpoint:  which API endpoint was called
  - request:   the user's input  (JSON text)
  - response:  the model's output (JSON text)
  - feedback:  optional user rating (JSON text or NULL)
  - metadata:  demo_mode, temperature, etc. (JSON text)

Usage::

    from src.persistence.interaction_logger import InteractionLogger

    logger = InteractionLogger()
    iid = logger.log("nl2fix", request={"text": "Buy 100 AAPL"}, response={"fix_message": "..."})
    logger.add_feedback(iid, rating="positive")
    pairs = logger.export_training_pairs()

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.persistence.action_logger import log_debug
from src.utils import paths


# DB directory — isolated per environment so dev/qa/prod never share data.
try:
    from src.config.env_config import env as _env_cfg
    _LOG_DIR = paths.PROJECT_ROOT / "db" / _env_cfg.ENV_NAME
except Exception:
    _LOG_DIR = paths.PROJECT_ROOT / "db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id        TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    endpoint  TEXT NOT NULL,
    request   TEXT NOT NULL DEFAULT '{}',
    response  TEXT NOT NULL DEFAULT '{}',
    feedback  TEXT,
    metadata  TEXT NOT NULL DEFAULT '{}',
    user_id   TEXT,
    trained_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_endpoint  ON interactions(endpoint);
CREATE INDEX IF NOT EXISTS idx_timestamp ON interactions(timestamp);
"""


class InteractionLogger:
    """Thread-safe logger that persists web interactions to SQLite.

    SQLite in WAL mode supports concurrent readers and a single writer
    without external locking, making it safe for multi-threaded Flask.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        """Open (or create) the SQLite interactions database and run migrations."""
        self.log_dir = Path(log_dir) if log_dir else _LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.log_dir / "interactions.db"
        # Keep the old attribute so tests that check file existence still work
        self.log_file = self.db_path
        self._init_db()
        self._migrate_jsonl()

    # ── DB bootstrap ──────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Return a new SQLite connection with WAL mode enabled.

        Returns:
            A ``sqlite3.Connection`` with ``Row`` row-factory.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the schema if absent and migrate legacy columns."""
        with self._get_conn() as conn:
            conn.executescript(_SCHEMA)
            # Migration: add columns if missing (existing DBs)
            cols = [r[1] for r in conn.execute("PRAGMA table_info(interactions)").fetchall()]
            if "user_id" not in cols:
                conn.execute("ALTER TABLE interactions ADD COLUMN user_id TEXT")
            if "trained_at" not in cols:
                conn.execute("ALTER TABLE interactions ADD COLUMN trained_at TEXT")
            # Always ensure indexes exist (safe for fresh and migrated DBs)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON interactions(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trained_at ON interactions(trained_at)")

    def _migrate_jsonl(self) -> None:
        """One-time import of legacy ``interactions.jsonl`` into SQLite.

        Renames the JSONL file to ``.jsonl.migrated`` after processing.
        """
        jsonl_path = self.log_dir / "interactions.jsonl"
        if not jsonl_path.exists():
            return

        records: List[Dict[str, Any]] = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not records:
            jsonl_path.rename(jsonl_path.with_suffix(".jsonl.migrated"))
            return

        with self._get_conn() as conn:
            for rec in records:
                # Skip if already imported (idempotent)
                exists = conn.execute(
                    "SELECT 1 FROM interactions WHERE id = ?", (rec["id"],)
                ).fetchone()
                if exists:
                    continue
                conn.execute(
                    "INSERT INTO interactions (id, timestamp, endpoint, request, response, feedback, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        rec["id"],
                        rec.get("timestamp", ""),
                        rec.get("endpoint", ""),
                        json.dumps(rec.get("request", {})),
                        json.dumps(rec.get("response", {})),
                        json.dumps(rec["feedback"]) if rec.get("feedback") else None,
                        json.dumps(rec.get("metadata", {})),
                    ),
                )

        # Rename old file so migration doesn't re-run
        jsonl_path.rename(jsonl_path.with_suffix(".jsonl.migrated"))

    # ── Helpers ───────────────────────────────────────────────────

    def _next_interaction_id(self) -> str:
        """Generate the next sequential interaction ID (IN00000001, IN00000002, ...).

        Returns:
            A string like ``IN00000001``.
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT id FROM interactions WHERE id LIKE 'IN%' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return "IN00000001"
        last_num = int(row["id"][2:])  # strip the 'IN' prefix
        return f"IN{last_num + 1:08d}"

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a ``sqlite3.Row`` to a plain dict.

        JSON-encoded columns are automatically parsed.

        Args:
            row: A single database row.

        Returns:
            Dict with parsed request/response/feedback/metadata.
        """
        d = {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "endpoint": row["endpoint"],
            "request": json.loads(row["request"]) if row["request"] else {},
            "response": json.loads(row["response"]) if row["response"] else {},
            "feedback": json.loads(row["feedback"]) if row["feedback"] else None,
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
        }
        # Include user_id if the column exists
        try:
            d["user_id"] = row["user_id"]
        except (IndexError, KeyError):
            pass
        # Include trained_at if the column exists
        try:
            d["trained_at"] = row["trained_at"]
        except (IndexError, KeyError):
            pass
        return d

    # ── Write ─────────────────────────────────────────────────────

    def log(
        self,
        endpoint: str,
        request: Dict[str, Any],
        response: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Log a single interaction and return its unique ID.

        Parameters
        ----------
        endpoint : str
            API endpoint name (e.g. "nl2fix", "generate", "explain").
        request : dict
            The user's request payload.
        response : dict
            The model/demo response payload.
        metadata : dict, optional
            Extra info (demo_mode, temperature, etc.).
        user_id : str, optional
            ID of the authenticated user.

        Returns
        -------
        str
            The interaction's sequential ID (e.g. ``IN00000001``).
        """
        interaction_id = self._next_interaction_id()
        ts = datetime.now(timezone.utc).isoformat()
        log_debug("db_interaction_log", detail={
            "interaction_id": interaction_id, "endpoint": endpoint,
            "user_id": user_id,
        })
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO interactions (id, timestamp, endpoint, request, response, feedback, metadata, user_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    interaction_id,
                    ts,
                    endpoint,
                    json.dumps(request),
                    json.dumps(response),
                    None,
                    json.dumps(metadata or {}),
                    user_id,
                ),
            )
        return interaction_id

    def add_feedback(
        self,
        interaction_id: str,
        rating: str,
        correction: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> bool:
        """
        Attach user feedback to an existing interaction.

        Parameters
        ----------
        interaction_id : str
            ID returned by ``log()``.
        rating : str
            ``"positive"`` or ``"negative"``.
        correction : str, optional
            User-supplied corrected FIX message.
        comment : str, optional
            Free-text user comment.

        Returns
        -------
        bool
            True if the interaction was found and updated.
        """
        if rating not in ("positive", "negative"):
            return False

        feedback = {
            "rating": rating,
            "correction": correction,
            "comment": comment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with self._get_conn() as conn:
            cur = conn.execute(
                "UPDATE interactions SET feedback = ? WHERE id = ?",
                (json.dumps(feedback), interaction_id),
            )
        return cur.rowcount > 0

    # ── Read ──────────────────────────────────────────────────────

    def get_interactions(
        self,
        endpoint: Optional[str] = None,
        rated_only: bool = False,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve logged interactions with optional filters.

        Parameters
        ----------
        endpoint : str, optional
            Filter by endpoint name.
        rated_only : bool
            If True, return only interactions that have feedback.
        limit : int
            Maximum number to return (0 = all). Most recent first.

        Returns
        -------
        list of dict
        """
        clauses: List[str] = []
        params: List[Any] = []

        if endpoint:
            clauses.append("endpoint = ?")
            params.append(endpoint)
        if rated_only:
            clauses.append("feedback IS NOT NULL")

        sql = "SELECT * FROM interactions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp DESC"
        if limit > 0:
            sql += " LIMIT ?"
            params.append(limit)

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics of logged interactions.

        Returns:
            Dict with ``total_interactions``, ``by_endpoint``,
            ``with_feedback``, ``positive``, ``negative``, and
            ``with_corrections``.
        """
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]

            by_endpoint: Dict[str, int] = {}
            for row in conn.execute("SELECT endpoint, COUNT(*) FROM interactions GROUP BY endpoint"):
                by_endpoint[row[0]] = row[1]

            with_feedback = conn.execute(
                "SELECT COUNT(*) FROM interactions WHERE feedback IS NOT NULL"
            ).fetchone()[0]

            # For rating counts we need to parse JSON — use Python
            rows = conn.execute(
                "SELECT feedback FROM interactions WHERE feedback IS NOT NULL"
            ).fetchall()

        positive = 0
        negative = 0
        with_corrections = 0
        for row in rows:
            fb = json.loads(row[0]) if row[0] else {}
            if fb.get("rating") == "positive":
                positive += 1
            elif fb.get("rating") == "negative":
                negative += 1
            if fb.get("correction"):
                with_corrections += 1

        return {
            "total_interactions": total,
            "by_endpoint": by_endpoint,
            "with_feedback": with_feedback,
            "positive": positive,
            "negative": negative,
            "with_corrections": with_corrections,
        }

    # ── Export for fine-tuning ────────────────────────────────────

    def export_training_pairs(self, untrained_only: bool = False) -> List[str]:
        """
        Convert logged interactions into training-ready text lines.

        Produces lines in the formats the model was trained on:

        * **NL → FIX pairs** from ``nl2fix`` interactions with positive
          feedback or user corrections.
        * **FIX messages** from ``generate`` / ``complete`` interactions
          with positive feedback.
        * **Message type descriptions** extracted from ``explain``
          interactions that received positive feedback.

        Parameters
        ----------
        untrained_only : bool
            If True, only export interactions that haven't been used
            for fine-tuning yet (``trained_at IS NULL``).

        Returns
        -------
        list of str
            Training lines ready to append to ``train.txt``.
        """
        sql = "SELECT * FROM interactions WHERE feedback IS NOT NULL"
        if untrained_only:
            sql += " AND trained_at IS NULL"
        with self._get_conn() as conn:
            rows = conn.execute(sql).fetchall()

        records = [self._row_to_dict(r) for r in rows]
        lines: List[str] = []

        for r in records:
            fb = r.get("feedback")
            ep = r.get("endpoint", "")
            req = r.get("request", {})
            resp = r.get("response", {})

            # ── NL2FIX interactions ───────────────────────────────
            if ep == "nl2fix":
                nl_text = req.get("text", "")
                fix_msg = resp.get("fix_message", "")
                if not nl_text or not fix_msg:
                    continue

                # User correction overrides model output
                if fb and fb.get("correction"):
                    fix_msg = fb["correction"]
                    lines.append(f"{nl_text}\n{fix_msg}")
                elif fb and fb.get("rating") == "positive":
                    lines.append(f"{nl_text}\n{fix_msg}")

            # ── Generate / Complete ───────────────────────────────
            elif ep in ("generate", "complete"):
                fix_msg = resp.get("generated", "") or resp.get("completed", "")
                if fb and fb.get("correction"):
                    fix_msg = fb["correction"]
                if fix_msg and fb and fb.get("rating") == "positive":
                    lines.append(fix_msg)

            # ── Explain ───────────────────────────────────────────
            elif ep == "explain":
                if fb and fb.get("rating") == "positive":
                    explanation = resp.get("explanation", {})
                    mt = explanation.get("message_type", {})
                    if mt.get("code") and mt.get("name") and mt.get("description"):
                        cat = mt.get("category", "")
                        cat_part = f" ({cat})" if cat else ""
                        line = (
                            f"FIX Message Type {mt['code']}: "
                            f"{mt['name']}{cat_part}. {mt['description']}"
                        )
                        lines.append(line)

        return lines

    def export_to_file(self, output_path: Optional[Path] = None) -> Path:
        """Export training pairs to a text file.

        Args:
            output_path: Destination path.  Defaults to
                ``log_dir / finetune_pairs.txt``.

        Returns:
            The resolved path of the exported file.
        """
        lines = self.export_training_pairs()
        if output_path is None:
            output_path = self.log_dir / "finetune_pairs.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")
        return output_path

    # ── Fine-tune tracking ─────────────────────────────────────────

    def mark_trained(self, interaction_ids: List[str]) -> int:
        """Mark interactions as consumed by a fine-tuning run.

        Parameters
        ----------
        interaction_ids : list of str
            IDs of interactions that were included in training data.

        Returns
        -------
        int
            Number of rows updated.
        """
        if not interaction_ids:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            placeholders = ",".join("?" for _ in interaction_ids)
            cur = conn.execute(
                f"UPDATE interactions SET trained_at = ? "
                f"WHERE id IN ({placeholders}) AND trained_at IS NULL",
                [now] + list(interaction_ids),
            )
        return cur.rowcount

    def get_trainable_ids(self) -> List[str]:
        """Return IDs of positive-feedback interactions not yet trained.

        Returns:
            List of interaction ID strings.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id FROM interactions "
                "WHERE feedback IS NOT NULL AND trained_at IS NULL"
            ).fetchall()
        return [r["id"] for r in rows]

    # ── Delete ────────────────────────────────────────────────────

    def clear(self) -> int:
        """Remove all interactions.

        Returns:
            Number of rows deleted.
        """
        with self._get_conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            conn.execute("DELETE FROM interactions")
        return count

    def delete(self, interaction_id: str) -> bool:
        """Delete a single interaction by ID.

        Args:
            interaction_id: Sequential ID of the interaction (e.g. ``IN00000001``).

        Returns:
            ``True`` if the row was found and removed.
        """
        with self._get_conn() as conn:
            cur = conn.execute(
                "DELETE FROM interactions WHERE id = ?", (interaction_id,)
            )
        return cur.rowcount > 0
