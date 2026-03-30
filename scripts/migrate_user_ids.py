"""One-time migration: convert UUID-based user IDs to sequential U00001-style IDs
and interaction IDs to sequential IN00000001-style IDs."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "model_store" / "interactions" / "interactions.db"


def migrate_users(conn):
    """Migrate user IDs from UUID to sequential U00001 format."""
    rows = conn.execute("SELECT id, username FROM users ORDER BY created_at").fetchall()
    # Skip if already migrated
    if rows and rows[0]["id"].startswith("U"):
        print(f"Users already migrated ({len(rows)} users with U-prefix IDs). Skipping.")
        return

    print(f"Migrating {len(rows)} user IDs...")
    mapping = {}
    for i, row in enumerate(rows, start=1):
        new_id = f"U{i:05d}"
        mapping[row["id"]] = new_id
        print(f"  {row['username']:15s}  {row['id']}  ->  {new_id}")

    conn.execute("PRAGMA foreign_keys = OFF")
    for old_id, new_id in mapping.items():
        conn.execute("UPDATE users SET id = ? WHERE id = ?", (new_id, old_id))
        conn.execute("UPDATE token_usage SET user_id = ? WHERE user_id = ?", (new_id, old_id))
        conn.execute("UPDATE interactions SET user_id = ? WHERE user_id = ?", (new_id, old_id))
    conn.execute("PRAGMA foreign_keys = ON")

    print("User ID migration done.")


def migrate_interactions(conn):
    """Migrate interaction IDs from UUID to sequential IN00000001 format."""
    rows = conn.execute("SELECT id FROM interactions ORDER BY timestamp").fetchall()
    # Skip if already migrated
    if rows and rows[0]["id"].startswith("IN"):
        print(f"Interactions already migrated ({len(rows)} with IN-prefix IDs). Skipping.")
        return

    print(f"Migrating {len(rows)} interaction IDs...")
    mapping = {}
    for i, row in enumerate(rows, start=1):
        new_id = f"IN{i:08d}"
        mapping[row["id"]] = new_id

    conn.execute("PRAGMA foreign_keys = OFF")
    for old_id, new_id in mapping.items():
        conn.execute("UPDATE interactions SET id = ? WHERE id = ?", (new_id, old_id))
    conn.execute("PRAGMA foreign_keys = ON")

    # Verify
    print("Verification (first 10):")
    for row in conn.execute("SELECT id, endpoint, user_id FROM interactions ORDER BY id LIMIT 10").fetchall():
        print(f"  {row['id']}  {row['endpoint']:10s}  {row['user_id']}")
    total = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    print(f"  ... {total} interactions total")
    orphans = conn.execute("SELECT COUNT(*) FROM interactions WHERE id NOT LIKE 'IN%'").fetchone()[0]
    print(f"Orphan interaction rows: {orphans}")
    print("Interaction ID migration done!")


def migrate():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("BEGIN")

    migrate_users(conn)
    migrate_interactions(conn)

    conn.execute("COMMIT")
    conn.close()
    print("\nAll migrations complete!")


if __name__ == "__main__":
    migrate()
