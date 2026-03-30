"""
Unit tests for UserManager (src/data/user_manager.py).

Tests registration, authentication, queries, updates, and edge cases.
"""

import sys
import tempfile
import shutil
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.persistence.user_manager import UserManager


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test databases."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def mgr(tmp_dir):
    """Return a fresh UserManager backed by a temp database."""
    return UserManager(db_dir=tmp_dir)


# ── Registration ──────────────────────────────────────────────────

class TestRegistration:
    def test_register_success(self, mgr):
        result = mgr.register("alice", "alice@example.com", "secret123", "Alice A.")
        assert result["success"] is True
        user = result["user"]
        assert user["username"] == "alice"
        assert user["email"] == "alice@example.com"
        assert user["full_name"] == "Alice A."
        assert user["role"] == "user"

    def test_sequential_user_ids(self, mgr):
        """User IDs should be sequential starting from U00001."""
        # 3 default users are seeded (U00001, U00002, U00003)
        users = mgr.list_users()
        ids = sorted(u["id"] for u in users)
        assert ids == ["U00001", "U00002", "U00003"]

        # Next registered user gets U00004
        r = mgr.register("seqtest", "seq@test.com", "pass123")
        assert r["success"] is True
        assert r["user"]["id"] == "U00004"

        r2 = mgr.register("seqtest2", "seq2@test.com", "pass123")
        assert r2["user"]["id"] == "U00005"

    def test_register_normalizes_username_email(self, mgr):
        result = mgr.register("  Alice  ", "  Alice@Example.COM  ", "pass123")
        assert result["success"] is True
        assert result["user"]["username"] == "alice"
        assert result["user"]["email"] == "alice@example.com"

    def test_register_duplicate_username(self, mgr):
        mgr.register("bob", "bob@one.com", "pass123")
        result = mgr.register("bob", "bob@two.com", "pass123")
        assert result["success"] is False
        assert "already taken" in result["error"].lower()

    def test_register_duplicate_email(self, mgr):
        mgr.register("user1", "same@mail.com", "pass123")
        result = mgr.register("user2", "same@mail.com", "pass123")
        assert result["success"] is False
        assert "already registered" in result["error"].lower()

    def test_register_empty_fields(self, mgr):
        result = mgr.register("", "a@b.com", "pass123")
        assert result["success"] is False

    def test_register_short_username(self, mgr):
        result = mgr.register("ab", "ab@b.com", "pass123")
        assert result["success"] is False
        assert "3 characters" in result["error"]

    def test_register_short_password(self, mgr):
        result = mgr.register("validuser", "v@b.com", "12345")
        assert result["success"] is False
        assert "6 characters" in result["error"]

    def test_register_invalid_email(self, mgr):
        result = mgr.register("validuser", "not-an-email", "pass123")
        assert result["success"] is False
        assert "email" in result["error"].lower()

    def test_register_admin_role(self, mgr):
        result = mgr.register("admin1", "admin@a.com", "pass123", role="admin")
        assert result["success"] is True
        assert result["user"]["role"] == "admin"


# ── Authentication ────────────────────────────────────────────────

class TestAuthentication:
    def test_auth_success(self, mgr):
        mgr.register("charlie", "c@c.com", "mypassword")
        result = mgr.authenticate("charlie", "mypassword")
        assert result["success"] is True
        assert result["user"]["username"] == "charlie"
        assert result["user"]["last_login"] is not None

    def test_auth_case_insensitive_username(self, mgr):
        mgr.register("delta", "d@d.com", "pass123")
        result = mgr.authenticate("DELTA", "pass123")
        assert result["success"] is True

    def test_auth_wrong_password(self, mgr):
        mgr.register("echo", "e@e.com", "correct")
        result = mgr.authenticate("echo", "wrong")
        assert result["success"] is False
        assert "invalid" in result["error"].lower()

    def test_auth_nonexistent_user(self, mgr):
        result = mgr.authenticate("nobody", "pass")
        assert result["success"] is False

    def test_auth_disabled_account(self, mgr):
        r = mgr.register("foxtrot", "f@f.com", "pass123")
        mgr.set_active(r["user"]["id"], False)
        result = mgr.authenticate("foxtrot", "pass123")
        assert result["success"] is False
        assert "disabled" in result["error"].lower()


# ── Queries ──────────────────────────────────────────────────────

class TestQueries:
    def test_get_by_id(self, mgr):
        r = mgr.register("golf", "g@g.com", "pass123")
        user = mgr.get_user_by_id(r["user"]["id"])
        assert user is not None
        assert user["username"] == "golf"

    def test_get_by_username(self, mgr):
        mgr.register("hotel", "h@h.com", "pass123")
        user = mgr.get_user_by_username("hotel")
        assert user is not None
        assert user["email"] == "h@h.com"

    def test_get_nonexistent(self, mgr):
        assert mgr.get_user_by_id("no-such-id") is None
        assert mgr.get_user_by_username("no-such") is None

    def test_list_users(self, mgr):
        baseline = mgr.get_user_count()          # 3 seeded defaults
        mgr.register("user1", "user1@u.com", "pass123")
        mgr.register("user2", "user2@u.com", "pass123")
        users = mgr.list_users()
        assert len(users) == baseline + 2

    def test_user_count(self, mgr):
        baseline = mgr.get_user_count()           # 3 seeded defaults
        mgr.register("count1", "c1@c.com", "pass123")
        assert mgr.get_user_count() == baseline + 1

    def test_password_not_in_dict(self, mgr):
        r = mgr.register("secure", "s@s.com", "pass123")
        user = mgr.get_user_by_id(r["user"]["id"])
        assert "password" not in user


# ── Updates ───────────────────────────────────────────────────────

class TestUpdates:
    def test_update_password(self, mgr):
        r = mgr.register("india", "i@i.com", "oldpass")
        uid = r["user"]["id"]
        assert mgr.update_password(uid, "newpass1") is True
        # Old password should fail
        assert mgr.authenticate("india", "oldpass")["success"] is False
        # New password should work
        assert mgr.authenticate("india", "newpass1")["success"] is True

    def test_update_password_too_short(self, mgr):
        r = mgr.register("juliet", "j@j.com", "pass123")
        assert mgr.update_password(r["user"]["id"], "ab") is False

    def test_set_active(self, mgr):
        r = mgr.register("kilo", "k@k.com", "pass123")
        uid = r["user"]["id"]
        assert mgr.set_active(uid, False) is True
        assert mgr.get_user_by_id(uid)["is_active"] is False
        assert mgr.set_active(uid, True) is True
        assert mgr.get_user_by_id(uid)["is_active"] is True

    def test_delete_user(self, mgr):
        r = mgr.register("lima", "l@l.com", "pass123")
        uid = r["user"]["id"]
        assert mgr.delete_user(uid) is True
        assert mgr.get_user_by_id(uid) is None
        assert mgr.delete_user("no-such-id") is False


# ── Token Usage Tracking ─────────────────────────────────────────

class TestTokenUsage:
    def test_record_and_get(self, mgr):
        r = mgr.register("mike", "m@m.com", "pass123")
        uid = r["user"]["id"]
        mgr.record_token_usage(uid, "generate", 10, 20)
        usage = mgr.get_user_token_usage(uid)
        assert usage["total_tokens"] == 30
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 20
        assert usage["request_count"] == 1

    def test_multiple_records_accumulate(self, mgr):
        r = mgr.register("november", "n@n.com", "pass123")
        uid = r["user"]["id"]
        mgr.record_token_usage(uid, "generate", 10, 20)
        mgr.record_token_usage(uid, "nl2fix", 15, 25)
        mgr.record_token_usage(uid, "generate", 5, 10)
        usage = mgr.get_user_token_usage(uid)
        assert usage["total_tokens"] == 85
        assert usage["request_count"] == 3

    def test_by_endpoint_breakdown(self, mgr):
        r = mgr.register("oscar", "o@o.com", "pass123")
        uid = r["user"]["id"]
        mgr.record_token_usage(uid, "generate", 10, 20)
        mgr.record_token_usage(uid, "nl2fix", 15, 25)
        usage = mgr.get_user_token_usage(uid)
        assert "generate" in usage["by_endpoint"]
        assert "nl2fix" in usage["by_endpoint"]
        assert usage["by_endpoint"]["generate"]["input_tokens"] == 10
        assert usage["by_endpoint"]["nl2fix"]["requests"] == 1

    def test_empty_usage(self, mgr):
        r = mgr.register("papa", "p@p.com", "pass123")
        uid = r["user"]["id"]
        usage = mgr.get_user_token_usage(uid)
        assert usage["total_tokens"] == 0
        assert usage["request_count"] == 0
        assert usage["by_endpoint"] == {}

    def test_all_users_token_usage(self, mgr):
        r1 = mgr.register("quebec", "q@q.com", "pass123")
        r2 = mgr.register("romeo", "r@r.com", "pass123")
        mgr.record_token_usage(r1["user"]["id"], "generate", 100, 200)
        mgr.record_token_usage(r2["user"]["id"], "nl2fix", 50, 50)
        all_usage = mgr.get_all_users_token_usage()
        # Includes 3 seeded users + 2 new users
        usernames = {u["username"] for u in all_usage}
        assert "quebec" in usernames
        assert "romeo" in usernames
        # Sorted by total_tokens descending
        quebec = next(u for u in all_usage if u["username"] == "quebec")
        assert quebec["total_tokens"] == 300
        assert quebec["request_count"] == 1

    def test_all_users_includes_zero_usage(self, mgr):
        """Users with no token usage still appear with 0 counts."""
        all_usage = mgr.get_all_users_token_usage()
        for u in all_usage:
            assert u["total_tokens"] == 0
            assert u["request_count"] == 0
