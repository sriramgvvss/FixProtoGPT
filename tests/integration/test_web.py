"""
Integration tests for the Flask web application (web/app.py).

Tests Flask routes using the test client.
The model is mocked to avoid loading a real checkpoint.

Covers:
- Authentication (register, login, logout, me)
- GET / (index page, requires auth)
- GET /api/status
- GET /api/examples
- POST /api/generate (demo mode)
- POST /api/validate
- POST /api/explain
- POST /api/convert/to-json
- POST /api/convert/to-xml
"""

import json
import sys
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


@pytest.fixture
def client(tmp_path):
    """Create Flask test client with isolated DB in a temp directory."""
    from src.persistence.interaction_logger import InteractionLogger
    from src.persistence.user_manager import UserManager
    import src.api.state as state

    # Point both managers at the temp directory so tests don't touch the real DB
    state.interaction_log = InteractionLogger(log_dir=tmp_path)
    state.user_manager = UserManager(db_dir=tmp_path)

    from src.api.app import create_app
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _register_and_login(client, username="testuser", email="test@example.com", password="testpass123"):
    """Helper: register a user and log in, returning the session-cookie-bearing client."""
    client.post("/auth/register", json={
        "username": username,
        "email": email,
        "password": password,
    }, content_type="application/json")
    client.post("/auth/login", json={
        "username": username,
        "password": password,
    }, content_type="application/json")
    return client


# ── Authentication Tests ───────────────────────────────────────────

class TestAuth:
    def test_login_page(self, client):
        resp = client.get("/auth/login")
        assert resp.status_code == 200
        assert b"Sign In" in resp.data or b"FixProtoGPT" in resp.data

    def test_register_success(self, client):
        resp = client.post("/auth/register", json={
            "username": "newuser",
            "email": "new@example.com",
            "password": "pass123",
            "full_name": "New User",
        }, content_type="application/json")
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["success"] is True
        assert data["user"]["username"] == "newuser"

    def test_register_duplicate_username(self, client):
        client.post("/auth/register", json={
            "username": "dupuser", "email": "a@a.com", "password": "pass123"
        }, content_type="application/json")
        resp = client.post("/auth/register", json={
            "username": "dupuser", "email": "b@b.com", "password": "pass123"
        }, content_type="application/json")
        assert resp.status_code == 400
        assert "already taken" in resp.get_json()["error"].lower()

    def test_register_short_password(self, client):
        resp = client.post("/auth/register", json={
            "username": "shortpw", "email": "s@s.com", "password": "ab"
        }, content_type="application/json")
        assert resp.status_code == 400

    def test_login_success(self, client):
        _register_and_login(client, "loguser", "log@e.com", "mypass1")
        resp = client.get("/auth/me")
        assert resp.status_code == 200
        assert resp.get_json()["user"]["username"] == "loguser"

    def test_login_wrong_password(self, client):
        client.post("/auth/register", json={
            "username": "wrongpw", "email": "wp@e.com", "password": "correct"
        }, content_type="application/json")
        resp = client.post("/auth/login", json={
            "username": "wrongpw", "password": "incorrect"
        }, content_type="application/json")
        assert resp.status_code == 401

    def test_logout(self, client):
        _register_and_login(client, "logoutuser", "lo@e.com", "pass123")
        resp = client.post("/auth/logout")
        assert resp.status_code == 200
        # After logout, /auth/me should fail
        resp = client.get("/auth/me", headers={"Accept": "application/json"})
        assert resp.status_code == 401

    def test_unauthenticated_api_returns_401(self, client):
        resp = client.get("/api/status", headers={"Accept": "application/json"})
        assert resp.status_code == 401

    def test_unauthenticated_index_redirects(self, client):
        resp = client.get("/")
        assert resp.status_code == 302
        assert "/auth/login" in resp.headers.get("Location", "")


# ── Static Routes (authenticated) ─────────────────────────────────

class TestStaticRoutes:
    def test_index_page(self, client):
        _register_and_login(client, "idx1", "idx1@e.com", "pass123")
        resp = client.get("/")
        assert resp.status_code == 200

    def test_status_endpoint(self, client):
        _register_and_login(client, "stat1", "stat1@e.com", "pass123")
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "model_loaded" in data
        assert "model_available" in data
        assert "version" in data

    def test_examples_endpoint(self, client):
        _register_and_login(client, "ex1", "ex1@e.com", "pass123")
        resp = client.get("/api/examples")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, (dict, list))


# ── Generation Endpoints (demo mode, no model loaded) ─────────────

class TestGenerationEndpoints:
    def test_generate_without_model(self, client):
        _register_and_login(client, "gen1", "gen1@e.com", "pass123")
        resp = client.post("/api/generate",
                           json={"prompt": "8=FIXT.1.1|35=D|"},
                           content_type="application/json")
        assert resp.status_code in (200, 503)

    def test_nl2fix_without_model(self, client):
        _register_and_login(client, "nl1", "nl1@e.com", "pass123")
        resp = client.post("/api/nl2fix",
                           json={"text": "Buy 100 AAPL"},
                           content_type="application/json")
        assert resp.status_code in (200, 503)


# ── Validation Endpoint ───────────────────────────────────────────

class TestValidationEndpoint:
    def test_validate_endpoint(self, client):
        _register_and_login(client, "val1", "val1@e.com", "pass123")
        resp = client.post("/api/validate",
                           json={"message": "8=FIXT.1.1|35=D|55=AAPL|10=123|"},
                           content_type="application/json")
        assert resp.status_code in (200, 503)


# ── Conversion Endpoints ──────────────────────────────────────────

class TestConversionEndpoints:
    def test_convert_to_json(self, client):
        _register_and_login(client, "conv1", "conv1@e.com", "pass123")
        resp = client.post("/api/convert/to-json",
                           json={"message": "8=FIXT.1.1|35=D|55=AAPL|10=123|"},
                           content_type="application/json")
        if resp.status_code == 200:
            data = resp.get_json()
            assert "json_output" in data or "success" in data

    def test_convert_to_xml(self, client):
        _register_and_login(client, "conv2", "conv2@e.com", "pass123")
        resp = client.post("/api/convert/to-xml",
                           json={"message": "8=FIXT.1.1|35=D|55=AAPL|10=123|"},
                           content_type="application/json")
        if resp.status_code == 200:
            data = resp.get_json()
            assert data is not None


# ── Error Handling ─────────────────────────────────────────────────

class TestErrorHandling:
    def test_missing_json_body(self, client):
        _register_and_login(client, "err1", "err1@e.com", "pass123")
        resp = client.post("/api/generate", content_type="application/json")
        assert resp.status_code in (400, 500, 503)

    def test_invalid_route(self, client):
        resp = client.get("/api/nonexistent")
        assert resp.status_code == 404


# ── Feedback & Learning Endpoints ──────────────────────────────────

class TestFeedbackEndpoints:
    def test_feedback_missing_id(self, client):
        _register_and_login(client, "fb1", "fb1@e.com", "pass123")
        resp = client.post("/api/feedback",
                           json={"rating": "positive"},
                           content_type="application/json")
        assert resp.status_code == 400

    def test_feedback_invalid_rating(self, client):
        _register_and_login(client, "fb2", "fb2@e.com", "pass123")
        resp = client.post("/api/feedback",
                           json={"interaction_id": "fake-id", "rating": "maybe"},
                           content_type="application/json")
        assert resp.status_code == 400

    def test_feedback_unknown_id(self, client):
        _register_and_login(client, "fb3", "fb3@e.com", "pass123")
        resp = client.post("/api/feedback",
                           json={"interaction_id": "no-such-uuid", "rating": "positive"},
                           content_type="application/json")
        assert resp.status_code == 404

    def test_interactions_endpoint(self, client):
        _register_and_login(client, "int1", "int1@e.com", "pass123")
        resp = client.get("/api/interactions")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "interactions" in data
        assert "stats" in data

    def test_learning_status_endpoint(self, client):
        _register_and_login(client, "ls1", "ls1@e.com", "pass123")
        resp = client.get("/api/learning/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total_interactions" in data
        assert "exportable_pairs" in data

    def test_learning_export_endpoint(self, client):
        _register_and_login(client, "le1", "le1@e.com", "pass123")
        resp = client.post("/api/learning/export",
                           content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "pair_count" in data

    def test_status_includes_interactions(self, client):
        _register_and_login(client, "si1", "si1@e.com", "pass123")
        resp = client.get("/api/status")
        data = resp.get_json()
        assert "interactions" in data
        assert "total_interactions" in data["interactions"]


class TestDeleteEndpoints:
    def test_delete_interaction_not_found(self, client):
        _register_and_login(client, "del1", "del1@e.com", "pass123")
        resp = client.delete("/api/interactions/no-such-uuid")
        assert resp.status_code == 404

    def test_clear_all_interactions(self, client):
        _register_and_login(client, "clr1", "clr1@e.com", "pass123")
        resp = client.delete("/api/interactions")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "deleted_count" in data


# ── Token Usage Endpoints ──────────────────────────────────────────

class TestTokenUsageEndpoints:
    def test_status_includes_token_usage(self, client):
        _register_and_login(client, "tu1", "tu1@e.com", "pass123")
        resp = client.get("/api/status")
        data = resp.get_json()
        assert "token_usage" in data
        assert "total_tokens" in data["token_usage"]

    def test_my_token_usage(self, client):
        _register_and_login(client, "tu2", "tu2@e.com", "pass123")
        resp = client.get("/auth/token-usage")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_tokens"] == 0
        assert data["request_count"] == 0

    def test_my_token_usage_requires_auth(self, client):
        resp = client.get("/auth/token-usage",
                          headers={"Accept": "application/json"})
        assert resp.status_code == 401

    def test_admin_token_usage_requires_admin(self, client):
        _register_and_login(client, "tu3", "tu3@e.com", "pass123")
        resp = client.get("/auth/admin/token-usage")
        assert resp.status_code == 403

    def test_admin_all_users_token_usage(self, client):
        """Admin can see all users' token usage."""
        # Log in as the seeded admin user
        client.post("/auth/login", json={
            "username": "admin",
            "password": "admin123",
        }, content_type="application/json")
        resp = client.get("/auth/admin/token-usage")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "users" in data
        assert isinstance(data["users"], list)
        # Should include at least the 3 seeded users
        assert len(data["users"]) >= 3
        for u in data["users"]:
            assert "username" in u
            assert "total_tokens" in u
            assert "request_count" in u


# ── Version Management Tests ──────────────────────────────────────

class TestVersionEndpoints:
    """Tests for GET /api/versions and POST /api/version."""

    def test_list_versions_requires_auth(self, client):
        resp = client.get("/api/versions",
                          headers={"Accept": "application/json"})
        assert resp.status_code == 401

    def test_list_versions(self, client):
        _register_and_login(client, "ver1", "ver1@e.com", "pass123")
        resp = client.get("/api/versions")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "versions" in data
        assert "current" in data
        versions = data["versions"]
        assert len(versions) == 8  # 4.0–5.0SP2
        # Each version must have required fields
        for v in versions:
            assert "version" in v
            assert "label" in v
            assert "has_data" in v
            assert "has_model" in v
            assert "selected" in v
            assert "begin_string" in v

    def test_list_versions_default_selected(self, client):
        """Default selected version should be 5.0SP2."""
        _register_and_login(client, "ver2", "ver2@e.com", "pass123")
        resp = client.get("/api/versions")
        data = resp.get_json()
        selected = [v for v in data["versions"] if v["selected"]]
        assert len(selected) == 1
        assert selected[0]["version"] == "5.0SP2"

    def test_set_version_requires_auth(self, client):
        resp = client.post("/api/version",
                           json={"version": "4.4"},
                           content_type="application/json",
                           headers={"Accept": "application/json"})
        assert resp.status_code == 401

    def test_set_version_success(self, client):
        _register_and_login(client, "ver3", "ver3@e.com", "pass123")
        resp = client.post("/api/version",
                           json={"version": "4.4"},
                           content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["version"] == "4.4"
        assert data["label"] == "FIX 4.4"

    def test_set_version_persists_in_session(self, client):
        _register_and_login(client, "ver4", "ver4@e.com", "pass123")
        # Set to 4.4
        client.post("/api/version",
                     json={"version": "4.4"},
                     content_type="application/json")
        # Verify list_versions reflects the change
        resp = client.get("/api/versions")
        data = resp.get_json()
        assert data["current"] == "4.4"
        selected = [v for v in data["versions"] if v["selected"]]
        assert len(selected) == 1
        assert selected[0]["version"] == "4.4"

    def test_set_version_invalid(self, client):
        _register_and_login(client, "ver5", "ver5@e.com", "pass123")
        resp = client.post("/api/version",
                           json={"version": "6.0"},
                           content_type="application/json")
        assert resp.status_code == 400
        assert "Unknown" in resp.get_json()["error"]

    def test_set_version_empty(self, client):
        _register_and_login(client, "ver6", "ver6@e.com", "pass123")
        resp = client.post("/api/version",
                           json={"version": ""},
                           content_type="application/json")
        assert resp.status_code == 400

    def test_status_includes_version(self, client):
        _register_and_login(client, "ver7", "ver7@e.com", "pass123")
        resp = client.get("/api/status")
        data = resp.get_json()
        assert "fix_version" in data
        assert "fix_version_label" in data
        assert data["fix_version"] == "5.0SP2"
        assert data["fix_version_label"] == "FIX 5.0 SP2"

    def test_status_reflects_version_change(self, client):
        _register_and_login(client, "ver8", "ver8@e.com", "pass123")
        client.post("/api/version",
                     json={"version": "5.0SP1"},
                     content_type="application/json")
        resp = client.get("/api/status")
        data = resp.get_json()
        assert data["fix_version"] == "5.0SP1"
        assert data["fix_version_label"] == "FIX 5.0 SP1"

    def test_generate_includes_version(self, client):
        """Demo-mode generate should include fix_version in response."""
        _register_and_login(client, "ver9", "ver9@e.com", "pass123")
        # Switch to FIX 4.4 (no model on disk)
        client.post("/api/version",
                     json={"version": "4.4"},
                     content_type="application/json")
        resp = client.post("/api/generate",
                           json={"prompt": "Buy 100 AAPL"},
                           content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("fix_version") == "4.4"
        # Demo response should use FIX.4.4 BeginString
        assert "FIX.4.4" in data["generated"]
        # FIX 4.x should NOT have ApplVerID (tag 1128)
        assert "1128=" not in data["generated"]

    def test_generate_fix5_includes_appl_ver_id(self, client):
        """Demo-mode generate for FIX 5.x should include ApplVerID (unless a real model is loaded)."""
        _register_and_login(client, "ver10", "ver10@e.com", "pass123")
        # Switch to 5.0 (no model on disk, guaranteed demo mode)
        client.post("/api/version",
                     json={"version": "5.0"},
                     content_type="application/json")
        resp = client.post("/api/generate",
                           json={"prompt": "Buy 100 AAPL"},
                           content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("fix_version") == "5.0"
        # Demo response for 5.x should contain ApplVerID
        if data.get("demo_mode"):
            assert "1128=" in data["generated"]

    def test_set_version_to_all_valid_versions(self, client):
        """Cycle through all 8 versions — all should succeed."""
        _register_and_login(client, "ver11", "ver11@e.com", "pass123")
        for ver in ["4.0", "4.1", "4.2", "4.3", "4.4", "5.0", "5.0SP1", "5.0SP2"]:
            resp = client.post("/api/version",
                               json={"version": ver},
                               content_type="application/json")
            assert resp.status_code == 200, f"Failed to set version {ver}"
            assert resp.get_json()["version"] == ver
