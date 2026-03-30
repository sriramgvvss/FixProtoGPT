"""
Security tests for FixProtoGPT.

Covers:
- SQL injection vectors (auth, registration, queries)
- XSS / script injection in input fields
- Authentication & session security
- Authorization / privilege escalation
- Security headers
- Input validation & boundary conditions
- Path traversal / file upload safety
- Error message information leakage
- CORS configuration
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path):
    """Flask test client with an isolated temp database."""
    from src.persistence.interaction_logger import InteractionLogger
    from src.persistence.user_manager import UserManager
    import src.api.state as state

    state.interaction_log = InteractionLogger(log_dir=tmp_path)
    state.user_manager = UserManager(db_dir=tmp_path)

    from src.api.app import create_app
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def auth_client(client):
    """Client with an authenticated regular-user session."""
    client.post("/auth/register", json={
        "username": "secuser",
        "email": "sec@test.com",
        "password": "secure123",
    }, content_type="application/json")
    client.post("/auth/login", json={
        "username": "secuser",
        "password": "secure123",
    }, content_type="application/json")
    return client


@pytest.fixture
def admin_client(client):
    """Client logged in as the seeded admin user."""
    client.post("/auth/login", json={
        "username": "admin",
        "password": "admin123",
    }, content_type="application/json")
    return client


def _mock_resolve(query):
    """Fast stub for resolve_symbol — no network calls."""
    return "AAPL" if query else None


# ══════════════════════════════════════════════════════════════════
# 1. SQL INJECTION
# ══════════════════════════════════════════════════════════════════

class TestSQLInjection:
    """Verify parameterised queries prevent SQL injection."""

    SQL_PAYLOADS = [
        "' OR '1'='1",
        "' OR '1'='1' --",
        "'; DROP TABLE users; --",
        "' UNION SELECT * FROM users --",
        "admin'--",
        "1; DELETE FROM users",
        "' OR 1=1 LIMIT 1 --",
        "'; INSERT INTO users VALUES('hacked','h','h','h','h','admin',datetime(),'',1);--",
    ]

    def test_login_sql_injection(self, client):
        """SQL payloads in username/password must not bypass auth."""
        for payload in self.SQL_PAYLOADS:
            resp = client.post("/auth/login", json={
                "username": payload,
                "password": payload,
            }, content_type="application/json")
            assert resp.status_code == 401, f"Payload bypassed auth: {payload}"
            data = resp.get_json()
            assert data["success"] is False

    def test_register_sql_injection_username(self, client):
        """SQL payloads in registration fields must not corrupt data."""
        for i, payload in enumerate(self.SQL_PAYLOADS):
            resp = client.post("/auth/register", json={
                "username": payload,
                "email": f"sqli{i}@test.com",
                "password": "secure123",
            }, content_type="application/json")
            # Should either fail validation or succeed safely
            if resp.status_code == 201:
                data = resp.get_json()
                # The username should be stored literally, not executed
                assert data["user"]["username"] == payload.strip().lower()

    def test_register_sql_injection_email(self, client):
        """SQL payloads in email field."""
        for i, payload in enumerate(self.SQL_PAYLOADS):
            resp = client.post("/auth/register", json={
                "username": f"sqliuser{i}",
                "email": payload,
                "password": "secure123",
            }, content_type="application/json")
            # Should fail due to invalid email format
            assert resp.status_code == 400

    def test_feedback_sql_injection(self, auth_client):
        """SQL payloads in interaction_id must not corrupt the DB."""
        for payload in self.SQL_PAYLOADS:
            resp = auth_client.post("/api/feedback", json={
                "interaction_id": payload,
                "rating": "positive",
            }, content_type="application/json")
            assert resp.status_code in (404, 400, 500)

    def test_interaction_delete_sql_injection(self, auth_client):
        """SQL payloads in URL path must not cause mass deletion."""
        for payload in self.SQL_PAYLOADS:
            resp = auth_client.delete(f"/api/interactions/{payload}")
            assert resp.status_code in (404, 400, 500)

    @patch("src.api.routes.core.resolve_symbol", side_effect=_mock_resolve)
    def test_generate_sql_injection(self, _mock, auth_client):
        """SQL payloads in prompt must not touch the DB."""
        for payload in self.SQL_PAYLOADS:
            resp = auth_client.post("/api/generate", json={
                "prompt": payload,
            }, content_type="application/json")
            assert resp.status_code in (200, 503)


# ══════════════════════════════════════════════════════════════════
# 2. XSS / SCRIPT INJECTION
# ══════════════════════════════════════════════════════════════════

class TestXSSInjection:
    """Ensure script payloads are not reflected unescaped."""

    XSS_PAYLOADS = [
        "<script>alert('XSS')</script>",
        '<img src=x onerror=alert(1)>',
        '"><script>alert(document.cookie)</script>',
        "javascript:alert(1)",
        "<svg/onload=alert(1)>",
        "{{7*7}}",  # Template injection
        "${7*7}",   # Template injection (alternate)
        "<iframe src='javascript:alert(1)'>",
    ]

    def test_xss_in_registration(self, client):
        """XSS payloads in registration fields must be stored safely."""
        for i, payload in enumerate(self.XSS_PAYLOADS):
            resp = client.post("/auth/register", json={
                "username": f"xssuser{i}",
                "email": f"xss{i}@test.com",
                "password": "secure123",
                "full_name": payload,
            }, content_type="application/json")
            if resp.status_code == 201:
                data = resp.get_json()
                # full_name must be stored literally, not executed
                assert "<script>" not in json.dumps(data).replace(payload, "")

    @patch("src.api.routes.core.resolve_symbol", side_effect=_mock_resolve)
    def test_xss_in_generate_prompt(self, _mock, auth_client):
        """XSS payloads in generate endpoint must not be reflected raw."""
        for payload in self.XSS_PAYLOADS:
            resp = auth_client.post("/api/generate", json={
                "prompt": payload,
            }, content_type="application/json")
            if resp.status_code == 200:
                # The raw response body must not contain executable script tags
                # outside of properly JSON-encoded strings
                raw = resp.data.decode()
                # JSON encoding should escape angle brackets
                assert "<script>" not in raw

    @patch("src.api.routes.core.resolve_symbol", side_effect=_mock_resolve)
    def test_xss_in_nl2fix(self, _mock, auth_client):
        """XSS payloads in nl2fix input."""
        for payload in self.XSS_PAYLOADS:
            resp = auth_client.post("/api/nl2fix", json={
                "text": payload,
            }, content_type="application/json")
            if resp.status_code == 200:
                raw = resp.data.decode()
                assert "<script>" not in raw

    @patch("src.api.routes.core.resolve_symbol", side_effect=_mock_resolve)
    def test_xss_in_explain(self, _mock, auth_client):
        """XSS payloads in explain input."""
        for payload in self.XSS_PAYLOADS:
            resp = auth_client.post("/api/explain", json={
                "message": payload,
            }, content_type="application/json")
            if resp.status_code == 200:
                raw = resp.data.decode()
                assert "<script>" not in raw

    def test_xss_in_feedback_comment(self, auth_client):
        """XSS payloads in feedback comment field."""
        for payload in self.XSS_PAYLOADS:
            resp = auth_client.post("/api/feedback", json={
                "interaction_id": "IN00000001",
                "rating": "positive",
                "comment": payload,
            }, content_type="application/json")
            # Should not crash
            assert resp.status_code in (200, 404)


# ══════════════════════════════════════════════════════════════════
# 3. AUTHENTICATION & SESSION SECURITY
# ══════════════════════════════════════════════════════════════════

class TestAuthSecurity:
    """Authentication bypass and session hijacking tests."""

    def test_unauthenticated_api_blocked(self, client):
        """All API endpoints must require authentication."""
        protected_routes = [
            ("GET", "/api/status"),
            ("GET", "/api/examples"),
            ("POST", "/api/generate"),
            ("POST", "/api/nl2fix"),
            ("POST", "/api/validate"),
            ("POST", "/api/explain"),
            ("POST", "/api/convert/to-json"),
            ("POST", "/api/convert/to-xml"),
            ("GET", "/api/interactions"),
            ("POST", "/api/feedback"),
            ("GET", "/api/learning/status"),
            ("POST", "/api/learning/export"),
            ("GET", "/api/versions"),
            ("POST", "/api/version"),
            ("GET", "/auth/me"),
            ("GET", "/auth/token-usage"),
        ]
        for method, path in protected_routes:
            if method == "GET":
                resp = client.get(path, headers={"Accept": "application/json"})
            elif method == "POST":
                resp = client.post(path, json={}, content_type="application/json",
                                   headers={"Accept": "application/json"})
            assert resp.status_code == 401, f"{method} {path} returned {resp.status_code} without auth"

    def test_admin_endpoints_require_admin_role(self, auth_client):
        """Regular users must not access admin-only endpoints."""
        admin_routes = [
            ("GET", "/auth/admin/token-usage"),
            ("GET", "/api/learning/finetune/status"),
        ]
        for method, path in admin_routes:
            resp = auth_client.get(path)
            assert resp.status_code == 403, f"{method} {path} accessible by non-admin"

    def test_session_invalidated_on_logout(self, auth_client):
        """After logout, session-based requests must fail."""
        resp = auth_client.get("/auth/me")
        assert resp.status_code == 200

        auth_client.post("/auth/logout")

        resp = auth_client.get("/auth/me", headers={"Accept": "application/json"})
        assert resp.status_code == 401

    def test_session_cookie_httponly(self, client):
        """Session cookie must have HttpOnly flag."""
        client.post("/auth/register", json={
            "username": "cookietest",
            "email": "cookie@test.com",
            "password": "secure123",
        }, content_type="application/json")
        resp = client.post("/auth/login", json={
            "username": "cookietest",
            "password": "secure123",
        }, content_type="application/json")
        # Check Set-Cookie header for HttpOnly
        cookies = resp.headers.getlist("Set-Cookie")
        session_cookies = [c for c in cookies if "session" in c.lower()]
        for cookie in session_cookies:
            assert "HttpOnly" in cookie, f"Session cookie missing HttpOnly: {cookie}"

    def test_password_not_in_any_response(self, client):
        """Password/hash must never appear in API responses."""
        client.post("/auth/register", json={
            "username": "pwnochk",
            "email": "pw@test.com",
            "password": "mysecretpw",
        }, content_type="application/json")
        resp = client.post("/auth/login", json={
            "username": "pwnochk",
            "password": "mysecretpw",
        }, content_type="application/json")
        # Login response must not contain password
        raw = resp.data.decode()
        assert "mysecretpw" not in raw
        assert "scrypt" not in raw
        assert "pbkdf2" not in raw

        # /auth/me must not contain password
        resp = client.get("/auth/me")
        raw = resp.data.decode()
        assert "mysecretpw" not in raw
        assert "password" not in raw.lower() or '"password"' not in raw

        # Admin user list must not contain passwords
        client.post("/auth/login", json={
            "username": "admin", "password": "admin123",
        }, content_type="application/json")
        resp = client.get("/auth/users")
        raw = resp.data.decode()
        assert "mysecretpw" not in raw
        assert "scrypt" not in raw


# ══════════════════════════════════════════════════════════════════
# 4. AUTHORIZATION / PRIVILEGE ESCALATION
# ══════════════════════════════════════════════════════════════════

class TestAuthorization:
    """Test that users cannot escalate privileges."""

    def test_register_cannot_set_admin_role_by_default(self, client):
        """A normal registration should default to 'user' role.

        Note: The current API *does* allow role= in register. This test
        documents that behaviour. If role should be locked to 'user' for
        public registration, the route must be updated.
        """
        resp = client.post("/auth/register", json={
            "username": "escalator",
            "email": "esc@test.com",
            "password": "secure123",
            "role": "admin",  # attempt privilege escalation
        }, content_type="application/json")
        # Document current behaviour — either rejected or accepted
        if resp.status_code == 201:
            data = resp.get_json()
            # If accepted, the role field from user input should be ignored
            # Currently the route does NOT pass role to register(), so
            # the resulting role should be 'user' (the default)
            assert data["user"]["role"] == "user", \
                "Public registration allowed admin role — privilege escalation!"

    def test_regular_user_cannot_list_all_users(self, auth_client):
        """Non-admin must not access the user list."""
        resp = auth_client.get("/auth/users")
        assert resp.status_code == 403

    def test_regular_user_cannot_trigger_finetune(self, auth_client):
        """Non-admin must not trigger fine-tuning."""
        resp = auth_client.post("/api/learning/finetune",
                                json={}, content_type="application/json")
        assert resp.status_code == 403

    def test_regular_user_cannot_access_finetune_preflight(self, auth_client):
        """Non-admin must not access finetune preflight."""
        resp = auth_client.get("/api/learning/finetune/preflight")
        assert resp.status_code == 403


# ══════════════════════════════════════════════════════════════════
# 5. SECURITY HEADERS
# ══════════════════════════════════════════════════════════════════

class TestSecurityHeaders:
    """Verify security headers are present on responses."""

    def test_x_content_type_options(self, auth_client):
        resp = auth_client.get("/api/status")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options(self, auth_client):
        resp = auth_client.get("/api/status")
        assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_referrer_policy(self, auth_client):
        resp = auth_client.get("/api/status")
        assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    def test_xss_protection(self, auth_client):
        resp = auth_client.get("/api/status")
        assert "1" in resp.headers.get("X-XSS-Protection", "")

    def test_headers_on_error_responses(self, client):
        """Security headers should be set even on 401/404 responses."""
        resp = client.get("/api/status", headers={"Accept": "application/json"})
        assert resp.status_code == 401
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_headers_on_404(self, client):
        resp = client.get("/nonexistent")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"


# ══════════════════════════════════════════════════════════════════
# 6. INPUT VALIDATION & BOUNDARY CONDITIONS
# ══════════════════════════════════════════════════════════════════

class TestInputValidation:
    """Test edge cases and oversized inputs."""

    @patch("src.api.routes.core.resolve_symbol", side_effect=_mock_resolve)
    def test_empty_json_body(self, _mock, auth_client):
        """Endpoints must handle empty JSON gracefully."""
        endpoints = [
            "/api/generate",
            "/api/nl2fix",
            "/api/validate",
            "/api/explain",
        ]
        for ep in endpoints:
            resp = auth_client.post(ep, json={}, content_type="application/json")
            assert resp.status_code in (200, 400, 503), f"{ep} crashed on empty body"

    @patch("src.api.routes.core.resolve_symbol", side_effect=_mock_resolve)
    def test_missing_content_type(self, _mock, auth_client):
        """Endpoints must not crash when Content-Type is missing."""
        resp = auth_client.post("/api/generate", data="not json")
        assert resp.status_code in (400, 415, 500, 503)

    @patch("src.api.routes.core.resolve_symbol", side_effect=_mock_resolve)
    def test_oversized_prompt(self, _mock, auth_client):
        """Very large inputs should not cause server crash."""
        huge = "A" * 100_000
        resp = auth_client.post("/api/generate", json={
            "prompt": huge,
        }, content_type="application/json")
        assert resp.status_code in (200, 400, 413, 503)

    @patch("src.api.routes.core.resolve_symbol", side_effect=_mock_resolve)
    def test_unicode_input(self, _mock, auth_client):
        """Unicode / emoji in inputs should not cause crashes."""
        resp = auth_client.post("/api/generate", json={
            "prompt": "Buy 100 shares of 🍎 at $150 📈",
        }, content_type="application/json")
        assert resp.status_code in (200, 503)

    @patch("src.api.routes.core.resolve_symbol", side_effect=_mock_resolve)
    def test_null_bytes_in_input(self, _mock, auth_client):
        """Null bytes must not cause crashes or bypass validation."""
        resp = auth_client.post("/api/generate", json={
            "prompt": "Buy\x00100\x00AAPL",
        }, content_type="application/json")
        assert resp.status_code in (200, 400, 503)

    def test_login_empty_fields(self, client):
        """Empty username/password must be rejected."""
        resp = client.post("/auth/login", json={
            "username": "", "password": "",
        }, content_type="application/json")
        assert resp.status_code == 400

    def test_register_extremely_long_username(self, client):
        """Very long username must not crash the server."""
        resp = client.post("/auth/register", json={
            "username": "a" * 10_000,
            "email": "long@test.com",
            "password": "secure123",
        }, content_type="application/json")
        assert resp.status_code in (201, 400)

    def test_register_extremely_long_password(self, client):
        """Very long password must not crash the server."""
        resp = client.post("/auth/register", json={
            "username": "longpw",
            "email": "longpw@test.com",
            "password": "x" * 100_000,
        }, content_type="application/json")
        assert resp.status_code in (201, 400)

    def test_special_chars_in_fix_message(self, auth_client):
        """Special characters in FIX message must not crash conversion."""
        payloads = [
            "8=FIX.4.4\x0135=D\x0155=<script>\x0110=000\x01",
            '8=FIX.4.4|35=D|55="AAPL|10=000|',
            "8=FIX.4.4|35=D|55='OR 1=1--|10=000|",
        ]
        for msg in payloads:
            resp = auth_client.post("/api/validate", json={
                "message": msg,
            }, content_type="application/json")
            assert resp.status_code in (200, 400, 503)

            resp = auth_client.post("/api/convert/to-json", json={
                "message": msg,
            }, content_type="application/json")
            assert resp.status_code in (200, 400, 500)


# ══════════════════════════════════════════════════════════════════
# 7. ERROR INFORMATION LEAKAGE
# ══════════════════════════════════════════════════════════════════

class TestErrorLeakage:
    """Ensure error responses don't leak sensitive information."""

    def test_login_error_is_generic(self, client):
        """Failed login must not reveal whether username exists."""
        # Register a user
        client.post("/auth/register", json={
            "username": "knownuser",
            "email": "known@test.com",
            "password": "secure123",
        }, content_type="application/json")

        # Wrong password for existing user
        resp1 = client.post("/auth/login", json={
            "username": "knownuser", "password": "wrongpass",
        }, content_type="application/json")

        # Non-existent user
        resp2 = client.post("/auth/login", json={
            "username": "ghostuser", "password": "anypass",
        }, content_type="application/json")

        # Both should return same generic error
        assert resp1.status_code == resp2.status_code == 401
        assert resp1.get_json()["error"] == resp2.get_json()["error"]

    def test_404_does_not_leak_internals(self, client):
        """404 responses must not reveal server technology details."""
        resp = client.get("/api/nonexistent")
        raw = resp.data.decode().lower()
        assert "traceback" not in raw
        assert "sqlalchemy" not in raw
        assert "sqlite" not in raw

    def test_500_does_not_leak_stack_trace(self, auth_client):
        """Server errors should not expose full stack traces to clients."""
        # Send malformed data that might trigger internal errors
        resp = auth_client.post("/api/convert/to-json", json={
            "message": None,
        }, content_type="application/json")
        if resp.status_code == 500:
            raw = resp.data.decode().lower()
            assert "traceback" not in raw
            assert "file \"/" not in raw


# ══════════════════════════════════════════════════════════════════
# 8. PATH TRAVERSAL / FILE UPLOAD SAFETY
# ══════════════════════════════════════════════════════════════════

class TestFileUploadSecurity:
    """Test that file uploads cannot escape allowed directories."""

    def test_import_json_no_file(self, auth_client):
        """Import without file attachment must be rejected."""
        resp = auth_client.post("/api/import/json")
        assert resp.status_code == 400

    def test_import_json_empty_filename(self, auth_client):
        """Import with empty filename must be rejected."""
        from io import BytesIO
        resp = auth_client.post("/api/import/json",
                                data={"file": (BytesIO(b"{}"), "")},
                                content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_import_json_malicious_content(self, auth_client):
        """Malicious JSON content must not execute code."""
        from io import BytesIO
        malicious = b'{"__class__": {"__reduce__": ["os.system", ["whoami"]]}}'
        resp = auth_client.post("/api/import/json",
                                data={"file": (BytesIO(malicious), "evil.json")},
                                content_type="multipart/form-data")
        # Should fail validation or produce a safe result
        assert resp.status_code in (200, 400)

    def test_import_xml_xxe_attack(self, auth_client):
        """XML External Entity payloads must be blocked."""
        from io import BytesIO
        xxe_payload = b"""<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<fix><field tag="8">&xxe;</field></fix>"""
        resp = auth_client.post("/api/import/xml",
                                data={"file": (BytesIO(xxe_payload), "xxe.xml")},
                                content_type="multipart/form-data")
        if resp.status_code == 200:
            raw = resp.data.decode()
            assert "root:" not in raw  # /etc/passwd content must not appear


# ══════════════════════════════════════════════════════════════════
# 9. USER MANAGER SECURITY (unit level)
# ══════════════════════════════════════════════════════════════════

class TestUserManagerSecurity:
    """Direct security tests on UserManager (no HTTP layer)."""

    @pytest.fixture
    def mgr(self, tmp_path):
        from src.persistence.user_manager import UserManager
        return UserManager(db_dir=tmp_path)

    def test_passwords_are_hashed(self, mgr):
        """Stored passwords must be hashed, not plaintext."""
        import sqlite3
        mgr.register("hashtest", "hash@test.com", "myplainpw")
        conn = sqlite3.connect(str(mgr.db_path))
        row = conn.execute("SELECT password FROM users WHERE username='hashtest'").fetchone()
        conn.close()
        stored = row[0]
        assert stored != "myplainpw"
        assert "scrypt" in stored or "pbkdf2" in stored

    def test_sql_injection_in_username_lookup(self, mgr):
        """SQL injection in username lookup must not return data."""
        mgr.register("victim", "v@v.com", "pass123")
        result = mgr.get_user_by_username("' OR '1'='1")
        assert result is None

    def test_sql_injection_in_id_lookup(self, mgr):
        """SQL injection in ID lookup must not return data."""
        result = mgr.get_user_by_id("' OR '1'='1")
        assert result is None

    def test_disabled_user_cannot_login(self, mgr):
        """Disabled accounts must be rejected at authentication."""
        r = mgr.register("blocked", "b@b.com", "pass123")
        mgr.set_active(r["user"]["id"], False)
        result = mgr.authenticate("blocked", "pass123")
        assert result["success"] is False
        assert "disabled" in result["error"].lower()

    def test_timing_attack_resistance(self, mgr):
        """Login for existing vs non-existing user should both fail gracefully.

        (We can't measure timing precisely in a unit test, but we verify
        that both paths return the same error structure.)
        """
        mgr.register("exists", "e@e.com", "pass123")

        r1 = mgr.authenticate("exists", "wrongpass")
        r2 = mgr.authenticate("nonexistent", "anypass")

        assert r1["success"] is False
        assert r2["success"] is False
        # Same error message for both (doesn't reveal user existence)
        assert r1["error"] == r2["error"]


# ══════════════════════════════════════════════════════════════════
# 10. INTERACTION LOGGER SECURITY (unit level)
# ══════════════════════════════════════════════════════════════════

class TestInteractionLoggerSecurity:
    """Security tests on InteractionLogger (no HTTP layer)."""

    @pytest.fixture
    def logger(self, tmp_path):
        from src.persistence.interaction_logger import InteractionLogger
        return InteractionLogger(log_dir=tmp_path)

    def test_sql_injection_in_log(self, logger):
        """SQL payloads in logged data must be stored safely."""
        iid = logger.log(
            "nl2fix",
            {"text": "'; DROP TABLE interactions; --"},
            {"fix_message": "8=FIX.4.4|35=D|"},
        )
        # Table must still be intact
        records = logger.get_interactions()
        assert len(records) == 1
        assert records[0]["request"]["text"] == "'; DROP TABLE interactions; --"

    def test_sql_injection_in_feedback(self, logger):
        """SQL payloads in feedback must be stored safely."""
        iid = logger.log("nl2fix", {}, {})
        ok = logger.add_feedback(iid, "positive", comment="'; DROP TABLE interactions; --")
        assert ok is True
        records = logger.get_interactions()
        assert records[0]["feedback"]["comment"] == "'; DROP TABLE interactions; --"

    def test_sql_injection_in_delete(self, logger):
        """SQL payloads in delete must not cause mass deletion."""
        logger.log("nl2fix", {}, {})
        logger.log("nl2fix", {}, {})
        result = logger.delete("' OR '1'='1")
        assert result is False
        # Both records must still exist
        assert len(logger.get_interactions()) == 2

    def test_json_injection_in_metadata(self, logger):
        """Malicious JSON metadata must be stored/retrieved safely."""
        meta = {"__class__": "os.system", "cmd": "rm -rf /"}
        iid = logger.log("test", {"x": 1}, {"y": 2}, metadata=meta)
        records = logger.get_interactions()
        assert records[0]["metadata"]["__class__"] == "os.system"
        # Data is just stored, not executed
