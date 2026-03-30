"""
Tests for the FixProtoGPT action logging system.

Covers:
- Server request logging (JSON lines in server.log)
- User action logging (JSON lines in user_actions.log)
- Log rotation configuration
- Log entry structure and content
- Integration with Flask routes
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def log_dir(tmp_path):
    """Temporary log directory."""
    return tmp_path / "logs"


@pytest.fixture
def client(tmp_path, log_dir):
    """Flask test client with isolated temp directories for DB and logs."""
    from src.persistence.interaction_logger import InteractionLogger
    from src.persistence.user_manager import UserManager
    import src.api.state as state

    state.interaction_log = InteractionLogger(log_dir=tmp_path)
    state.user_manager = UserManager(db_dir=tmp_path)

    # Patch LOG_DIR so logs go to our temp directory
    with patch("src.persistence.action_logger.LOG_DIR", log_dir):
        # Clear any cached loggers from prior tests
        import logging
        for name in ("fixprotogpt.server", "fixprotogpt.actions"):
            lgr = logging.getLogger(name)
            lgr.handlers.clear()

        from src.api.app import create_app
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


@pytest.fixture
def auth_client(client):
    """Client with an authenticated regular-user session."""
    client.post("/auth/register", json={
        "username": "loguser",
        "email": "log@test.com",
        "password": "secure123",
    }, content_type="application/json")
    client.post("/auth/login", json={
        "username": "loguser",
        "password": "secure123",
    }, content_type="application/json")
    return client


def _read_log_lines(log_dir, filename):
    """Read a log file and return parsed JSON entries."""
    path = log_dir / filename
    if not path.exists():
        return []
    lines = path.read_text().strip().split("\n")
    return [json.loads(line) for line in lines if line.strip()]


def _mock_resolve(query):
    """Fast stub for resolve_symbol — no network calls."""
    return "AAPL" if query else None


# ══════════════════════════════════════════════════════════════════
# 1. SERVER REQUEST LOGGING
# ══════════════════════════════════════════════════════════════════

class TestServerLogging:
    """Verify that server.log captures HTTP requests as JSON lines."""

    def test_server_log_created(self, auth_client, log_dir):
        """server.log file must be created after a request."""
        auth_client.get("/api/status")
        assert (log_dir / "server.log").exists()

    def test_server_log_json_format(self, auth_client, log_dir):
        """Each line in server.log must be valid JSON."""
        auth_client.get("/api/status")
        entries = _read_log_lines(log_dir, "server.log")
        assert len(entries) >= 1
        for entry in entries:
            assert isinstance(entry, dict)

    def test_server_log_entry_fields(self, auth_client, log_dir):
        """Server log entries must contain required fields."""
        auth_client.get("/api/status")
        entries = _read_log_lines(log_dir, "server.log")
        status_entries = [e for e in entries if e.get("path") == "/api/status"]
        assert len(status_entries) >= 1
        entry = status_entries[0]
        assert entry["event"] == "http_request"
        assert entry["method"] == "GET"
        assert entry["path"] == "/api/status"
        assert entry["status"] == 200
        assert "duration_ms" in entry
        assert "timestamp" in entry
        assert "username" in entry

    def test_server_log_records_user_context(self, auth_client, log_dir):
        """Authenticated requests must log the user_id and username."""
        auth_client.get("/api/status")
        entries = _read_log_lines(log_dir, "server.log")
        status_entries = [e for e in entries if e.get("path") == "/api/status"]
        entry = status_entries[0]
        assert entry["username"] == "loguser"
        assert entry["user_id"] is not None

    def test_server_log_records_status_code(self, client, log_dir):
        """Unauthenticated requests must log 401 status."""
        client.get("/api/status", headers={"Accept": "application/json"})
        entries = _read_log_lines(log_dir, "server.log")
        status_entries = [e for e in entries if e.get("path") == "/api/status"]
        assert len(status_entries) >= 1
        assert status_entries[0]["status"] == 401

    def test_static_requests_not_logged(self, auth_client, log_dir):
        """Requests to /static/ should be skipped to reduce noise."""
        auth_client.get("/static/nonexistent.js")
        entries = _read_log_lines(log_dir, "server.log")
        static_entries = [e for e in entries if "/static/" in e.get("path", "")]
        assert len(static_entries) == 0

    def test_server_log_records_duration(self, auth_client, log_dir):
        """Duration must be a non-negative number."""
        auth_client.get("/api/status")
        entries = _read_log_lines(log_dir, "server.log")
        status_entries = [e for e in entries if e.get("path") == "/api/status"]
        assert status_entries[0]["duration_ms"] >= 0


# ══════════════════════════════════════════════════════════════════
# 2. USER ACTION LOGGING
# ══════════════════════════════════════════════════════════════════

class TestUserActionLogging:
    """Verify that user_actions.log captures meaningful actions."""

    def test_login_action_logged(self, client, log_dir):
        """Successful login must appear in user_actions.log."""
        client.post("/auth/register", json={
            "username": "actionuser",
            "email": "action@test.com",
            "password": "secure123",
        }, content_type="application/json")
        client.post("/auth/login", json={
            "username": "actionuser",
            "password": "secure123",
        }, content_type="application/json")
        entries = _read_log_lines(log_dir, "user_actions.log")
        login_entries = [e for e in entries if e.get("action") == "login"]
        assert len(login_entries) >= 1
        assert login_entries[-1]["status"] == "success"
        assert login_entries[-1]["username"] == "actionuser"

    def test_failed_login_logged(self, client, log_dir):
        """Failed login must be logged with status=failure."""
        client.post("/auth/login", json={
            "username": "nonexistent",
            "password": "wrongpass",
        }, content_type="application/json")
        entries = _read_log_lines(log_dir, "user_actions.log")
        failed = [e for e in entries if e.get("action") == "login" and e.get("status") == "failure"]
        assert len(failed) >= 1

    def test_register_action_logged(self, client, log_dir):
        """Registration must appear in user_actions.log."""
        client.post("/auth/register", json={
            "username": "newuser",
            "email": "new@test.com",
            "password": "secure123",
        }, content_type="application/json")
        entries = _read_log_lines(log_dir, "user_actions.log")
        reg_entries = [e for e in entries if e.get("action") == "register"]
        assert len(reg_entries) >= 1
        assert reg_entries[-1]["status"] == "success"
        assert reg_entries[-1]["detail"]["email"] == "new@test.com"

    def test_logout_action_logged(self, auth_client, log_dir):
        """Logout must appear in user_actions.log."""
        auth_client.post("/auth/logout")
        entries = _read_log_lines(log_dir, "user_actions.log")
        logout_entries = [e for e in entries if e.get("action") == "logout"]
        assert len(logout_entries) >= 1

    @patch("src.api.routes.core.resolve_symbol", side_effect=_mock_resolve)
    def test_generate_action_logged(self, _mock, auth_client, log_dir):
        """Generate endpoint must log user action."""
        auth_client.post("/api/generate", json={
            "prompt": "Buy 100 AAPL",
        }, content_type="application/json")
        entries = _read_log_lines(log_dir, "user_actions.log")
        gen_entries = [e for e in entries if e.get("action") == "generate"]
        assert len(gen_entries) >= 1
        assert "prompt" in gen_entries[-1].get("detail", {})

    @patch("src.api.routes.core.resolve_symbol", side_effect=_mock_resolve)
    def test_nl2fix_action_logged(self, _mock, auth_client, log_dir):
        """NL2FIX endpoint must log user action."""
        auth_client.post("/api/nl2fix", json={
            "text": "Sell 50 GOOGL at 150",
        }, content_type="application/json")
        entries = _read_log_lines(log_dir, "user_actions.log")
        nl_entries = [e for e in entries if e.get("action") == "nl2fix"]
        assert len(nl_entries) >= 1

    def test_feedback_action_logged(self, auth_client, log_dir):
        """Feedback submission must log user action."""
        # First create an interaction so feedback has a valid target
        import src.api.state as state
        iid = state.interaction_log.log("nl2fix", {"text": "test"}, {"result": "ok"})

        auth_client.post("/api/feedback", json={
            "interaction_id": iid,
            "rating": "positive",
        }, content_type="application/json")
        entries = _read_log_lines(log_dir, "user_actions.log")
        fb_entries = [e for e in entries if e.get("action") == "feedback"]
        assert len(fb_entries) >= 1

    def test_action_log_has_required_fields(self, client, log_dir):
        """Every action entry must have event, action, timestamp, status."""
        client.post("/auth/register", json={
            "username": "fieldtest",
            "email": "field@test.com",
            "password": "secure123",
        }, content_type="application/json")
        entries = _read_log_lines(log_dir, "user_actions.log")
        assert len(entries) >= 1
        for entry in entries:
            assert entry["event"] == "user_action"
            assert "action" in entry
            assert "timestamp" in entry
            assert "status" in entry


# ══════════════════════════════════════════════════════════════════
# 3. LOG INTEGRATION
# ══════════════════════════════════════════════════════════════════

class TestLogIntegration:
    """End-to-end: verify server + action logs work together."""

    def test_login_produces_both_logs(self, client, log_dir):
        """Login should produce entries in both server.log and user_actions.log."""
        client.post("/auth/register", json={
            "username": "bothtest",
            "email": "both@test.com",
            "password": "secure123",
        }, content_type="application/json")
        client.post("/auth/login", json={
            "username": "bothtest",
            "password": "secure123",
        }, content_type="application/json")

        server_entries = _read_log_lines(log_dir, "server.log")
        action_entries = _read_log_lines(log_dir, "user_actions.log")

        # Server log has the POST /auth/login request
        login_req = [e for e in server_entries if e.get("path") == "/auth/login"]
        assert len(login_req) >= 1

        # Action log has the login action
        login_act = [e for e in action_entries if e.get("action") == "login"]
        assert len(login_act) >= 1

    def test_multiple_actions_logged_sequentially(self, auth_client, log_dir):
        """Multiple user actions in sequence must all be logged."""
        auth_client.get("/api/status")
        auth_client.get("/api/versions")
        auth_client.get("/api/examples")

        server_entries = _read_log_lines(log_dir, "server.log")
        paths = [e.get("path") for e in server_entries]

        assert "/api/status" in paths
        assert "/api/versions" in paths
        assert "/api/examples" in paths

    def test_admin_action_logged(self, client, log_dir):
        """Admin actions must be logged with user context."""
        client.post("/auth/login", json={
            "username": "admin",
            "password": "admin123",
        }, content_type="application/json")
        client.get("/auth/users")

        entries = _read_log_lines(log_dir, "user_actions.log")
        admin_entries = [e for e in entries if e.get("action") == "admin_list_users"]
        assert len(admin_entries) >= 1


# ══════════════════════════════════════════════════════════════════
# 4. UNIT TESTS FOR ACTION LOGGER MODULE
# ══════════════════════════════════════════════════════════════════

class TestActionLoggerUnit:
    """Direct tests on the action_logger module."""

    def test_json_formatter_produces_valid_json(self):
        """_JSONFormatter must produce parseable JSON."""
        from src.persistence.action_logger import _JSONFormatter
        import logging

        fmt = _JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="hello", args=(), exc_info=None,
        )
        line = fmt.format(record)
        parsed = json.loads(line)
        assert parsed["message"] == "hello"
        assert parsed["level"] == "INFO"
        assert "timestamp" in parsed

    def test_json_formatter_with_data_extra(self):
        """_JSONFormatter must merge extra data dict into the entry."""
        from src.persistence.action_logger import _JSONFormatter
        import logging

        fmt = _JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="action", args=(), exc_info=None,
        )
        record.data = {"action": "login", "user_id": "U00001"}
        line = fmt.format(record)
        parsed = json.loads(line)
        assert parsed["action"] == "login"
        assert parsed["user_id"] == "U00001"
        assert "message" not in parsed  # data replaces message

    def test_log_user_action_outside_request_context(self, log_dir):
        """log_user_action must work outside Flask request context."""
        import logging
        for name in ("fixprotogpt.server", "fixprotogpt.actions"):
            lgr = logging.getLogger(name)
            lgr.handlers.clear()

        with patch("src.persistence.action_logger.LOG_DIR", log_dir):
            from src.persistence.action_logger import log_user_action
            log_user_action(
                "test_action",
                user_id="U00001",
                username="tester",
                detail={"key": "value"},
            )

        entries = _read_log_lines(log_dir, "user_actions.log")
        assert len(entries) >= 1
        assert entries[-1]["action"] == "test_action"
        assert entries[-1]["user_id"] == "U00001"
        assert entries[-1]["ip"] is None  # no request context


# ══════════════════════════════════════════════════════════════════
# 5. DEBUG LOGGING
# ══════════════════════════════════════════════════════════════════

class TestDebugLogging:
    """Verify the env-var-configurable debug log system."""

    def test_debug_log_not_written_at_info_level(self, log_dir):
        """At INFO level (default), debug.log should not receive entries."""
        import logging
        for name in ("fixprotogpt.server", "fixprotogpt.actions", "fixprotogpt.debug"):
            lgr = logging.getLogger(name)
            lgr.handlers.clear()

        with (
            patch("src.persistence.action_logger.LOG_DIR", log_dir),
            patch.dict("os.environ", {"FIXPROTOGPT_LOG_LEVEL": "INFO"}),
        ):
            from src.persistence.action_logger import log_debug
            log_debug("test_info_event", detail={"key": "value"})

        debug_file = log_dir / "debug.log"
        # debug.log either doesn't exist or has no matching DEBUG entries
        if debug_file.exists():
            entries = _read_log_lines(log_dir, "debug.log")
            debug_entries = [e for e in entries if e.get("event") == "test_info_event"]
            assert len(debug_entries) == 0

    def test_debug_log_written_at_debug_level(self, log_dir):
        """When FIXPROTOGPT_LOG_LEVEL=DEBUG, log_debug writes to debug.log."""
        import logging
        for name in ("fixprotogpt.server", "fixprotogpt.actions", "fixprotogpt.debug"):
            lgr = logging.getLogger(name)
            lgr.handlers.clear()

        with (
            patch("src.persistence.action_logger.LOG_DIR", log_dir),
            patch.dict("os.environ", {"FIXPROTOGPT_LOG_LEVEL": "DEBUG"}),
        ):
            from src.persistence.action_logger import log_debug
            log_debug("test_debug_event", detail={"foo": "bar"})

        entries = _read_log_lines(log_dir, "debug.log")
        debug_entries = [e for e in entries if e.get("event") == "test_debug_event"]
        assert len(debug_entries) >= 1
        assert debug_entries[0]["detail"]["foo"] == "bar"

    def test_get_log_level_defaults_to_env_config(self):
        """_get_log_level falls back to env_config.LOG_LEVEL when env var is unset."""
        import logging
        from src.config.env_config import env as _env_cfg
        with patch.dict("os.environ", {}, clear=False):
            # Remove FIXPROTOGPT_LOG_LEVEL if it exists
            import os
            os.environ.pop("FIXPROTOGPT_LOG_LEVEL", None)
            from src.persistence.action_logger import _get_log_level
            expected = getattr(logging, _env_cfg.LOG_LEVEL, logging.INFO)
            assert _get_log_level() == expected

    def test_get_log_level_respects_env(self):
        """_get_log_level returns DEBUG when env var is set to DEBUG."""
        import logging
        with patch.dict("os.environ", {"FIXPROTOGPT_LOG_LEVEL": "DEBUG"}):
            from src.persistence.action_logger import _get_log_level
            assert _get_log_level() == logging.DEBUG

    def test_get_log_level_invalid_falls_back_to_env_config(self):
        """_get_log_level falls back to env_config for invalid env var values."""
        import logging
        from src.config.env_config import env as _env_cfg
        with patch.dict("os.environ", {"FIXPROTOGPT_LOG_LEVEL": "VERBOSE"}):
            from src.persistence.action_logger import _get_log_level
            expected = getattr(logging, _env_cfg.LOG_LEVEL, logging.INFO)
            assert _get_log_level() == expected

    def test_debug_logger_name_constant(self):
        """DEBUG_LOGGER_NAME must be 'fixprotogpt.debug'."""
        from src.persistence.action_logger import DEBUG_LOGGER_NAME
        assert DEBUG_LOGGER_NAME == "fixprotogpt.debug"

    def test_debug_log_entry_structure(self, log_dir):
        """Debug log entries must have event and timestamp fields."""
        import logging
        for name in ("fixprotogpt.server", "fixprotogpt.actions", "fixprotogpt.debug"):
            lgr = logging.getLogger(name)
            lgr.handlers.clear()

        with (
            patch("src.persistence.action_logger.LOG_DIR", log_dir),
            patch.dict("os.environ", {"FIXPROTOGPT_LOG_LEVEL": "DEBUG"}),
        ):
            from src.persistence.action_logger import log_debug
            log_debug("struct_check", detail={"x": 1})

        entries = _read_log_lines(log_dir, "debug.log")
        struct_entries = [e for e in entries if e.get("event") == "struct_check"]
        assert len(struct_entries) >= 1
        entry = struct_entries[0]
        assert "timestamp" in entry
        assert entry["level"] == "DEBUG"
        assert entry["logger"] == "fixprotogpt.debug"

    def test_log_debug_without_detail(self, log_dir):
        """log_debug should work when detail is omitted."""
        import logging
        for name in ("fixprotogpt.server", "fixprotogpt.actions", "fixprotogpt.debug"):
            lgr = logging.getLogger(name)
            lgr.handlers.clear()

        with (
            patch("src.persistence.action_logger.LOG_DIR", log_dir),
            patch.dict("os.environ", {"FIXPROTOGPT_LOG_LEVEL": "DEBUG"}),
        ):
            from src.persistence.action_logger import log_debug
            log_debug("simple_event")

        entries = _read_log_lines(log_dir, "debug.log")
        simple_entries = [e for e in entries if e.get("event") == "simple_event"]
        assert len(simple_entries) >= 1
        assert "detail" not in simple_entries[0]

    def test_server_log_feeds_debug_at_debug_level(self, log_dir):
        """With DEBUG level, server events should also appear in debug.log."""
        import logging
        for name in ("fixprotogpt.server", "fixprotogpt.actions", "fixprotogpt.debug"):
            lgr = logging.getLogger(name)
            lgr.handlers.clear()

        with (
            patch("src.persistence.action_logger.LOG_DIR", log_dir),
            patch.dict("os.environ", {"FIXPROTOGPT_LOG_LEVEL": "DEBUG"}),
        ):
            from src.persistence.action_logger import _get_server_logger
            logger = _get_server_logger()
            # The server logger should have 2 handlers at DEBUG level
            assert len(logger.handlers) == 2
