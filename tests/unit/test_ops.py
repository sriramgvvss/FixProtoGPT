"""
Tests for the /ops/* operational endpoints.

Covers:
- /ops/health — always available, returns env + model + DB status
- /ops/config — dev/qa only, redacted secret key
- /ops/logs   — dev/qa only, tails log files
- Access control: /ops/config and /ops/logs return 404 in prod/preprod
- Env-aware log directory (logs/<env_name>/)
- Env-aware DB directory (db/<env_name>/)
- Directory scaffolding (ensure_all_env_dirs)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────

def _make_env_config(env_name: str = "dev", **overrides):
    """Create an EnvConfig with sensible defaults and overrides."""
    from src.config.env_config import EnvConfig
    defaults = {
        "ENV_NAME": env_name,
        "DEBUG": True,
        "SECRET_KEY": "test-secret",
        "SESSION_COOKIE_SECURE": False,
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        "SESSION_LIFETIME": 86400,
        "SEED_DEMO_USERS": True,
        "CORS_ORIGINS": [],
        "LOG_LEVEL": "DEBUG",
        "HOST": "0.0.0.0",
        "PORT": 8080,
        "CSP_POLICY": "default-src 'self'",
        "HSTS_ENABLED": False,
        "HSTS_MAX_AGE": 0,
    }
    defaults.update(overrides)
    return EnvConfig(**defaults)


@pytest.fixture
def dev_client(tmp_path):
    """Flask test client in dev environment."""
    cfg = _make_env_config("dev")
    with patch("src.config.env_config.env", cfg), \
         patch("src.api.app.env_config", cfg), \
         patch("src.api.routes.ops.env_config", cfg), \
         patch("src.api.routes.ops.LOG_DIR", tmp_path):
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
def qa_client(tmp_path):
    """Flask test client in qa environment."""
    cfg = _make_env_config("qa")
    with patch("src.config.env_config.env", cfg), \
         patch("src.api.app.env_config", cfg), \
         patch("src.api.routes.ops.env_config", cfg), \
         patch("src.api.routes.ops.LOG_DIR", tmp_path):
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
def prod_client(tmp_path):
    """Flask test client in prod environment."""
    cfg = _make_env_config("prod", DEBUG=False, SECRET_KEY="real-prod-key",
                           SESSION_COOKIE_SECURE=True, HSTS_ENABLED=True,
                           HSTS_MAX_AGE=31536000, LOG_LEVEL="WARNING",
                           SEED_DEMO_USERS=False)
    with patch("src.config.env_config.env", cfg), \
         patch("src.api.app.env_config", cfg), \
         patch("src.api.routes.ops.env_config", cfg), \
         patch("src.api.routes.ops.LOG_DIR", tmp_path):
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


# ══════════════════════════════════════════════════════════════════
# 1. /ops/health
# ══════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    """Verify /ops/health is always available and returns correct info."""

    def test_health_returns_200_in_dev(self, dev_client):
        resp = dev_client.get("/ops/health")
        assert resp.status_code == 200

    def test_health_returns_200_in_prod(self, prod_client):
        resp = prod_client.get("/ops/health")
        assert resp.status_code == 200

    def test_health_returns_200_in_qa(self, qa_client):
        resp = qa_client.get("/ops/health")
        assert resp.status_code == 200

    def test_health_contains_env_name(self, dev_client):
        data = dev_client.get("/ops/health").get_json()
        assert data["env"] == "dev"

    def test_health_contains_uptime(self, dev_client):
        data = dev_client.get("/ops/health").get_json()
        assert "uptime_s" in data
        assert isinstance(data["uptime_s"], (int, float))

    def test_health_contains_model_status(self, dev_client):
        data = dev_client.get("/ops/health").get_json()
        assert "model" in data
        assert "status" in data["model"]

    def test_health_contains_db_status(self, dev_client):
        data = dev_client.get("/ops/health").get_json()
        assert "db" in data
        assert data["db"]["status"] == "ok"
        assert "users" in data["db"]
        assert "interactions" in data["db"]

    def test_health_contains_started_at(self, dev_client):
        data = dev_client.get("/ops/health").get_json()
        assert "started_at" in data

    def test_health_contains_log_level(self, dev_client):
        data = dev_client.get("/ops/health").get_json()
        assert data["log_level"] == "DEBUG"

    def test_health_prod_shows_warning_log_level(self, prod_client):
        data = prod_client.get("/ops/health").get_json()
        assert data["log_level"] == "WARNING"


# ══════════════════════════════════════════════════════════════════
# 2. /ops/config
# ══════════════════════════════════════════════════════════════════

class TestConfigEndpoint:
    """Verify /ops/config is only available in dev/qa and redacts secrets."""

    def test_config_available_in_dev(self, dev_client):
        resp = dev_client.get("/ops/config")
        assert resp.status_code == 200

    def test_config_available_in_qa(self, qa_client):
        resp = qa_client.get("/ops/config")
        assert resp.status_code == 200

    def test_config_returns_404_in_prod(self, prod_client):
        resp = prod_client.get("/ops/config")
        assert resp.status_code == 404

    def test_config_redacts_secret_key(self, dev_client):
        data = dev_client.get("/ops/config").get_json()
        assert data["secret_key"] == "***"

    def test_config_shows_env_name(self, dev_client):
        data = dev_client.get("/ops/config").get_json()
        assert data["env"] == "dev"

    def test_config_includes_log_dir(self, dev_client):
        data = dev_client.get("/ops/config").get_json()
        assert "log_dir" in data

    def test_config_includes_all_fields(self, dev_client):
        data = dev_client.get("/ops/config").get_json()
        expected_keys = {
            "env", "debug", "secret_key", "session_cookie_secure",
            "session_cookie_httponly", "session_cookie_samesite",
            "session_lifetime", "seed_demo_users", "cors_origins",
            "log_level", "host", "port", "csp_policy", "hsts_enabled",
            "hsts_max_age", "log_dir",
        }
        assert expected_keys.issubset(set(data.keys()))


# ══════════════════════════════════════════════════════════════════
# 3. /ops/logs
# ══════════════════════════════════════════════════════════════════

class TestLogsEndpoint:
    """Verify /ops/logs is only available in dev/qa and tails files."""

    def test_logs_available_in_dev(self, dev_client):
        resp = dev_client.get("/ops/logs")
        assert resp.status_code == 200

    def test_logs_available_in_qa(self, qa_client):
        resp = qa_client.get("/ops/logs")
        assert resp.status_code == 200

    def test_logs_returns_404_in_prod(self, prod_client):
        resp = prod_client.get("/ops/logs")
        assert resp.status_code == 404

    def test_logs_returns_all_streams(self, dev_client):
        data = dev_client.get("/ops/logs").get_json()
        assert "server" in data
        assert "user_actions" in data
        assert "debug" in data

    def test_logs_returns_empty_when_no_files(self, dev_client):
        data = dev_client.get("/ops/logs").get_json()
        assert data["server"] == []
        assert data["user_actions"] == []
        assert data["debug"] == []

    def test_logs_respects_lines_param(self, dev_client, tmp_path):
        """Write a small log file and verify 'lines' param works."""
        log_file = tmp_path / "server.log"
        log_file.write_text("\n".join(f"line {i}" for i in range(20)))
        data = dev_client.get("/ops/logs?lines=5").get_json()
        assert len(data["server"]) == 5
        assert data["server"][-1] == "line 19"

    def test_logs_limits_to_500(self, dev_client, tmp_path):
        """Passing lines > 500 should be capped to 500."""
        log_file = tmp_path / "server.log"
        log_file.write_text("\n".join(f"line {i}" for i in range(600)))
        data = dev_client.get("/ops/logs?lines=999").get_json()
        assert len(data["server"]) == 500

    def test_logs_includes_env_name(self, dev_client):
        data = dev_client.get("/ops/logs").get_json()
        assert data["env"] == "dev"


# ══════════════════════════════════════════════════════════════════
# 4. ENV-AWARE LOG DIRECTORY
# ══════════════════════════════════════════════════════════════════

class TestEnvAwareLogDir:
    """Verify that action_logger writes to logs/<env_name>/."""

    def test_log_dir_includes_env_name(self):
        from src.persistence.action_logger import LOG_DIR
        # Should end with logs/<something> where <something> is an env name
        assert LOG_DIR.parent.name == "logs"
        assert LOG_DIR.name in ("dev", "qa", "preprod", "prod")

    def test_ops_log_dir_matches_action_logger(self):
        from src.api.routes.ops import LOG_DIR as ops_log_dir
        from src.persistence.action_logger import LOG_DIR as action_log_dir
        assert ops_log_dir == action_log_dir


# ══════════════════════════════════════════════════════════════════
# 5. ENV-AWARE DB DIRECTORIES
# ══════════════════════════════════════════════════════════════════

class TestEnvAwareDbDir:
    """Verify that UserManager and InteractionLogger use env-specific DB paths."""

    def test_user_manager_default_db_dir_includes_env(self):
        from src.persistence.user_manager import _DB_DIR
        assert _DB_DIR.parent.name == "db"
        assert _DB_DIR.name in ("dev", "qa", "preprod", "prod")

    def test_interaction_logger_default_dir_includes_env(self):
        from src.persistence.interaction_logger import _LOG_DIR
        assert _LOG_DIR.parent.name == "db"
        assert _LOG_DIR.name in ("dev", "qa", "preprod", "prod")

    def test_user_manager_db_separate_per_env(self, tmp_path):
        """Two UserManagers with different db_dirs write to different files."""
        from src.persistence.user_manager import UserManager

        dev_dir = tmp_path / "dev"
        qa_dir = tmp_path / "qa"
        mgr_dev = UserManager(db_dir=dev_dir)
        mgr_qa = UserManager(db_dir=qa_dir)

        assert mgr_dev.db_path != mgr_qa.db_path
        assert mgr_dev.db_path.parent.name == "dev"
        assert mgr_qa.db_path.parent.name == "qa"

    def test_user_manager_uses_users_db_file(self, tmp_path):
        """UserManager should write to users.db, not interactions.db."""
        from src.persistence.user_manager import UserManager
        mgr = UserManager(db_dir=tmp_path)
        assert mgr.db_path.name == "users.db"
        assert mgr.db_path.exists()

    def test_interaction_logger_uses_interactions_db_file(self, tmp_path):
        """InteractionLogger should write to interactions.db."""
        from src.persistence.interaction_logger import InteractionLogger
        il = InteractionLogger(log_dir=tmp_path)
        assert il.db_path.name == "interactions.db"
        assert il.db_path.exists()

    def test_user_and_interaction_dbs_are_separate_files(self, tmp_path):
        """UserManager and InteractionLogger must write to different DB files."""
        from src.persistence.user_manager import UserManager
        from src.persistence.interaction_logger import InteractionLogger
        mgr = UserManager(db_dir=tmp_path)
        il = InteractionLogger(log_dir=tmp_path)
        assert mgr.db_path != il.db_path
        assert mgr.db_path.name == "users.db"
        assert il.db_path.name == "interactions.db"

    def test_interaction_logger_db_separate_per_env(self, tmp_path):
        """Two InteractionLoggers with different dirs write to different files."""
        from src.persistence.interaction_logger import InteractionLogger

        dev_dir = tmp_path / "dev"
        qa_dir = tmp_path / "qa"
        il_dev = InteractionLogger(log_dir=dev_dir)
        il_qa = InteractionLogger(log_dir=qa_dir)

        assert il_dev.db_path != il_qa.db_path
        assert il_dev.db_path.parent.name == "dev"
        assert il_qa.db_path.parent.name == "qa"

    def test_health_endpoint_shows_user_db_path(self, dev_client):
        data = dev_client.get("/ops/health").get_json()
        assert "db_path" in data["db"]["users"]
        assert data["db"]["users"]["db_path"].endswith("users.db")

    def test_health_endpoint_shows_interactions_db_path(self, dev_client):
        data = dev_client.get("/ops/health").get_json()
        assert "db_path" in data["db"]["interactions"]
        assert data["db"]["interactions"]["db_path"].endswith("interactions.db")

    def test_health_db_paths_are_in_same_env_dir(self, dev_client):
        data = dev_client.get("/ops/health").get_json()
        from pathlib import Path
        user_dir = Path(data["db"]["users"]["db_path"]).parent
        int_dir = Path(data["db"]["interactions"]["db_path"]).parent
        assert user_dir == int_dir

    def test_legacy_migration_copies_users(self, tmp_path):
        """UserManager migrates data from legacy interactions.db to users.db."""
        import sqlite3
        from src.persistence.user_manager import UserManager, _USER_SCHEMA

        # Create a legacy interactions.db with user data
        legacy_db = tmp_path / "interactions.db"
        conn = sqlite3.connect(str(legacy_db))
        conn.executescript(_USER_SCHEMA)
        conn.execute(
            "INSERT INTO users (id, username, email, password, role, created_at) "
            "VALUES ('U00001', 'legacy', 'l@l.com', 'hash', 'user', '2026-01-01T00:00:00Z')"
        )
        conn.commit()
        conn.close()

        # Create UserManager — should auto-migrate
        mgr = UserManager(db_dir=tmp_path)
        assert mgr.db_path.name == "users.db"
        user = mgr.get_user_by_username("legacy")
        assert user is not None
        assert user["email"] == "l@l.com"

    def test_no_migration_when_users_db_exists(self, tmp_path):
        """If users.db already exists, skip migration from interactions.db."""
        import sqlite3
        from src.persistence.user_manager import UserManager, _USER_SCHEMA

        # Create both legacy and new DB
        legacy_db = tmp_path / "interactions.db"
        conn = sqlite3.connect(str(legacy_db))
        conn.executescript(_USER_SCHEMA)
        conn.execute(
            "INSERT INTO users (id, username, email, password, role, created_at) "
            "VALUES ('U00001', 'old_user', 'o@o.com', 'hash', 'user', '2026-01-01T00:00:00Z')"
        )
        conn.commit()
        conn.close()

        # Pre-create users.db so migration is skipped
        mgr_first = UserManager(db_dir=tmp_path)
        mgr_first.register("new_user", "n@n.com", "password123")

        # Create second manager — should NOT re-migrate
        mgr_second = UserManager(db_dir=tmp_path)
        assert mgr_second.get_user_by_username("new_user") is not None


# ══════════════════════════════════════════════════════════════════
# 6. DIRECTORY SCAFFOLDING (ensure_all_env_dirs)
# ══════════════════════════════════════════════════════════════════

class TestEnsureAllEnvDirs:
    """Verify that ensure_all_env_dirs creates all env directories."""

    def test_all_log_dirs_exist(self):
        from src.config.env_config import VALID_ENVS
        from src.utils import paths

        for env_name in VALID_ENVS:
            log_dir = paths.PROJECT_ROOT / "logs" / env_name
            assert log_dir.is_dir(), f"logs/{env_name}/ missing"

    def test_all_db_dirs_exist(self):
        from src.config.env_config import VALID_ENVS
        from src.utils import paths

        for env_name in VALID_ENVS:
            db_dir = paths.PROJECT_ROOT / "db" / env_name
            assert db_dir.is_dir(), f"db/{env_name}/ missing"

    def test_all_log_dirs_have_gitkeep(self):
        from src.config.env_config import VALID_ENVS
        from src.utils import paths

        for env_name in VALID_ENVS:
            gitkeep = paths.PROJECT_ROOT / "logs" / env_name / ".gitkeep"
            assert gitkeep.is_file(), f"logs/{env_name}/.gitkeep missing"

    def test_all_db_dirs_have_gitkeep(self):
        from src.config.env_config import VALID_ENVS
        from src.utils import paths

        for env_name in VALID_ENVS:
            gitkeep = paths.PROJECT_ROOT / "db" / env_name / ".gitkeep"
            assert gitkeep.is_file(), f"db/{env_name}/.gitkeep missing"

    def test_ensure_all_env_dirs_is_idempotent(self, tmp_path):
        """Calling ensure_all_env_dirs twice does not fail or duplicate."""
        from src.config.env_config import ensure_all_env_dirs
        # First call already ran at import time; second should be safe
        ensure_all_env_dirs()  # no error

    def test_no_root_level_log_files_remain(self):
        """Stale root-level log files should have been migrated to dev/."""
        from src.utils import paths

        root_logs = paths.PROJECT_ROOT / "logs"
        stale = [
            f.name for f in root_logs.iterdir()
            if f.is_file() and f.suffix in (".log",) and f.name != ".gitkeep"
        ]
        assert stale == [], f"Stale root-level log files: {stale}"

    def test_no_root_level_db_files_remain(self):
        """Stale root-level .db files should not exist in db/."""
        from src.utils import paths

        root_db = paths.PROJECT_ROOT / "db"
        stale = [
            f.name for f in root_db.iterdir()
            if f.is_file() and f.suffix == ".db"
        ]
        assert stale == [], f"Stale root-level DB files: {stale}"
