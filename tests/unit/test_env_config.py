"""
Tests for the FixProtoGPT environment configuration system.

Covers:
- EnvConfig dataclass and its properties
- Environment variable parsing (bool, list, defaults)
- Per-environment defaults (dev, qa, preprod, prod)
- Secret key enforcement in prod/preprod
- SEED_DEMO_USERS gating in user_manager
- CSP / HSTS header injection via app factory
- CORS origin restriction
"""

import json
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ══════════════════════════════════════════════════════════════════
# 1. HELPER PARSING
# ══════════════════════════════════════════════════════════════════

class TestBoolParser:
    """Verify the _bool() helper handles all edge cases."""

    def test_true_values(self):
        from src.config.env_config import _bool
        for val in ("true", "True", "TRUE", "1", "yes", "Yes", "YES"):
            assert _bool(val) is True, f"Expected True for {val!r}"

    def test_false_values(self):
        from src.config.env_config import _bool
        for val in ("false", "False", "0", "no", "No", "", "anything"):
            assert _bool(val) is False, f"Expected False for {val!r}"

    def test_whitespace_stripped(self):
        from src.config.env_config import _bool
        assert _bool("  true  ") is True
        assert _bool("  false  ") is False


class TestStrListParser:
    """Verify the _str_list() helper for comma-separated values."""

    def test_empty_string(self):
        from src.config.env_config import _str_list
        assert _str_list("") == []

    def test_single_value(self):
        from src.config.env_config import _str_list
        assert _str_list("http://localhost:3000") == ["http://localhost:3000"]

    def test_multiple_values(self):
        from src.config.env_config import _str_list
        result = _str_list("http://a.com, http://b.com , http://c.com")
        assert result == ["http://a.com", "http://b.com", "http://c.com"]

    def test_trailing_commas_ignored(self):
        from src.config.env_config import _str_list
        result = _str_list("http://a.com,,http://b.com,")
        assert result == ["http://a.com", "http://b.com"]


# ══════════════════════════════════════════════════════════════════
# 2. EnvConfig DATACLASS
# ══════════════════════════════════════════════════════════════════

class TestEnvConfigDataclass:
    """Verify EnvConfig properties and immutability."""

    def _make_config(self, **overrides):
        from src.config.env_config import EnvConfig
        defaults = {
            "ENV_NAME": "dev",
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
            "HSTS_MAX_AGE": 31536000,
        }
        defaults.update(overrides)
        return EnvConfig(**defaults)

    def test_is_production_for_prod(self):
        cfg = self._make_config(ENV_NAME="prod")
        assert cfg.is_production is True

    def test_is_production_for_preprod(self):
        cfg = self._make_config(ENV_NAME="preprod")
        assert cfg.is_production is True

    def test_is_production_for_dev(self):
        cfg = self._make_config(ENV_NAME="dev")
        assert cfg.is_production is False

    def test_is_production_for_qa(self):
        cfg = self._make_config(ENV_NAME="qa")
        assert cfg.is_production is False

    def test_is_secure_for_prod(self):
        cfg = self._make_config(ENV_NAME="prod")
        assert cfg.is_secure is True

    def test_is_secure_for_qa(self):
        cfg = self._make_config(ENV_NAME="qa")
        assert cfg.is_secure is False

    def test_is_secure_for_dev(self):
        cfg = self._make_config(ENV_NAME="dev")
        assert cfg.is_secure is False

    def test_frozen_raises_on_mutation(self):
        cfg = self._make_config()
        with pytest.raises(AttributeError):
            cfg.DEBUG = False


# ══════════════════════════════════════════════════════════════════
# 3. _build_config() — DEFAULTS PER ENVIRONMENT
# ══════════════════════════════════════════════════════════════════

class TestBuildConfigDefaults:
    """Verify that _build_config picks correct defaults per environment."""

    def _build(self, env_name, **extra_env):
        """Build config with a specific environment and optional overrides.

        We scrub ALL existing ``FIXPROTOGPT_*`` vars first so that
        values loaded by the module-level singleton (or a previous test)
        don't leak through.
        """
        from src.config.env_config import _build_config

        # Build a clean env: remove every FIXPROTOGPT_* key, then add ours
        clean = {k: v for k, v in os.environ.items() if not k.startswith("FIXPROTOGPT_")}
        clean["FIXPROTOGPT_ENV"] = env_name
        clean["FIXPROTOGPT_SECRET_KEY"] = "test-secret-for-ci"
        clean.update(extra_env)
        with patch.dict(os.environ, clean, clear=True):
            return _build_config()

    def test_dev_debug_is_true(self):
        cfg = self._build("dev")
        assert cfg.DEBUG is True

    def test_dev_seed_demo_is_true(self):
        cfg = self._build("dev")
        assert cfg.SEED_DEMO_USERS is True

    def test_dev_cookie_secure_is_false(self):
        cfg = self._build("dev")
        assert cfg.SESSION_COOKIE_SECURE is False

    def test_qa_seed_demo_is_true(self):
        cfg = self._build("qa")
        assert cfg.SEED_DEMO_USERS is True

    def test_qa_cookie_secure_is_false(self):
        cfg = self._build("qa")
        assert cfg.SESSION_COOKIE_SECURE is False

    def test_prod_debug_is_false(self):
        cfg = self._build("prod")
        assert cfg.DEBUG is False

    def test_prod_seed_demo_is_false(self):
        cfg = self._build("prod")
        assert cfg.SEED_DEMO_USERS is False

    def test_prod_cookie_secure_is_true(self):
        cfg = self._build("prod")
        assert cfg.SESSION_COOKIE_SECURE is True

    def test_prod_hsts_enabled(self):
        cfg = self._build("prod")
        assert cfg.HSTS_ENABLED is True

    def test_preprod_hsts_enabled(self):
        cfg = self._build("preprod")
        assert cfg.HSTS_ENABLED is True

    def test_dev_hsts_disabled(self):
        cfg = self._build("dev")
        assert cfg.HSTS_ENABLED is False

    def test_invalid_env_falls_back_to_dev(self):
        cfg = self._build("invalid_env")
        assert cfg.ENV_NAME == "dev"

    def test_env_name_preserved(self):
        cfg = self._build("qa")
        assert cfg.ENV_NAME == "qa"

    def test_custom_port_override(self):
        cfg = self._build("dev", FIXPROTOGPT_PORT="9090")
        assert cfg.PORT == 9090

    def test_custom_cors_origins(self):
        cfg = self._build("prod", FIXPROTOGPT_CORS_ORIGINS="http://a.com,http://b.com")
        assert cfg.CORS_ORIGINS == ["http://a.com", "http://b.com"]


# ══════════════════════════════════════════════════════════════════
# 4. SECRET KEY ENFORCEMENT
# ══════════════════════════════════════════════════════════════════

class TestSecretKeyEnforcement:
    """Verify that prod/preprod refuse to start with the dev-default secret."""

    def _clean_env(self, overrides):
        """Return a patched env dict with all FIXPROTOGPT_* keys removed
        except those specified in *overrides*."""
        clean = {k: v for k, v in os.environ.items() if not k.startswith("FIXPROTOGPT_")}
        clean.update(overrides)
        return clean

    def test_prod_rejects_dev_secret(self):
        env = self._clean_env({
            "FIXPROTOGPT_ENV": "prod",
            "FIXPROTOGPT_SECRET_KEY": "fixprotogpt-dev-secret-change-in-prod",
        })
        with patch.dict(os.environ, env, clear=True):
            from src.config.env_config import _build_config
            with pytest.raises(SystemExit):
                _build_config()

    def test_preprod_rejects_dev_secret(self):
        env = self._clean_env({
            "FIXPROTOGPT_ENV": "preprod",
            "FIXPROTOGPT_SECRET_KEY": "fixprotogpt-dev-secret-change-in-prod",
        })
        with patch.dict(os.environ, env, clear=True):
            from src.config.env_config import _build_config
            with pytest.raises(SystemExit):
                _build_config()

    def test_prod_accepts_real_secret(self):
        env = self._clean_env({
            "FIXPROTOGPT_ENV": "prod",
            "FIXPROTOGPT_SECRET_KEY": "a-real-production-secret-key-here",
        })
        with patch.dict(os.environ, env, clear=True):
            from src.config.env_config import _build_config
            cfg = _build_config()
            assert cfg.SECRET_KEY == "a-real-production-secret-key-here"

    def test_dev_allows_dev_secret(self):
        env = self._clean_env({
            "FIXPROTOGPT_ENV": "dev",
            "FIXPROTOGPT_SECRET_KEY": "fixprotogpt-dev-secret-change-in-prod",
        })
        with patch.dict(os.environ, env, clear=True):
            from src.config.env_config import _build_config
            cfg = _build_config()
            assert cfg.ENV_NAME == "dev"


# ══════════════════════════════════════════════════════════════════
# 5. SEED_DEMO_USERS GATING IN USER MANAGER
# ══════════════════════════════════════════════════════════════════

class TestSeedDemoUsersGating:
    """Verify user_manager respects SEED_DEMO_USERS config."""

    def test_dev_seeds_demo_users(self, tmp_path):
        """In dev, default demo users should be created."""
        from src.config.env_config import EnvConfig
        mock_cfg = EnvConfig(
            ENV_NAME="dev", DEBUG=True, SECRET_KEY="test",
            SESSION_COOKIE_SECURE=False, SESSION_COOKIE_HTTPONLY=True,
            SESSION_COOKIE_SAMESITE="Lax", SESSION_LIFETIME=86400,
            SEED_DEMO_USERS=True, CORS_ORIGINS=[], LOG_LEVEL="DEBUG",
            HOST="0.0.0.0", PORT=8080, CSP_POLICY="", HSTS_ENABLED=False,
            HSTS_MAX_AGE=0,
        )
        with patch("src.config.env_config.env", mock_cfg):
            from src.persistence.user_manager import UserManager
            mgr = UserManager(db_dir=tmp_path)
            assert mgr.get_user_count() == 3  # admin, trader, developer

    def test_prod_skips_demo_users(self, tmp_path):
        """In prod, no demo users should be created."""
        from src.config.env_config import EnvConfig
        mock_cfg = EnvConfig(
            ENV_NAME="prod", DEBUG=False, SECRET_KEY="real-secret",
            SESSION_COOKIE_SECURE=True, SESSION_COOKIE_HTTPONLY=True,
            SESSION_COOKIE_SAMESITE="Lax", SESSION_LIFETIME=28800,
            SEED_DEMO_USERS=False, CORS_ORIGINS=[], LOG_LEVEL="WARNING",
            HOST="0.0.0.0", PORT=8080, CSP_POLICY="", HSTS_ENABLED=True,
            HSTS_MAX_AGE=31536000,
        )
        with patch("src.config.env_config.env", mock_cfg):
            from src.persistence.user_manager import UserManager
            mgr = UserManager(db_dir=tmp_path)
            assert mgr.get_user_count() == 0

    def test_qa_seeds_demo_users(self, tmp_path):
        """In qa, demo users should be created (same as dev)."""
        from src.config.env_config import EnvConfig
        mock_cfg = EnvConfig(
            ENV_NAME="qa", DEBUG=True, SECRET_KEY="qa-secret",
            SESSION_COOKIE_SECURE=False, SESSION_COOKIE_HTTPONLY=True,
            SESSION_COOKIE_SAMESITE="Lax", SESSION_LIFETIME=86400,
            SEED_DEMO_USERS=True, CORS_ORIGINS=[], LOG_LEVEL="DEBUG",
            HOST="0.0.0.0", PORT=8080, CSP_POLICY="", HSTS_ENABLED=False,
            HSTS_MAX_AGE=0,
        )
        with patch("src.config.env_config.env", mock_cfg):
            from src.persistence.user_manager import UserManager
            mgr = UserManager(db_dir=tmp_path)
            assert mgr.get_user_count() == 3  # admin, trader, developer


# ══════════════════════════════════════════════════════════════════
# 6. APP FACTORY — SECURITY HEADERS
# ══════════════════════════════════════════════════════════════════

class TestAppSecurityHeaders:
    """Verify CSP and HSTS headers from env config in Flask responses."""

    @pytest.fixture
    def prod_client(self, tmp_path):
        """Flask test client configured with prod-like settings."""
        from src.config.env_config import EnvConfig
        mock_cfg = EnvConfig(
            ENV_NAME="prod", DEBUG=False, SECRET_KEY="secure-prod-secret",
            SESSION_COOKIE_SECURE=True, SESSION_COOKIE_HTTPONLY=True,
            SESSION_COOKIE_SAMESITE="Lax", SESSION_LIFETIME=28800,
            SEED_DEMO_USERS=True, CORS_ORIGINS=["https://app.example.com"],
            LOG_LEVEL="WARNING", HOST="0.0.0.0", PORT=8080,
            CSP_POLICY="default-src 'self'; script-src 'self'",
            HSTS_ENABLED=True, HSTS_MAX_AGE=31536000,
        )
        with patch("src.config.env_config.env", mock_cfg), \
             patch("src.api.app.env_config", mock_cfg):
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
    def dev_client(self, tmp_path):
        """Flask test client configured with dev settings."""
        from src.config.env_config import EnvConfig
        mock_cfg = EnvConfig(
            ENV_NAME="dev", DEBUG=True, SECRET_KEY="dev-secret",
            SESSION_COOKIE_SECURE=False, SESSION_COOKIE_HTTPONLY=True,
            SESSION_COOKIE_SAMESITE="Lax", SESSION_LIFETIME=86400,
            SEED_DEMO_USERS=True, CORS_ORIGINS=[],
            LOG_LEVEL="DEBUG", HOST="0.0.0.0", PORT=8080,
            CSP_POLICY="default-src 'self'; script-src 'self' 'unsafe-inline'",
            HSTS_ENABLED=False, HSTS_MAX_AGE=0,
        )
        with patch("src.config.env_config.env", mock_cfg), \
             patch("src.api.app.env_config", mock_cfg):
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

    def test_csp_header_present_in_prod(self, prod_client):
        """Prod responses must include Content-Security-Policy."""
        resp = prod_client.get("/auth/login")
        assert "Content-Security-Policy" in resp.headers
        assert "default-src 'self'" in resp.headers["Content-Security-Policy"]

    def test_hsts_header_present_in_prod(self, prod_client):
        """Prod responses must include Strict-Transport-Security."""
        resp = prod_client.get("/auth/login")
        assert "Strict-Transport-Security" in resp.headers
        assert "max-age=31536000" in resp.headers["Strict-Transport-Security"]

    def test_hsts_header_absent_in_dev(self, dev_client):
        """Dev responses should NOT include HSTS."""
        resp = dev_client.get("/auth/login")
        assert "Strict-Transport-Security" not in resp.headers

    def test_csp_header_present_in_dev(self, dev_client):
        """Dev responses should still have CSP (but more permissive)."""
        resp = dev_client.get("/auth/login")
        assert "Content-Security-Policy" in resp.headers
        assert "'unsafe-inline'" in resp.headers["Content-Security-Policy"]

    def test_x_frame_options_always_present(self, prod_client):
        resp = prod_client.get("/auth/login")
        assert resp.headers["X-Frame-Options"] == "DENY"

    def test_x_content_type_options_always_present(self, dev_client):
        resp = dev_client.get("/auth/login")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"


# ══════════════════════════════════════════════════════════════════
# 7. ENV FILE EXISTENCE
# ══════════════════════════════════════════════════════════════════

class TestEnvFilesExist:
    """Verify that all expected env files are checked in."""

    _ENV_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "env"

    @pytest.mark.parametrize("env_name", ["dev", "qa", "preprod", "prod"])
    def test_env_file_exists(self, env_name):
        env_file = self._ENV_DIR / f".env.{env_name}"
        assert env_file.is_file(), f"Missing {env_file}"

    def test_example_file_exists(self):
        example = self._ENV_DIR.parent / ".env.example"
        assert example.is_file()


# ══════════════════════════════════════════════════════════════════
# 8. VALID_ENVS CONSTANT
# ══════════════════════════════════════════════════════════════════

class TestValidEnvs:
    """Verify the set of accepted environment names."""

    def test_valid_envs_contains_all_expected(self):
        from src.config.env_config import VALID_ENVS
        assert set(VALID_ENVS) == {"dev", "qa", "preprod", "prod"}


# ══════════════════════════════════════════════════════════════════
# 9. ensure_all_env_dirs FUNCTION
# ══════════════════════════════════════════════════════════════════

class TestEnsureAllEnvDirs:
    """Verify the directory scaffolding function."""

    def test_function_is_exported(self):
        from src.config.env_config import ensure_all_env_dirs
        assert callable(ensure_all_env_dirs)

    def test_creates_dirs_in_fresh_tree(self, tmp_path):
        """When pointed at a fresh tree, creates all env subdirs."""
        from src.config.env_config import VALID_ENVS
        import src.config.env_config as mod

        original_root = mod._PROJECT_ROOT
        try:
            mod._PROJECT_ROOT = tmp_path
            mod.ensure_all_env_dirs()

            for env_name in VALID_ENVS:
                assert (tmp_path / "logs" / env_name).is_dir()
                assert (tmp_path / "db" / env_name).is_dir()
                assert (tmp_path / "logs" / env_name / ".gitkeep").is_file()
                assert (tmp_path / "db" / env_name / ".gitkeep").is_file()
        finally:
            mod._PROJECT_ROOT = original_root
