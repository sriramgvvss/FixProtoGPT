"""
Module: src.api.app
====================

FixProtoGPT web interface — Flask application factory.

Registers Flask Blueprints for admin, auth, core, export, and learning routes.

All environment-specific configuration (secret key, debug mode,
cookie security, CORS origins, etc.) is driven by
``src.config.env_config.env`` — see ``config/env/.env.*`` files.

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

import os
from flask import Flask, render_template, session
from flask_cors import CORS
import sys
import logging
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work everywhere
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.api.routes.admin import admin_bp
from src.api.routes.auth import auth_bp, login_required
from src.api.routes.core import core_bp
from src.api.routes.export import export_bp
from src.api.routes.learning import learning_bp
from src.api.routes.assets import assets_bp
from src.api.routes.ops import ops_bp
from src.persistence.action_logger import setup_logging
from src.config.env_config import env as env_config

logging.basicConfig(level=getattr(logging, env_config.LOG_LEVEL, logging.INFO))


def create_app() -> Flask:
    """Application factory — create and configure the Flask app.

    Configuration is read from :data:`src.config.env_config.env`
    which is populated from ``FIXPROTOGPT_ENV`` and the matching
    ``config/env/.env.*`` file.

    Returns:
        Configured ``Flask`` application instance.
    """
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent.parent.parent / "ui" / "templates"),
        static_folder=str(Path(__file__).parent.parent.parent / "ui" / "static"),
    )

    # ── Session / cookie configuration (from env_config) ──────────
    app.secret_key = env_config.SECRET_KEY
    app.config.update(
        SESSION_COOKIE_HTTPONLY=env_config.SESSION_COOKIE_HTTPONLY,
        SESSION_COOKIE_SECURE=env_config.SESSION_COOKIE_SECURE,
        SESSION_COOKIE_SAMESITE=env_config.SESSION_COOKIE_SAMESITE,
        PERMANENT_SESSION_LIFETIME=env_config.SESSION_LIFETIME,
    )

    # ── CORS (restricted origins in non-dev environments) ─────────
    cors_kwargs = {"supports_credentials": True}
    if env_config.CORS_ORIGINS:
        cors_kwargs["origins"] = env_config.CORS_ORIGINS
    CORS(app, **cors_kwargs)

    # Register blueprints
    app.register_blueprint(admin_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(core_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(learning_bp)
    app.register_blueprint(assets_bp)
    app.register_blueprint(ops_bp)

    # Set up structured server + user-action logging
    setup_logging(app)

    # Security headers on every response
    @app.after_request
    def _security_headers(resp):
        """Attach security headers to every response."""
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        resp.headers["X-XSS-Protection"] = "1; mode=block"
        # Content-Security-Policy
        if env_config.CSP_POLICY:
            resp.headers["Content-Security-Policy"] = env_config.CSP_POLICY
        # HTTP Strict Transport Security (only when HTTPS is active)
        if env_config.HSTS_ENABLED:
            resp.headers["Strict-Transport-Security"] = (
                f"max-age={env_config.HSTS_MAX_AGE}; includeSubDomains"
            )
        return resp

    @app.route("/")
    @login_required
    def index():
        """Main page — requires authentication."""
        from src.api.routes.assets import SOURCE_PROTECTION_JS
        return render_template(
            "index.html",
            user=session.get("username", ""),
            source_protection=SOURCE_PROTECTION_JS,
        )

    return app


# Module-level app instance — only created for `flask run` or direct execution.
# Gunicorn / test clients should call create_app() directly.
def _get_app() -> Flask:
    """Lazy app factory for module-level access."""
    return create_app()


def main():
    """Run the development web server."""
    app = create_app()

    print("\n" + "=" * 60)
    print(f"FixProtoGPT Web Interface  [env={env_config.ENV_NAME}]")
    print("=" * 60)
    print(f"\nEnvironment : {env_config.ENV_NAME}")
    print(f"Debug       : {env_config.DEBUG}")
    print(f"Seed demo   : {env_config.SEED_DEMO_USERS}")
    print(f"Cookie secure: {env_config.SESSION_COOKIE_SECURE}")
    print(f"\nStarting server on http://{env_config.HOST}:{env_config.PORT}")
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    app.run(
        host=env_config.HOST,
        port=env_config.PORT,
        debug=env_config.DEBUG,
    )


# For `flask run` CLI and backward compatibility
app = _get_app()


if __name__ == "__main__":
    main()
