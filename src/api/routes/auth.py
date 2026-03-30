"""
Module: src.api.routes.auth
============================

Authentication Blueprint for FixProtoGPT.

Routes: login, register, logout, me (session info), admin user-list.
Uses Flask sessions with server-side secret key.

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from functools import wraps
from flask import Blueprint, request, jsonify, session, redirect, url_for, render_template
import logging

import src.api.state as state
from src.persistence.action_logger import log_user_action, log_debug

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


# ── Decorator ─────────────────────────────────────────────────────

def login_required(f):
    """Reject unauthenticated requests.

    * For JSON API calls (Accept or Content-Type contains 'json')
      → respond with 401 JSON.
    * For page requests → redirect to /auth/login.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        """Redirect or 401 if the user is not authenticated."""
        if "user_id" not in session:
            if _wants_json():
                return jsonify({"error": "Authentication required"}), 401
            return redirect(url_for("auth.login_page"))
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    """Only allow users with ``role == 'admin'``."""
    @wraps(f)
    @login_required
    def wrapper(*args, **kwargs):
        """Return 403 if the authenticated user does not have the admin role."""
        user = state.user_manager.get_user_by_id(session["user_id"])
        if not user or user.get("role") != "admin":
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)
    return wrapper


def _wants_json() -> bool:
    """Return ``True`` if the client prefers a JSON response."""
    accept = request.headers.get("Accept", "")
    ct = request.content_type or ""
    return "json" in accept or "json" in ct


# ── Pages ─────────────────────────────────────────────────────────

@auth_bp.route("/login", methods=["GET"])
def login_page():
    """Render the login/register page."""
    if "user_id" in session:
        return redirect(url_for("index"))
    from src.api.routes.assets import SOURCE_PROTECTION_JS
    return render_template("login.html", source_protection=SOURCE_PROTECTION_JS)


# ── API Endpoints ─────────────────────────────────────────────────

@auth_bp.route("/login", methods=["POST"])
def login():
    """Authenticate user and start session."""
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"success": False, "error": "Username and password required"}), 400

    log_debug("auth_attempt", detail={"username": username})
    result = state.user_manager.authenticate(username, password)
    if not result["success"]:
        log_user_action("login", username=username, status="failure",
                        detail={"reason": result.get("error", "unknown")})
        return jsonify(result), 401

    user = result["user"]
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["role"] = user["role"]

    log_user_action("login", user_id=user["id"], username=user["username"],
                    detail={"role": user["role"]})
    return jsonify({"success": True, "user": user})


@auth_bp.route("/register", methods=["POST"])
def register():
    """Register a new user and auto-login."""
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "")
    full_name = data.get("full_name", "").strip()

    log_debug("register_attempt", detail={"username": username, "email": email})
    result = state.user_manager.register(username, email, password, full_name)
    if not result["success"]:
        log_user_action("register", username=username, status="failure",
                        detail={"reason": result.get("error", "unknown")})
        return jsonify(result), 400

    # Auto-login after registration
    user = result["user"]
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["role"] = user.get("role", "user")

    log_user_action("register", user_id=user["id"], username=user["username"],
                    detail={"email": email})
    return jsonify({"success": True, "user": user}), 201


@auth_bp.route("/logout", methods=["POST"])
def logout():
    """End session."""
    user_id = session.get("user_id")
    username = session.get("username", "?")
    log_user_action("logout", user_id=user_id, username=username)
    session.clear()
    return jsonify({"success": True})


@auth_bp.route("/me", methods=["GET"])
@login_required
def me():
    """Return the current user's profile."""
    user = state.user_manager.get_user_by_id(session["user_id"])
    if not user:
        session.clear()
        return jsonify({"error": "User not found"}), 401
    return jsonify({"user": user})


# ── Admin ─────────────────────────────────────────────────────────

@auth_bp.route("/users", methods=["GET"])
@admin_required
def list_users():
    """Admin: list all users."""
    log_user_action("admin_list_users", detail={"action": "list_all_users"})
    users = state.user_manager.list_users()
    return jsonify({"users": users, "count": len(users)})


@auth_bp.route("/token-usage", methods=["GET"])
@login_required
def my_token_usage():
    """Return the current user's token usage stats."""
    user_id = session.get("user_id")
    usage = state.user_manager.get_user_token_usage(user_id)
    return jsonify(usage)


@auth_bp.route("/admin/token-usage", methods=["GET"])
@admin_required
def all_users_token_usage():
    """Admin: token usage breakdown for every user."""
    data = state.user_manager.get_all_users_token_usage()
    return jsonify({"users": data})
