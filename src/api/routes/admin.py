"""
Module: src.api.routes.admin
==============================

Admin API endpoints for managing spec ingestion, client overlays,
and client-specific fine-tuning.

All endpoints live under ``/admin/`` and require admin role
(enforced via ``@admin_required`` decorator).

Available only in **dev / qa** environments by default.

Endpoints
---------
* ``GET  /admin/clients``           — list all client overlays
* ``POST /admin/clients``           — create a new client
* ``GET  /admin/clients/<id>``      — client stats
* ``DELETE /admin/clients/<id>``    — remove client + data
* ``POST /admin/clients/<id>/upload`` — upload spec files
* ``POST /admin/clients/<id>/ingest`` — parse uploaded specs
* ``POST /admin/clients/<id>/train``  — trigger client fine-tune
* ``POST /admin/specs/upload``      — upload base spec file
* ``POST /admin/specs/ingest``      — re-ingest all base specs
* ``GET  /admin/specs/canonical``   — view canonical records

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from __future__ import annotations

import logging
import threading
from functools import wraps
from pathlib import Path
from typing import Optional

from flask import Blueprint, abort, jsonify, render_template, request, session

from src.config.env_config import env as env_config
from src.utils import paths

logger = logging.getLogger(__name__)

admin_bp = Blueprint("admin", __name__)

# Track background fine-tuning threads
_active_trains: dict[str, threading.Thread] = {}


# ── Auth decorator ────────────────────────────────────────────────

def _valid_admin_key(req) -> bool:
    """Check if the request carries a valid ``X-Admin-Key`` header.

    Only accepted from ``127.0.0.1`` / ``::1`` (localhost) to prevent
    remote callers from bypassing session auth.
    """
    key = req.headers.get("X-Admin-Key", "")
    if not key:
        return False
    remote = req.remote_addr or ""
    if remote not in ("127.0.0.1", "::1"):
        return False
    return key == env_config.SECRET_KEY


def admin_required(f):
    """Require admin role.  Returns 403 if not admin.

    Also accepts a valid ``X-Admin-Key`` header from localhost
    (used by the control-panel scripts).
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        """Gate access: accept admin key, env-based role check, or 401/403."""
        # Internal key from localhost (control-panel / scripts)
        if _valid_admin_key(request):
            return f(*args, **kwargs)
        # In dev/qa allow all authenticated users; in prod check role
        if env_config.ENV_NAME in ("preprod", "prod"):
            role = session.get("role", "")
            if role != "admin":
                return jsonify({"error": "Admin access required"}), 403
        elif not session.get("username"):
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


# ── Admin page ────────────────────────────────────────────────────

@admin_bp.route("/admin")
@admin_required
def admin_page():
    """Render the admin panel UI."""
    return render_template("admin.html")


# ══════════════════════════════════════════════════════════════════
#  CLIENT OVERLAY ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@admin_bp.route("/admin/clients", methods=["GET"])
@admin_required
def list_clients():
    """List all client IDs with overlay directories."""
    from src.data.ingest.client_overlay import (
        list_clients as _list_clients,
        get_client_stats,
    )
    clients = _list_clients()
    return jsonify({
        "clients": [get_client_stats(c) for c in clients],
        "count": len(clients),
    })


@admin_bp.route("/admin/clients", methods=["POST"])
@admin_required
def create_client():
    """Create a new client overlay.

    JSON body: ``{"client_id": "acme"}``
    """
    from src.data.ingest.client_overlay import create_client as _create

    data = request.get_json(silent=True) or {}
    client_id = data.get("client_id", "").strip()
    if not client_id or not client_id.isalnum():
        return jsonify({
            "error": "client_id must be non-empty and alphanumeric",
        }), 400

    overlay_dir = _create(client_id)
    return jsonify({
        "message": f"Client {client_id!r} created",
        "overlay_dir": str(overlay_dir),
    }), 201


@admin_bp.route("/admin/clients/<client_id>", methods=["GET"])
@admin_required
def client_detail(client_id: str):
    """Return stats for a specific client."""
    from src.data.ingest.client_overlay import get_client_stats
    return jsonify(get_client_stats(client_id))


@admin_bp.route("/admin/clients/<client_id>", methods=["DELETE"])
@admin_required
def delete_client(client_id: str):
    """Delete a client overlay and all associated data."""
    from src.data.ingest.client_overlay import delete_client as _delete
    deleted = _delete(client_id)
    if not deleted:
        return jsonify({"error": f"Client {client_id!r} not found"}), 404
    return jsonify({"message": f"Client {client_id!r} deleted"})


@admin_bp.route("/admin/clients/<client_id>/upload", methods=["POST"])
@admin_required
def upload_client_spec(client_id: str):
    """Upload one or more spec files for a client.

    Expects ``multipart/form-data`` with one or more ``file`` fields.
    """
    from src.data.ingest.client_overlay import save_uploaded_file, create_client

    # Ensure client exists
    create_client(client_id)

    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    saved = []
    for f in files:
        if not f.filename:
            continue
        path = save_uploaded_file(client_id, f.filename, f.read())
        saved.append(str(path.name))

    return jsonify({
        "message": f"Uploaded {len(saved)} files for client {client_id!r}",
        "files": saved,
    }), 201


@admin_bp.route("/admin/clients/<client_id>/ingest", methods=["POST"])
@admin_required
def ingest_client(client_id: str):
    """Parse all uploaded specs for a client into canonical format."""
    from src.data.ingest.client_overlay import ingest_client_specs

    specs = ingest_client_specs(client_id)
    return jsonify({
        "message": f"Ingested {len(specs)} records for client {client_id!r}",
        "record_count": len(specs),
    })


@admin_bp.route("/admin/clients/<client_id>/train", methods=["POST"])
@admin_required
def train_client(client_id: str):
    """Trigger a client-specific fine-tuning run (async).

    Optional JSON body for config overrides:
    ``{"max_steps": 500, "learning_rate": 1e-4}``
    """
    from src.training.finetune import FineTuner, FinetuneConfig

    # Prevent duplicate runs
    thread = _active_trains.get(client_id)
    if thread and thread.is_alive():
        return jsonify({
            "error": f"Training already in progress for client {client_id!r}",
        }), 409

    data = request.get_json(silent=True) or {}
    config = FinetuneConfig(
        max_steps=data.get("max_steps", 500),
        learning_rate=data.get("learning_rate", 1e-4),
        client_id=client_id,
    )
    tuner = FineTuner(config=config)

    def _run():
        """Execute client-scoped fine-tuning in a background thread."""
        result = tuner.run_client(client_id)
        logger.info(
            "Client %s fine-tune result: %s",
            client_id, result.to_dict(),
        )

    t = threading.Thread(target=_run, daemon=True, name=f"finetune-{client_id}")
    t.start()
    _active_trains[client_id] = t

    return jsonify({
        "message": f"Fine-tuning started for client {client_id!r}",
        "max_steps": config.max_steps,
    }), 202


@admin_bp.route("/admin/clients/<client_id>/train/status", methods=["GET"])
@admin_required
def train_client_status(client_id: str):
    """Check if a client fine-tune is in progress."""
    thread = _active_trains.get(client_id)
    running = thread.is_alive() if thread else False

    # Check if checkpoint exists
    ckpt = paths.client_best_model(client_id)

    return jsonify({
        "client_id": client_id,
        "training_in_progress": running,
        "has_checkpoint": ckpt.exists(),
    })


# ══════════════════════════════════════════════════════════════════
#  BASE SPEC ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@admin_bp.route("/admin/specs/upload", methods=["POST"])
@admin_required
def upload_base_spec():
    """Upload spec files to the base (official) specs directory.

    Files are saved to ``model_store/data/<ver>/specs/uploads/``.
    """
    spec_uploads = paths.data_dir() / "specs" / "uploads"
    spec_uploads.mkdir(parents=True, exist_ok=True)

    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    saved = []
    for f in files:
        if not f.filename:
            continue
        dest = spec_uploads / f.filename
        f.save(str(dest))
        saved.append(f.filename)

    return jsonify({
        "message": f"Uploaded {len(saved)} base spec files",
        "files": saved,
    }), 201


@admin_bp.route("/admin/specs/ingest", methods=["POST"])
@admin_required
def ingest_base_specs():
    """Re-ingest all uploaded base spec files into canonical JSON."""
    from src.data.ingest import ingest_directory

    spec_uploads = paths.data_dir() / "specs" / "uploads"
    if not spec_uploads.is_dir():
        return jsonify({"error": "No uploads directory found"}), 404

    specs = ingest_directory(spec_uploads, save=True)
    return jsonify({
        "message": f"Ingested {len(specs)} base spec records",
        "record_count": len(specs),
    })


@admin_bp.route("/admin/specs/canonical", methods=["GET"])
@admin_required
def view_canonical():
    """Return the canonical spec records as JSON.

    Query params: ``?kind=field&limit=100``
    """
    from src.data.ingest.normalizer import load_canonical

    specs = load_canonical()
    kind_filter = request.args.get("kind")
    if kind_filter:
        specs = [s for s in specs if s.kind.value == kind_filter]

    limit = int(request.args.get("limit", 500))
    truncated = len(specs) > limit
    specs = specs[:limit]

    return jsonify({
        "records": [s.to_dict() for s in specs],
        "count": len(specs),
        "truncated": truncated,
    })


# ══════════════════════════════════════════════════════════════════
#  MODEL ENGINE MANAGEMENT
# ══════════════════════════════════════════════════════════════════

@admin_bp.route("/admin/models", methods=["GET"])
@admin_required
def list_models():
    """List all FIX version engines with load status."""
    import src.api.state as state

    engines = state.list_loaded_engines()
    return jsonify({
        "engines": engines,
        "any_loaded": any(e["loaded"] for e in engines.values()),
    })


@admin_bp.route("/admin/models/load", methods=["POST"])
@admin_required
def load_model_version():
    """Load (or reload) a model engine for a FIX version.

    JSON body: ``{"version": "4.4"}``
    """
    import src.api.state as state
    from src.core.version_registry import is_valid_version

    data = request.get_json(silent=True) or {}
    version = data.get("version")
    if not version or not is_valid_version(version):
        return jsonify({"error": f"Invalid version: {version}"}), 400

    engine = state.load_model(version)
    loaded = engine is not None
    return jsonify({
        "version": version,
        "loaded": loaded,
        "message": f"Engine for FIX {version} {'loaded' if loaded else 'failed to load'}",
    }), 200 if loaded else 500


@admin_bp.route("/admin/models/unload", methods=["POST"])
@admin_required
def unload_model_version():
    """Unload a model engine for a FIX version.

    JSON body: ``{"version": "4.4"}``  or  ``{"all": true}``
    """
    import src.api.state as state

    data = request.get_json(silent=True) or {}

    if data.get("all"):
        count = state.unload_all_models()
        return jsonify({"unloaded": count, "message": f"Unloaded {count} engine(s)"})

    version = data.get("version")
    if not version:
        return jsonify({"error": "Provide 'version' or 'all': true"}), 400

    state.unload_model(version)
    return jsonify({"version": version, "message": f"Engine for FIX {version} unloaded"})
