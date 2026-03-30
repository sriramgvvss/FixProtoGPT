"""
Module: src.api.routes.export
==============================

Export / Import / Convert Blueprint for FixProtoGPT.

Endpoints: export (JSON/XML), import (JSON/XML), convert (to-json, to-xml).

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from io import BytesIO

from flask import Blueprint, request, jsonify, send_file

import src.api.state as state
from src.api.routes.auth import login_required
from src.persistence.action_logger import log_user_action, log_debug
from src.utils.fix_converter import FixMessageConverter

export_bp = Blueprint("export", __name__, url_prefix="/api")


# ── Export ────────────────────────────────────────────────────────

@export_bp.route("/export/json", methods=["POST"])
@login_required
def export_json():
    """Export FIX message to downloadable JSON file."""
    try:
        fix_message = (request.json or {}).get("message", "")
        if not fix_message:
            return jsonify({"error": "FIX message is required"}), 400

        converter = FixMessageConverter()
        json_output = converter.fix_to_json(fix_message, pretty=True)

        buf = BytesIO(json_output.encode("utf-8"))
        log_debug("export_json", detail={"msg_len": len(fix_message)})
        log_user_action("export_json", detail={"message_length": len(fix_message)})
        return send_file(buf, mimetype="application/json", as_attachment=True, download_name="fix_message.json")

    except Exception as e:
        state.logger.error("JSON export error: %s", e)
        return jsonify({"error": str(e)}), 500


@export_bp.route("/export/xml", methods=["POST"])
@login_required
def export_xml():
    """Export FIX message to downloadable XML file."""
    try:
        fix_message = (request.json or {}).get("message", "")
        if not fix_message:
            return jsonify({"error": "FIX message is required"}), 400

        converter = FixMessageConverter()
        xml_output = converter.fix_to_xml(fix_message, pretty=True)

        buf = BytesIO(xml_output.encode("utf-8"))
        log_debug("export_xml", detail={"msg_len": len(fix_message)})
        log_user_action("export_xml", detail={"message_length": len(fix_message)})
        return send_file(buf, mimetype="application/xml", as_attachment=True, download_name="fix_message.xml")

    except Exception as e:
        state.logger.error("XML export error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Import ────────────────────────────────────────────────────────

@export_bp.route("/import/json", methods=["POST"])
@login_required
def import_json():
    """Import FIX message from uploaded JSON file."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        content = file.read().decode("utf-8")
        converter = FixMessageConverter()
        validation = converter.validate_json_structure(content)

        if not validation["valid"]:
            return jsonify({"error": "Invalid JSON structure", "details": validation["errors"]}), 400

        fix_message = converter.json_to_fix(content)
        log_user_action("import_json", detail={"filename": file.filename})
        return jsonify({
            "success": True,
            "fix_message": fix_message,
            "warnings": validation.get("warnings", []),
        })

    except Exception as e:
        state.logger.error("JSON import error: %s", e)
        return jsonify({"error": str(e)}), 500


@export_bp.route("/import/xml", methods=["POST"])
@login_required
def import_xml():
    """Import FIX message from uploaded XML file."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        content = file.read().decode("utf-8")
        converter = FixMessageConverter()
        validation = converter.validate_xml_structure(content)

        if not validation["valid"]:
            return jsonify({"error": "Invalid XML structure", "details": validation["errors"]}), 400

        fix_message = converter.xml_to_fix(content)
        log_user_action("import_xml", detail={"filename": file.filename})
        return jsonify({
            "success": True,
            "fix_message": fix_message,
            "warnings": validation.get("warnings", []),
        })

    except Exception as e:
        state.logger.error("XML import error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Convert (inline display) ─────────────────────────────────────

@export_bp.route("/convert/to-json", methods=["POST"])
@login_required
def convert_to_json():
    """Convert FIX message to JSON (returns data, not file)."""
    try:
        fix_message = (request.json or {}).get("message", "")
        if not fix_message:
            return jsonify({"error": "FIX message is required"}), 400

        converter = FixMessageConverter()
        json_output = converter.fix_to_json(fix_message, pretty=True)
        return jsonify({"success": True, "json_output": json_output})

    except Exception as e:
        state.logger.error("JSON conversion error: %s", e)
        return jsonify({"error": str(e)}), 500


@export_bp.route("/convert/to-xml", methods=["POST"])
@login_required
def convert_to_xml():
    """Convert FIX message to XML (returns data, not file)."""
    try:
        fix_message = (request.json or {}).get("message", "")
        if not fix_message:
            return jsonify({"error": "FIX message is required"}), 400

        converter = FixMessageConverter()
        xml_output = converter.fix_to_xml(fix_message, pretty=True)
        return jsonify({"success": True, "xml_output": xml_output})

    except Exception as e:
        state.logger.error("XML conversion error: %s", e)
        return jsonify({"error": str(e)}), 500
