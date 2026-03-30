"""
Module: src.api.routes.learning
================================

Learning & Feedback Blueprint for FixProtoGPT.

Endpoints: feedback, interactions list/delete/clear, learning
           export/status, finetune (preflight + trigger).

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

import threading
from flask import Blueprint, request, jsonify, session

import src.api.state as state
from src.api.routes.auth import login_required, admin_required
from src.persistence.action_logger import log_user_action, log_debug

learning_bp = Blueprint("learning", __name__, url_prefix="/api")


# ── Feedback ──────────────────────────────────────────────────────

@learning_bp.route("/feedback", methods=["POST"])
@login_required
def submit_feedback():
    """Submit user feedback (thumbs-up/down) for an interaction."""
    try:
        data = request.json
        interaction_id = data.get("interaction_id", "")
        rating = data.get("rating", "")
        correction = data.get("correction")
        comment = data.get("comment")

        if not interaction_id:
            return jsonify({"error": "interaction_id is required"}), 400
        if rating not in ("positive", "negative"):
            return jsonify({"error": 'rating must be "positive" or "negative"'}), 400

        ok = state.interaction_log.add_feedback(interaction_id, rating, correction, comment)
        if not ok:
            return jsonify({"error": "Interaction not found"}), 404

        log_debug("feedback_submitted", detail={
            "interaction_id": interaction_id, "rating": rating,
            "has_correction": correction is not None,
        })
        log_user_action("feedback", detail={
            "interaction_id": interaction_id, "rating": rating,
        })
        return jsonify({"success": True, "interaction_id": interaction_id})

    except Exception as e:
        state.logger.error("Feedback error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Interactions CRUD ─────────────────────────────────────────────

@learning_bp.route("/interactions", methods=["GET"])
@login_required
def list_interactions():
    """List logged interactions with optional filters."""
    try:
        endpoint = request.args.get("endpoint")
        rated_only = request.args.get("rated_only", "").lower() == "true"
        limit = int(request.args.get("limit", 50))

        records = state.interaction_log.get_interactions(
            endpoint=endpoint, rated_only=rated_only, limit=limit
        )
        stats = state.interaction_log.get_stats()

        return jsonify({"interactions": records, "stats": stats})

    except Exception as e:
        state.logger.error("Interactions error: %s", e)
        return jsonify({"error": str(e)}), 500


@learning_bp.route("/interactions/<interaction_id>", methods=["DELETE"])
@login_required
def delete_interaction(interaction_id):
    """Delete a single interaction by ID."""
    try:
        ok = state.interaction_log.delete(interaction_id)
        if not ok:
            return jsonify({"error": "Interaction not found"}), 404
        log_user_action("delete_interaction", detail={
            "interaction_id": interaction_id,
        })
        return jsonify({"success": True, "deleted": interaction_id})
    except Exception as e:
        state.logger.error("Delete error: %s", e)
        return jsonify({"error": str(e)}), 500


@learning_bp.route("/interactions", methods=["DELETE"])
@login_required
def clear_interactions():
    """Delete all interactions."""
    try:
        count = state.interaction_log.clear()
        log_user_action("clear_interactions", detail={
            "deleted_count": count,
        })
        return jsonify({"success": True, "deleted_count": count})
    except Exception as e:
        state.logger.error("Clear error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Learning pipeline ────────────────────────────────────────────

@learning_bp.route("/learning/export", methods=["POST"])
@login_required
def export_finetune_data():
    """Export positive-feedback interactions as training-ready text."""
    try:
        output_path = state.interaction_log.export_to_file()
        pairs = state.interaction_log.export_training_pairs()

        log_user_action("export_finetune_data", detail={
            "pair_count": len(pairs),
        })
        return jsonify({
            "success": True,
            "exported_file": str(output_path),
            "pair_count": len(pairs),
            "pairs_preview": pairs[:10],
        })

    except Exception as e:
        state.logger.error("Export error: %s", e)
        return jsonify({"error": str(e)}), 500


@learning_bp.route("/learning/status", methods=["GET"])
@login_required
def learning_status():
    """Show how many interactions are available for fine-tuning."""
    try:
        stats = state.interaction_log.get_stats()
        pairs = state.interaction_log.export_training_pairs()
        new_pairs = state.interaction_log.export_training_pairs(untrained_only=True)

        return jsonify({
            "total_interactions": stats["total_interactions"],
            "with_feedback": stats["with_feedback"],
            "positive": stats["positive"],
            "negative": stats["negative"],
            "with_corrections": stats["with_corrections"],
            "exportable_pairs": len(pairs),
            "new_pairs": len(new_pairs),
            "by_endpoint": stats["by_endpoint"],
        })

    except Exception as e:
        state.logger.error("Learning status error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Fine-tuning pipeline ─────────────────────────────────────────

# Module-level lock: only one finetune run at a time
_finetune_lock = threading.Lock()
_finetune_result: dict | None = None


@learning_bp.route("/learning/finetune/preflight", methods=["GET"])
@admin_required
def finetune_preflight():
    """Check whether fine-tuning can be started (admin only)."""
    try:
        from src.training.finetune import FineTuner, FinetuneConfig
        ver = session.get("fix_version", None)
        cfg = FinetuneConfig(fix_version=ver)
        ft = FineTuner(config=cfg, interaction_log=state.interaction_log)
        info = ft.preflight()
        return jsonify(info)
    except Exception as e:
        state.logger.error("Finetune preflight error: %s", e)
        return jsonify({"error": str(e)}), 500


@learning_bp.route("/learning/finetune", methods=["POST"])
@admin_required
def trigger_finetune():
    """Trigger an incremental fine-tuning run (admin only).

    The training runs in a background thread. Poll
    ``GET /api/learning/finetune/status`` for progress.
    """
    global _finetune_result

    if not _finetune_lock.acquire(blocking=False):
        return jsonify({
            "error": "A fine-tuning run is already in progress"
        }), 409

    try:
        from src.training.finetune import FineTuner, FinetuneConfig

        data = request.json or {}
        ver = data.get("fix_version") or session.get("fix_version", None)
        cfg = FinetuneConfig(
            max_steps=data.get("max_steps", 500),
            learning_rate=data.get("learning_rate", 1e-4),
            min_new_pairs=data.get("min_new_pairs", 5),
            fix_version=ver,
        )
        ft = FineTuner(config=cfg, interaction_log=state.interaction_log)

        # Preflight check
        pre = ft.preflight()
        if not pre["ready"]:
            _finetune_lock.release()
            return jsonify({
                "error": "Preflight failed",
                "details": pre["reasons"],
            }), 400

        _finetune_result = {"status": "running", "started_at": _now_iso()}

        def _run():
            """Run global fine-tuning in a background thread and update result state."""
            global _finetune_result
            try:
                result = ft.run()
                _finetune_result = {
                    "status": "completed" if result.success else "failed",
                    **result.to_dict(),
                }
            except Exception as exc:
                state.logger.exception("Background finetune failed")
                _finetune_result = {"status": "failed", "error": str(exc)}
            finally:
                _finetune_lock.release()

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        log_user_action("trigger_finetune", detail={
            "new_pairs": pre["new_pairs"],
            "fix_version": ver,
        })
        return jsonify({
            "success": True,
            "message": "Fine-tuning started in background",
            "new_pairs": pre["new_pairs"],
        })

    except Exception as e:
        _finetune_lock.release()
        state.logger.error("Finetune trigger error: %s", e)
        return jsonify({"error": str(e)}), 500


@learning_bp.route("/learning/finetune/status", methods=["GET"])
@admin_required
def finetune_status():
    """Check the status of the current/last fine-tuning run."""
    if _finetune_result is None:
        return jsonify({"status": "idle", "message": "No fine-tuning run yet"})
    return jsonify(_finetune_result)


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
