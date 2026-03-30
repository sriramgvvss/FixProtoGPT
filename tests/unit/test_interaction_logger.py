"""
Unit tests for src/data/interaction_logger.py

Tests the InteractionLogger class:
- Logging interactions to SQLite
- Adding feedback
- Reading back interactions with filters
- Exporting training pairs
- Stats
- Concurrency basics
"""

import json
import sqlite3
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


@pytest.fixture
def logger(tmp_path):
    """Create a fresh InteractionLogger writing to a temp directory."""
    from src.persistence.interaction_logger import InteractionLogger
    return InteractionLogger(log_dir=tmp_path)


# ── Basic Logging ──────────────────────────────────────────────────

class TestLogInteraction:
    def test_log_returns_sequential_id(self, logger):
        iid = logger.log("nl2fix", {"text": "Buy 100 AAPL"}, {"fix_message": "8=FIXT.1.1|35=D|"})
        assert isinstance(iid, str)
        assert iid == "IN00000001"

    def test_log_sequential_ids_increment(self, logger):
        id1 = logger.log("nl2fix", {}, {})
        id2 = logger.log("generate", {}, {})
        id3 = logger.log("explain", {}, {})
        assert id1 == "IN00000001"
        assert id2 == "IN00000002"
        assert id3 == "IN00000003"

    def test_log_creates_db(self, logger):
        logger.log("nl2fix", {"text": "test"}, {"fix_message": "msg"})
        assert logger.db_path.exists()

    def test_log_inserts_row(self, logger):
        logger.log("nl2fix", {"text": "test"}, {"fix_message": "msg"})
        conn = sqlite3.connect(str(logger.db_path))
        rows = conn.execute("SELECT * FROM interactions").fetchall()
        conn.close()
        assert len(rows) == 1

    def test_log_stores_correct_data(self, logger):
        logger.log("nl2fix", {"text": "test"}, {"fix_message": "msg"})
        records = logger.get_interactions()
        assert records[0]["endpoint"] == "nl2fix"
        assert records[0]["request"]["text"] == "test"
        assert records[0]["response"]["fix_message"] == "msg"

    def test_log_multiple(self, logger):
        logger.log("nl2fix", {}, {})
        logger.log("generate", {}, {})
        logger.log("explain", {}, {})
        records = logger.get_interactions()
        assert len(records) == 3

    def test_log_has_timestamp(self, logger):
        iid = logger.log("nl2fix", {}, {})
        records = logger.get_interactions()
        ts = records[0]["timestamp"]
        assert ts.endswith("Z") or ts.endswith("+00:00")

    def test_log_feedback_initially_none(self, logger):
        logger.log("nl2fix", {}, {})
        records = logger.get_interactions()
        assert records[0]["feedback"] is None

    def test_log_with_metadata(self, logger):
        logger.log("generate", {}, {}, {"temperature": 0.8, "demo_mode": False})
        records = logger.get_interactions()
        assert records[0]["metadata"]["temperature"] == 0.8


# ── Feedback ───────────────────────────────────────────────────────

class TestAddFeedback:
    def test_positive_feedback(self, logger):
        iid = logger.log("nl2fix", {}, {})
        ok = logger.add_feedback(iid, "positive")
        assert ok is True
        records = logger.get_interactions()
        assert records[0]["feedback"]["rating"] == "positive"

    def test_negative_feedback(self, logger):
        iid = logger.log("nl2fix", {}, {})
        ok = logger.add_feedback(iid, "negative")
        assert ok is True
        records = logger.get_interactions()
        assert records[0]["feedback"]["rating"] == "negative"

    def test_feedback_with_correction(self, logger):
        iid = logger.log("nl2fix", {"text": "Buy 100 AAPL"}, {"fix_message": "wrong"})
        logger.add_feedback(iid, "negative", correction="8=FIXT.1.1|35=D|55=AAPL|54=1|38=100|")
        records = logger.get_interactions()
        assert records[0]["feedback"]["correction"] == "8=FIXT.1.1|35=D|55=AAPL|54=1|38=100|"

    def test_feedback_with_comment(self, logger):
        iid = logger.log("nl2fix", {}, {})
        logger.add_feedback(iid, "positive", comment="Great response!")
        records = logger.get_interactions()
        assert records[0]["feedback"]["comment"] == "Great response!"

    def test_feedback_invalid_rating(self, logger):
        iid = logger.log("nl2fix", {}, {})
        ok = logger.add_feedback(iid, "maybe")
        assert ok is False

    def test_feedback_unknown_id(self, logger):
        logger.log("nl2fix", {}, {})
        ok = logger.add_feedback("nonexistent-uuid", "positive")
        assert ok is False

    def test_feedback_has_timestamp(self, logger):
        iid = logger.log("nl2fix", {}, {})
        logger.add_feedback(iid, "positive")
        records = logger.get_interactions()
        ts = records[0]["feedback"]["timestamp"]
        assert ts.endswith("Z") or ts.endswith("+00:00")


# ── Get Interactions ───────────────────────────────────────────────

class TestGetInteractions:
    def test_empty_log(self, logger):
        records = logger.get_interactions()
        assert records == []

    def test_filter_by_endpoint(self, logger):
        logger.log("nl2fix", {}, {})
        logger.log("generate", {}, {})
        logger.log("nl2fix", {}, {})
        records = logger.get_interactions(endpoint="nl2fix")
        assert len(records) == 2

    def test_filter_rated_only(self, logger):
        id1 = logger.log("nl2fix", {}, {})
        logger.log("nl2fix", {}, {})
        logger.add_feedback(id1, "positive")
        records = logger.get_interactions(rated_only=True)
        assert len(records) == 1

    def test_limit(self, logger):
        for i in range(10):
            logger.log("nl2fix", {"i": i}, {})
        records = logger.get_interactions(limit=3)
        assert len(records) == 3

    def test_most_recent_first(self, logger):
        logger.log("nl2fix", {"order": 1}, {})
        logger.log("nl2fix", {"order": 2}, {})
        records = logger.get_interactions()
        assert records[0]["request"]["order"] == 2


# ── Stats ──────────────────────────────────────────────────────────

class TestGetStats:
    def test_empty_stats(self, logger):
        stats = logger.get_stats()
        assert stats["total_interactions"] == 0
        assert stats["positive"] == 0
        assert stats["negative"] == 0

    def test_counts(self, logger):
        id1 = logger.log("nl2fix", {}, {})
        id2 = logger.log("generate", {}, {})
        id3 = logger.log("nl2fix", {}, {})
        logger.add_feedback(id1, "positive")
        logger.add_feedback(id2, "negative")
        stats = logger.get_stats()
        assert stats["total_interactions"] == 3
        assert stats["by_endpoint"] == {"nl2fix": 2, "generate": 1}
        assert stats["positive"] == 1
        assert stats["negative"] == 1
        assert stats["with_feedback"] == 2

    def test_corrections_counted(self, logger):
        iid = logger.log("nl2fix", {}, {})
        logger.add_feedback(iid, "negative", correction="fixed")
        stats = logger.get_stats()
        assert stats["with_corrections"] == 1


# ── Export Training Pairs ──────────────────────────────────────────

class TestExportTrainingPairs:
    def test_empty_export(self, logger):
        pairs = logger.export_training_pairs()
        assert pairs == []

    def test_nl2fix_positive(self, logger):
        iid = logger.log("nl2fix", {"text": "Buy 100 AAPL"}, {"fix_message": "8=FIXT.1.1|35=D|55=AAPL|"})
        logger.add_feedback(iid, "positive")
        pairs = logger.export_training_pairs()
        assert len(pairs) == 1
        assert "Buy 100 AAPL" in pairs[0]
        assert "8=FIXT.1.1|35=D|55=AAPL|" in pairs[0]

    def test_nl2fix_correction_overrides(self, logger):
        iid = logger.log("nl2fix", {"text": "Buy 100 AAPL"}, {"fix_message": "wrong"})
        logger.add_feedback(iid, "negative", correction="8=FIXT.1.1|35=D|55=AAPL|54=1|38=100|")
        pairs = logger.export_training_pairs()
        assert len(pairs) == 1
        assert "8=FIXT.1.1|35=D|55=AAPL|54=1|38=100|" in pairs[0]

    def test_nl2fix_no_feedback_skipped(self, logger):
        logger.log("nl2fix", {"text": "Buy 100 AAPL"}, {"fix_message": "msg"})
        pairs = logger.export_training_pairs()
        assert pairs == []

    def test_generate_positive(self, logger):
        iid = logger.log("generate", {"prompt": "8=FIXT.1.1|"}, {"generated": "8=FIXT.1.1|35=D|55=AAPL|"})
        logger.add_feedback(iid, "positive")
        pairs = logger.export_training_pairs()
        assert len(pairs) == 1
        assert "8=FIXT.1.1|35=D|55=AAPL|" in pairs[0]

    def test_explain_positive(self, logger):
        resp = {
            "explanation": {
                "message_type": {
                    "code": "D",
                    "name": "NewOrderSingle",
                    "category": "pre-trade",
                    "description": "Submit a new single order",
                }
            }
        }
        iid = logger.log("explain", {"message": "fix"}, resp)
        logger.add_feedback(iid, "positive")
        pairs = logger.export_training_pairs()
        assert len(pairs) == 1
        assert "FIX Message Type D:" in pairs[0]
        assert "NewOrderSingle" in pairs[0]

    def test_export_to_file(self, logger, tmp_path):
        iid = logger.log("nl2fix", {"text": "Buy 100 AAPL"}, {"fix_message": "msg"})
        logger.add_feedback(iid, "positive")
        path = logger.export_to_file()
        assert path.exists()
        content = path.read_text()
        assert "Buy 100 AAPL" in content


# ── Clear ──────────────────────────────────────────────────────────

class TestClear:
    def test_clear_returns_count(self, logger):
        logger.log("nl2fix", {}, {})
        logger.log("nl2fix", {}, {})
        count = logger.clear()
        assert count == 2

    def test_clear_empties_table(self, logger):
        logger.log("nl2fix", {}, {})
        logger.clear()
        assert logger.get_interactions() == []

    def test_clear_empty(self, logger):
        count = logger.clear()
        assert count == 0


# ── Delete ─────────────────────────────────────────────────────────

class TestDelete:
    def test_delete_existing(self, logger):
        iid = logger.log("nl2fix", {"text": "hello"}, {"fix": "msg"})
        result = logger.delete(iid)
        assert result is True
        items = logger.get_interactions()
        assert len(items) == 0

    def test_delete_nonexistent(self, logger):
        logger.log("nl2fix", {}, {})
        result = logger.delete("no-such-uuid")
        assert result is False
        items = logger.get_interactions()
        assert len(items) == 1

    def test_delete_preserves_others(self, logger):
        id1 = logger.log("nl2fix", {"text": "first"}, {})
        id2 = logger.log("generate", {"text": "second"}, {})
        id3 = logger.log("explain", {"text": "third"}, {})
        logger.delete(id2)
        items = logger.get_interactions()
        ids = {i["id"] for i in items}
        assert id1 in ids
        assert id3 in ids
        assert id2 not in ids

    def test_delete_empty_log(self, logger):
        result = logger.delete("any-id")
        assert result is False
