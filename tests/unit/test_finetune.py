"""
Unit tests for the automated fine-tuning pipeline.

Covers:
- InteractionLogger: mark_trained, get_trainable_ids, untrained_only filter
- FineTuner: preflight, run (mocked training loop)
- API endpoints: preflight, trigger, status (admin-required checks)
"""

import sys
import json
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def logger(tmp_path):
    """Create a fresh InteractionLogger in a temp directory."""
    from src.persistence.interaction_logger import InteractionLogger
    return InteractionLogger(log_dir=tmp_path)


@pytest.fixture
def logger_with_pairs(logger):
    """InteractionLogger with several interactions, some with positive feedback."""
    # 3 positive-feedback nl2fix interactions
    for i in range(3):
        iid = logger.log("nl2fix", {"text": f"Buy {i} AAPL"}, {"fix_message": f"8=FIXT.1.1|35=D|55=AAPL|38={i}|"})
        logger.add_feedback(iid, "positive")

    # 2 with correction
    for i in range(2):
        iid = logger.log("nl2fix", {"text": f"Sell {i} MSFT"}, {"fix_message": "wrong"})
        logger.add_feedback(iid, "negative", correction=f"8=FIXT.1.1|35=D|55=MSFT|38={i}|")

    # 1 without feedback (should not be trainable)
    logger.log("nl2fix", {"text": "no feedback"}, {"fix_message": "msg"})

    # 1 negative without correction (should still be skipped for training pairs)
    iid = logger.log("nl2fix", {"text": "bad"}, {"fix_message": "msg"})
    logger.add_feedback(iid, "negative")

    return logger


@pytest.fixture
def client(tmp_path):
    """Flask test client with isolated DB."""
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


def _login_admin(client):
    """Log in as the seeded admin user."""
    client.post("/auth/login", json={
        "username": "admin",
        "password": "admin123",
    }, content_type="application/json")
    return client


def _login_user(client):
    """Log in as a non-admin seeded user."""
    client.post("/auth/login", json={
        "username": "trader",
        "password": "trader123",
    }, content_type="application/json")
    return client


# ══════════════════════════════════════════════════════════════════
# InteractionLogger — trained_at tracking
# ══════════════════════════════════════════════════════════════════

class TestMarkTrained:
    def test_mark_trained_returns_count(self, logger_with_pairs):
        ids = logger_with_pairs.get_trainable_ids()
        count = logger_with_pairs.mark_trained(ids)
        assert count == len(ids)

    def test_mark_trained_sets_timestamp(self, logger_with_pairs):
        ids = logger_with_pairs.get_trainable_ids()
        logger_with_pairs.mark_trained(ids[:1])
        records = logger_with_pairs.get_interactions()
        trained = [r for r in records if r.get("trained_at")]
        assert len(trained) == 1
        assert trained[0]["trained_at"] is not None

    def test_mark_trained_empty_list(self, logger):
        count = logger.mark_trained([])
        assert count == 0

    def test_mark_trained_nonexistent_ids(self, logger):
        count = logger.mark_trained(["no-such-id-1", "no-such-id-2"])
        assert count == 0


class TestGetTrainableIds:
    def test_trainable_ids_count(self, logger_with_pairs):
        ids = logger_with_pairs.get_trainable_ids()
        # 3 positive + 2 with correction + 1 negative = 6 (all with feedback)
        assert len(ids) == 6

    def test_trainable_ids_are_strings(self, logger_with_pairs):
        ids = logger_with_pairs.get_trainable_ids()
        for i in ids:
            assert isinstance(i, str)

    def test_no_trainable_empty_log(self, logger):
        ids = logger.get_trainable_ids()
        assert ids == []

    def test_trainable_excluded_after_mark(self, logger_with_pairs):
        ids = logger_with_pairs.get_trainable_ids()
        logger_with_pairs.mark_trained(ids)
        remaining = logger_with_pairs.get_trainable_ids()
        assert remaining == []


class TestExportUntrainedOnly:
    def test_untrained_only_returns_all_initially(self, logger_with_pairs):
        all_pairs = logger_with_pairs.export_training_pairs()
        untrained = logger_with_pairs.export_training_pairs(untrained_only=True)
        assert len(untrained) == len(all_pairs)

    def test_untrained_only_excludes_trained(self, logger_with_pairs):
        ids = logger_with_pairs.get_trainable_ids()
        logger_with_pairs.mark_trained(ids[:2])
        untrained = logger_with_pairs.export_training_pairs(untrained_only=True)
        all_pairs = logger_with_pairs.export_training_pairs()
        assert len(untrained) == len(all_pairs) - 2

    def test_untrained_only_empty_after_all_trained(self, logger_with_pairs):
        ids = logger_with_pairs.get_trainable_ids()
        logger_with_pairs.mark_trained(ids)
        untrained = logger_with_pairs.export_training_pairs(untrained_only=True)
        assert untrained == []

    def test_default_returns_all_including_trained(self, logger_with_pairs):
        ids = logger_with_pairs.get_trainable_ids()
        logger_with_pairs.mark_trained(ids)
        all_pairs = logger_with_pairs.export_training_pairs()
        assert len(all_pairs) > 0


# ══════════════════════════════════════════════════════════════════
# FineTuner — preflight checks
# ══════════════════════════════════════════════════════════════════

class TestFineTunerPreflight:
    def test_not_ready_insufficient_pairs(self, logger):
        """Preflight fails when there aren't enough training pairs."""
        from src.training.finetune import FineTuner, FinetuneConfig
        cfg = FinetuneConfig(min_new_pairs=5)
        ft = FineTuner(config=cfg, interaction_log=logger)
        result = ft.preflight()
        assert result["ready"] is False
        assert result["new_pairs"] == 0
        assert any("at least" in r for r in result["reasons"])

    def test_ready_with_enough_pairs(self, logger_with_pairs):
        """Preflight passes when there are enough untrained pairs."""
        from src.training.finetune import FineTuner, FinetuneConfig
        cfg = FinetuneConfig(min_new_pairs=3)
        ft = FineTuner(config=cfg, interaction_log=logger_with_pairs)
        with patch("src.training.finetune.paths") as mock_paths:
            mock_paths.best_model.return_value = MagicMock(exists=MagicMock(return_value=True))
            mock_paths.MODEL_STORE = Path("/tmp/fake_store")
            result = ft.preflight()
        assert result["ready"] is True
        assert result["new_pairs"] >= 3

    def test_preflight_no_checkpoint(self, logger_with_pairs):
        """Preflight fails if there's no checkpoint to resume from."""
        from src.training.finetune import FineTuner, FinetuneConfig
        cfg = FinetuneConfig(min_new_pairs=1)
        ft = FineTuner(config=cfg, interaction_log=logger_with_pairs)
        with patch.object(ft, "_find_latest_checkpoint", return_value=None):
            with patch("src.training.finetune.paths") as mock_paths:
                mock_paths.best_model.return_value = MagicMock(exists=MagicMock(return_value=False))
                result = ft.preflight()
        assert result["ready"] is False
        assert any("checkpoint" in r.lower() for r in result["reasons"])

    def test_preflight_returns_pair_count(self, logger_with_pairs):
        """Preflight response includes the new pair count."""
        from src.training.finetune import FineTuner, FinetuneConfig
        ft = FineTuner(config=FinetuneConfig(min_new_pairs=100), interaction_log=logger_with_pairs)
        result = ft.preflight()
        assert "new_pairs" in result
        assert isinstance(result["new_pairs"], int)
        assert result["new_pairs"] == 5  # 3 positive + 2 corrections


class TestFineTunerRun:
    def test_run_not_enough_pairs(self, logger):
        """Run returns error when there aren't enough pairs."""
        from src.training.finetune import FineTuner, FinetuneConfig
        cfg = FinetuneConfig(min_new_pairs=5)
        ft = FineTuner(config=cfg, interaction_log=logger)
        result = ft.run()
        assert result.success is False
        assert "not enough" in result.error.lower() or result.new_pairs == 0

    def test_run_concurrent_guard(self, logger):
        """Only one run can execute at a time."""
        from src.training.finetune import FineTuner, FinetuneConfig
        ft = FineTuner(config=FinetuneConfig(), interaction_log=logger)
        ft._running = True
        result = ft.run()
        assert result.success is False
        assert "already running" in result.error.lower()

    def test_run_sets_running_flag(self, logger_with_pairs):
        """The _running flag is set during execution."""
        from src.training.finetune import FineTuner, FinetuneConfig
        cfg = FinetuneConfig(min_new_pairs=1)
        ft = FineTuner(config=cfg, interaction_log=logger_with_pairs)

        was_running = []

        original_do = ft._do_finetune
        def mock_do():
            was_running.append(ft.is_running)
            raise RuntimeError("abort test")

        ft._do_finetune = mock_do
        result = ft.run()
        assert was_running == [True]
        assert ft.is_running is False  # reset after exception

    def test_result_to_dict(self):
        """FinetuneResult.to_dict() returns expected keys."""
        from src.training.finetune import FinetuneResult
        r = FinetuneResult(
            success=True,
            new_pairs=10,
            steps_trained=100,
            final_loss=0.123,
            best_val_loss=0.100,
            checkpoint_path="/tmp/best.pt",
            duration_secs=42.567,
        )
        d = r.to_dict()
        assert d["success"] is True
        assert d["new_pairs"] == 10
        assert d["steps_trained"] == 100
        assert d["duration_secs"] == 42.57
        assert "trained_ids" not in d  # excluded from dict


# ══════════════════════════════════════════════════════════════════
# API Endpoints — fine-tuning routes
# ══════════════════════════════════════════════════════════════════

class TestFinetunePreflightEndpoint:
    def test_requires_auth(self, client):
        resp = client.get("/api/learning/finetune/preflight",
                          headers={"Accept": "application/json"})
        assert resp.status_code == 401

    def test_requires_admin(self, client):
        _login_user(client)
        resp = client.get("/api/learning/finetune/preflight")
        assert resp.status_code == 403

    def test_admin_can_access(self, client):
        _login_admin(client)
        resp = client.get("/api/learning/finetune/preflight")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "ready" in data
        assert "new_pairs" in data

    def test_not_ready_when_empty(self, client):
        _login_admin(client)
        resp = client.get("/api/learning/finetune/preflight")
        data = resp.get_json()
        assert data["ready"] is False


class TestTriggerFinetuneEndpoint:
    def test_requires_auth(self, client):
        resp = client.post("/api/learning/finetune",
                           json={},
                           content_type="application/json",
                           headers={"Accept": "application/json"})
        assert resp.status_code == 401

    def test_requires_admin(self, client):
        _login_user(client)
        resp = client.post("/api/learning/finetune",
                           json={},
                           content_type="application/json")
        assert resp.status_code == 403

    def test_fails_preflight_when_empty(self, client):
        _login_admin(client)
        resp = client.post("/api/learning/finetune",
                           json={},
                           content_type="application/json")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "preflight" in data.get("error", "").lower() or "details" in data


class TestFinetuneStatusEndpoint:
    def test_requires_auth(self, client):
        resp = client.get("/api/learning/finetune/status",
                          headers={"Accept": "application/json"})
        assert resp.status_code == 401

    def test_requires_admin(self, client):
        _login_user(client)
        resp = client.get("/api/learning/finetune/status")
        assert resp.status_code == 403

    def test_idle_when_no_run(self, client):
        _login_admin(client)
        resp = client.get("/api/learning/finetune/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "idle"


class TestLearningStatusNewPairs:
    def test_status_includes_new_pairs(self, client):
        """The learning/status endpoint now includes new_pairs count."""
        _login_admin(client)
        resp = client.get("/api/learning/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "new_pairs" in data
        assert data["new_pairs"] == 0
