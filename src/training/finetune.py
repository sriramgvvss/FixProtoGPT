"""
Module: src.training.finetune
==============================

Automated fine-tuning pipeline for FixProtoGPT.

Uses positive-feedback interactions from the SQLite interaction log to
incrementally fine-tune the model from its latest checkpoint.

Workflow
--------
1.  Export untrained interactions that have positive user feedback.
2.  Append the training lines to a temporary augmented ``train.txt``.
3.  Re-tokenize into a ``.bin`` file.
4.  Resume training from the latest checkpoint for a configurable
    number of additional steps (default 500).
5.  Save a new ``best.pt`` and mark the interactions as ``trained_at``.
6.  Hot-reload the model in the running inference engine.

Usage::

    from src.training.finetune import FineTuner
    ft = FineTuner()
    result = ft.run()       # blocking — runs on the calling thread

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

import json
import shutil
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils import paths

logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    """Configuration for a fine-tuning run."""

    # How many optimizer steps to train for
    max_steps: int = 500
    # Learning rate (lower than pre-training to avoid catastrophic forgetting)
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    warmup_steps: int = 50
    batch_size: int = 8
    micro_batch_size: int = 2
    grad_clip: float = 1.0
    eval_interval: int = 50
    save_interval: int = 100
    # Minimum new interactions before allowing a run
    min_new_pairs: int = 5
    # Target FIX version (None → default from YAML)
    fix_version: Optional[str] = None
    # Client ID for per-client fine-tuning (None → base model)
    client_id: Optional[str] = None


@dataclass
class FinetuneResult:
    """Outcome of a fine-tuning run."""
    success: bool = False
    new_pairs: int = 0
    total_pairs: int = 0
    steps_trained: int = 0
    final_loss: float = 0.0
    best_val_loss: float = 0.0
    checkpoint_path: str = ""
    duration_secs: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise result to a JSON-safe dict."""
        return {
            "success": self.success,
            "new_pairs": self.new_pairs,
            "total_pairs": self.total_pairs,
            "steps_trained": self.steps_trained,
            "final_loss": self.final_loss,
            "best_val_loss": self.best_val_loss,
            "checkpoint_path": self.checkpoint_path,
            "duration_secs": round(self.duration_secs, 2),
            "error": self.error,
        }


class FineTuner:
    """Automated incremental fine-tuning from user-feedback interactions."""

    def __init__(
        self,
        config: Optional[FinetuneConfig] = None,
        interaction_log=None,
    ):
        """Initialise the fine-tuner.

        Args:
            config: Tuning hyper-parameters.  Falls back to defaults.
            interaction_log: Optional :class:`InteractionLogger` to use
                instead of the global one.
        """
        self.config = config or FinetuneConfig()
        self._version = self.config.fix_version or paths.active_version()
        self._client_id = self.config.client_id  # None → base model

        # Allow injection for testing; default to the global one
        if interaction_log is not None:
            self._log = interaction_log
        else:
            import src.api.state as state
            self._log = state.interaction_log

        self._running = False

    # ── Public API ────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """Whether a fine-tuning run is currently in progress."""
        return self._running

    def preflight(self) -> Dict[str, Any]:
        """Check readiness without actually training.

        Returns a dict with ``ready``, ``new_pairs``, and ``reason``.
        """
        new_pairs = self._log.export_training_pairs(untrained_only=True)
        trainable_ids = self._log.get_trainable_ids()
        checkpoint = self._find_latest_checkpoint()

        ready = True
        reasons: List[str] = []

        if len(new_pairs) < self.config.min_new_pairs:
            ready = False
            reasons.append(
                f"Need at least {self.config.min_new_pairs} new training pairs "
                f"(have {len(new_pairs)})"
            )

        if checkpoint is None and not paths.best_model(self._version).exists():
            ready = False
            reasons.append(f"No checkpoint found for FIX {self._version}")

        if self._running:
            ready = False
            reasons.append("A fine-tuning run is already in progress")

        return {
            "ready": ready,
            "new_pairs": len(new_pairs),
            "trainable_interactions": len(trainable_ids),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "fix_version": self._version,
            "reasons": reasons,
        }

    def run(self) -> FinetuneResult:
        """Execute the full fine-tuning pipeline (blocking).

        Returns a :class:`FinetuneResult` with metrics.
        """
        result = FinetuneResult()
        t0 = time.time()

        if self._running:
            result.error = "Fine-tuning is already running"
            return result

        self._running = True
        try:
            result = self._do_finetune()
        except Exception as exc:
            logger.exception("Fine-tuning failed")
            result.error = str(exc)
        finally:
            self._running = False
            result.duration_secs = time.time() - t0

        return result

    # ── Client-specific fine-tuning ───────────────────────────────

    def run_client(self, client_id: Optional[str] = None) -> FinetuneResult:
        """Fine-tune a client-specific model from overlay training data.

        This builds training data from base + client overlay, then
        runs the full training loop starting from the base checkpoint,
        saving the result to a client-specific checkpoint directory.

        Args:
            client_id: Client identifier.  Defaults to ``config.client_id``.

        Returns:
            :class:`FinetuneResult` with metrics.
        """
        cid = client_id or self._client_id
        if not cid:
            result = FinetuneResult()
            result.error = "client_id is required for client fine-tuning"
            return result

        result = FinetuneResult()
        t0 = time.time()

        if self._running:
            result.error = "Fine-tuning is already running"
            return result

        self._running = True
        try:
            result = self._do_client_finetune(cid)
        except Exception as exc:
            logger.exception("Client fine-tuning failed for %s", cid)
            result.error = str(exc)
        finally:
            self._running = False
            result.duration_secs = time.time() - t0

        return result

    def _do_client_finetune(self, client_id: str) -> FinetuneResult:
        """Internal pipeline for client-specific fine-tuning."""
        from src.data.ingest.client_overlay import (
            build_client_training_data,
            client_training_path,
        )

        result = FinetuneResult()

        # 1. Build merged training data (base + overlay)
        training_txt = build_client_training_data(client_id, self._version)
        with open(training_txt) as fh:
            result.total_pairs = sum(1 for _ in fh)

        if result.total_pairs == 0:
            result.error = f"No training data for client {client_id}"
            return result

        logger.info(
            "Client %s: %d training lines from %s",
            client_id, result.total_pairs, training_txt,
        )

        # 2. Tokenize into .bin
        bin_path = self._tokenize(training_txt)

        # 3. Train (start from base checkpoint, save to client dir)
        train_result = self._train_loop(
            bin_path,
            client_id=client_id,
        )
        result.steps_trained = train_result["steps"]
        result.final_loss = train_result["final_loss"]
        result.best_val_loss = train_result["best_val_loss"]
        result.checkpoint_path = train_result["checkpoint"]

        # 4. Hot-reload the client model
        self._hot_reload(client_id=client_id)

        result.success = True
        return result

    # ── Internal pipeline ─────────────────────────────────────────

    def _do_finetune(self) -> FinetuneResult:
        """Internal pipeline that exports, tokenizes, trains, and reloads."""
        result = FinetuneResult()

        # 1. Export new training pairs
        new_pairs = self._log.export_training_pairs(untrained_only=True)
        trainable_ids = self._log.get_trainable_ids()
        result.new_pairs = len(new_pairs)

        if len(new_pairs) < self.config.min_new_pairs:
            result.error = (
                f"Not enough new pairs ({len(new_pairs)}); "
                f"minimum is {self.config.min_new_pairs}"
            )
            return result

        logger.info("Fine-tuning: %d new training pairs from %d interactions",
                     len(new_pairs), len(trainable_ids))

        # 2. Build augmented training data
        augmented_path = self._build_augmented_data(new_pairs)
        result.total_pairs = sum(1 for _ in open(augmented_path))

        # 3. Tokenize into .bin
        bin_path = self._tokenize(augmented_path)

        # 4. Train
        train_result = self._train_loop(bin_path)
        result.steps_trained = train_result["steps"]
        result.final_loss = train_result["final_loss"]
        result.best_val_loss = train_result["best_val_loss"]
        result.checkpoint_path = train_result["checkpoint"]

        # 5. Mark interactions as trained
        self._log.mark_trained(trainable_ids)
        logger.info("Marked %d interactions as trained", len(trainable_ids))

        # 6. Hot-reload the model
        self._hot_reload()

        result.success = True
        return result

    def _build_augmented_data(self, new_pairs: List[str]) -> Path:
        """Append new pairs to a copy of the existing training text.

        Args:
            new_pairs: Training lines to add.

        Returns:
            Path to the augmented training file.
        """
        original = paths.processed_data_dir(self._version) / "train.txt"
        ft_dir = paths.PROJECT_ROOT / "model_store" / "finetune"
        ft_dir.mkdir(parents=True, exist_ok=True)

        slug = paths._version_to_slug(self._version)
        augmented = ft_dir / f"train_augmented_{slug}.txt"

        # Copy original training data
        if original.exists():
            shutil.copy2(original, augmented)
        else:
            augmented.write_text("")

        # Append new pairs
        with open(augmented, "a") as f:
            for line in new_pairs:
                f.write(line.rstrip("\n") + "\n")

        logger.info("Augmented training file: %s (%d new lines added)",
                     augmented, len(new_pairs))
        return augmented

    def _tokenize(self, text_path: Path) -> Path:
        """Tokenize the augmented text file into a ``.bin`` file.

        Args:
            text_path: Path to the ``.txt`` training file.

        Returns:
            Path to the ``.bin`` binary token file.
        """
        from src.core.tokenizer import FixProtocolTokenizer

        tokenizer = FixProtocolTokenizer()
        tokenizer.load(str(paths.tokenizer_dir(self._version)))

        with open(text_path) as f:
            texts = f.readlines()

        all_tokens: List[int] = []
        for text in texts:
            tokens = tokenizer.encode(text.strip())
            all_tokens.extend(tokens)

        bin_path = text_path.with_suffix(".bin")
        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(str(bin_path))

        logger.info("Tokenized %d lines → %d tokens → %s",
                     len(texts), len(all_tokens), bin_path)
        return bin_path

    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent ``.pt`` checkpoint for the target version.

        Returns:
            Path to the checkpoint, or ``None`` if none exists.
        """
        from src.training.train import find_latest_checkpoint

        ckpt_dir = str(paths.checkpoint_dir(self._version))
        ckpt = find_latest_checkpoint(ckpt_dir)
        if ckpt:
            return Path(ckpt)

        best = paths.best_model(self._version)
        if best.exists():
            return best
        return None

    def _train_loop(self, train_bin: Path, client_id: Optional[str] = None) -> Dict[str, Any]:
        """Run the training loop for ``max_steps`` steps.

        Delegates to :class:`Trainer` to avoid duplicating the core
        training logic.  Overrides only the checkpoint directory and
        step limits for fine-tuning.

        Args:
            train_bin: Path to the binary token file.
            client_id: If provided, save checkpoints to the client dir.

        Returns:
            Dict with ``steps``, ``final_loss``, ``best_val_loss``,
            and ``checkpoint`` path.
        """
        import torch
        from src.core.transformer import ModelConfig, create_model
        from src.core.tokenizer import FixProtocolTokenizer
        from src.training.train import (
            FixProtocolDataset,
            Trainer,
            TrainConfig,
        )

        # Load model config
        model_config = ModelConfig.from_yaml(paths.CONFIG_PATH)

        # Fine-tune training config (lower LR, fewer steps)
        # Determine checkpoint directory for fine-tune outputs
        if client_id:
            checkpoint_dir = str(paths.client_checkpoint_dir(client_id, self._version))
        else:
            checkpoint_dir = str(paths.checkpoint_dir(self._version))

        train_config = TrainConfig(
            batch_size=self.config.batch_size,
            micro_batch_size=self.config.micro_batch_size,
            max_epochs=1,   # we control by max_steps instead
            learning_rate=self.config.learning_rate,
            min_lr=self.config.min_lr,
            warmup_steps=self.config.warmup_steps,
            grad_clip=self.config.grad_clip,
            save_interval=self.config.save_interval,
            eval_interval=self.config.eval_interval,
            checkpoint_dir=checkpoint_dir,
        )

        # Device selection
        from src.utils.device import detect_device
        device = detect_device()

        logger.info("Fine-tuning on device: %s", device)

        # Create model
        model = create_model(model_config)

        # Tokenizer (needed by Dataset)
        tokenizer = FixProtocolTokenizer(vocab_size=model_config.vocab_size)
        tokenizer.load(str(paths.tokenizer_dir(self._version)))

        # Datasets
        train_dataset = FixProtocolDataset(
            str(train_bin), tokenizer, model_config.max_seq_len
        )
        val_bin = paths.val_data(self._version)
        val_dataset = (
            FixProtocolDataset(str(val_bin), tokenizer, model_config.max_seq_len)
            if val_bin.exists()
            else None
        )

        # ── Create Trainer (reuses full Trainer infrastructure) ───
        trainer = Trainer(model, train_dataset, val_dataset, train_config, device)

        # Resume from latest checkpoint
        ckpt = self._find_latest_checkpoint()
        if ckpt:
            logger.info("Resuming from checkpoint: %s", ckpt)
            trainer.load_checkpoint(str(ckpt))

        # ── Fine-tune for config.max_steps ────────────────────────
        start_step = trainer.step
        target_step = start_step + self.config.max_steps

        logger.info("Fine-tuning from step %d → %d", start_step, target_step)

        model.train()
        model.to(device)

        grad_accum_steps = train_config.batch_size // train_config.micro_batch_size
        final_loss = 0.0

        for batch_idx, batch in enumerate(trainer.train_loader):
            loss = trainer.train_step(batch)
            final_loss = loss

            if (batch_idx + 1) % grad_accum_steps == 0:
                # Gradient update — clip, step, zero
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.grad_clip,
                )
                trainer.optimizer.step()
                trainer.optimizer.zero_grad(set_to_none=True)
                trainer.step += 1

                if trainer.step % 10 == 0:
                    logger.info(
                        "Finetune step %d/%d | loss %.4f",
                        trainer.step, target_step, loss,
                    )

                # Periodic evaluation (uses Trainer.evaluate)
                if (
                    trainer.step % train_config.eval_interval == 0
                    and val_dataset is not None
                ):
                    val_loss = trainer.evaluate()
                    logger.info("Finetune val_loss: %.4f", val_loss)
                    if val_loss < trainer.best_val_loss:
                        trainer.best_val_loss = val_loss
                        trainer.save_checkpoint("best.pt")

                # Periodic save (uses Trainer.save_checkpoint)
                if trainer.step % train_config.save_interval == 0:
                    trainer.save_checkpoint(f"step_{trainer.step}.pt")

                # Stop after max_steps
                if trainer.step >= target_step:
                    break

        # Save final fine-tuned checkpoint (via Trainer)
        trainer.save_checkpoint("best.pt")
        best_path = str(Path(checkpoint_dir) / "best.pt")

        # Write finetune metadata
        meta_path = Path(checkpoint_dir) / "finetune_meta.json"
        meta = {
            "started_at_step": start_step,
            "finished_at_step": trainer.step,
            "new_pairs": self.config.max_steps,
            "final_loss": final_loss,
            "best_val_loss": trainer.best_val_loss,
            "client_id": client_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return {
            "steps": trainer.step - start_step,
            "final_loss": final_loss,
            "best_val_loss": trainer.best_val_loss,
            "checkpoint": best_path,
        }

    def _hot_reload(self, client_id: Optional[str] = None) -> None:
        """Reload the inference engine for the target version + client.

        Non-fatal: logs a warning if hot-reload fails.
        """
        try:
            import src.api.state as state
            if client_id:
                state.unload_client_model(client_id, self._version)
                state.load_client_model(client_id, self._version)
                logger.info(
                    "Client %s model hot-reloaded for FIX %s",
                    client_id, self._version,
                )
            else:
                state.unload_model(self._version)
                state.load_model(self._version)
                logger.info(
                    "Model hot-reloaded for FIX %s after fine-tuning",
                    self._version,
                )
        except Exception as exc:
            logger.warning("Hot-reload failed (non-fatal): %s", exc)
