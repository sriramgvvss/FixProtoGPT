"""
Module: src.training.train
===========================

Training pipeline for FixProtoGPT.

Defines the training configuration (:class:`TrainConfig`), learning-rate
schedule (:func:`get_lr`), optimizer factory (:func:`configure_optimizers`),
the main :class:`Trainer` loop, and the CLI entry-point.

The dataset class lives in :mod:`src.training.dataset` and is re-exported
here for backward compatibility.

Coding Standards
----------------
- PEP 8  : Python Style Guide — naming, spacing, line length ≤ 120
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils import paths

# Force unbuffered output so log files are written in real-time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from src.utils.device import configure_mps, detect_device
configure_mps()

from src.core.transformer import FixProtoGPT, ModelConfig
from src.core.tokenizer import FixProtocolTokenizer
from src.data.version_detector import FIXVersionDetector

# Re-export for backward compatibility (finetune.py, etc.)
from src.training.dataset import FixProtocolDataset  # noqa: F401


@dataclass
class TrainConfig:
    """Training hyper-parameters and runtime settings.

    Attributes:
        batch_size:              Effective batch size (via gradient accumulation).
        micro_batch_size:        Per-step (GPU) batch size.
        max_epochs:              Maximum training epochs.
        learning_rate:           Peak learning rate.
        weight_decay:            AdamW weight-decay coefficient.
        beta1:                   Adam β₁.
        beta2:                   Adam β₂.
        grad_clip:               Max gradient norm for clipping.
        warmup_steps:            Linear warmup duration.
        lr_schedule:             ``"cosine"`` or ``"constant"``.
        min_lr:                  Minimum learning rate after decay.
        use_mixed_precision:     Enable AMP (CUDA only).
        compile_model:           Use ``torch.compile`` (PyTorch 2.0+).
        save_interval:           Steps between checkpoint saves.
        eval_interval:           Steps between validation evaluations.
        max_checkpoints_to_keep: Retain only the *N* most recent step files.
        emergency_save_on_error: Save a checkpoint on unhandled exceptions.
        checkpoint_dir:          Root checkpoint directory.
        log_dir:                 Logging directory.
    """

    # Training
    batch_size: int = 8
    micro_batch_size: int = 2
    max_epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 1000

    # Learning rate schedule
    lr_schedule: str = "cosine"
    min_lr: float = 3e-5

    # Optimisation
    use_mixed_precision: bool = True
    compile_model: bool = False

    # Checkpointing
    save_interval: int = 500
    eval_interval: int = 100
    max_checkpoints_to_keep: int = 2

    # Crash recovery
    emergency_save_on_error: bool = True

    # Early stopping
    early_stopping_patience: int = 0  # 0 = disabled; >0 = stop after N evals with no improvement
    target_val_loss: float = 0.0      # 0 = disabled; >0 = stop when val loss reaches this target

    # Paths
    checkpoint_dir: str = "model_store/checkpoints"
    log_dir: str = "logs"

    @classmethod
    def from_yaml(
        cls,
        path: str | Path = "config/model_config.yaml",
        **overrides,
    ) -> "TrainConfig":
        """Construct a ``TrainConfig`` from the ``training`` section of a YAML file.

        Args:
            path:       Path to the YAML config file.
            **overrides: Keyword overrides applied after loading
                         (e.g. ``early_stopping_patience=10``).

        Returns:
            Populated :class:`TrainConfig` instance.
        """
        with open(path) as f:
            cfg = yaml.safe_load(f)["training"]
        kwargs = {
            "batch_size": cfg["batch_size"],
            "micro_batch_size": cfg["micro_batch_size"],
            "max_epochs": cfg["max_epochs"],
            "learning_rate": cfg["learning_rate"],
            "weight_decay": cfg["weight_decay"],
            "beta1": cfg["beta1"],
            "beta2": cfg["beta2"],
            "grad_clip": cfg["grad_clip"],
            "warmup_steps": cfg["warmup_steps"],
            "save_interval": cfg.get("save_interval", 500),
            "eval_interval": cfg.get("eval_interval", 100),
            "early_stopping_patience": cfg.get("early_stopping_patience", 0),
            "target_val_loss": cfg.get("target_val_loss", 0.0),
        }
        kwargs.update(overrides)
        return cls(**kwargs)


def get_lr(step: int, config: TrainConfig, max_steps: int) -> float:
    """Calculate the learning rate with linear warmup and cosine decay.

    Args:
        step:      Current training step.
        config:    Training configuration.
        max_steps: Total number of training steps.

    Returns:
        Current learning rate.
    """
    # Warmup
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    
    # Cosine decay
    if config.lr_schedule == "cosine":
        decay_ratio = (step - config.warmup_steps) / (max_steps - config.warmup_steps)
        decay_ratio = max(0.0, min(1.0, decay_ratio))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)
    
    # Constant learning rate
    return config.learning_rate


def configure_optimizers(
    model: nn.Module,
    config: TrainConfig,
) -> torch.optim.Optimizer:
    """Configure an AdamW optimiser with per-parameter weight decay.

    Biases and LayerNorm parameters receive zero weight decay.

    Args:
        model:  The model to optimise.
        config: Training configuration.

    Returns:
        Configured :class:`torch.optim.AdamW` instance.
    """
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No weight decay for biases and layer norms
        if param.ndim < 2 or 'ln' in name or 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        fused=True if torch.cuda.is_available() else False  # Fused optimizer for speed
    )
    
    print(f"Optimizer configured: {len(decay_params)} decay params, {len(no_decay_params)} no-decay params")
    
    return optimizer


class Trainer:
    """Full training loop for FixProtoGPT.

    Manages gradient accumulation, mixed-precision, checkpointing,
    evaluation, and graceful shutdown via signal handlers.

    Args:
        model:         The :class:`FixProtoGPT` model instance.
        train_dataset: Training :class:`~torch.utils.data.Dataset`.
        val_dataset:   Optional validation dataset.
        config:        :class:`TrainConfig` instance.
        device:        Compute device string.
    """
    
    def __init__(
        self,
        model: FixProtoGPT,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        config: TrainConfig,
        device: str = "cuda"
    ):
        """Initialise trainer with model, datasets, config, and target device."""
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Compile model (PyTorch 2.0+)
        if config.compile_model:
            print("Compiling model...")
            self.model = torch.compile(self.model)
        
        # Setup optimizer
        self.optimizer = configure_optimizers(model, config)
        
        # Setup data loaders
        # Use pin_memory with CUDA; use multiprocessing workers on all platforms
        use_pin_memory = device == "cuda"
        if device == "cuda":
            n_workers = 4
        elif device == "mps":
            n_workers = 4   # MPS benefits from parallel data loading
        else:
            n_workers = 2   # CPU-only: still use workers for prefetching

        loader_kwargs = dict(
            num_workers=n_workers,
            pin_memory=use_pin_memory,
        )
        if n_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.micro_batch_size,
            shuffle=True,
            **loader_kwargs,
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.micro_batch_size,
                shuffle=False,
                **loader_kwargs,
            )
        
        # Mixed precision training (only works on CUDA)
        use_amp = config.use_mixed_precision and device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
        self.use_amp = use_amp
        self.amp_device = "cuda" if device == "cuda" else "cpu"
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self._shutting_down = False
        self._bundle_done = False  # Track if we already bundled data
        self._early_stop_counter = 0  # Counts evals with no improvement
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, _frame):
        """Handle interrupt signals — save checkpoint before exiting."""
        sig_name = signal.Signals(signum).name
        print(f"\n⚠ Received {sig_name}. Saving emergency checkpoint...")
        self._shutting_down = True
        try:
            self.save_checkpoint(f"emergency_step_{self.step}.pt")
            print(f"Emergency checkpoint saved at step {self.step}.")
        except Exception as e:
            print(f"Failed to save emergency checkpoint: {e}")
        sys.exit(0)
    
    def _clear_mps_cache(self):
        """Clear MPS memory cache periodically to prevent OOM."""
        if self.device == "mps" and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    
    def _cleanup_old_checkpoints(self):
        """Keep only the N most recent step_*.pt files. best.pt and emergency checkpoints are never deleted."""
        import re as _re
        try:
            detector = FIXVersionDetector()
            checkpoint_dir = detector.get_checkpoint_dir(self.config.checkpoint_dir)
        except Exception:
            checkpoint_dir = Path(self.config.checkpoint_dir)
        
        # Find all step_*.pt files (not best.pt, not emergency_*)
        step_files = []
        for f in checkpoint_dir.glob("step_*.pt"):
            m = _re.match(r"step_(\d+)\.pt", f.name)
            if m:
                step_files.append((int(m.group(1)), f))
        
        # Sort by step number descending and delete old ones
        step_files.sort(key=lambda x: x[0], reverse=True)
        keep = self.config.max_checkpoints_to_keep
        for _, fpath in step_files[keep:]:
            fpath.unlink()
            print(f"Cleaned up old checkpoint: {fpath.name}")
        
        # Also clean up emergency checkpoints older than the latest step checkpoint
        if step_files:
            latest_step = step_files[0][0]
            for f in checkpoint_dir.glob("emergency_step_*.pt"):
                m = _re.match(r"emergency_step_(\d+)\.pt", f.name)
                if m and int(m.group(1)) < latest_step:
                    f.unlink()
                    print(f"Cleaned up old emergency checkpoint: {f.name}")
    
    def train_step(self, batch: tuple) -> float:
        """Execute a single forward + backward pass.

        Args:
            batch: ``(input_ids, target_ids)`` tensors.

        Returns:
            Scalar loss value for this micro-batch.
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        # Forward pass with mixed precision
        with torch.amp.autocast(self.amp_device, enabled=self.use_amp):
            logits, loss, _ = self.model(x, targets=y)
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item()
    
    def train_epoch(self) -> float:
        """Train for one full epoch.

        Returns:
            Average training loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Calculate gradient accumulation steps
        grad_accum_steps = self.config.batch_size // self.config.micro_batch_size
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Train step
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                # Clip gradients
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                # Update weights
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                
                # Update learning rate
                max_steps = len(self.train_loader) * self.config.max_epochs // grad_accum_steps
                lr = get_lr(self.step, self.config, max_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
                self.step += 1
                
                # Logging
                if self.step % 10 == 0:
                    print(f"Step {self.step} | Loss: {loss:.4f} | LR: {lr:.6f}")
                
                # Clear MPS cache periodically to prevent memory buildup
                if self.step % 50 == 0:
                    self._clear_mps_cache()
                
                # Evaluation
                if self.step % self.config.eval_interval == 0 and self.val_dataset:
                    val_loss = self.evaluate()
                    print(f"Step {self.step} | Val Loss: {val_loss:.4f}")
                    
                    # Check target val loss threshold
                    if self.config.target_val_loss > 0 and val_loss <= self.config.target_val_loss:
                        print(f"Target val loss {self.config.target_val_loss:.4f} reached! (actual: {val_loss:.4f})")
                        self.save_checkpoint(f"target_step_{self.step}.pt")
                        self._shutting_down = True

                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._early_stop_counter = 0
                        self.save_checkpoint("best.pt")
                    else:
                        # Track early stopping patience
                        if self.config.early_stopping_patience > 0:
                            self._early_stop_counter += 1
                            print(
                                f"  No improvement for {self._early_stop_counter}"
                                f"/{self.config.early_stopping_patience} evals"
                            )
                            if self._early_stop_counter >= self.config.early_stopping_patience:
                                print("Early stopping triggered — saving and exiting.")
                                self.save_checkpoint(f"early_stop_step_{self.step}.pt")
                                self._shutting_down = True
                
                # Save checkpoint
                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint(f"step_{self.step}.pt")
                    self._cleanup_old_checkpoints()
                
                # Check for shutdown request
                if self._shutting_down:
                    return total_loss / max(num_batches, 1)
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on the validation set.

        Returns:
            Average validation loss, or ``inf`` if no val set.
        """
        if not self.val_dataset:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            
            with torch.amp.autocast(self.amp_device, enabled=self.use_amp):
                logits, loss, _ = self.model(x, targets=y)
            
            total_loss += loss.item()
            num_batches += 1
        
        self.model.train()
        return total_loss / num_batches
    
    def train(self) -> None:
        """Run the full training loop with crash recovery.

        Iterates over epochs, evaluates periodically, and saves
        emergency checkpoints on unhandled exceptions.
        """
        start_epoch = self.epoch
        print(f"Starting training for {self.config.max_epochs} epochs...")
        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch + 1}, step {self.step}")
        print(f"Total steps: ~{len(self.train_loader) * self.config.max_epochs // (self.config.batch_size // self.config.micro_batch_size)}")
        
        start_time = time.time()
        
        try:
            for epoch in range(start_epoch, self.config.max_epochs):
                self.epoch = epoch
                print(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
                
                epoch_loss = self.train_epoch()
                print(f"Epoch {epoch + 1} | Train Loss: {epoch_loss:.4f}")
                
                # Evaluate
                if self.val_dataset:
                    val_loss = self.evaluate()
                    print(f"Epoch {epoch + 1} | Val Loss: {val_loss:.4f}")
                
                if self._shutting_down:
                    print("Shutdown requested. Stopping training.")
                    break
        except Exception as e:
            print(f"\n✗ Training error at step {self.step}: {e}")
            traceback.print_exc()
            if self.config.emergency_save_on_error:
                print("Saving emergency checkpoint...")
                try:
                    self.save_checkpoint(f"emergency_step_{self.step}.pt")
                    print(f"Emergency checkpoint saved at step {self.step}.")
                except Exception as save_err:
                    print(f"Failed to save emergency checkpoint: {save_err}")
            raise
        
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed / 3600:.2f} hours")
        
        # Always save final checkpoint with bundled data
        self.save_checkpoint("final.pt")
        print("Final checkpoint saved with training data bundle.")
    
    def save_checkpoint(self, filename: str, use_versioning: bool = True):
        """Save model checkpoint with full training dataset bundle.
        
        Each checkpoint directory contains:
          - model weights (.pt file)
          - tokenizer/ (vocab.json, merges.pkl, fix_tags.json)
          - data/ (train.bin, val.bin, metadata.json)
          - config/ (model_config.yaml)
          - checkpoint_meta.json (training state summary)
        """
        if use_versioning:
            try:
                detector = FIXVersionDetector()
                version_metadata = detector.get_version_metadata()
                checkpoint_dir = detector.get_checkpoint_dir(self.config.checkpoint_dir)
            except Exception as e:
                print(f"Warning: Version detection failed ({e}), using default directory")
                checkpoint_dir = Path(self.config.checkpoint_dir)
        else:
            checkpoint_dir = Path(self.config.checkpoint_dir)
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / filename
        
        # --- Save model weights ---
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        if use_versioning:
            try:
                checkpoint['fix_version'] = version_metadata['primary_version']
                checkpoint['fix_versions_detected'] = version_metadata['detected_versions']
            except:
                pass
        
        torch.save(checkpoint, checkpoint_path)
        
        # --- Bundle training data into checkpoint dir ---
        self._bundle_training_data(checkpoint_dir)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def _bundle_training_data(self, checkpoint_dir: Path):
        """Copy tokenizer, training data, and config into the checkpoint directory.
        
        Only does a full copy on the first save to avoid heavy I/O on every checkpoint.
        Subsequent saves only update the metadata.
        """
        if self._bundle_done:
            # Skip heavy I/O — data hasn't changed since last bundle
            return
        
        project_root = Path(__file__).parent.parent
        
        # 1. Copy tokenizer
        tok_src = project_root / "data" / "processed" / "tokenizer"
        tok_dst = checkpoint_dir / "tokenizer"
        if tok_src.exists():
            if tok_dst.exists():
                shutil.rmtree(tok_dst)
            shutil.copytree(tok_src, tok_dst)
        
        # 2. Copy training data (bin files + metadata)
        data_dst = checkpoint_dir / "data"
        data_dst.mkdir(parents=True, exist_ok=True)
        data_src = project_root / "data" / "processed"
        for fname in ["train.bin", "val.bin", "metadata.json"]:
            src = data_src / fname
            if src.exists():
                shutil.copy2(src, data_dst / fname)
        
        # 3. Copy model config
        cfg_dst = checkpoint_dir / "config"
        cfg_dst.mkdir(parents=True, exist_ok=True)
        cfg_src = project_root / "config" / "model_config.yaml"
        if cfg_src.exists():
            shutil.copy2(cfg_src, cfg_dst / "model_config.yaml")
        
        self._bundle_done = True
        
        # 4. Write checkpoint metadata summary
        meta = {
            "step": self.step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "train_samples": len(self.train_dataset) if self.train_dataset else 0,
            "val_samples": len(self.val_dataset) if self.val_dataset else 0,
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            detector = FIXVersionDetector()
            vmeta = detector.get_version_metadata()
            meta["fix_version"] = vmeta.get("primary_version", "unknown")
            meta["fix_versions_trained"] = vmeta.get("detected_versions", [meta["fix_version"]])
        except Exception:
            meta["fix_version"] = paths.active_version()
            meta["fix_versions_trained"] = ["4.2", "4.4", "5.0SP2", "Latest"]
        
        with open(checkpoint_dir / "checkpoint_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        If the checkpoint directory contains bundled tokenizer/data/config,
        those paths are logged so the user can restore the full training state.
        """
        checkpoint_path = Path(checkpoint_path)
        # Use weights_only with safe globals for TrainConfig
        try:
            import torch.serialization
            with torch.serialization.safe_globals([TrainConfig]):
                checkpoint = torch.load(
                    checkpoint_path, map_location=self.device, weights_only=True,
                )
        except Exception:
            # Fallback for very old checkpoints
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False,
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Report bundled assets
        ckpt_dir = checkpoint_path.parent
        bundled = []
        if (ckpt_dir / "tokenizer").exists():
            bundled.append("tokenizer")
        if (ckpt_dir / "data").exists():
            bundled.append("training data")
        if (ckpt_dir / "config").exists():
            bundled.append("config")
        if (ckpt_dir / "checkpoint_meta.json").exists():
            with open(ckpt_dir / "checkpoint_meta.json") as f:
                meta = json.load(f)
            print(f"  FIX version: {meta.get('fix_version', 'unknown')}")
            print(f"  Saved at: {meta.get('saved_at', 'unknown')}")
        
        if bundled:
            print(f"  Bundled assets: {', '.join(bundled)}")
        
        print(f"Checkpoint loaded: {checkpoint_path} (step={self.step}, epoch={self.epoch})")


def main() -> None:
    """CLI entry-point for training."""
    import argparse
    parser = argparse.ArgumentParser(description="FixProtoGPT Training")
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint .pt file to resume training from')
    args = parser.parse_args()

    # Load config
    model_config = ModelConfig.from_yaml()
    train_config = TrainConfig.from_yaml()
    
    # Initialize tokenizer
    print("\nLoading tokenizer...")
    tokenizer = FixProtocolTokenizer(vocab_size=model_config.vocab_size)
    tokenizer.load(str(paths.tokenizer_dir()))
    print(f"Tokenizer loaded with vocab size: {len(tokenizer.token_to_id)}")
    
    # Create model
    from src.core.transformer import create_model
    model = create_model(model_config)
    
    # Setup device (prefer CUDA > MPS > CPU)
    device = detect_device()
    print(f"\nUsing device: {device}")
    
    # Log model size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Create datasets
    print("\nLoading training data...")
    train_path = config_dict['data'].get('train_path') or str(paths.train_data())
    train_dataset = FixProtocolDataset(
        train_path,
        tokenizer,
        model_config.max_seq_len
    )
    print(f"Training samples: {len(train_dataset)}")
    
    val_path = config_dict['data'].get('val_path') or str(paths.val_data())
    val_dataset = FixProtocolDataset(
        val_path,
        tokenizer,
        model_config.max_seq_len
    )
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(model, train_dataset, val_dataset, train_config, device)
    
    # Resume from checkpoint if specified, or auto-find the latest one
    if args.resume:
        resume_path = args.resume
    else:
        # Auto-find the latest checkpoint
        resume_path = find_latest_checkpoint(train_config.checkpoint_dir)
    
    if resume_path:
        print(f"\nResuming from checkpoint: {resume_path}")
        trainer.load_checkpoint(resume_path)
    
    # Start training
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    trainer.train()
    
    print("\n✓ Training completed!")


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint by step number across all version dirs.

    Searches recursively for ``step_*.pt`` and ``emergency_step_*.pt``
    files, returning the path with the highest step number.

    Args:
        checkpoint_dir: Root checkpoint directory to search.

    Returns:
        Absolute path string of the latest checkpoint, or ``None``.
    """
    ckpt_root = Path(checkpoint_dir)
    if not ckpt_root.exists():
        return None
    
    best_step = -1
    best_path = None
    
    # Search all subdirectories for step_*.pt and emergency_step_*.pt files
    for pt_file in ckpt_root.rglob("*.pt"):
        # Skip backup directories
        if "backup" in str(pt_file):
            continue
        name = pt_file.stem
        if name.startswith("step_") or name.startswith("emergency_step_"):
            try:
                step = int(name.split("_")[-1])
                if step > best_step:
                    best_step = step
                    best_path = str(pt_file)
            except ValueError:
                continue
    
    if best_path:
        print(f"Auto-detected latest checkpoint: {best_path} (step {best_step})")
    return best_path


if __name__ == "__main__":
    main()
