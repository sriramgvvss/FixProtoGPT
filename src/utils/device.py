"""
Module: src.utils.device
=========================

Centralised PyTorch device detection and MPS configuration.

Eliminates duplicated ``cuda → mps → cpu`` fallback blocks that were
previously copy-pasted across training, fine-tuning, and inference
modules.

Author : FixProtoGPT Team
"""

from __future__ import annotations

import os


def configure_mps() -> None:
    """Allow MPS to use more system memory before falling back.

    Safe to call multiple times — uses ``os.environ.setdefault`` so it
    will not overwrite an explicitly-set value.
    """
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def detect_device() -> str:
    """Return the best available PyTorch device string.

    Priority:  ``cuda`` → ``mps`` (Apple Silicon) → ``cpu``.

    Returns:
        One of ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
