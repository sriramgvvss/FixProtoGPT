"""
Module: src.utils.quantization
================================

Model quantization utilities for FixProtoGPT.

Provides dynamic INT8 quantization for faster CPU inference and
reduced model size, as well as half-precision (FP16) conversion
for GPU inference.

Supported strategies:
    - **Dynamic INT8** (CPU): ``torch.quantization.quantize_dynamic``
      — quantises ``nn.Linear`` layers on-the-fly at inference time.
    - **FP16** (CUDA/MPS): Simple ``model.half()`` conversion.
    - **Static INT8** (experimental): calibration-based quantisation.

Usage::

    from src.utils.quantization import quantize_model, QuantizationConfig

    config = QuantizationConfig(strategy="dynamic_int8")
    quant_model = quantize_model(model, config)

    # Run inference as normal — the model is now quantised
    logits, _, _ = quant_model(input_ids)

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
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization.

    Attributes:
        strategy:     ``"dynamic_int8"``, ``"fp16"``, or ``"none"``.
        target_layers: Tuple of module types to quantise (dynamic only).
        device:       Target device after quantisation.
    """

    strategy: str = "dynamic_int8"
    target_layers: tuple = (nn.Linear,)
    device: str = "cpu"


def quantize_model(
    model: nn.Module,
    config: Optional[QuantizationConfig] = None,
) -> nn.Module:
    """Quantize a model according to the given configuration.

    The model is returned in eval mode.  The original model object
    may be modified in place for some strategies.

    Args:
        model:  The model to quantise (should be in eval mode).
        config: Quantisation configuration.

    Returns:
        The quantised model (may be the same object or a new one).
    """
    if config is None:
        config = QuantizationConfig()

    model.eval()

    if config.strategy == "dynamic_int8":
        return _dynamic_int8(model, config)
    elif config.strategy == "fp16":
        return _fp16(model, config)
    elif config.strategy == "none":
        return model
    else:
        logger.warning("Unknown quantization strategy: %s", config.strategy)
        return model


def _dynamic_int8(model: nn.Module, config: QuantizationConfig) -> nn.Module:
    """Apply PyTorch dynamic INT8 quantization.

    Replaces ``nn.Linear`` modules with quantised versions that
    compute INT8 matrix multiplications at inference time.

    Only works on CPU.

    Args:
        model:  Model in eval mode.
        config: Quantisation config.

    Returns:
        Quantised model on CPU.
    """
    model = model.cpu()

    t0 = time.time()
    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec=set(config.target_layers),
        dtype=torch.qint8,
    )
    elapsed = time.time() - t0

    # Report size reduction
    orig_size = _model_size_mb(model)
    quant_size = _model_size_mb(quantized)
    reduction = (1.0 - quant_size / max(orig_size, 0.01)) * 100

    print(f"Dynamic INT8 quantization complete ({elapsed:.2f}s)")
    print(f"  Original size: {orig_size:.2f} MB")
    print(f"  Quantized size: {quant_size:.2f} MB")
    print(f"  Reduction: {reduction:.1f}%")

    return quantized


def _fp16(model: nn.Module, config: QuantizationConfig) -> nn.Module:
    """Convert model to FP16 (half precision).

    Useful for GPU/MPS inference with ~2x memory reduction.

    Args:
        model:  Model in eval mode.
        config: Quantisation config.

    Returns:
        FP16 model on the target device.
    """
    device = config.device
    if device == "cpu":
        logger.warning("FP16 on CPU may be slower than FP32 — consider dynamic_int8")

    model = model.half().to(device)

    orig_size = _model_size_mb(model) * 2  # Approximate FP32 size
    fp16_size = _model_size_mb(model)

    print(f"FP16 conversion complete")
    print(f"  Approximate size: {fp16_size:.2f} MB (was ~{orig_size:.2f} MB)")

    return model


# ── Benchmarking ──────────────────────────────────────────────────


def benchmark_inference(
    model: nn.Module,
    input_ids: torch.Tensor,
    num_runs: int = 10,
    warmup_runs: int = 3,
) -> Dict[str, Any]:
    """Benchmark inference latency for the model.

    Args:
        model:       Model to benchmark.
        input_ids:   Sample input tensor.
        num_runs:    Number of timed iterations.
        warmup_runs: Warmup iterations before timing.

    Returns:
        Dict with ``mean_ms``, ``std_ms``, ``min_ms``, ``max_ms``.
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Handle FP16 model input
    timings = []

    with torch.no_grad():
        # Warmup
        for _ in range(warmup_runs):
            _ = model(input_ids)

        # Timed runs
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_ids)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # ms

    import numpy as np
    timings_arr = np.array(timings)

    return {
        "mean_ms": round(float(timings_arr.mean()), 2),
        "std_ms": round(float(timings_arr.std()), 2),
        "min_ms": round(float(timings_arr.min()), 2),
        "max_ms": round(float(timings_arr.max()), 2),
        "num_runs": num_runs,
    }


# ── Model I/O ────────────────────────────────────────────────────


def save_quantized_model(model: nn.Module, path: str) -> None:
    """Save a quantised model to disk.

    Args:
        model: The quantised model.
        path:  Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))
    size_mb = os.path.getsize(str(path)) / (1024 * 1024)
    print(f"Quantized model saved: {path} ({size_mb:.2f} MB)")


def get_model_stats(model: nn.Module) -> Dict[str, Any]:
    """Get summary statistics for a model.

    Args:
        model: The model to inspect.

    Returns:
        Dict with ``total_params``, ``size_mb``, ``dtype``,
        ``quantized_layers``, ``device``.
    """
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = _model_size_mb(model)

    # Count quantized layers
    quantized_layers = 0
    for module in model.modules():
        module_name = type(module).__name__
        if "Quantized" in module_name or "Dynamic" in module_name:
            quantized_layers += 1

    # Determine dominant dtype
    dtypes = set()
    for p in model.parameters():
        dtypes.add(str(p.dtype))

    device = "unknown"
    try:
        device = str(next(model.parameters()).device)
    except StopIteration:
        pass

    return {
        "total_params": total_params,
        "size_mb": round(size_mb, 2),
        "dtypes": list(dtypes),
        "quantized_layers": quantized_layers,
        "device": device,
    }


# ── Helpers ───────────────────────────────────────────────────────


def _model_size_mb(model: nn.Module) -> float:
    """Estimate model size in megabytes from parameter storage."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        total_bytes += b.nelement() * b.element_size()
    return total_bytes / (1024 * 1024)
