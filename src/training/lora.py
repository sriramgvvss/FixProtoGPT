"""
Module: src.training.lora
==========================

Low-Rank Adaptation (LoRA) for efficient client-specific fine-tuning.

Wraps existing ``nn.Linear`` layers with low-rank adapters, freezing
the original weights and training only the small adapter matrices
``A`` and ``B`` (where the adapted weight is ``W + αBA``).

This reduces the number of trainable parameters by ~95% for
per-client fine-tuning while preserving the base model's capacity.

Usage::

    from src.training.lora import apply_lora, merge_lora, LoRAConfig

    config = LoRAConfig(rank=8, alpha=16, target_modules=["q_proj", "v_proj"])
    lora_model, lora_params = apply_lora(base_model, config)

    # Fine-tune only lora_params ...

    # Merge adapters back into base weights for deployment
    merge_lora(lora_model)

Reference:
    Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021.
    https://arxiv.org/abs/2106.09685

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
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation.

    Attributes:
        rank:            Rank of the low-rank matrices (r).
        alpha:           Scaling factor (α).  Effective scale = α/r.
        dropout:         Dropout applied to the LoRA path.
        target_modules:  List of submodule name patterns to adapt.
                          Default targets the attention projections.
        bias:            Whether to train biases (``"none"``, ``"lora_only"``,
                          or ``"all"``).
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "out_proj",
    ])
    bias: str = "none"  # "none", "lora_only", "all"


class LoRALinear(nn.Module):
    """A ``nn.Linear`` wrapper with a LoRA low-rank adapter.

    Computes::

        y = x @ W^T + b + (x @ A^T @ B^T) * scale

    where ``A ∈ R^{r×in}`` and ``B ∈ R^{out×r}`` are the trainable
    adapter matrices, and ``scale = α / r``.

    The original weight ``W`` (and optionally bias ``b``) are frozen.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        """Wrap an existing ``nn.Linear`` with a LoRA adapter.

        Args:
            original: The original linear layer to adapt.
            rank:     Low-rank dimension.
            alpha:    Scaling factor.
            dropout:  Dropout rate on the LoRA path.
        """
        super().__init__()

        self.in_features = original.in_features
        self.out_features = original.out_features
        self.rank = rank
        self.scale = alpha / rank

        # Freeze the original weight
        self.weight = original.weight
        self.weight.requires_grad = False
        self.bias = original.bias
        if self.bias is not None:
            self.bias.requires_grad = False

        # LoRA adapter matrices
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Initialise A with Kaiming uniform, B with zeros
        # This ensures the adapter starts as identity (no change)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B is already zeros

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Flag for merged state
        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: original linear + LoRA adapter.

        Args:
            x: Input tensor ``(..., in_features)``.

        Returns:
            Output tensor ``(..., out_features)``.
        """
        # Original path
        out = nn.functional.linear(x, self.weight, self.bias)

        if not self._merged:
            # LoRA path: x @ A^T @ B^T * scale
            lora_out = self.lora_dropout(x)
            lora_out = lora_out @ self.lora_A.T  # (..., rank)
            lora_out = lora_out @ self.lora_B.T  # (..., out_features)
            out = out + lora_out * self.scale

        return out

    def merge(self) -> None:
        """Merge LoRA weights into the original weight for inference.

        After merging, the LoRA path is skipped, and the adapted weight
        is used directly.  This is more efficient for deployment.
        """
        if self._merged:
            return
        with torch.no_grad():
            # W' = W + scale * B @ A
            self.weight.add_(self.scale * (self.lora_B @ self.lora_A))
        self._merged = True

    def unmerge(self) -> None:
        """Reverse a previous merge (for continued fine-tuning)."""
        if not self._merged:
            return
        with torch.no_grad():
            self.weight.sub_(self.scale * (self.lora_B @ self.lora_A))
        self._merged = False


# ── Application utilities ─────────────────────────────────────────


def apply_lora(
    model: nn.Module,
    config: LoRAConfig,
) -> Tuple[nn.Module, List[nn.Parameter]]:
    """Apply LoRA adapters to target modules in the model.

    Replaces matched ``nn.Linear`` layers with :class:`LoRALinear`
    wrappers.  The original weights are frozen; only the adapter
    parameters (and optionally biases) are trainable.

    Args:
        model:  The base model to adapt.
        config: LoRA configuration.

    Returns:
        Tuple of ``(adapted_model, lora_parameters)`` where
        ``lora_parameters`` is a flat list of all trainable adapter
        parameters (for the optimizer).
    """
    adapted_count = 0
    lora_params: List[nn.Parameter] = []

    # First, freeze all base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Then apply LoRA to target modules
    for name, module in model.named_modules():
        for target in config.target_modules:
            if target in name and isinstance(module, nn.Linear):
                # Get parent module and attribute name
                parent_name, attr_name = _split_module_name(name)
                parent = _get_module_by_name(model, parent_name) if parent_name else model

                # Replace with LoRA wrapper
                lora_layer = LoRALinear(
                    module,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                setattr(parent, attr_name, lora_layer)

                lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
                adapted_count += 1

                logger.debug("Applied LoRA to %s", name)
                break

    # Optionally unfreeze biases
    if config.bias == "all":
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
                lora_params.append(param)
    elif config.bias == "lora_only":
        for mod in model.modules():
            if isinstance(mod, LoRALinear) and mod.bias is not None:
                mod.bias.requires_grad = True
                lora_params.append(mod.bias)

    total_lora = sum(p.numel() for p in lora_params)
    total_all = sum(p.numel() for p in model.parameters())
    pct = total_lora / max(total_all, 1) * 100

    print(f"LoRA applied: {adapted_count} layers adapted")
    print(f"  Trainable params: {total_lora:,} / {total_all:,} ({pct:.2f}%)")

    return model, lora_params


def merge_lora(model: nn.Module) -> None:
    """Merge all LoRA adapters into base weights for deployment.

    After merging, the model behaves as a regular model with no
    adapter overhead.

    Args:
        model: Model with LoRA adapters applied.
    """
    merged_count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            merged_count += 1
    logger.info("Merged %d LoRA layers into base weights", merged_count)


def unmerge_lora(model: nn.Module) -> None:
    """Unmerge all LoRA adapters (reverse of :func:`merge_lora`).

    Args:
        model: Model with merged LoRA adapters.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
            count += 1
    logger.info("Unmerged %d LoRA layers", count)


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only the LoRA adapter weights to a file.

    Args:
        model: Model with LoRA adapters.
        path:  File path to save to (``.pt``).
    """
    lora_state: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.data.clone()
            lora_state[f"{name}.lora_B"] = module.lora_B.data.clone()

    torch.save(lora_state, path)
    logger.info("Saved LoRA weights (%d tensors) to %s", len(lora_state), path)


def load_lora_weights(model: nn.Module, path: str) -> None:
    """Load LoRA adapter weights from a file.

    The model must already have LoRA adapters applied (via :func:`apply_lora`).

    Args:
        model: Model with LoRA adapters.
        path:  File path to load from.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in state:
                module.lora_A.data.copy_(state[a_key])
                loaded += 1
            if b_key in state:
                module.lora_B.data.copy_(state[b_key])

    logger.info("Loaded LoRA weights for %d layers from %s", loaded, path)


def get_lora_param_count(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters in a LoRA-adapted model.

    Args:
        model: The model.

    Returns:
        Dict with ``trainable``, ``frozen``, ``total``, ``lora_layers``.
    """
    trainable = 0
    frozen = 0
    lora_layers = 0

    for p in model.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()

    for m in model.modules():
        if isinstance(m, LoRALinear):
            lora_layers += 1

    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": trainable + frozen,
        "lora_layers": lora_layers,
    }


# ── Internal helpers ──────────────────────────────────────────────


def _split_module_name(name: str) -> Tuple[str, str]:
    """Split ``a.b.c`` into ``(\"a.b\", \"c\")``."""
    parts = name.rsplit(".", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])


def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """Get a submodule by dotted name."""
    parts = name.split(".")
    current = model
    for part in parts:
        current = getattr(current, part)
    return current
