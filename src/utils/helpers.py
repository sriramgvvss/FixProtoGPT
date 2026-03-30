"""
Module: src.utils.helpers
==========================

Utility functions for FixProtoGPT.

Includes seed management, model parameter counting, memory estimation,
and an :class:`AverageMeter` for tracking metrics.

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
        'total_millions': total_params / 1e6,
        'trainable_millions': trainable_params / 1e6,
    }


def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate model size in MB
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with size information
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        'param_size_mb': param_size / (1024 ** 2),
        'buffer_size_mb': buffer_size / (1024 ** 2),
        'total_size_mb': total_size / (1024 ** 2),
    }


def format_time(seconds: float) -> str:
    """
    Format time in human-readable format
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f}m"
    else:
        return f"{seconds / 3600:.2f}h"


def save_config(config: Dict[str, Any], path: str):
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        path: Configuration file path
    
    Returns:
        Configuration dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


def estimate_memory_usage(
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_layers: int,
    vocab_size: int
) -> Dict[str, float]:
    """
    Estimate GPU memory usage for training
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model dimension
        n_layers: Number of layers
        vocab_size: Vocabulary size
    
    Returns:
        Dictionary with memory estimates in GB
    """
    # Activation memory (forward pass)
    activation_memory = batch_size * seq_len * d_model * n_layers * 4  # 4 bytes per float
    
    # Parameter memory
    # Rough estimate: embeddings + attention + FFN weights
    param_memory = (
        vocab_size * d_model +  # Token embeddings
        n_layers * (3 * d_model * d_model + 4 * d_model * d_model)  # Attention + FFN
    ) * 4  # 4 bytes per float
    
    # Gradient memory (same as parameters)
    gradient_memory = param_memory
    
    # Optimizer state (Adam): 2x parameters (momentum + variance)
    optimizer_memory = param_memory * 2
    
    # Total
    total_memory = activation_memory + param_memory + gradient_memory + optimizer_memory
    
    return {
        'activation_gb': activation_memory / (1024 ** 3),
        'parameter_gb': param_memory / (1024 ** 3),
        'gradient_gb': gradient_memory / (1024 ** 3),
        'optimizer_gb': optimizer_memory / (1024 ** 3),
        'total_gb': total_memory / (1024 ** 3),
    }


def print_model_summary(model: torch.nn.Module):
    """
    Print model summary
    
    Args:
        model: PyTorch model
    """
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    
    # Parameter counts
    param_counts = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Total: {param_counts['total']:,} ({param_counts['total_millions']:.2f}M)")
    print(f"  Trainable: {param_counts['trainable']:,} ({param_counts['trainable_millions']:.2f}M)")
    
    # Model size
    model_size = get_model_size(model)
    print(f"\nModel Size:")
    print(f"  Parameters: {model_size['param_size_mb']:.2f} MB")
    print(f"  Buffers: {model_size['buffer_size_mb']:.2f} MB")
    print(f"  Total: {model_size['total_size_mb']:.2f} MB")
    
    # Layer breakdown
    print(f"\nLayer Breakdown:")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {num_params:,} parameters")
    
    print("=" * 60 + "\n")


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        """Initialise meter and reset all accumulators."""
        self.reset()

    def reset(self) -> None:
        """Reset the meter to initial state."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """Record a new value.

        Args:
            val: The observed value.
            n: Batch size / weight for this observation.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test seed setting
    set_seed(42)
    print(f"Random numbers: {[random.random() for _ in range(3)]}")
    
    # Test time formatting
    print(f"45 seconds: {format_time(45)}")
    print(f"90 seconds: {format_time(90)}")
    print(f"7200 seconds: {format_time(7200)}")
    
    # Test memory estimation
    memory = estimate_memory_usage(
        batch_size=8,
        seq_len=512,
        d_model=512,
        n_layers=6,
        vocab_size=1024
    )
    print(f"\nEstimated memory usage: {memory['total_gb']:.2f} GB")
