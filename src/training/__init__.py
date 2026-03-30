"""Package: src.training
========================

Model training pipeline, dataset loading, and fine-tuning.

Coding Standards: PEP 8, PEP 257, PEP 484, Google Python Style Guide.
"""

from .dataset import FixProtocolDataset

__all__ = ["FixProtocolDataset"]
