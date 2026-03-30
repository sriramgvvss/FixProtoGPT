"""
Module: src.training.dataset
==============================

FIX Protocol training dataset.

Provides :class:`FixProtocolDataset`, a PyTorch
:class:`~torch.utils.data.Dataset` that produces overlapping
(input, target) token-ID pairs from pre-tokenised ``.bin`` files
or raw text.

Coding Standards
----------------
- PEP 8  : Python Style Guide — naming, spacing, line length ≤ 120
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.core.tokenizer import FixProtocolTokenizer


class FixProtocolDataset(Dataset):
    """Dataset for next-token prediction on FIX Protocol text.

    Supports two input formats:

    * **Binary (``.bin``)** — pre-tokenised token IDs stored as
      ``uint16`` via :func:`numpy.memmap`.
    * **Text (``.txt``)** — raw text that is tokenised on load.

    Each sample returns a contiguous window of ``max_seq_len`` input
    IDs and the offset-by-one target IDs used for language-model
    training.

    Args:
        data_path:   Path to ``.bin`` or ``.txt`` training data.
        tokenizer:   Loaded :class:`FixProtocolTokenizer`.
        max_seq_len: Context length (tokens per sample).
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: FixProtocolTokenizer,
        max_seq_len: int = 512,
    ) -> None:
        """Load training data from a .bin or .txt file and tokenize it into memory."""
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        data_path = Path(data_path)
        if data_path.suffix == ".bin":
            self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        else:
            with open(data_path, "r") as fh:
                texts = fh.readlines()
            all_tokens: list[int] = []
            for text in texts:
                all_tokens.extend(tokenizer.encode(text.strip()))
            self.data = np.array(all_tokens, dtype=np.uint16)

        print(f"Loaded dataset with {len(self.data)} tokens")

    def __len__(self) -> int:
        """Number of contiguous windows available."""
        return len(self.data) // self.max_seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(input_ids, target_ids)`` for window *idx*.

        Args:
            idx: Window index.

        Returns:
            Tuple of ``int64`` tensors each of length ``max_seq_len``.
        """
        start = idx * self.max_seq_len
        end = start + self.max_seq_len + 1
        chunk = self.data[start:end]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y
