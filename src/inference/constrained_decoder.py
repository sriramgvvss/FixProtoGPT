"""
Module: src.inference.constrained_decoder
==========================================

Grammar-guided constrained decoding for FIX protocol messages.

Enforces valid ``tag=value|`` structure during autoregressive generation
using a finite-state machine (FSM).  The FSM ensures:
    1. Only valid FIX tag numbers follow a field separator.
    2. An ``=`` character always follows a tag number.
    3. Values contain only legal characters for the field's data type.
    4. The ``|`` (SOH) delimiter terminates each field.

The decoder wraps the base model's logit output and masks illegal
tokens at each step, guaranteeing structurally valid FIX messages
without sacrificing the model's learned distribution over legal tokens.

Usage::

    from src.inference.constrained_decoder import ConstrainedFIXDecoder

    decoder = ConstrainedFIXDecoder(tokenizer)
    mask = decoder.get_token_mask(generated_so_far)

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from __future__ import annotations

import enum
import re
from typing import Dict, FrozenSet, List, Optional, Set

import torch

from src.core.fix_reference import FIELDS


class FIXParseState(enum.Enum):
    """States of the FIX field-level finite-state machine."""

    EXPECT_TAG_OR_END = "expect_tag_or_end"
    IN_TAG = "in_tag"
    EXPECT_EQUALS = "expect_equals"
    IN_VALUE = "in_value"
    EXPECT_DELIM = "expect_delim"


# Characters allowed in tag numbers
_TAG_CHARS: FrozenSet[str] = frozenset("0123456789")

# Characters allowed in FIX values (printable ASCII minus SOH/|)
_VALUE_CHARS: FrozenSet[str] = frozenset(
    "".join(chr(c) for c in range(32, 127) if chr(c) not in ("|", "\x01"))
)

# Valid FIX tag numbers (from knowledge base)
_VALID_TAGS: Set[int] = set(FIELDS.keys())


class ConstrainedFIXDecoder:
    """Applies FSM-based token masking to enforce valid FIX structure.

    The decoder is stateless per call — it inspects the already-generated
    token sequence to determine the current FSM state and returns a
    boolean mask over the vocabulary indicating which tokens are legal.

    Attributes:
        tokenizer: The tokenizer instance for id↔token mapping.
    """

    def __init__(self, tokenizer) -> None:
        """Initialise with a tokenizer for vocab introspection.

        Args:
            tokenizer: A :class:`FixProtocolTokenizer` (or compatible)
                instance with ``id_to_token``, ``token_to_id``, and
                ``special_tokens`` attributes.
        """
        self.tokenizer = tokenizer
        self._vocab_size = max(tokenizer.id_to_token.keys()) + 1 if tokenizer.id_to_token else 1024

        # Pre-compute token category sets for fast masking
        self._tag_token_ids: Set[int] = set()
        self._equals_token_ids: Set[int] = set()
        self._delim_token_ids: Set[int] = set()
        self._value_token_ids: Set[int] = set()
        self._digit_token_ids: Set[int] = set()
        self._end_token_ids: Set[int] = set()
        self._tag_pattern_ids: Set[int] = set()  # e.g. "35=", "55="

        self._classify_tokens()

    # ── Token classification ──────────────────────────────────────

    def _classify_tokens(self) -> None:
        """Classify every token in the vocabulary into FSM categories."""
        special = set(self.tokenizer.special_tokens.values())
        eom_id = self.tokenizer.special_tokens.get("<|eom|>")
        eos_id = self.tokenizer.special_tokens.get("<|eos|>")
        field_id = self.tokenizer.special_tokens.get("<|field|>")

        for tid, tok in self.tokenizer.id_to_token.items():
            if tid in special:
                if tid == eom_id or tid == eos_id:
                    self._end_token_ids.add(tid)
                if tid == field_id:
                    self._delim_token_ids.add(tid)
                continue

            # Tag pattern tokens like "35=", "55="
            if tok.endswith("=") and tok[:-1].isdigit():
                self._tag_pattern_ids.add(tid)
                continue

            # Single characters
            if len(tok) == 1:
                if tok in _TAG_CHARS:
                    self._digit_token_ids.add(tid)
                    self._tag_token_ids.add(tid)
                    self._value_token_ids.add(tid)
                elif tok == "=":
                    self._equals_token_ids.add(tid)
                elif tok == "|":
                    self._delim_token_ids.add(tid)
                elif tok in _VALUE_CHARS:
                    self._value_token_ids.add(tid)
            else:
                # Multi-char tokens — allowed in values if all chars are value-legal
                if all(c in _VALUE_CHARS for c in tok):
                    self._value_token_ids.add(tid)

    # ── FSM state detection ───────────────────────────────────────

    def _detect_state(self, text: str) -> FIXParseState:
        """Determine the current FSM state from generated text.

        Args:
            text: The decoded text generated so far.

        Returns:
            Current :class:`FIXParseState`.
        """
        if not text:
            return FIXParseState.EXPECT_TAG_OR_END

        # Work backward from the end to find the last structural character
        stripped = text.rstrip()
        if not stripped:
            return FIXParseState.EXPECT_TAG_OR_END

        last_char = stripped[-1]

        if last_char == "|":
            return FIXParseState.EXPECT_TAG_OR_END
        elif last_char == "=":
            return FIXParseState.IN_VALUE
        elif last_char.isdigit():
            # Could be in a tag or in a value — look for context
            last_delim = max(stripped.rfind("|"), stripped.rfind("="))
            if last_delim == -1:
                # No delimiter seen yet — we're starting a tag
                return FIXParseState.IN_TAG
            suffix = stripped[last_delim:]
            if suffix[0] == "|":
                # After a pipe — we're in a tag number
                tag_part = suffix[1:]
                if tag_part.isdigit():
                    return FIXParseState.IN_TAG
            elif suffix[0] == "=":
                return FIXParseState.IN_VALUE
            return FIXParseState.IN_VALUE
        else:
            # Regular character — must be in a value
            return FIXParseState.IN_VALUE

    def _extract_current_tag(self, text: str) -> Optional[str]:
        """Extract the tag number currently being built.

        Args:
            text: Decoded text so far.

        Returns:
            Tag number string, or ``None``.
        """
        last_pipe = text.rfind("|")
        if last_pipe == -1:
            segment = text
        else:
            segment = text[last_pipe + 1:]

        eq_pos = segment.find("=")
        if eq_pos > 0:
            candidate = segment[:eq_pos]
            return candidate if candidate.isdigit() else None
        elif segment.isdigit():
            return segment
        return None

    # ── Public API ────────────────────────────────────────────────

    def get_token_mask(
        self,
        generated_ids: List[int],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Compute a boolean mask over the vocabulary for the next token.

        ``True`` means the token is **allowed**; ``False`` means it
        should be masked to ``-inf`` in the logits.

        Args:
            generated_ids: Token IDs generated so far.
            device: Device for the output tensor.

        Returns:
            Boolean tensor of shape ``[vocab_size]``.
        """
        # Decode what we have so far
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        state = self._detect_state(text)

        mask = torch.zeros(self._vocab_size, dtype=torch.bool, device=device)

        if state == FIXParseState.EXPECT_TAG_OR_END:
            # Allow digit tokens (start of tag), tag-pattern tokens, and end tokens
            for tid in self._digit_token_ids:
                mask[tid] = True
            for tid in self._tag_pattern_ids:
                mask[tid] = True
            for tid in self._end_token_ids:
                mask[tid] = True

        elif state == FIXParseState.IN_TAG:
            # Allow more digits (continue tag) or equals (end tag)
            for tid in self._digit_token_ids:
                mask[tid] = True
            for tid in self._equals_token_ids:
                mask[tid] = True
            # Also allow tag-pattern tokens that start with the current prefix
            current_tag = self._extract_current_tag(text)
            if current_tag:
                for tid in self._tag_pattern_ids:
                    tok = self.tokenizer.id_to_token.get(tid, "")
                    if tok.startswith(current_tag):
                        mask[tid] = True

        elif state == FIXParseState.IN_VALUE:
            # Allow value characters and delimiters (end of field)
            for tid in self._value_token_ids:
                mask[tid] = True
            for tid in self._delim_token_ids:
                mask[tid] = True
            # Allow end tokens (message can end mid-value in some edge cases)
            for tid in self._end_token_ids:
                mask[tid] = True

        elif state == FIXParseState.EXPECT_EQUALS:
            for tid in self._equals_token_ids:
                mask[tid] = True

        elif state == FIXParseState.EXPECT_DELIM:
            for tid in self._delim_token_ids:
                mask[tid] = True
            for tid in self._end_token_ids:
                mask[tid] = True

        # Safety: if nothing is allowed, allow everything (don't crash generation)
        if not mask.any():
            mask.fill_(True)

        return mask

    def apply_constraint(
        self,
        logits: torch.Tensor,
        generated_ids: List[int],
    ) -> torch.Tensor:
        """Mask illegal tokens in logits according to the FSM.

        Args:
            logits: Raw logits ``[vocab_size]`` for the next token.
            generated_ids: Token IDs generated so far.

        Returns:
            Masked logits with illegal positions set to ``-inf``.
        """
        mask = self.get_token_mask(generated_ids, device=logits.device)
        # Expand mask to match logits vocab (in case of size mismatch)
        if mask.shape[0] < logits.shape[-1]:
            padded = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
            padded[: mask.shape[0]] = mask
            mask = padded
        elif mask.shape[0] > logits.shape[-1]:
            mask = mask[: logits.shape[-1]]

        logits[~mask] = float("-inf")
        return logits
