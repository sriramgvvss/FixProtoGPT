"""
Module: src.inference.beam_search
==================================

Beam search decoding for FixProtoGPT with FIX-validity scoring.

Provides :func:`beam_search_generate` — a beam search implementation
that selects the structurally best FIX message from multiple candidate
beams, using a FIX-validity scoring function to re-rank candidates.

Usage::

    from src.inference.beam_search import beam_search_generate

    results = beam_search_generate(
        model, input_ids, beam_width=4, max_new_tokens=256,
        eos_token_id=2, tokenizer=tok,
    )
    # results: list of (token_ids, log_prob, fix_score) sorted by combined score

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class BeamCandidate:
    """A single candidate in beam search."""

    token_ids: List[int] = field(default_factory=list)
    log_prob: float = 0.0
    finished: bool = False
    kv_cache: Optional[list] = None  # KV-cache for efficient decoding


# ── FIX validity scoring ─────────────────────────────────────────

# Required header/trailer tags for a valid FIX message
_REQUIRED_TAGS = {"8", "9", "35", "49", "56", "10"}

# Message type → required body tags (subset)
_MSG_TYPE_REQUIRED: Dict[str, set] = {
    "D": {"11", "21", "55", "54", "38", "40", "60"},          # NewOrderSingle
    "8": {"37", "11", "17", "150", "39", "55", "54", "38"},    # ExecutionReport
    "F": {"11", "41", "55", "54", "38", "60"},                 # OrderCancelRequest
    "G": {"11", "41", "55", "54", "38", "40", "60"},           # OrderCancelReplaceRequest
    "V": {"262", "263", "264"},                                 # MarketDataRequest
    "A": {"98", "108"},                                         # Logon
    "0": set(),                                                 # Heartbeat
}


def score_fix_validity(text: str) -> float:
    """Score the structural validity of a FIX message.

    Returns a value in ``[0.0, 1.0]`` where 1.0 is perfectly valid.

    Scoring criteria (weighted):
        - Tag=Value structure (0.30): every field matches ``\\d+=.+``
        - Required tags present (0.25): 8, 9, 35, 49, 56, 10
        - Message-type body tags (0.15): per MsgType required fields
        - Checksum present (0.05): tag 10 is present
        - BeginString correct (0.10): tag 8 starts with ``FIX`` or ``FIXT``
        - No duplicate tags (0.10): except repeating groups
        - Reasonable field count (0.05): 5-50 fields

    Args:
        text: Candidate FIX message string.

    Returns:
        Validity score between 0.0 and 1.0.
    """
    if not text or "=" not in text:
        return 0.0

    delimiter = "\x01" if "\x01" in text else "|"
    fields = [f for f in text.split(delimiter) if f.strip()]

    if not fields:
        return 0.0

    score = 0.0

    # 1. Tag=Value structure (0.30)
    well_formed = 0
    parsed_tags: List[str] = []
    parsed_values: Dict[str, str] = {}
    for f in fields:
        if "=" in f:
            tag, value = f.split("=", 1)
            if tag.isdigit() and value:
                well_formed += 1
                parsed_tags.append(tag)
                parsed_values[tag] = value
    structure_score = well_formed / len(fields) if fields else 0.0
    score += 0.30 * structure_score

    # 2. Required tags present (0.25)
    present = set(parsed_tags)
    required_found = len(_REQUIRED_TAGS & present)
    score += 0.25 * (required_found / len(_REQUIRED_TAGS))

    # 3. Message-type body tags (0.15)
    msg_type = parsed_values.get("35", "")
    body_required = _MSG_TYPE_REQUIRED.get(msg_type, set())
    if body_required:
        body_found = len(body_required & present)
        score += 0.15 * (body_found / len(body_required))
    else:
        score += 0.15  # No body requirements to check

    # 4. Checksum present (0.05)
    if "10" in present:
        score += 0.05

    # 5. BeginString correct (0.10)
    begin = parsed_values.get("8", "")
    if begin.startswith("FIX") or begin.startswith("FIXT"):
        score += 0.10

    # 6. No duplicate tags (0.10) — except known repeating group tags
    _repeating_tags = {"269", "270", "271", "146", "267", "454", "455"}
    non_repeating = [t for t in parsed_tags if t not in _repeating_tags]
    if non_repeating:
        unique_ratio = len(set(non_repeating)) / len(non_repeating)
        score += 0.10 * unique_ratio

    # 7. Reasonable field count (0.05)
    if 5 <= len(fields) <= 50:
        score += 0.05
    elif len(fields) > 0:
        score += 0.02

    return round(min(score, 1.0), 4)


# ── Beam search ──────────────────────────────────────────────────


@torch.no_grad()
def beam_search_generate(
    model: Any,
    input_ids: torch.Tensor,
    beam_width: int = 4,
    max_new_tokens: int = 256,
    eos_token_id: int = 2,
    length_penalty: float = 1.0,
    fix_validity_weight: float = 0.3,
    temperature: float = 1.0,
    tokenizer: Optional[Any] = None,
) -> List[Tuple[List[int], float, float]]:
    """Beam search with FIX-validity re-ranking.

    Generates multiple candidate sequences and ranks them by a weighted
    combination of model log-probability and FIX structural validity.

    Args:
        model:               FixProtoGPT model in eval mode.
        input_ids:           Input token IDs ``(1, seq_len)``.
        beam_width:          Number of beams to maintain.
        max_new_tokens:      Maximum tokens to generate per beam.
        eos_token_id:        End-of-sequence token ID.
        length_penalty:       Exponent for length normalisation.
        fix_validity_weight: Weight of FIX-validity score in ranking.
        temperature:         Softmax temperature for log-prob computation.
        tokenizer:           Tokenizer for decoding candidates (needed
                              for FIX-validity scoring).

    Returns:
        List of ``(token_ids, normalised_log_prob, fix_score)`` tuples,
        sorted by combined score (descending).
    """
    device = input_ids.device
    prompt_ids = input_ids[0].tolist()

    # Initialise beams — all start with the same prompt
    beams: List[BeamCandidate] = [
        BeamCandidate(token_ids=list(prompt_ids), log_prob=0.0),
    ]

    for _step in range(max_new_tokens):
        all_candidates: List[BeamCandidate] = []

        for beam in beams:
            if beam.finished:
                all_candidates.append(beam)
                continue

            # Prepare input — feed only the last token if we have KV-cache
            if beam.kv_cache is not None:
                inp = torch.tensor(
                    [[beam.token_ids[-1]]], dtype=torch.long, device=device,
                )
                logits, _, new_cache = model(
                    inp, use_cache=True, past_kv_caches=beam.kv_cache,
                )
            else:
                inp = torch.tensor(
                    [beam.token_ids], dtype=torch.long, device=device,
                )
                logits, _, new_cache = model(inp, use_cache=True)

            # Get logits for the last position
            next_logits = logits[0, -1, :]

            if temperature != 1.0:
                next_logits = next_logits / temperature

            log_probs = F.log_softmax(next_logits, dim=-1)

            # Select top-k candidates
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)

            for i in range(beam_width):
                token_id = topk_ids[i].item()
                token_log_prob = topk_log_probs[i].item()

                new_ids = beam.token_ids + [token_id]
                new_log_prob = beam.log_prob + token_log_prob
                finished = token_id == eos_token_id

                all_candidates.append(BeamCandidate(
                    token_ids=new_ids,
                    log_prob=new_log_prob,
                    finished=finished,
                    kv_cache=new_cache if not finished else None,
                ))

        # Prune to beam_width best candidates
        all_candidates.sort(key=lambda c: c.log_prob, reverse=True)
        beams = all_candidates[:beam_width]

        # Early termination if all beams are finished
        if all(b.finished for b in beams):
            break

    # ── Final ranking with FIX-validity ───────────────────────────

    results: List[Tuple[List[int], float, float]] = []
    for beam in beams:
        gen_len = len(beam.token_ids) - len(prompt_ids)
        if gen_len <= 0:
            gen_len = 1

        # Length-normalised log probability
        normalised_lp = beam.log_prob / (gen_len ** length_penalty)

        # FIX validity score (needs tokenizer)
        fix_score = 0.0
        if tokenizer is not None:
            try:
                text = tokenizer.decode(beam.token_ids, skip_special_tokens=True)
                fix_score = score_fix_validity(text)
            except Exception:
                fix_score = 0.0

        results.append((beam.token_ids, normalised_lp, fix_score))

    # Sort by combined score: (1 - w) * normalised_log_prob + w * fix_score
    # Normalise log_probs to [0, 1] range for fair combination
    if results:
        max_lp = max(r[1] for r in results)
        min_lp = min(r[1] for r in results)
        lp_range = max_lp - min_lp if max_lp != min_lp else 1.0

        results.sort(
            key=lambda r: (
                (1.0 - fix_validity_weight)
                * ((r[1] - min_lp) / lp_range)
                + fix_validity_weight * r[2]
            ),
            reverse=True,
        )

    return results
