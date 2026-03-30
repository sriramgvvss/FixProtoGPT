"""
Module: src.core.bpe_tokenizer
================================

True BPE tokenizer for FIX protocol messages using the HuggingFace
``tokenizers`` library.

This module provides :class:`FixProtocolBPETokenizer`, a drop-in
replacement for :class:`FixProtocolTokenizer` that performs *real*
byte-pair encoding via a pre-tokenisation step that respects FIX
message structure (splits on ``|`` and ``=``).

The existing :class:`FixProtocolTokenizer` is preserved for backward
compatibility.  Switch between them via ``tokenizer.type`` in
``model_config.yaml``:

    tokenizer:
      type: "huggingface_bpe"   # or "bpe" for legacy
      vocab_size: 4096

Usage::

    from src.core.bpe_tokenizer import FixProtocolBPETokenizer

    tok = FixProtocolBPETokenizer(vocab_size=4096)
    tok.train(texts)
    ids = tok.encode("8=FIXT.1.1|35=D|55=AAPL|")
    text = tok.decode(ids)

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import HuggingFace tokenizers — graceful fallback
try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
    from tokenizers.normalizers import NFC

    HAS_HF_TOKENIZERS = True
except ImportError:
    HAS_HF_TOKENIZERS = False
    logger.info("tokenizers library not installed — FixProtocolBPETokenizer unavailable. "
                 "Install with: pip install tokenizers")


class FixProtocolBPETokenizer:
    """True BPE tokenizer with FIX-aware pre-tokenisation.

    Uses the HuggingFace ``tokenizers`` library under the hood for
    proper byte-pair encoding with a configurable vocabulary size.

    The API is kept compatible with :class:`FixProtocolTokenizer`
    so that the rest of the codebase can use either interchangeably.
    """

    # Special tokens — same IDs as the legacy tokenizer for compatibility
    SPECIAL_TOKENS: List[str] = [
        "<|pad|>",    # 0
        "<|bos|>",    # 1
        "<|eos|>",    # 2
        "<|fix|>",    # 3
        "<|field|>",  # 4
        "<|eom|>",    # 5
        "<|unk|>",    # 6
    ]

    def __init__(self, vocab_size: int = 4096) -> None:
        """Initialise the BPE tokenizer.

        Args:
            vocab_size: Target vocabulary size (default 4096).

        Raises:
            ImportError: If the ``tokenizers`` library is not installed.
        """
        if not HAS_HF_TOKENIZERS:
            raise ImportError(
                "The 'tokenizers' library is required for FixProtocolBPETokenizer. "
                "Install it with: pip install tokenizers"
            )

        self.vocab_size = vocab_size
        self._tokenizer: Optional[Tokenizer] = None

        # Maintain legacy interface attributes
        self.special_tokens: Dict[str, int] = {
            tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)
        }
        self.token_to_id: Dict[str, int] = dict(self.special_tokens)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}
        self.bpe_merges: List[Tuple[str, str]] = []

        # FIX tags (for parse_fix_message compat)
        self.fix_tags = self._get_common_fix_tags()

    # ── FIX protocol tags (identical to legacy) ───────────────────

    @staticmethod
    def _get_common_fix_tags() -> Dict[str, str]:
        """Return common FIX tags and their names."""
        return {
            '8': 'BeginString', '9': 'BodyLength', '35': 'MsgType',
            '49': 'SenderCompID', '56': 'TargetCompID', '34': 'MsgSeqNum',
            '52': 'SendingTime', '11': 'ClOrdID', '21': 'HandlInst',
            '55': 'Symbol', '54': 'Side', '38': 'OrderQty',
            '40': 'OrdType', '44': 'Price', '59': 'TimeInForce',
            '150': 'ExecType', '151': 'LeavesQty', '14': 'CumQty',
            '6': 'AvgPx', '10': 'CheckSum', '37': 'OrderID',
            '17': 'ExecID', '39': 'OrdStatus', '60': 'TransactTime',
            '41': 'OrigClOrdID', '1': 'Account', '15': 'Currency',
            '48': 'SecurityID', '22': 'SecurityIDSource',
            '31': 'LastPx', '32': 'LastQty', '58': 'Text',
            '98': 'EncryptMethod', '108': 'HeartBtInt',
            '262': 'MDReqID', '263': 'SubscriptionRequestType',
            '264': 'MarketDepth', '269': 'MDEntryType',
            '270': 'MDEntryPx', '271': 'MDEntrySize',
            '1128': 'ApplVerID', '1137': 'DefaultApplVerID',
            '553': 'Username', '554': 'Password',
        }

    # ── Training ──────────────────────────────────────────────────

    def build_vocab(self, texts: List[str], min_frequency: int = 2) -> None:
        """Train a BPE model on *texts*.

        Alias for :meth:`train` to match the legacy tokenizer API.

        Args:
            texts: Training corpus.
            min_frequency: Minimum pair frequency for merges.
        """
        self.train(texts, min_frequency=min_frequency)

    def train(self, texts: List[str], min_frequency: int = 2) -> None:
        """Train the BPE tokenizer on a list of texts.

        Uses a custom pre-tokeniser that splits on FIX delimiters
        (``|``, ``=``, SOH) so that byte-pair merges learn subword
        patterns *within* tag numbers and values, but never across
        field boundaries.

        Args:
            texts: Training corpus.
            min_frequency: Minimum pair frequency for merges.
        """
        print("Training BPE tokenizer...")

        # Build a BPE model from scratch
        tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

        # Normaliser: NFC unicode normalisation
        tokenizer.normalizer = NFC()

        # Pre-tokeniser: split on FIX delimiters, whitespace, and punctuation
        # This ensures BPE merges never cross field boundaries
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern="|", behavior="isolated"),
            pre_tokenizers.Split(pattern="=", behavior="isolated"),
            pre_tokenizers.Split(pattern="\x01", behavior="removed"),
            pre_tokenizers.Whitespace(),
        ])

        # Decoder: BPE decoder
        tokenizer.decoder = decoders.BPEDecoder()

        # Trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.SPECIAL_TOKENS,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        # Train from in-memory iterator
        tokenizer.train_from_iterator(texts, trainer=trainer)

        # Post-processor: add BOS/EOS
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<|bos|> $A <|eos|>",
            pair="<|bos|> $A <|eos|> <|bos|> $B:1 <|eos|>:1",
            special_tokens=[
                ("<|bos|>", tokenizer.token_to_id("<|bos|>")),
                ("<|eos|>", tokenizer.token_to_id("<|eos|>")),
            ],
        )

        self._tokenizer = tokenizer
        self._sync_vocab()

        print(f"BPE tokenizer trained with {self._tokenizer.get_vocab_size()} tokens")

    def _sync_vocab(self) -> None:
        """Synchronise internal vocab dicts from the HF tokenizer."""
        if self._tokenizer is None:
            return
        vocab = self._tokenizer.get_vocab()
        self.token_to_id = dict(vocab)
        self.id_to_token = {v: k for k, v in vocab.items()}
        # Update special_tokens dict with actual IDs
        for tok in self.SPECIAL_TOKENS:
            tid = vocab.get(tok)
            if tid is not None:
                self.special_tokens[tok] = tid

    # ── Encoding ──────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        for_generation: bool = False,
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text or FIX message.
            add_special_tokens: Add BOS/EOS tokens (default True).
            for_generation: If True, omit trailing EOS/EOM so the
                model continues generating.

        Returns:
            List of token IDs.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not trained or loaded. Call train() or load() first.")

        # For FIX messages, prepend <|fix|> token and replace | with <|field|>
        is_fix = "=" in text and ("8=FIX" in text or "35=" in text)

        if is_fix:
            # Encode with special handling
            encoding = self._tokenizer.encode(text, add_special_tokens=False)
            ids = encoding.ids

            # Prepend BOS + FIX marker
            result = []
            if add_special_tokens:
                result.append(self.special_tokens["<|bos|>"])
            result.append(self.special_tokens["<|fix|>"])

            # Replace pipe-separator tokens with <|field|>
            pipe_id = self.token_to_id.get("|")
            field_id = self.special_tokens["<|field|>"]
            for tid in ids:
                if tid == pipe_id:
                    result.append(field_id)
                else:
                    result.append(tid)

            # Add terminators unless generating
            if not for_generation:
                result.append(self.special_tokens["<|eom|>"])
                if add_special_tokens:
                    result.append(self.special_tokens["<|eos|>"])

            return result

        # Regular text
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        ids = list(encoding.ids)

        if for_generation and add_special_tokens:
            # Remove trailing EOS for generation prompts
            eos = self.special_tokens.get("<|eos|>")
            if ids and ids[-1] == eos:
                ids.pop()

        return ids

    # ── Decoding ──────────────────────────────────────────────────

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: Sequence of token IDs.
            skip_special_tokens: Skip BOS/EOS/PAD tokens.

        Returns:
            Decoded text string.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not trained or loaded.")

        # Convert <|field|> back to | and handle <|eom|>
        field_id = self.special_tokens.get("<|field|>")
        eom_id = self.special_tokens.get("<|eom|>")
        fix_id = self.special_tokens.get("<|fix|>")
        pipe_id = self.token_to_id.get("|")

        processed: List[int] = []
        for tid in token_ids:
            if tid == eom_id:
                break
            if tid == fix_id:
                continue
            if tid == field_id:
                if pipe_id is not None:
                    processed.append(pipe_id)
                continue
            processed.append(tid)

        text = self._tokenizer.decode(processed, skip_special_tokens=skip_special_tokens)
        # Clean up extra whitespace around delimiters
        text = re.sub(r'\s*\|\s*', '|', text)
        text = re.sub(r'\s*=\s*', '=', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # ── FIX message parsing (compat) ─────────────────────────────

    SOH = '\x01'

    def parse_fix_message(self, message: str) -> List[Dict[str, str]]:
        """Parse a FIX message into fields (compat with legacy tokenizer).

        Args:
            message: Raw FIX message string.

        Returns:
            List of {tag, value, name} dicts.
        """
        delimiter = self.SOH if self.SOH in message else '|'
        fields = message.split(delimiter)
        parsed = []
        for field in fields:
            if '=' in field:
                tag, value = field.split('=', 1)
                parsed.append({
                    'tag': tag,
                    'value': value,
                    'name': self.fix_tags.get(tag, 'Unknown'),
                })
        return parsed

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the tokenizer to disk.

        Saves both the HF tokenizer JSON and legacy-compatible files
        so that the old loader can at least read the vocab.

        Args:
            path: Directory path to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._tokenizer is not None:
            # Save native HF tokenizer
            self._tokenizer.save(str(path / "tokenizer.json"))

        # Save legacy-compatible vocab.json
        with open(path / "vocab.json", "w") as f:
            json.dump(self.token_to_id, f, indent=2)

        # Save fix_tags.json
        with open(path / "fix_tags.json", "w") as f:
            json.dump(self.fix_tags, f, indent=2)

        # Save empty merges for compat
        import pickle
        with open(path / "merges.pkl", "wb") as f:
            pickle.dump([], f)

        # Marker file so loader knows which tokenizer to use
        with open(path / ".tokenizer_type", "w") as f:
            f.write("huggingface_bpe\n")

        print(f"BPE tokenizer saved to {path}")

    def load(self, path: str) -> None:
        """Load the tokenizer from disk.

        Args:
            path: Directory path to load from.
        """
        path = Path(path)
        hf_path = path / "tokenizer.json"

        if hf_path.exists():
            self._tokenizer = Tokenizer.from_file(str(hf_path))
            self._sync_vocab()
            print(f"BPE tokenizer loaded from {path}")
        else:
            # Fallback: load legacy vocab
            with open(path / "vocab.json", "r") as f:
                self.token_to_id = json.load(f)
            self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
            with open(path / "fix_tags.json", "r") as f:
                self.fix_tags = json.load(f)
            logger.warning("No tokenizer.json found — loaded legacy vocab only. "
                           "Encoding/decoding will not work until re-trained.")

    # ── Properties (compat) ───────────────────────────────────────

    @property
    def pad_token_id(self) -> int:
        """Return the integer ID of the padding token."""
        return self.special_tokens["<|pad|>"]

    @property
    def bos_token_id(self) -> int:
        """Return the integer ID of the beginning-of-sequence token."""
        return self.special_tokens["<|bos|>"]

    @property
    def eos_token_id(self) -> int:
        """Return the integer ID of the end-of-sequence token."""
        return self.special_tokens["<|eos|>"]


def create_tokenizer(
    tokenizer_type: str = "bpe",
    vocab_size: int = 1024,
) -> "FixProtocolBPETokenizer | object":
    """Factory function to create the appropriate tokenizer.

    Args:
        tokenizer_type: ``"bpe"`` for legacy or ``"huggingface_bpe"`` for
            the true BPE implementation.
        vocab_size: Target vocabulary size.

    Returns:
        Tokenizer instance.
    """
    if tokenizer_type == "huggingface_bpe":
        return FixProtocolBPETokenizer(vocab_size=vocab_size)
    else:
        from src.core.tokenizer import FixProtocolTokenizer
        return FixProtocolTokenizer(vocab_size=vocab_size)
