"""
Module: src.core.tokenizer
===========================

Custom tokenizer optimised for FIX protocol messages.

Supports BPE-style vocabulary, FIX-specific special tokens
(``<|fix|>``, ``<|eom|>``, field-level tags), and bidirectional
encode/decode.

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

import re
import json
import logging
from typing import List, Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class FixProtocolTokenizer:
    """
    Tokenizer specifically designed for FIX Protocol messages
    Combines character-level and field-level tokenization
    """
    
    # FIX Protocol field delimiters
    SOH = '\x01'  # Start of Header (field delimiter)
    
    def __init__(self, vocab_size: int = 1024):
        """Initialise tokenizer with special tokens and empty vocab.

        Args:
            vocab_size: Maximum vocabulary size (default 1024).
        """
        self.vocab_size = vocab_size
        
        # Special tokens
        self.special_tokens = {
            '<|pad|>': 0,
            '<|bos|>': 1,
            '<|eos|>': 2,
            '<|fix|>': 3,      # FIX message start
            '<|field|>': 4,    # Field separator
            '<|eom|>': 5,      # End of message
            '<|unk|>': 6,      # Unknown token
        }
        
        # FIX Protocol common tags (for special handling)
        self.fix_tags = self._get_common_fix_tags()
        
        # Vocabulary mappings
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Initialize with special tokens
        self.token_to_id.update(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        
        # BPE merges (will be built during training)
        self.bpe_merges: List[Tuple[str, str]] = []
        
    def _get_common_fix_tags(self) -> Dict[str, str]:
        """Return common FIX 5.0 SP2 protocol tags and their meanings.

        Returns:
            Dict mapping tag numbers (str) to human-readable field names.
        """
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
    
    def parse_fix_message(self, message: str) -> List[Dict[str, str]]:
        """
        Parse a FIX message into fields
        
        Args:
            message: Raw FIX message string
        
        Returns:
            List of field dictionaries with tag and value
        """
        import re
        # Strip version markers like [FIX-4.2], [FIX-4.4], [FIX-5.0SP2], etc.
        message = re.sub(r'\[FIX[^\]]*\]\s*', '', message)

        # Split by SOH or | (pipe is common in FIX logs)
        delimiter = self.SOH if self.SOH in message else '|'
        fields = message.split(delimiter)
        
        parsed_fields = []
        for field in fields:
            if '=' in field:
                tag, value = field.split('=', 1)
                parsed_fields.append({
                    'tag': tag,
                    'value': value,
                    'name': self.fix_tags.get(tag, 'Unknown')
                })
        
        return parsed_fields
    
    def build_vocab(self, texts: List[str], min_frequency: int = 2):
        """
        Build vocabulary from training texts using BPE algorithm
        
        Args:
            texts: List of training texts
            min_frequency: Minimum frequency for a token to be included
        """
        print("Building vocabulary...")
        
        # Start with character-level tokens
        char_vocab = set()
        for text in texts:
            char_vocab.update(text)
        
        # Add character tokens (reserve capacity for FIX-specific tokens)
        # Characters get at most 60% of remaining vocab slots; the rest
        # are saved for FIX tags and BPE merges.
        max_char_slots = int((self.vocab_size - len(self.special_tokens)) * 0.6)
        next_id = len(self.special_tokens)
        for char in sorted(char_vocab):
            if next_id >= len(self.special_tokens) + max_char_slots:
                break
            if char not in self.token_to_id:
                self.token_to_id[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1
        
        # Add FIX-specific tokens
        fix_tokens = []
        
        # Common FIX message types
        msg_types = ['8=FIX', 'D', 'F', 'G', 'V', 'W', 'X', '0', '1', '2', '3', '4', '5']
        for msg in msg_types:
            if msg not in self.token_to_id:
                fix_tokens.append(msg)
        
        # Common field patterns
        for tag in self.fix_tags.keys():
            pattern = f"{tag}="
            if pattern not in self.token_to_id:
                fix_tokens.append(pattern)
        
        # Add FIX tokens to vocabulary
        for token in fix_tokens:
            if next_id < self.vocab_size:
                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1
        
        # Simple BPE: merge most common adjacent pairs
        # For production, use a full BPE implementation
        word_freq = {}
        for text in texts:
            words = text.split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add common words as tokens
        common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in common_words:
            if freq >= min_frequency and next_id < self.vocab_size:
                if word not in self.token_to_id:
                    self.token_to_id[word] = next_id
                    self.id_to_token[next_id] = word
                    next_id += 1
        
        print(f"Vocabulary built with {len(self.token_to_id)} tokens")
    
    def encode(self, text: str, add_special_tokens: bool = True,
               for_generation: bool = False) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            for_generation: If True, omit trailing EOM/EOS so the model
                           continues generating instead of stopping.
        
        Returns:
            List of token IDs
        """
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.special_tokens['<|bos|>'])
        
        # Check if it's a FIX message
        if '=' in text and ('8=FIX' in text or '35=' in text):
            tokens.append(self.special_tokens['<|fix|>'])
            
            # Tokenize FIX message field by field
            fields = self.parse_fix_message(text)
            for field in fields:
                # Try to find field pattern in vocab
                field_str = f"{field['tag']}={field['value']}"
                
                if field_str in self.token_to_id:
                    tokens.append(self.token_to_id[field_str])
                else:
                    # Tokenize tag
                    tag_pattern = f"{field['tag']}="
                    if tag_pattern in self.token_to_id:
                        tokens.append(self.token_to_id[tag_pattern])
                    else:
                        # Character-level fallback
                        for char in tag_pattern:
                            tokens.append(self.token_to_id.get(char, self.special_tokens['<|unk|>']))
                    
                    # Tokenize value
                    for char in field['value']:
                        tokens.append(self.token_to_id.get(char, self.special_tokens['<|unk|>']))
                
                tokens.append(self.special_tokens['<|field|>'])
            
            # Only add EOM when encoding a complete message, not a generation prompt
            if not for_generation:
                tokens.append(self.special_tokens['<|eom|>'])
        else:
            # Regular text tokenization
            # Try word-level first, then character-level
            words = text.split()
            for word in words:
                if word in self.token_to_id:
                    tokens.append(self.token_to_id[word])
                else:
                    # Character-level fallback
                    for char in word:
                        tokens.append(self.token_to_id.get(char, self.special_tokens['<|unk|>']))
                # Add space token between words
                tokens.append(self.token_to_id.get(' ', self.special_tokens['<|unk|>']))
            # Remove trailing space
            if tokens and tokens[-1] == self.token_to_id.get(' ', -1):
                tokens.pop()
        
        # Only add EOS for training sequences, not generation prompts
        if add_special_tokens and not for_generation:
            tokens.append(self.special_tokens['<|eos|>'])
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text
        """
        tokens = []
        special_token_ids = set(self.special_tokens.values())
        
        # Build set of word-level tokens (multi-char, non-special) for spacing
        word_token_ids = set()
        for tok, tid in self.token_to_id.items():
            if tid not in special_token_ids and len(tok) > 1 and not tok.endswith('='):
                word_token_ids.add(tid)
        
        # Structural special tokens that map to visible characters
        field_token_id = self.special_tokens.get('<|field|>')
        eom_token_id = self.special_tokens.get('<|eom|>')
        fix_token_id = self.special_tokens.get('<|fix|>')
        
        for token_id in token_ids:
            # Always render structural FIX tokens even when skipping specials
            if token_id == field_token_id:
                tokens.append('|')
                continue
            if token_id == eom_token_id:
                # End-of-message — stop decoding
                break
            if token_id == fix_token_id:
                # FIX start marker — skip silently
                continue
            
            if skip_special_tokens and token_id in special_token_ids:
                continue
            
            token = self.id_to_token.get(token_id, '<|unk|>')
            # Add space before word-level tokens for readability
            if token_id in word_token_ids and tokens and tokens[-1] != ' ' and tokens[-1] != '|':
                tokens.append(' ')
            tokens.append(token)
        
        # Join tokens
        text = ''.join(tokens)
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def save(self, path: str) -> None:
        """Persist vocabulary, BPE merges, and FIX tags to disk.

        Args:
            path: Directory path where files will be written.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary
        with open(path / 'vocab.json', 'w') as f:
            json.dump(self.token_to_id, f, indent=2)
        
        # Save BPE merges (JSON for security; avoid pickle)
        with open(path / 'merges.json', 'w') as f:
            json.dump(self.bpe_merges, f)
        
        # Save FIX tags
        with open(path / 'fix_tags.json', 'w') as f:
            json.dump(self.fix_tags, f, indent=2)
        
        print(f"Tokenizer saved to {path}")
    
    def load(self, path: str) -> None:
        """Load vocabulary, BPE merges, and FIX tags from disk.

        Args:
            path: Directory path from which to read.
        """
        path = Path(path)
        
        # Load vocabulary
        with open(path / 'vocab.json', 'r') as f:
            self.token_to_id = json.load(f)
        
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        
        # Load BPE merges — support both JSON (preferred) and legacy pickle
        merges_json = path / 'merges.json'
        merges_pkl = path / 'merges.pkl'
        if merges_json.exists():
            with open(merges_json, 'r') as f:
                raw = json.load(f)
                self.bpe_merges = [tuple(pair) for pair in raw]
        elif merges_pkl.exists():
            import pickle
            logger.warning(
                "Loading BPE merges from legacy pickle file %s — "
                "re-save the tokenizer to migrate to JSON.",
                merges_pkl,
            )
            with open(merges_pkl, 'rb') as f:
                self.bpe_merges = pickle.load(f)
        else:
            self.bpe_merges = []
        
        # Load FIX tags
        with open(path / 'fix_tags.json', 'r') as f:
            self.fix_tags = json.load(f)
        
        print(f"Tokenizer loaded from {path}")
    
    @property
    def pad_token_id(self) -> int:
        """Token ID used for padding."""
        return self.special_tokens['<|pad|>']

    @property
    def bos_token_id(self) -> int:
        """Token ID for Beginning-Of-Sequence."""
        return self.special_tokens['<|bos|>']

    @property
    def eos_token_id(self) -> int:
        """Token ID for End-Of-Sequence."""
        return self.special_tokens['<|eos|>']


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = FixProtocolTokenizer()
    
    # Example FIX message
    fix_msg = "8=FIXT.1.1|9=178|35=D|49=SENDER|56=TARGET|34=1|52=20231201-12:30:00|11=ORDER123|21=1|55=AAPL|54=1|38=100|40=2|44=150.50|59=0|1128=9|10=123|"
    
    # Build vocab with sample data
    sample_texts = [
        fix_msg,
        "8=FIXT.1.1|35=8|55=GOOGL|54=2|38=50|1128=9|",
        "Create a new order for AAPL",
        "Market data request for symbols",
    ]
    
    tokenizer.build_vocab(sample_texts)
    
    # Test encoding
    encoded = tokenizer.encode(fix_msg)
    print(f"Encoded: {encoded[:20]}...")  # Show first 20 tokens
    
    # Test decoding
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded[:100]}...")  # Show first 100 chars
    
    # Test natural language
    nl_text = "Create a new order for 100 shares of AAPL"
    nl_encoded = tokenizer.encode(nl_text)
    nl_decoded = tokenizer.decode(nl_encoded)
    print(f"\nNatural Language: {nl_text}")
    print(f"Decoded: {nl_decoded}")
