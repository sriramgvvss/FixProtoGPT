"""
Module: src.core.transformer
=============================

GPT-style decoder-only transformer architecture for FixProtoGPT.

Includes RoPE (Rotary Position Embedding), multi-head self-attention
with optional flash attention, and a pre-LN transformer block.

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration"""
    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    vocab_size: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.1
    use_rotary: bool = True
    bias: bool = False  # Use bias in Linear layers

    @classmethod
    def from_yaml(cls, path: str | Path = "config/model_config.yaml") -> "ModelConfig":
        """Construct a ``ModelConfig`` from the ``model`` section of a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            Populated :class:`ModelConfig` instance.
        """
        import yaml

        with open(path) as f:
            cfg = yaml.safe_load(f)["model"]
        return cls(
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            d_model=cfg["d_model"],
            d_ff=cfg["d_ff"],
            vocab_size=cfg["vocab_size"],
            max_seq_len=cfg["max_seq_len"],
            dropout=cfg.get("dropout", 0.1),
            attention_dropout=cfg.get("attention_dropout", 0.1),
            use_rotary=cfg.get("use_rotary", True),
        )


class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        """Initialise RoPE with pre-computed frequency cache.

        Args:
            dim: Head dimension (must be even).
            max_seq_len: Maximum sequence length to cache.
            base: Base for the geometric frequency progression.
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, max_seq_len: int) -> None:
        """Build and cache cos/sin rotation matrices.

        Args:
            max_seq_len: Length of the sequence dimension to cache.
        """
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Don't duplicate - keep shape [seq_len, head_dim//2]
        
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :], persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached cos/sin tensors sliced to *seq_len*.

        Args:
            x: Input tensor (used only for device/dtype inference).
            seq_len: Actual sequence length of the current batch.

        Returns:
            Tuple of (cos, sin) tensors, each ``[1, 1, seq_len, dim//2]``.
        """
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :]
        )


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to queries or keys.

    Args:
        x: Tensor of shape ``[batch, n_heads, seq_len, head_dim]``.
        cos: Cosine rotation matrix.
        sin: Sine rotation matrix.

    Returns:
        Rotated tensor with the same shape as *x*.
    """
    # x shape: [batch, n_heads, seq_len, head_dim]
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    
    # Apply rotation
    rotated = torch.cat(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
        dim=-1
    )
    return rotated


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with optional RoPE and KV-cache."""

    def __init__(self, config: ModelConfig):
        """Initialise attention layer from *config*.

        Args:
            config: Model hyper-parameters.
        """
        super().__init__()
        assert config.d_model % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.attention_dropout
        
        # Q, K, V projections
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # RoPE
        self.use_rotary = config.use_rotary
        if config.use_rotary:
            self.rope = RoPE(self.head_dim, config.max_seq_len)
        
        # Flash attention compatibility
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        use_cache: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Compute multi-head self-attention with optional KV-cache.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.
            attention_mask: Optional additive mask.
            is_causal: If ``True``, apply a causal (lower-triangular) mask.
            use_cache: If ``True``, return updated KV tensors for caching.
            kv_cache: Previously cached ``(K, V)`` tensors from prior steps.

        Returns:
            Tuple of (output ``[batch, seq_len, d_model]``, optional kv_cache).
        """
        batch_size, seq_len, d_model = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if enabled
        if self.use_rotary:
            # When using KV-cache, offset the position for the new tokens
            if kv_cache is not None:
                cache_len = kv_cache[0].shape[2]
                cos, sin = self.rope(v, cache_len + seq_len)
                # Only apply rotation for the new positions
                cos = cos[:, :, cache_len:cache_len + seq_len, :]
                sin = sin[:, :, cache_len:cache_len + seq_len, :]
            else:
                cos, sin = self.rope(v, seq_len)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        
        # Concatenate with KV-cache if present
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        
        # Store updated cache
        new_cache = (k, v) if use_cache else None
        
        # Total sequence length after cache concatenation
        total_seq_len = k.shape[2]
        
        # Attention computation
        if self.flash and attention_mask is None and kv_cache is None:
            # Use Flash Attention if available (only without KV-cache for simplicity)
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal
            )
        else:
            # Manual attention computation
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                scores = scores + attention_mask
            
            if is_causal and kv_cache is None:
                causal_mask = torch.triu(
                    torch.ones(seq_len, total_seq_len, device=x.device) * float('-inf'),
                    diagonal=1
                )
                scores = scores + causal_mask
            # With KV-cache, the new query attends to all cached keys (no masking needed)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            out = torch.matmul(attn_weights, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        
        return out, new_cache


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation."""

    def __init__(self, config: ModelConfig):
        """Initialise two-layer FFN.

        Args:
            config: Model hyper-parameters.
        """
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFN: Linear → GELU → Linear → Dropout.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.

        Returns:
            Output tensor of the same shape.
        """
        x = self.fc1(x)
        x = F.gelu(x, approximate='tanh')  # Use tanh approximation for speed
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer Block with Pre-LayerNorm and optional KV-cache."""

    def __init__(self, config: ModelConfig):
        """Initialise transformer block.

        Args:
            config: Model hyper-parameters.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        use_cache: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with pre-LN residual connections.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.
            attention_mask: Optional additive attention mask.
            is_causal: Whether to apply causal masking.
            use_cache: If ``True``, return updated KV-cache.
            kv_cache: Previously cached KV tensors for this layer.

        Returns:
            Tuple of (output tensor, optional new KV-cache).
        """
        # Pre-LN: Attention block with residual
        attn_out, new_cache = self.attn(
            self.ln1(x), attention_mask, is_causal,
            use_cache=use_cache, kv_cache=kv_cache,
        )
        x = x + attn_out
        
        # Pre-LN: FFN block with residual
        x = x + self.ffn(self.ln2(x))
        
        return x, new_cache


class FixProtoGPT(nn.Module):
    """
    FixProtoGPT: Transformer Language Model for FIX Protocol
    GPT-style decoder-only architecture
    """
    
    def __init__(self, config: ModelConfig):
        """Build the full GPT-style decoder model.

        Args:
            config: Model hyper-parameters including vocab_size,
                d_model, n_heads, n_layers, etc.
        """
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Position embedding (optional, not used with RoPE)
        if not config.use_rotary:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Output projection (language modeling head)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying: share weights between token embedding and lm_head
        self.token_embedding.weight = self.lm_head.weight
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to residual projections
        for name, param in self.named_parameters():
            if name.endswith('out_proj.weight') or name.endswith('fc2.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialise weights for *module*.

        Applies truncated-normal init with σ=0.02 for Linear/Embedding
        layers and zero-bias / ones-weight for LayerNorm.

        Args:
            module: The ``nn.Module`` being initialised.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv_caches: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Forward pass with optional KV-cache for efficient inference.
        
        Args:
            input_ids: [batch_size, seq_len]
            targets: [batch_size, seq_len] (optional, for training)
            attention_mask: [batch_size, seq_len] (optional)
            use_cache: If True, compute and return KV-caches per layer.
            past_kv_caches: List of (K, V) tuples from previous steps.
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            loss: scalar (if targets provided)
            new_kv_caches: list of (K, V) per layer (if use_cache=True)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Position embeddings (if not using RoPE)
        if not self.config.use_rotary:
            if past_kv_caches is not None and past_kv_caches[0] is not None:
                offset = past_kv_caches[0][0].shape[2]
            else:
                offset = 0
            pos = torch.arange(offset, offset + seq_len, dtype=torch.long, device=input_ids.device)
            pos_emb = self.position_embedding(pos)
            x = x + pos_emb
        
        x = self.dropout(x)
        
        # Transformer blocks with optional KV-cache
        new_kv_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_cache = None
            if past_kv_caches is not None and i < len(past_kv_caches):
                layer_cache = past_kv_caches[i]
            x, new_cache = block(
                x, attention_mask, is_causal=True,
                use_cache=use_cache, kv_cache=layer_cache,
            )
            if use_cache:
                new_kv_caches.append(new_cache)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                label_smoothing=0.1,
            )
        
        return logits, loss, new_kv_caches
    
    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        constrained_decoder,
        gen_ids: Optional[list],
    ) -> torch.Tensor:
        """Apply temperature, constrained decoding, top-k/top-p, and sample.

        Args:
            logits: Raw logits ``[batch, vocab]`` for the last position.
            temperature: Sampling temperature (higher = more random).
            top_k: Keep only *k* highest-probability tokens.
            top_p: Nucleus-sampling cumulative probability threshold.
            constrained_decoder: Optional FSM-based decoder.
            gen_ids: Full generated token list (for constrained decoder).

        Returns:
            Sampled token IDs ``[batch, 1]``.
        """
        logits = logits / temperature

        # Apply constrained decoding mask
        if constrained_decoder is not None and gen_ids is not None:
            logits[0] = constrained_decoder.apply_constraint(logits[0], gen_ids)

        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1,
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove,
            )
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
        constrained_decoder=None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively with optional KV-cache and
        constrained decoding.
        
        Args:
            input_ids: [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            eos_token_id: End of sequence token ID
            use_cache: Use KV-cache for O(n) inference instead of O(n²)
            constrained_decoder: Optional ConstrainedFIXDecoder for
                grammar-guided generation
        
        Returns:
            generated: [batch_size, seq_len + max_new_tokens]
        """
        _truncation_logged = False
        past_kv_caches = None
        
        # Initial prefill pass (process entire prompt)
        if use_cache:
            if input_ids.size(1) > self.config.max_seq_len:
                if not _truncation_logged:
                    logger.warning(
                        "Sequence length %d exceeds max_seq_len %d — "
                        "truncating to last %d tokens (earliest context lost)",
                        input_ids.size(1), self.config.max_seq_len,
                        self.config.max_seq_len,
                    )
                    _truncation_logged = True
                input_ids = input_ids[:, -self.config.max_seq_len:]
            
            logits, _, past_kv_caches = self(input_ids, use_cache=True)
            next_token = self._sample_next_token(
                logits[:, -1, :], temperature, top_k, top_p,
                constrained_decoder, input_ids[0].tolist(),
            )
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                return input_ids
            
            # Decode loop — one token at a time using KV-cache
            for _ in range(max_new_tokens - 1):
                # Only feed the last token; cache has everything else
                logits, _, past_kv_caches = self(
                    next_token, use_cache=True, past_kv_caches=past_kv_caches,
                )
                next_token = self._sample_next_token(
                    logits[:, -1, :], temperature, top_k, top_p,
                    constrained_decoder, input_ids[0].tolist(),
                )
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
            
            return input_ids
        
        # Fallback: original O(n²) generation without cache
        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.config.max_seq_len:
                if not _truncation_logged:
                    logger.warning(
                        "Sequence length %d exceeds max_seq_len %d — "
                        "truncating to last %d tokens (earliest context lost)",
                        input_ids.size(1), self.config.max_seq_len,
                        self.config.max_seq_len,
                    )
                    _truncation_logged = True
                idx_cond = input_ids[:, -self.config.max_seq_len:]
            else:
                idx_cond = input_ids
            
            logits, _, _ = self(idx_cond)
            next_token = self._sample_next_token(
                logits[:, -1, :], temperature, top_k, top_p,
                constrained_decoder, input_ids[0].tolist(),
            )
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return input_ids
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model.

        Args:
            non_embedding: If ``True``, subtract the position embedding
                parameters (when not using RoPE).

        Returns:
            Total parameter count.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.use_rotary:
            n_params -= self.position_embedding.weight.numel()
        return n_params


def create_model(config: ModelConfig) -> FixProtoGPT:
    """Factory function to create a :class:`FixProtoGPT` model.

    Args:
        config: Model hyper-parameters.

    Returns:
        Newly initialised model instance.
    """
    model = FixProtoGPT(config)
    print(f"Model created with {model.get_num_params() / 1e6:.2f}M parameters")
    return model


if __name__ == "__main__":
    # Test model creation
    config = ModelConfig()
    model = create_model(config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, _, _ = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test generation
    generated = model.generate(input_ids[:, :10], max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
