# FixProtoGPT — AI Model Deep Dive

> A stage-by-stage breakdown of the AI model: what was chosen, why, and what the alternatives were.

---

## Table of Contents

1. [Stage 1 — Architecture Choice](#stage-1--architecture-choice)
2. [Stage 2 — Positional Encoding](#stage-2--positional-encoding)
3. [Stage 3 — Attention Mechanism](#stage-3--attention-mechanism)
4. [Stage 4 — Feed-Forward Network](#stage-4--feed-forward-network)
5. [Stage 5 — Tokenisation](#stage-5--tokenisation)
6. [Stage 6 — Weight Initialisation & Tying](#stage-6--weight-initialisation--tying)
7. [Stage 7 — Training Pipeline](#stage-7--training-pipeline)
8. [Stage 8 — Decoding & Inference](#stage-8--decoding--inference)
9. [Stage 9 — Fine-Tuning (LoRA)](#stage-9--fine-tuning-lora)
10. [Stage 10 — Post-Processing & Enrichment](#stage-10--post-processing--enrichment)
11. [Stage 11 — Quantisation](#stage-11--quantisation)
12. [Stage 12 — Implementation Language & Framework](#stage-12--implementation-language--framework)
13. [Full Configuration Reference](#full-configuration-reference)
14. [Parameter Count Breakdown](#parameter-count-breakdown)

---

## Stage 1 — Architecture Choice

> **The Big Picture:** The architecture is the foundation of everything. It determines *what kind of task* the model can perform. FIX message generation is a sequence-generation problem — the model must produce tokens left-to-right, each conditioned on all previous tokens. Choosing the wrong architecture (e.g., encoder-only) would make generation physically impossible. Choosing an oversized one (e.g., fine-tuning a 7B LLM) would make deployment impractical on commodity hardware. This single decision shapes every subsequent stage.

### What We Chose: GPT-Style Decoder-Only Transformer

The model is a **decoder-only Transformer** — the same architecture family as GPT-2, GPT-3, and GPT-4 — but built from scratch in PyTorch with **19.94 million parameters**.

It uses a **Pre-LayerNorm** design, meaning Layer Normalization is applied *before* each sub-layer (attention, feed-forward) rather than after. This is the standard used by GPT-2 and all modern decoder-only models because it stabilises training, especially for smaller models.

```
Input tokens
    │
    ▼
┌─────────────────────┐
│  Token Embedding     │   Each token → 512-dimensional vector
│  (2048 × 512)       │   No position embedding (RoPE handles position)
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  × 6 Transformer Blocks (Pre-LayerNorm)             │
│                                                     │
│  For each block:                                    │
│    x_norm  = LayerNorm(x)                           │
│    attn_out = MultiHeadAttention(x_norm)            │
│    x = x + attn_out            ← residual skip      │
│                                                     │
│    x_norm  = LayerNorm(x)                           │
│    ffn_out = FeedForward(x_norm)                    │
│    x = x + ffn_out             ← residual skip      │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Final LayerNorm     │
│  + LM Head (Linear)  │   512 → 2048 (vocab size)
│  (tied with embed)   │   Outputs logits for next token prediction
└──────────────────────┘
```

### What Each Layer Contains

Each of the 6 `TransformerBlock` instances is identical in structure. Here is exactly what lives inside one block (class: `TransformerBlock` in `src/core/transformer.py`):

| # | Component | Class / Op | Shape / Config | Purpose |
|---|---|---|---|---|
| 1 | **LayerNorm 1** | `nn.LayerNorm(512)` | Normalises over d_model=512 | Pre-norm before attention — stabilises gradients (Pre-LN design) |
| 2 | **Multi-Head Self-Attention** | `MultiHeadAttention` | 8 heads × 64 dim each | Fused QKV projection (`nn.Linear(512, 1536, bias=False)`), RoPE applied to Q and K, causal mask, optional Flash Attention via `scaled_dot_product_attention`, KV-cache for incremental generation. Output projected back via `nn.Linear(512, 512)` with residual dropout |
| 3 | **Residual skip** | `x = x + attn_out` | — | Adds the original input back — prevents gradient vanishing across depth |
| 4 | **LayerNorm 2** | `nn.LayerNorm(512)` | Normalises over d_model=512 | Pre-norm before FFN |
| 5 | **Feed-Forward Network** | `FeedForward` | 512 → 2048 → 512 | Two linear layers with GELU activation (tanh approximation) between them: `Linear(512, 2048)` → `GELU` → `Linear(2048, 512)` → `Dropout` |
| 6 | **Residual skip** | `x = x + ffn_out` | — | Adds the pre-FFN input back |

The 6 blocks are sandwiched between:
- **Token Embedding** (`nn.Embedding(2048, 512)`) — no learned position embedding (RoPE handles position in attention)
- **Final LayerNorm** (`nn.LayerNorm(512)`) + **LM Head** (`nn.Linear(512, 2048, bias=False)`, weight-tied with embeddings)

### Why Not the Alternatives?

| Architecture | What It Is | Why We Didn't Use It |
|---|---|---|
| **Encoder-Only** (BERT, RoBERTa) | Processes entire input at once; good for classification and understanding | Can't *generate* text. BERT predicts masked tokens, not the next token. FIX message generation is a sequence-generation task — encoder-only models fundamentally can't do this |
| **Encoder-Decoder** (T5, BART, original Transformer) | Encoder reads input, decoder generates output. Good for translation | Overkill for FIX. Our task is "continue this sequence" (autoregressive), not "translate language A to language B". The encoder adds parameters and latency for no benefit when the input and output share the same vocabulary |
| **Fine-Tuned Large LLM** (GPT-2 124M, LLaMA 7B) | Take an existing pre-trained model and adapt it to FIX | (1) Even GPT-2 small is 124M params — 6× our size, slower on CPU. (2) Pre-trained vocabulary wastes capacity on irrelevant tokens ("the", "pizza", etc.). (3) Pre-trained weights carry unknown biases. (4) Deployment requires cloud GPUs for larger models |
| **RNN / LSTM** | Older sequence models that process tokens one by one | Slower training (no parallelism), poor at capturing long-range dependencies. A Transformer processes all positions simultaneously during training. For even 50-field FIX messages, LSTMs struggle to remember the beginning |
| **State Space Models** (Mamba, S4) | Newer sub-quadratic alternatives to Transformers | Emerging technology, less mature tooling, limited community support. Transformers are battle-tested for sequence generation. SSMs shine at very long sequences (4K+) — FIX messages are ≤512 tokens, so the quadratic cost of attention is negligible |
| **Mixture of Experts** (MoE) | Only activates a subset of parameters per token | Designed for massive models (100B+) where you want efficiency. At 20M parameters, MoE adds routing complexity for zero benefit |

### Who Else Uses This?

Decoder-only Transformers dominate modern AI. Notable examples:

| Model | Organisation | Parameters | Notes |
|---|---|---|---|
| **GPT-2** | OpenAI (2019) | 117M – 1.5B | Popularised the decoder-only + Pre-LN architecture we follow |
| **GPT-3** | OpenAI (2020) | 175B | Scaled the same architecture to few-shot learning |
| **GPT-4** | OpenAI (2023) | Undisclosed | Powers ChatGPT — same architectural family |
| **LLaMA 1/2/3** | Meta (2023–24) | 7B – 405B | Open-weight models that became the community standard |
| **Chinchilla** | DeepMind (2022) | 70B | Proved "train longer on more data" beats "make model bigger" |
| **PaLM / PaLM 2** | Google (2022–23) | 540B / undisclosed | Powers Bard/Gemini features |
| **Claude** | Anthropic (2023–24) | Undisclosed | Constitutional AI, same decoder-only foundation |
| **Mistral / Mixtral** | Mistral AI (2023–24) | 7B – 8×22B | High-performance open models; Mixtral adds MoE |
| **Falcon** | TII (2023) | 7B – 180B | Among the first fully open large decoder-only models |
| **Phi-1/2/3** | Microsoft (2023–24) | 1.3B – 14B | "Small but mighty" — proves this architecture scales *down* too |
| **StarCoder** | BigCode (2023) | 15B | Code generation — decoder-only applied to a domain (like FixProtoGPT for FIX) |

FixProtoGPT sits at the small end of this family (19.94M), proving the architecture works from 20M to 500B+ parameters.

### Why Decoder-Only is the Right Fit

FIX message generation is **autoregressive**: the model generates one token at a time, left to right, each token depending on all previous tokens. This is exactly what decoder-only Transformers are designed for:

```
"35=D" → predict "|" → predict "55=" → predict "TSLA" → predict "|" → ...
```

The causal attention mask ensures the model can only "look left" — it never peeks at future tokens during training, which matches how generation works at inference time.

---

## Stage 2 — Positional Encoding

> **The Big Picture:** Without positional encoding, the model has no concept of *order* — it would see `"35=D|55=TSLA"` and `"55=TSLA|35=D"` as identical. FIX messages have strict ordering requirements: tag 8 (BeginString) must come first, tag 10 (CheckSum) must come last, and field groups follow conventional sequences. Position encoding is what lets the model learn that "tag 35 at position 3 means something different from tag 35 at position 20" and that certain tags *must precede* others.

### What We Chose: RoPE (Rotary Position Embeddings)

RoPE encodes position information by *rotating* the query and key vectors in attention space. It pre-computes sine and cosine frequencies using a base of 10,000, then applies them to split halves of Q and K:

```
For position p and dimension pair (i, i+1):
    q_rotated[i]   = q[i] * cos(p·θ) − q[i+1] * sin(p·θ)
    q_rotated[i+1] = q[i] * sin(p·θ) + q[i+1] * cos(p·θ)

where θ = 1 / (10000^(2i/d))
```

This makes the attention score between two tokens naturally depend on their **relative distance**, not just their absolute positions.

### Why Not the Alternatives?

| Method | What It Is | Why RoPE Wins |
|---|---|---|
| **Learned Positional Embedding** (GPT-2 style) | Adds a trainable vector per position (`nn.Embedding(max_seq_len, d_model)`) | Fixed to max_seq_len seen in training — completely breaks on longer sequences. Adds parameters (512 × 512 = 262K). Learns absolute positions, not relative relationships. RoPE generalises to unseen lengths |
| **Sinusoidal (Original Transformer)** | Handcrafted sin/cos at different frequencies, added to token embeddings | Fixed, non-learnable. Added to embeddings rather than applied in attention — information degrades through layers. RoPE applies directly in the attention computation where it matters |
| **ALiBi** (Attention with Linear Biases) | Adds a linear bias to attention scores based on distance | Simpler than RoPE but less expressive. ALiBi works well for extrapolation but encodes a fixed "nearby tokens matter more" assumption. FIX messages sometimes have long-range dependencies (e.g., MsgType tag at position 3 determines required tags at position 30+) where RoPE's flexibility helps |
| **Relative Position Bias** (T5 style) | Learns a bias table for relative distances, added to attention scores | Requires extra parameters and a lookup table. Bucketed distances lose precision for exact positions. RoPE achieves relative awareness with zero extra parameters |
| **NoPE** (No Position Encoding) | Some models work without explicit positions | Only viable for very specific tasks with position-agnostic structure. FIX messages have meaningful field ordering — tags like BeginString(8) *must* come first. Position matters |

### Who Else Uses This?

RoPE was introduced by Su et al. (2021) and quickly became the default for open-weight LLMs:

| Model | Positional Encoding | Notes |
|---|---|---|
| **LLaMA 1/2/3** | **RoPE** | The paper that made RoPE the de facto standard for open models |
| **Mistral / Mixtral** | **RoPE** | Follows LLaMA's design |
| **CodeLlama** | **RoPE** with extended context | Scales RoPE frequencies for 16K+ context |
| **Qwen (1/1.5/2)** | **RoPE** | Alibaba's flagship LLM family |
| **Yi** | **RoPE** | 01.AI's models |
| **DeepSeek** | **RoPE** | Chinese open LLM, including DeepSeek-Coder |
| **Falcon** | **RoPE** (v2) / ALiBi (v1) | Falcon 1 used ALiBi; Falcon 2 switched to RoPE |
| GPT-2 / GPT-3 | Learned embeddings | Older approach — fixed max length, no relative awareness |
| Original Transformer | Sinusoidal | Handcrafted, added to embeddings, not applied in attention |
| GPT-NeoX | **RoPE** | One of the earliest open models to adopt RoPE (EleutherAI, 2022) |

### Why RoPE is the Right Fit

FIX messages have **ordered structure** — tag 8 (BeginString) must be first, tag 10 (CheckSum) must be last, and tag groups follow conventional orderings. But the ordering is **relative**, not absolute: what matters is that BeginString comes *before* MsgType, not that it's at position 0 vs position 1. RoPE naturally captures this relative relationship without learning wasted absolute-position patterns.

---

## Stage 3 — Attention Mechanism

> **The Big Picture:** Attention is how the model learns *relationships between fields*. When generating a NewOrderSingle (35=D), the model needs to know that tag 55 (Symbol), tag 54 (Side), and tag 38 (OrderQty) are required — and that tag 35's value *determines* which body tags must follow. Each attention head can specialise in a different relationship: one head might track tag-value dependencies, another might track message structure patterns, another might learn checksum-relevant positions. Without attention, the model would process each position in isolation with no awareness of context.

### What We Chose: Multi-Head Self-Attention (8 heads, combined QKV)

The attention layer uses a **single linear projection** to compute Q, K, and V simultaneously:

```python
qkv = self.c_attn(x)          # Linear(512 → 1536) — projects to 3 × d_model
q, k, v = qkv.chunk(3, dim=-1) # Split into Q, K, V — each (batch, seq_len, 512)
```

Then reshaped to 8 heads of 64 dimensions each:
```
Q, K, V: (batch, 8 heads, seq_len, 64)
```

RoPE is applied to Q and K. Then attention is computed:

```
Attention(Q, K, V) = softmax(Q × K^T / √64) × V
```

**Flash Attention** (via `F.scaled_dot_product_attention`) is used when available and no KV-cache is active. This is an optimised fused kernel that avoids materialising the full attention matrix in memory.

When KV-cache is in use (autoregressive generation), a manual attention computation is used instead, with cached K/V tensors concatenated with new values.

### Key Design Decisions

| Decision | Our Choice | Alternative | Why |
|---|---|---|---|
| **QKV projection** | Combined single `Linear(512, 1536)` | Separate Q, K, V projections | Combined is ~10% faster due to one matrix multiply instead of three. Standard in GPT-2/3. No accuracy difference |
| **Head dimension** | 64 (= 512 / 8 heads) | Other splits (e.g., 16 heads × 32) | 64 is the proven sweet spot from the original Transformer paper. 32-dim heads lose representational power; more heads with smaller dims have diminishing returns |
| **Flash Attention** | Used when available | Always manual attention | Flash Attention is 2-4× faster and uses O(n) memory instead of O(n²). We fall back gracefully when it's unavailable |
| **KV-Cache** | Enabled by default for generation | Re-encode full sequence each step | Without KV-cache, generating a 100-token message requires 100 forward passes of increasing length (O(n²) total). With cache, each step processes only 1 new token (O(n) total). Critical for real-time performance |
| **Attention dropout** | 0.1 | No dropout / higher rates | 0.1 is the standard for models this size. Prevents attention patterns from becoming too "sharp" (over-relying on a single position) |
| **Bias** | `bias=False` in all Linear layers | `bias=True` | Follows modern practice (LLaMA, GPT-NeoX). Removing bias saves parameters and has no measurable accuracy impact for decoder-only models |

### Who Else Uses This?

| Model | Attention Variant | Notes |
|---|---|---|
| **GPT-2 / GPT-3** | **Standard MHA** (combined QKV) | Our architecture follows this exact pattern |
| **BERT / RoBERTa** | Standard MHA | Encoder-side, but same multi-head mechanism |
| **LLaMA 1** | Standard MHA | 32 heads at 7B; same as ours at smaller scale |
| **LLaMA 2 (70B)** | **GQA** (8 KV heads, 64 Q heads) | Grouped-Query Attention for inference efficiency at scale |
| **Mistral 7B** | **GQA** (8 KV groups) | Even at 7B, uses GQA for faster inference |
| **Falcon 180B** | **MQA** (1 shared KV) | Extreme KV sharing for inference speed |
| **GPT-J / GPT-NeoX** | Standard MHA + **parallel attention** | Computes attention and FFN in parallel (not sequential) |

**Flash Attention** (Dao et al., 2022) is used in the training and inference stacks of LLaMA 2, Mistral, Falcon 2, and most modern frameworks (vLLM, TGI, llama.cpp). It's an *implementation optimisation*, not an architecture change — the mathematics are identical to standard attention.

### Why Not Multi-Query or Grouped-Query Attention?

| Variant | What It Is | Why We Use Standard MHA |
|---|---|---|
| **Multi-Query Attention** (MQA) | All heads share one K and V, only Q is per-head | Designed for inference-speed optimisation in very large models (30B+). At 20M params with 8 heads, the memory savings are negligible and the quality loss from sharing K/V is not worth it |
| **Grouped-Query Attention** (GQA) | K/V are shared within groups of heads | Same reasoning as MQA — it's a compromise for large models. Our model is small enough that full MHA has no performance bottleneck |

---

## Stage 4 — Feed-Forward Network

> **The Big Picture:** If attention is the model's "eyes" (deciding *what to look at*), the FFN is its "memory" (deciding *what it knows* about what it saw). The FFN layers store factual knowledge: that tag 55 expects ticker symbols like `AAPL` or `TSLA`, that tag 54 only accepts `1` (Buy) or `2` (Sell), that a Market order (40=1) doesn't need a Price tag but a Limit order (40=2) does. Without sufficient FFN capacity, the model would understand message *structure* but produce nonsensical *values*.

### What We Chose: 2-Layer FFN with GELU Activation

Each Transformer block contains a position-wise feed-forward network:

```
FFN(x) = Dropout( Linear₂( GELU( Linear₁(x) ) ) )

Where:
  Linear₁: 512 → 2048  (expand)
  GELU:    activation function (tanh approximation)
  Linear₂: 2048 → 512  (contract)
```

The FFN is where the model stores **factual knowledge** — it's the "memory" of the network. The attention layers decide *what* to look at; the FFN layers decide *what to output* based on what was seen.

### Why Not the Alternatives?

| Activation | Formula | Why GELU Wins |
|---|---|---|
| **ReLU** | max(0, x) | Simplest option. Used in the original Transformer. Problem: "dead neurons" — once a neuron outputs 0, it may never recover during training. GELU avoids this by allowing small negative values through |
| **GELU** (our choice) | x · Φ(x) where Φ is the Gaussian CDF | Smooth, non-zero gradient everywhere. Standard in GPT-2, BERT, and most modern Transformers. The `tanh` approximation we use is faster than the exact computation with negligible accuracy difference |
| **SwiGLU** (LLaMA, PaLM) | `Swish(xW₁) ⊙ (xW₂)` — gated linear unit | More expressive than GELU. LLaMA uses this with great results. BUT: it requires 3 weight matrices instead of 2 (adds ~50% more FFN parameters). For a 20M-param model where we're optimising for size, GELU with 2 matrices is the right trade-off |
| **Sigmoid** | 1/(1+e^(-x)) | Saturates at 0 and 1, causing vanishing gradients. Essentially obsolete for deep networks |
| **Swish** | x · σ(x) | Very similar to GELU in practice. GELU has slightly more theoretical backing and wider ecosystem support (default in PyTorch for Transformer architectures) |

### Who Else Uses This?

| Model | Activation | FFN Design | Notes |
|---|---|---|---|
| **GPT-2 / GPT-3** | **GELU** | 2-layer, 4× expansion | Our exact design — 2 matrices, GELU, 4:1 ratio |
| **BERT / RoBERTa** | **GELU** | 2-layer, 4× expansion | Same activation, encoder-side |
| **LLaMA 1/2/3** | **SwiGLU** | 3-layer (gated), ~2.7× | More expressive but 50% more FFN parameters |
| **PaLM / PaLM 2** | **SwiGLU** | 3-layer (gated) | Google followed the SwiGLU trend |
| **Mistral / Mixtral** | **SwiGLU** | 3-layer (gated) | Industry shift post-LLaMA |
| **Original Transformer** | **ReLU** | 2-layer, 4× expansion | The baseline that started it all |
| **GPT-J** | **GELU** | 2-layer, 4× expansion | EleutherAI followed GPT-2's design |

The trend in 2023–24 has been towards SwiGLU (used by LLaMA, PaLM, Mistral), but GELU remains the standard for GPT-2-style architectures and smaller models where the extra gating matrix isn't justified.

### Why 4× Expansion (512 → 2048)?

The FFN expansion ratio of 4:1 comes from the original Transformer paper ("Attention Is All You Need"). The expanded dimension gives the network more capacity to store factual knowledge in the intermediate layer, while the contraction back to `d_model` keeps the residual stream compact. Most models from GPT-2 to LLaMA-2 use this ratio (or 8/3× with SwiGLU to match parameter count).

---

## Stage 5 — Tokenisation

> **The Big Picture:** Tokenisation is the bridge between human-readable FIX messages and the numbers the model actually processes. A bad tokenizer can *destroy* information before the model ever sees it — splitting `"49=SENDER8"` into `["49", "=S", "ENDER", "8"]` forces the model to waste capacity reassembling fragments. A good tokenizer preserves the structural boundaries that matter: `=` separates tags from values, `|` separates fields. For a 20M-param model that can't brute-force its way through poor tokenisation (unlike a 175B model), getting this right is critical.

### What We Chose: Custom FIX-Aware BPE Tokenizer

Two tokenizer implementations are provided (selectable via config):

**Primary: `FixProtocolTokenizer`** — Custom BPE with FIX-specific vocabulary building:
1. Character-level base vocabulary (up to 60% of vocab slots)
2. FIX-specific tokens added: common tags (`35=`, `55=`, `49=`, etc.), message types (`D`, `8`, `V`, `W`), and structural patterns (`8=FIX`)
3. Word-frequency BPE merges fill remaining slots

**Alternative: `FixProtocolBPETokenizer`** — HuggingFace Tokenizers library with true byte-pair encoding, FIX-aware pre-tokenization (splits on `|` and `=`), and NFC Unicode normalisation.

**Both share the same 7 special tokens:**

| Token | ID | Purpose |
|---|---|---|
| `<\|pad\|>` | 0 | Padding for batching |
| `<\|bos\|>` | 1 | Beginning of sequence |
| `<\|eos\|>` | 2 | End of sequence |
| `<\|fix\|>` | 3 | "Now generating FIX" trigger |
| `<\|field\|>` | 4 | FIX field boundary marker |
| `<\|eom\|>` | 5 | End of FIX message |
| `<\|unk\|>` | 6 | Unknown token fallback |

**Encoding a FIX message:**
```
Input:  "35=D|55=TSLA|54=1|38=100|"

Tokens: [<|fix|>] [35=] [D] [<|field|>] [55=] [TSLA] [<|field|>] [54=] [1]
        [<|field|>] [38=] [100] [<|field|>] [<|eom|>]
```

### Why Not the Alternatives?

| Tokenizer | What It Is | Why Custom FIX-Aware Wins |
|---|---|---|
| **GPT-2 BPE (tiktoken)** | General-purpose tokenizer trained on internet text | Tokenizes `"49=SENDER8"` as `["49", "=S", "ENDER", "8"]` — completely destroys FIX structure. Tag numbers split unpredictably. The model would need to learn to *reassemble* fragments before understanding them |
| **WordPiece** (BERT) | Subword tokenizer using likelihood-based merges | Same problem as GPT-2 BPE — not FIX-aware. Would split `"8=FIX.4.4"` into subwords that cross the `=` boundary |
| **SentencePiece** (Unigram/BPE) | Language-agnostic subword model | Better than GPT-2 BPE for uncommon domains, but still doesn't understand that `=` and `|` are structural delimiters in FIX. Would merge `"=FIX"` as one token, obscuring the tag-value boundary |
| **Character-level** | Each character is a token | Would work correctly but is extremely inefficient: a 120-character FIX message = 120 tokens. Our tokenizer compresses the same message to ~30 tokens, making training 4× more efficient and generation 4× faster |
| **Byte-level BPE** (GPT-3.5/4) | BPE on raw bytes (256 base tokens) | Handles any input but with no domain awareness. FIX tags would be split across multi-byte merge boundaries. Works for general-purpose 100B+ models that can afford to brute-force structure learning; a 20M model cannot |

### Who Else Uses This?

| Model | Tokenizer | Vocab Size | Notes |
|---|---|---|---|
| **GPT-2** | Byte-level BPE | 50,257 | General-purpose English text |
| **GPT-3 / GPT-3.5** | Byte-level BPE | 50,257 | Same tokenizer as GPT-2 |
| **GPT-4** | Byte-level BPE (cl100k) | ~100,000 | Larger vocab for multilingual + code |
| **LLaMA 1/2** | SentencePiece (BPE) | 32,000 | Language-agnostic subword model |
| **LLaMA 3** | tiktoken BPE | 128,000 | Massive vocab for multilingual |
| **BERT** | WordPiece | 30,522 | Subword splits using `##` prefix |
| **Mistral** | SentencePiece (BPE) | 32,000 | Follows LLaMA's tokenizer design |
| **StarCoder** | BPE (code-aware) | 49,152 | Domain-specific tokenizer for code — conceptually similar to our FIX-aware approach |
| **FixProtoGPT** | **Custom FIX-aware BPE** | **2,048** | Tiny vocab tuned for a single domain |

Our approach is most similar to **StarCoder's philosophy**: build a domain-aware tokenizer that understands the structural delimiters of your data (for them: code syntax; for us: FIX `=` and `|` delimiters). The key difference is scale — general-purpose models need 30K–128K tokens to cover human language, while FIX protocol's constrained vocabulary allows just 2,048.

### Vocab Size: 2,048

The model's vocabulary is 2,048 tokens. This covers:
- ~200 FIX tag numbers
- ~50 message type codes
- ~100 enumeration values
- Common identifiers, timestamps, and numbers
- BPE sub-word units for less common patterns

Larger vocabularies (e.g., 32K for GPT-2) waste embedding parameters on tokens that never appear in FIX data. Smaller vocabularies (e.g., 256) over-fragment tokens and increase sequence lengths.

---

## Stage 6 — Weight Initialisation & Tying

> **The Big Picture:** Initialisation determines whether training *starts at all*. Neural networks learn by adjusting weights from their initial values — if those values are too large, gradients explode and training diverges instantly; too small, gradients vanish and the model learns nothing. For a 6-layer Transformer, the residual connections compound this effect: without scaled initialisation, signals grow exponentially through layers. Weight tying additionally saves over 1M parameters (~5% of the model) — meaningful at our 20M scale, where every parameter must earn its place.

### Initialisation Strategy

| What | Method | Why |
|---|---|---|
| **Linear layers** | `Normal(μ=0, σ=0.02)`, bias → zeros | Standard GPT-2 initialisation. Small standard deviation prevents early saturation |
| **Embedding layer** | `Normal(μ=0, σ=0.02)` | Same as linear layers — embeddings are effectively a lookup of learned vectors |
| **LayerNorm** | Weight → ones, bias → zeros | Starts as identity transformation — no scaling or shifting initially |
| **Residual projections** (`out_proj.weight`, `fc2.weight`) | `Normal(μ=0, σ=0.02 / √(2·n_layers))` | **GPT-2 scaled init** — prevents the residual stream from growing with depth. With 6 layers: σ = 0.02 / √12 ≈ 0.0058. This ensures early training stability |

### Why This and Not Others?

| Strategy | What It Is | Why Ours is Better for This Model |
|---|---|---|
| **Xavier/Glorot** | Scales init by `1/√(fan_in + fan_out)` | Designed for sigmoid/tanh activations. Works poorly with ReLU/GELU families and Pre-LN Transformers |
| **Kaiming (He)** | Scales by `√(2/fan_in)` — designed for ReLU | Better than Xavier for ReLU, but GPT-2 style Normal(0, 0.02) with residual scaling is the proven standard for decoder-only Transformers |
| **Zero init** | All weights = 0 | Fatal for neural networks — all neurons would compute identical gradients (symmetry problem) |
| **Large random** | Normal(0, 1) or larger | Causes immediate exploding gradients in deep networks. LayerNorm can partially compensate but training is unstable |

### Who Else Uses This?

| Model | Initialisation | Weight Tying | Notes |
|---|---|---|---|
| **GPT-2** | Normal(0, 0.02) + residual scaling | ✅ Yes | Our *exact* initialisation strategy comes from GPT-2 |
| **GPT-3** | Same as GPT-2 | ✅ Yes | Scaled up but kept the same init |
| **LLaMA 1/2** | Normal(0, 0.02) | ❌ No | LLaMA does *not* tie weights (separate LM head) — affordable at 7B+ scale |
| **LLaMA 3** | RMSNorm + custom | ✅ Yes (8B only) | Ties at 8B, untied at 70B+ |
| **ALBERT** | — | ✅ Yes (extreme) | Ties weights *across layers* too — maximum parameter sharing |
| **T5** | — | ✅ Yes | Encoder-decoder, but ties embeddings with decoder output |
| **Mistral 7B** | — | ❌ No | Follows LLaMA's untied approach |

Weight tying is almost universal for models under ~7B parameters. For our 20M-param model, tying saves 5% of total parameters — a significant win.

### Weight Tying

The **token embedding layer** and the **output projection (LM head)** share the same weight matrix:

```python
self.token_embedding.weight = self.lm_head.weight   # Same tensor, not a copy
```

**Why:** The embedding maps tokens → vectors, and the LM head maps vectors → token probabilities. These are inverse operations — sharing weights makes them learn a consistent mapping. This saves 512 × 2048 = **1,048,576 parameters** (~5% of total) with no quality loss. Used by GPT-2, LLaMA, and virtually all modern language models.

---

## Stage 7 — Training Pipeline

> **The Big Picture:** Training is where the model actually *learns* FIX protocol patterns. Every other stage defines the model's potential — training realises it. The optimiser, learning rate schedule, batch size, and loss function collectively determine whether the model converges to a good solution, overfits to the training data, or fails to learn at all. For a domain-specific model trained on relatively small FIX datasets (vs. internet-scale corpora), every training decision is amplified — there's less data to compensate for suboptimal hyperparameters.

### Optimiser: AdamW

```
For each parameter:
    m = β₁ · m + (1-β₁) · gradient           ← moving average of gradients
    v = β₂ · v + (1-β₂) · gradient²          ← moving average of squared gradients
    param = param - lr · (m/√v + λ · param)   ← update with weight decay
```

| Parameter | Value | Why |
|---|---|---|
| **β₁** | 0.9 | How much the optimiser "remembers" gradient direction. 0.9 is standard — smooths out noisy gradients |
| **β₂** | 0.95 | How much it remembers gradient magnitude. Lower than default 0.999 — follows GPT-3/Chinchilla findings that 0.95 converges better for small-batch Transformer training |
| **Weight decay** | 0.1 | Gently shrinks weights to prevent overfitting. Applied only to 2D+ parameters (weight matrices), not biases or LayerNorm |

### Why Not Other Optimisers?

| Optimiser | What It Is | Why AdamW Wins for This Task |
|---|---|---|
| **SGD** (Stochastic Gradient Descent) | Simplest: `param -= lr × gradient` | No momentum or adaptive rates. Requires very careful LR tuning and many more epochs to converge. Practically unusable for Transformers without extensive hyperparameter search |
| **Adam** (original) | Like AdamW but weight decay is applied *inside* the adaptive rate | L2 regularisation in Adam interacts poorly with the adaptive learning rates — some parameters get regularised more than intended. AdamW decouples weight decay, fixing this issue |
| **AdaFactor** | Memory-efficient Adam variant that factorises second moments | Saves memory on very large models (1B+). At 20M params, memory is not a concern and AdamW's full second-moment tracking gives better convergence |
| **LAMB / LARS** | Large-batch optimisers with per-layer learning rate scaling | Designed for batch sizes of 4K–64K. Our effective batch is 32 — these add complexity for no benefit |
| **Lion** | Newer optimiser using sign of momentum | Promising but less battle-tested than AdamW. For a small model where training cost is low anyway, the risk of an under-studied optimiser outweighs potential gains |
| **Sophia** | Second-order optimiser using Hessian diagonal | Theoretically converges in fewer steps. Practically, the Hessian computation adds overhead per step. For our training scale (~100K tokens), AdamW with cosine schedule converges in reasonable time |

### Who Else Uses This?

| Model | Optimiser | β₂ | Weight Decay | Notes |
|---|---|---|---|---|
| **GPT-3** | AdamW | 0.95 | 0.1 | Our *exact* hyperparameters come from the GPT-3 paper |
| **Chinchilla** | AdamW | 0.95 | 0.1 | Confirmed β₂=0.95 outperforms 0.999 for Transformers |
| **LLaMA 1/2** | AdamW | 0.95 | 0.1 | Meta followed the same recipe |
| **PaLM** | Adafactor (modified) | — | — | Google used a memory-efficient variant for 540B scale |
| **Mistral** | AdamW | 0.95 | 0.1 | Same recipe as LLaMA |
| **GPT-2** | Adam (not W) | 0.999 | — | Older setup — AdamW with β₂=0.95 is the refined version |

The **AdamW + β₂=0.95 + wd=0.1** combination has become the standard recipe for Transformer training, validated across GPT-3, Chinchilla, and LLaMA.

### Learning Rate Schedule: Cosine Annealing with Warmup

```
LR
 │
 │     1.5e-4 ─────╲
 │    /              ╲
 │   / warmup         ╲  cosine decay
 │  /  (500 steps)     ╲
 │ /                    ╲
 │/                      ╲───── 1e-5 (floor)
 └─────────────────────────────────── step
      500        Training Steps →
```

| Phase | What Happens | Why |
|---|---|---|
| **Warmup (0 → 500 steps)** | LR ramps linearly from 0 to 1.5e-4 | Prevents early training instability. The model's initial random weights produce large gradients — small learning rates during warmup avoid overshooting |
| **Cosine decay (500 → end)** | LR smoothly decreases following a cosine curve to 1e-5 | Starts fast (exploring the loss landscape broadly) then slows down for fine-grained convergence. Smoother than step decay — no sudden LR drops that can destabilize training |

### Why Not Other Schedules?

| Schedule | How It Works | Why Cosine is Better Here |
|---|---|---|
| **Constant** | Same LR throughout | No adaptation — wastes early steps (LR too small for initial random weights) and late steps (LR too large, prevents convergence) |
| **Step decay** | Drops LR by a factor at fixed epochs | Requires choosing *when* to drop — a hyperparameter that changes with dataset size. Sudden drops can cause loss spikes |
| **Linear decay** | LR decreases linearly | Works, but decays too quickly in the middle of training. Cosine keeps a higher LR longer, allowing more exploration before convergence |
| **Inverse square root** (original Transformer) | `LR = 1/√step` | Decays too aggressively for small models. Designed for the original Transformer's "big model, big data" regime |

### Who Else Uses This?

| Model | LR Schedule | Warmup | Notes |
|---|---|---|---|
| **GPT-3** | Cosine → min LR | 375M tokens | Our schedule directly follows GPT-3's approach |
| **LLaMA 1/2** | Cosine → 10% of peak | 2000 steps | Same cosine shape, longer warmup for larger batches |
| **Chinchilla** | Cosine → min LR | — | DeepMind validated cosine as optimal for compute-efficient training |
| **Mistral** | Cosine | — | Industry standard |
| **Original Transformer** | Inverse sqrt | 4000 steps | Older approach — too aggressive for small models |
| **BERT** | Linear decay | 10K steps | Simpler but less effective than cosine |

Cosine annealing with warmup is the most widely used schedule in modern LLM training. Our choice of 500 warmup steps is proportional to our smaller training scale (vs. thousands for billion-parameter models).

### Gradient Accumulation & Batch Size

```
Effective batch = micro_batch_size × gradient_accumulation_steps
              32 = 8 × 4

Step 1: Forward+backward on 8 samples → accumulate gradients
Step 2: Forward+backward on 8 samples → accumulate gradients
Step 3: Forward+backward on 8 samples → accumulate gradients
Step 4: Forward+backward on 8 samples → accumulate gradients
         → Clip gradients (max norm 1.0)
         → Optimizer step (updates weights)
         → Zero gradients
```

**Why batch size 32?** Too small (4-8) → noisy gradients, unstable training. Too large (256+) → each step is expensive, and small datasets may not have enough diversity per batch. 32 is a proven sweet spot for Transformer training at this scale.

**Why gradient accumulation instead of a real batch of 32?** Memory. Processing 32 sequences of 512 tokens simultaneously requires ~2GB+ of activation memory. By processing 4 micro-batches of 8, peak memory usage is ~500MB — feasible on consumer hardware.

### Mixed Precision (FP16)

On CUDA GPUs, the training loop uses automatic mixed precision:
- **Forward pass** runs in FP16 (16-bit floats) → 2× less memory, faster math
- **Loss scaling** via `GradScaler` prevents underflow in FP16 gradients
- **Optimizer step** runs in FP32 (full precision) to avoid accumulated rounding errors

On Apple Silicon (MPS) and CPU, mixed precision is disabled — MPS doesn't fully support GradScaler, and CPU has no FP16 speed benefit.

### Loss Function: Cross-Entropy with Label Smoothing

```python
F.cross_entropy(logits, targets, ignore_index=-1, label_smoothing=0.1)
```

| Component | What It Does |
|---|---|
| **Cross-entropy** | Measures how different the model's predicted probability distribution is from the true next token. Standard for language modelling |
| **Label smoothing (0.1)** | Instead of training the model to put 100% probability on the correct token, targets become 90% correct + 10% spread across all tokens. This reduces overconfidence and improves generation diversity |
| **ignore_index=-1** | Padding tokens are excluded from the loss — the model isn't penalized for positions that don't contain real data |

### Checkpointing & Safety

- **Regular checkpoints** every 500 training steps
- **Best model** saved whenever validation loss improves
- **Emergency save** on SIGINT (Ctrl+C) or SIGTERM — no training progress is lost
- **Exception handling** — `emergency_save_on_error=True` catches crashes and saves before exiting
- Only the 2 most recent step-checkpoints are kept (disk space management)
- Each checkpoint bundles: model weights, optimizer state, tokenizer, config, training data, and metadata

---

## Stage 8 — Decoding & Inference

> **The Big Picture:** A perfectly trained model is useless without the right decoding strategy. Decoding controls the *quality, validity, and diversity* of generated FIX messages. Too much randomness → invalid messages with wrong tag-value combinations. Too little → repetitive, degenerate output. For FIX protocol specifically, structural validity is non-negotiable — a malformed message will be rejected by any counterparty's FIX engine. The three decoding strategies offer different trade-offs between speed, quality, and guaranteed validity, making the system adaptable to different use cases.

At inference time, the trained model generates FIX messages using one of three strategies:

### Strategy 1: Sampling (Default)

The model predicts a probability distribution over all 2,048 tokens, then **samples** from it with controlled randomness:

```
Raw logits from model
       │
       ▼
  ┌──────────────────────┐
  │ Temperature Scaling   │   logits = logits / 0.8
  │ (temp = 0.8)          │   Lower → more deterministic
  │                       │   Higher → more random
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ Constrained Mask      │   (optional) FSM masks illegal tokens
  │                       │   Only if constrained decoding is ON
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ Top-k Filtering       │   Keep only top 50 tokens
  │ (k = 50)              │   Set rest to -infinity
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ Top-p (Nucleus)       │   Sort by probability, keep tokens
  │ (p = 0.95)            │   until cumulative prob reaches 95%
  │                       │   Discard the long tail
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ Softmax → Sample      │   Convert to probabilities
  │                       │   Random draw from distribution
  └──────────────────────┘
```

| Parameter | Value | Effect |
|---|---|---|
| **Temperature** | 0.8 | Moderate sharpening — favours higher-probability tokens while allowing some variety |
| **Top-k** | 50 | Hard cutoff at 50 candidates. Eliminates very unlikely tokens that could produce invalid FIX |
| **Top-p** | 0.95 | Dynamic cutoff — keeps however many tokens are needed to cover 95% of probability mass. Adapts to situations where the model is confident (few tokens) vs uncertain (many tokens) |

### Strategy 2: Beam Search

Explores multiple candidate messages in parallel:

```
Start: "<|fix|> 35="
       │
       ├── Beam 1: "35=D"     (log_prob: -0.12)
       ├── Beam 2: "35=8"     (log_prob: -0.35)
       ├── Beam 3: "35=V"     (log_prob: -0.58)
       └── Beam 4: "35=0"     (log_prob: -0.72)
                    │
            Each beam continues expanding...
                    │
                    ▼
         Finished beams are re-ranked by:
         score = (1 - 0.3) × normalised_log_prob + 0.3 × fix_validity_score
```

**FIX Validity Scoring** (0.0 to 1.0):

| Criterion | Weight | What It Checks |
|---|---|---|
| Tag=Value structure | 0.30 | Every field matches the pattern `digits=value` |
| Required tags present | 0.25 | Has tags 8, 9, 35, 49, 56, 10 (header/trailer essentials) |
| Message-type body tags | 0.15 | Correct fields for the specific MsgType (e.g., NewOrderSingle needs 55, 54, 38) |
| BeginString correct | 0.10 | Tag 8 starts with `FIX` or `FIXT` |
| No duplicate tags | 0.10 | Same tag doesn't appear twice (except in repeating groups) |
| Checksum present | 0.05 | Tag 10 exists |
| Reasonable field count | 0.05 | Between 5 and 50 fields |

### Strategy 3: Constrained Decoding (FSM-Guided)

A **finite state machine** forces the model to produce structurally valid FIX messages at every step:

```
  ┌──────────────────┐     digit/tag-pattern     ┌──────────┐
  │  EXPECT_TAG      │ ─────────────────────────► │  IN_TAG  │
  │  (or END)        │                            └────┬─────┘
  └──────────────────┘                                 │
          ▲                                     "=" character
          │                                            │
     "|" delimiter                                     ▼
          │                                     ┌──────────────┐
  ┌───────┴──────────┐                          │ EXPECT_EQUALS│
  │  EXPECT_DELIM    │                          └──────┬───────┘
  └───────▲──────────┘                                 │
          │                                     any value token
     value tokens                                      │
          │                                            ▼
          └────────────────────────────────────  ┌──────────┐
                                                 │ IN_VALUE │
                                                 └──────────┘
```

At each decoding step:
1. Determine the current FSM state from the generated text so far
2. Identify which tokens are **legal** in that state (pre-computed sets)
3. Set all **illegal** token logits to `-infinity` before sampling
4. The model can only output structurally valid FIX syntax

### Who Else Uses This?

| Technique | Used By | Notes |
|---|---|---|
| **Top-p (nucleus) sampling** | GPT-3 API, ChatGPT, Claude, Gemini | Introduced by Holtzman et al. (2020, "The Curious Case of Neural Text Degeneration"). Now the default for all major chat APIs |
| **Top-k sampling** | GPT-2 (original demo), early Hugging Face | Simpler but less adaptive than top-p; often combined with it (as we do) |
| **Temperature scaling** | All major LLMs | Universal technique — every inference API exposes a temperature parameter |
| **Beam search** | Google Translate, MarianMT, BART summarisation | Standard for machine translation where one "best" output is wanted. Less common for open-ended generation |
| **Constrained/guided decoding** | Microsoft Guidance, Outlines (dottxt), LMQL, jsonformer | Growing trend for structured output. Outlines uses FSM-guided decoding very similar to our approach |

Our combination of top-k + top-p + temperature is the same stack used by the ChatGPT and Claude APIs. The constrained decoding FSM approach parallels the open-source **Outlines** library (2023), which uses finite state machines to guarantee JSON/regex-conforming LLM output.

### Comparison of Strategies

| | Sampling | Beam Search | Constrained |
|---|---|---|---|
| **Speed** | Fast (1 forward pass / token) | 4× slower (beam_width=4) | Fast (tiny overhead per step) |
| **Quality** | Good — usually valid FIX | Best — explores multiple options | Guaranteed valid structure |
| **Variety** | High — different outputs each time | Low — deterministic search | Medium — model picks values, FSM enforces structure |
| **Use case** | General-purpose generation | High-stakes, accuracy-critical | When structural validity is non-negotiable |

---

## Stage 9 — Fine-Tuning (LoRA)

> **The Big Picture:** The initial model is trained on generic FIX specification data, but real-world usage reveals gaps — a user's firm may use non-standard tags, prefer specific field orderings, or need messages for uncommon instrument types. Fine-tuning lets the model *adapt to its users* without full retraining. LoRA makes this practical: a user upvotes or downvotes generated messages, and within minutes the model adjusts. Without fine-tuning, the model is frozen at deployment — unable to improve from feedback or adapt to firm-specific conventions.

### What We Chose: Low-Rank Adaptation (LoRA)

After deployment, user feedback (👍/👎) drives incremental improvement. Instead of retraining all 19.94M parameters, LoRA adds small **adapter matrices** to the attention layers:

```
Original:   y = x × W              (W is frozen — not updated)

With LoRA:  y = x × W + x × A × B × (α/r)
                         └───┬───┘
                     Small trainable matrices
                     A: (8 × 512)     ← rank × input
                     B: (512 × 8)     ← output × rank
```

| Parameter | Value | Meaning |
|---|---|---|
| **Rank (r)** | 8 | Dimensionality of the adapter. Higher = more capacity, more parameters. 8 is the standard starting point |
| **Alpha (α)** | 16.0 | Scaling factor. Effective scale = α/r = 2.0. Controls how much the adapter influences the output |
| **Dropout** | 0.0 | No dropout on the LoRA path — fine-tuning data is already curated (positive examples) |
| **Target modules** | q_proj, k_proj, v_proj, out_proj | All four attention projections. These control *what the model attends to* — the most impactful place to adapt |
| **Steps** | 500 | Quick fine-tune — enough to learn from 5+ positive examples without overfitting |
| **Learning rate** | 1e-4 | Lower than initial training (1.5e-4) for gentle adaptation |

**Initialisation:** A is Kaiming uniform (random), B is all zeros → the adapter starts as an identity (zero effect), then gradually learns the correction.

**Merge:** After fine-tuning, `W' = W + (α/r) × B × A` folds the adapter back into the base weight. The model returns to its original architecture with no inference overhead.

### Who Else Uses This?

LoRA was introduced by **Hu et al. (2021)** at Microsoft and has become the dominant fine-tuning method:

| Application | Method | Notes |
|---|---|---|
| **Alpaca-LoRA** | LoRA on LLaMA 7B/13B | One of the first open LLaMA fine-tunes; proved LoRA works for instruction tuning |
| **QLoRA** (Dettmers et al., 2023) | LoRA on 4-bit quantised LLaMA | Enabled fine-tuning 65B models on a single 48GB GPU |
| **Stable Diffusion LoRAs** | LoRA on U-Net attention layers | Massive community of custom LoRA adapters for image generation (same technique, different domain) |
| **LLaMA-Factory** | LoRA/QLoRA on any LLM | Popular open-source fine-tuning framework — LoRA is the default method |
| **OpenAI Fine-Tuning API** | Undisclosed (likely LoRA-like) | Even proprietary APIs offer parameter-efficient fine-tuning |
| **Hugging Face PEFT** | LoRA, AdaLoRA, prefix tuning | `peft` library makes LoRA a one-line addition — industry standard toolkit |
| **Mistral fine-tunes** | LoRA (community) | Most Mistral adaptations use LoRA via Axolotl or LLaMA-Factory |

Our LoRA config (rank=8, alpha=16, targeting Q/K/V/O projections) matches the most common community defaults for LLaMA fine-tuning.

### Why Not the Alternatives?

| Method | What It Is | Why LoRA Wins Here |
|---|---|---|
| **Full fine-tuning** | Update all 19.94M parameters | (1) Needs all the original training data to avoid catastrophic forgetting. (2) Takes hours instead of minutes. (3) Requires full optimizer state (3× memory). For 5 user corrections, this is massive overkill |
| **Prompt tuning** | Learn a "soft prompt" (trainable prefix vectors) prepended to input | Only ~1K trainable parameters — too limited to meaningfully change FIX generation patterns. Works for steering style, not for learning new tag-value relationships |
| **Adapter layers** (Houlsby-style) | Insert small feedforward layers between Transformer blocks | Adds inference latency — every forward pass goes through extra layers. LoRA adds zero latency after merging. Adapters can't be merged into the base model |
| **QLoRA** | LoRA on quantised (4-bit) base model | Designed for fine-tuning 65B+ models on consumer GPUs. Our 20M model fits in memory with full precision — quantising would only hurt quality |
| **Prefix tuning** | Prepend learnable key-value pairs to attention | Limited capacity, doesn't adapt the actual attention computation. LoRA directly modifies Q, K, V projections for deeper adaptation |
| **BitFit** | Only fine-tune bias terms | Extremely parameter-efficient but can only make minor adjustments. Can't learn new FIX patterns from user corrections |

---

## Stage 10 — Post-Processing & Enrichment

> **The Big Picture:** The model generates the *semantic core* of FIX messages (which fields, which values), but a valid FIX message requires precise mechanical elements: a correct BodyLength (tag 9) computed from the message body, a CheckSum (tag 10) computed as the byte sum mod 256, sequential MsgSeqNum values, and UTC timestamps accurate to the millisecond. These are deterministic calculations, not predictions — asking a neural network to compute checksums would waste model capacity on arithmetic. The enrichment pipeline ensures every output is a standards-compliant, transmission-ready FIX message.

The model's raw output is a sequence of tokens that approximates a FIX message. The **enrichment pipeline** transforms it into a production-ready, standards-compliant message:

```
Model output:   "35=D|55=TSLA|54=1|38=100|40=1|"

After enrichment:
  8=FIX.4.4|        ← BeginString (from active FIX version)
  9=154|             ← BodyLength (computed from message content)
  35=D|              ← MsgType (from model output)
  49=SENDER8|        ← SenderCompID (generated)
  56=TARGET10|       ← TargetCompID (generated)
  34=1385|           ← MsgSeqNum (sequential)
  52=20260329-...|   ← SendingTime (current UTC timestamp)
  11=ORD14090|       ← ClOrdID (generated if missing)
  21=1|              ← HandlInst (default: automated)
  55=TSLA|           ← Symbol (from model output)
  54=1|              ← Side (from model output)
  38=100|            ← OrderQty (from model output)
  40=1|              ← OrdType (from model output)
  59=4|              ← TimeInForce (added if missing)
  60=20260329-...|   ← TransactTime (current UTC timestamp)
  10=222|            ← CheckSum (computed: sum of all bytes mod 256)
```

This two-stage approach (model generates core content, pipeline adds headers/trailers) is deliberate: it lets the model focus on learning the **semantic** parts of FIX messages (which fields go together, correct values) while the pipeline handles the **mechanical** parts (checksums, timestamps, sequence numbers) that are deterministic calculations, not predictions.

---

## Stage 11 — Quantisation

> **The Big Picture:** A trained model in FP32 uses ~80MB of memory. While manageable, reducing this to ~40MB (FP16) or ~20MB (INT8) makes deployment on edge devices, containers, and shared servers significantly more efficient. In financial environments where FIX engines run alongside dozens of other processes, minimising the model's memory footprint matters. Quantisation also accelerates inference — INT8 operations are faster on CPUs, and FP16 leverages GPU tensor cores — reducing the latency between a user's request and a generated FIX message.

For deployment on resource-constrained hardware, the trained model can be quantised:

| Mode | What It Does | Size Reduction | Speed Impact | Quality Impact |
|---|---|---|---|---|
| **Dynamic INT8** (CPU) | Replaces 32-bit floating point weights with 8-bit integers, quantised on-the-fly per inference | ~4× smaller | Faster (CPU) | Minimal — attention patterns with 2048-vocab are robust to quantisation |
| **FP16** (GPU/MPS) | Uses 16-bit half-precision floats | ~2× smaller | 2× faster with hardware support | Negligible — FP16 has sufficient precision for inference |
| **None** (default) | Full FP32 precision | Baseline | Baseline | Baseline |

### Who Else Uses This?

| Technique | Used By | Notes |
|---|---|---|
| **Dynamic INT8** | PyTorch native (`torch.quantization`) | Our approach — simple, no calibration data needed, good for small models |
| **FP16 inference** | All major GPU-served LLMs | Standard for GPU deployment — 2× memory savings, hardware-accelerated |
| **GPTQ** (Frantar et al., 2022) | TheBloke's Hugging Face models, llama.cpp | Post-training 4-bit quantisation for large models |
| **AWQ** (Lin et al., 2023) | vLLM, TGI serving | Activation-aware 4-bit quantisation — better quality than GPTQ |
| **bitsandbytes** (Dettmers) | QLoRA, Hugging Face Transformers | NF4/INT8 quantisation enabling fine-tuning of quantised models |
| **GGUF / llama.cpp** | Local LLM community | Various quantisation levels (Q4_K_M, Q5_K_M, etc.) for CPU inference |

For models under 100M parameters, dynamic INT8 or FP16 is standard practice. The more aggressive methods (GPTQ, AWQ, GGUF) are designed for models that *wouldn't fit in memory* at full precision — a problem we don't have at 20M parameters.

**Why not INT4 or GPTQ?** These aggressive quantisation methods are designed for 7B-70B models where memory is critical. At 20M parameters (~80MB), our model fits comfortably in memory at full precision. Aggressive quantisation risks disproportionate quality loss on a small model.

---

## Stage 12 — Implementation Language & Framework

> **The Big Picture:** The choice of programming language and framework determines development velocity, ecosystem access, hardware compatibility, and long-term maintainability. A model that can't be trained, served, and debugged efficiently is a model that never ships. For an AI project spanning custom architecture definition, GPU-accelerated training, REST API serving, and interactive fine-tuning, the language/framework choice affects every other stage — from how the model is defined (Stage 1) to how it's quantised and deployed (Stage 11).

### What We Chose: Python + PyTorch

The entire model — architecture, training, inference, tokenisation, and serving — is implemented in **Python 3.10+** using **PyTorch ≥ 2.0** as the deep learning framework.

| Layer | Choice | Role |
|---|---|---|
| **Language** | Python 3.10+ | All model code, training scripts, API server, CLI |
| **DL Framework** | PyTorch ≥ 2.0 | Tensor ops, autograd, GPU/MPS acceleration, model definition |
| **Serving** | Flask + Gunicorn | REST API for inference |
| **Tokenizer lib** | HuggingFace `tokenizers` (Rust core) | Fast BPE via `FixProtocolBPETokenizer` alternative |
| **Data handling** | Pure Python + JSON | FIX spec parsing, dataset preparation |
| **Database** | SQLite (via `sqlite3` stdlib) | User interactions, model metadata, sessions |

### Why Python?

| Language | What It Offers | Why Python Wins for This Project |
|---|---|---|
| **Python** (our choice) | Dominant ML/AI ecosystem, massive library support, rapid prototyping | PyTorch, Hugging Face, NumPy, Flask — the entire ML toolchain is Python-native. Development speed is critical for a research-to-production project |
| **C++** | Maximum runtime performance | PyTorch's core *is* C++ (libtorch). Writing the model in C++ gives ~10–20% inference speedup but 5–10× slower development. For a 20M-param model, Python's overhead is negligible |
| **Rust** | Memory safety, performance | Growing ML ecosystem (candle, burn) but immature compared to PyTorch. HuggingFace's `tokenizers` library *is* Rust under the hood — we get Rust speed where it matters most (tokenisation) via Python bindings |
| **Julia** | Designed for scientific computing | Excellent for numerical work but tiny DL ecosystem. No equivalent to PyTorch's community, pre-built layers, or GPU kernels |
| **Java / Scala** | Enterprise ecosystem, JVM performance | DL4J exists but has a fraction of PyTorch's features. FIX protocol *is* common in Java shops, but model training tooling is far weaker |
| **JavaScript / TypeScript** | Web-native | TensorFlow.js exists but is limited to inference of small models. No serious training support. Our JS frontend calls the Python API instead |

### Why PyTorch?

| Framework | What It Is | Why PyTorch Wins |
|---|---|---|
| **PyTorch** (our choice) | Dynamic computation graph, eager execution, Pythonic API | Industry standard for research *and* production. `torch.compile` (2.0+) bridges the eager/compiled gap. Native support for MPS (Apple Silicon), CUDA, and CPU |
| **TensorFlow / Keras** | Static graph (v1) → eager (v2), Google-backed | Was dominant pre-2020 but PyTorch overtook it in research and increasingly in production. TF2/Keras is viable but has a smaller community for custom architectures |
| **JAX** | Functional transforms (grad, jit, vmap), Google Research | Excellent for research (used by DeepMind for Gemini). Steeper learning curve, less mature serving ecosystem. Best for teams already in the Google/TPU ecosystem |
| **MLX** | Apple Silicon-native, NumPy-like API | New (2023), optimised for M-series chips. Promising but tiny ecosystem — no equivalent to PyTorch's library of pre-built modules, optimisers, and data loaders |
| **ONNX Runtime** | Cross-platform inference | Not a training framework — we export to ONNX for production inference if needed, but the model is *built* in PyTorch |

### Who Else Uses This?

| Model / Project | Language | Framework | Notes |
|---|---|---|---|
| **GPT-2 / GPT-3** | Python | PyTorch (later JAX) | OpenAI started with TensorFlow, switched to PyTorch |
| **LLaMA 1/2/3** | Python | PyTorch | Meta AI's flagship models — pure PyTorch |
| **Mistral / Mixtral** | Python | PyTorch | Mistral AI uses PyTorch throughout |
| **Stable Diffusion** | Python | PyTorch | Stability AI, Runway — image generation in PyTorch |
| **Hugging Face Transformers** | Python | PyTorch (primary) + TF/JAX | The ecosystem standard — PyTorch is the default backend |
| **Gemini / Gemma** | Python | JAX/Flax | Google's models use JAX for TPU optimisation |
| **PaLM** | Python | JAX | Google Research — JAX for large-scale TPU training |
| **DeepSeek** | Python | PyTorch | Chinese open LLMs — PyTorch |
| **llama.cpp** | **C++** | Custom | Inference-only reimplementation for CPU/edge — no training |
| **MLX examples** | **Python** | MLX | Apple's framework — growing but niche |

PyTorch dominates the LLM landscape. The notable exception is Google (JAX for Gemini/PaLM), while C++ is used only for optimised inference runtimes (llama.cpp, vLLM's CUDA kernels). For a project spanning training, fine-tuning, and serving, Python + PyTorch is the industry-standard choice.

---

## Full Configuration Reference

All values from `config/model_config.yaml`:

### Model Architecture
| Parameter | Value |
|---|---|
| `n_layers` | 6 |
| `n_heads` | 8 |
| `d_model` | 512 |
| `d_ff` | 2048 |
| `vocab_size` | 2048 |
| `max_seq_len` | 512 |
| `dropout` | 0.1 |
| `attention_dropout` | 0.1 |
| `use_rotary` | true |

### Training
| Parameter | Value |
|---|---|
| `batch_size` | 32 |
| `micro_batch_size` | 8 |
| `max_epochs` | 100 |
| `learning_rate` | 1.5e-4 |
| `min_lr` | 1e-5 |
| `weight_decay` | 0.1 |
| `beta1` | 0.9 |
| `beta2` | 0.95 |
| `grad_clip` | 1.0 |
| `warmup_steps` | 500 |
| `lr_schedule` | cosine |
| `use_mixed_precision` | true |
| `save_interval` | 500 steps |
| `eval_interval` | 100 steps |

### Inference
| Parameter | Value |
|---|---|
| `temperature` | 0.8 |
| `top_k` | 50 |
| `top_p` | 0.95 |
| `max_new_tokens` | 512 |
| `use_kv_cache` | true |
| `use_constrained_decoding` | false |
| `beam_width` | 0 (sampling mode) |
| `fix_validity_weight` | 0.3 |
| `quantization` | none |

### Tokenizer
| Parameter | Value |
|---|---|
| `type` | bpe |
| `vocab_size` | 1024 (tokenizer default; model uses 2048 from YAML override) |
| Special tokens | 7 (`<\|pad\|>` through `<\|unk\|>`) |

### LoRA
| Parameter | Value |
|---|---|
| `rank` | 8 |
| `alpha` | 16.0 |
| `dropout` | 0.0 |
| `target_modules` | q_proj, k_proj, v_proj, out_proj |

---

## Parameter Count Breakdown

**Total: ~19.94 million parameters**

| Component | Parameters | Calculation | % of Total |
|---|---|---|---|
| **Token Embedding** | 1,048,576 | 2048 × 512 | 5.3% |
| **LM Head** | *(tied with embedding)* | — | 0% (shared) |
| **Per Transformer Block** | | | |
| → QKV Projection | 786,432 | 512 × (3 × 512) | 3.9% |
| → Output Projection | 262,144 | 512 × 512 | 1.3% |
| → FFN Linear₁ | 1,048,576 | 512 × 2048 | 5.3% |
| → FFN Linear₂ | 1,048,576 | 2048 × 512 | 5.3% |
| → 2 × LayerNorm | 2,048 | 2 × (512 + 512) | ~0% |
| **× 6 blocks subtotal** | 18,885,120 | 6 × 3,147,520 | 94.7% |
| **Final LayerNorm** | 1,024 | 512 + 512 | ~0% |
| **RoPE** | 0 | Pre-computed, not learned | 0% |
| **Grand Total** | **~19,934,720** | | **100%** |

The vast majority (94.7%) of parameters are in the 6 Transformer blocks. The embedding is shared with the LM head via weight tying, saving ~1M parameters. RoPE adds zero trainable parameters — it's computed from a fixed formula.

---

*Generated: 29 March 2026*
