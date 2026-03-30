# FixProtoGPT — A Simple Guide

> **An AI assistant that speaks the language of trading — so you don't have to.**

---

## What Problem Does This Solve?

When banks and brokers trade stocks, they send messages to each other using a system called **FIX Protocol**. These messages look like this:

```
8=FIX.4.4|35=D|49=BROKER|56=EXCHANGE|55=TSLA|54=1|38=100|40=1|10=123|
```

That says "Buy 100 shares of Tesla at market price" — but in a code that's hard for humans to read or write. There are **200+ tag numbers** to remember, strict formatting rules, and multiple versions of the standard.

**FixProtoGPT lets you just type:**

> "Buy 100 shares of Tesla at market price"

...and it generates the correct FIX message for you. It can also explain, validate, and auto-complete FIX messages.

---

## What Can It Do?

| Feature | What You Do | What You Get |
|---------|------------|--------------|
| **Generate** | Type a request in plain English | A complete, valid FIX message |
| **Explain** | Paste a FIX message | A human-readable breakdown of every field |
| **Validate** | Paste a FIX message | A report on whether it's correctly formed |
| **Complete** | Paste a partial FIX message | The missing fields filled in automatically |

---

## How Does It Work? (The Simple Version)

```
  YOU                        THE SYSTEM                    RESULT
  ───                        ──────────                    ──────

  "Buy 100 shares     ┌─────────────────────────┐
   of Tesla at    ───►│  1. Understand English   │
   market price"      │     Tesla → TSLA         │
                      │     100 shares, Buy side  │
                      │     market order          │
                      ├─────────────────────────┤
                      │  2. AI Model thinks...   │     (like autocomplete
                      │     Predicts the FIX     │      on your phone, but
                      │     message token by     │      for trading messages)
                      │     token                │
                      ├─────────────────────────┤
                      │  3. Clean up & validate  │
                      │     Add headers          │
                      │     Add timestamps       │
                      │     Calculate checksum   │
                      └────────────┬────────────┘
                                   │
                                   ▼
                      8=FIX.4.4|9=154|35=D|
                      49=SENDER|56=TARGET|
                      55=TSLA|54=1|38=100|
                      40=1|59=4|10=222|

                      ✅ Valid FIX message, ready to use
```

---

## The AI Model — Plain English Explanation

### What Is It?

FixProtoGPT uses a **Transformer** neural network — the same type of technology behind ChatGPT, but *much* smaller and built specifically for one job.

Think of it like this:

```
  ChatGPT  =  A general-purpose doctor who knows a bit about everything
  
  FixProtoGPT  =  A specialist who knows ONE thing extremely well
                   (FIX protocol messages)
```

### How Does It Think?

The model works like **predictive text on your phone**, but for FIX messages:

```
  Step 1:  The model sees:   "35=D | 55=TSLA | 54=1 | 38="
  Step 2:  It predicts:      "100"     ← (the quantity)
  Step 3:  Now it sees:      "35=D | 55=TSLA | 54=1 | 38=100 | 40="
  Step 4:  It predicts:      "1"       ← (market order)
  Step 5:  ...and so on, one piece at a time, until the message is done.
```

Each "piece" is called a **token**. The model has learned from thousands of example FIX messages what token should come next.

---

## Why Not Just Use ChatGPT?

| | FixProtoGPT (ours) | Using ChatGPT / GPT-4 |
|---|---|---|
| **Size** | 20 million parameters — runs on a laptop | 175+ billion parameters — needs powerful servers |
| **Speed** | Under 1 second | 1-3 seconds (needs internet) |
| **Cost** | Free after setup | Costs money per message ($0.01-$0.10 each) |
| **Privacy** | Everything stays on YOUR machine | Your trading data goes to OpenAI's cloud |
| **Internet** | Works completely offline | Requires internet connection |
| **Accuracy on FIX** | Trained ONLY on FIX data — no confusion | Knows FIX *and* cooking recipes, poetry, etc. |
| **Financial compliance** | Can run in air-gapped networks (SOX, MiFID II) | Cloud API may violate data regulations |

**Bottom line:** For this specific job, a small specialist beats a giant generalist.

---

## Model Architecture — The Technical Details

Here's what's inside the AI model:

```
  ┌──────────────────────────────────────────────────────────┐
  │                    FixProtoGPT Model                      │
  │                                                          │
  │   INPUT: Tokens (pieces of text)                         │
  │     │                                                    │
  │     ▼                                                    │
  │   ┌──────────────────────────────────────┐               │
  │   │  TOKEN EMBEDDING                     │               │
  │   │  Turns each token into a list of     │               │
  │   │  512 numbers (a "vector")            │               │
  │   │  Vocabulary: 2,048 possible tokens   │               │
  │   └──────────────┬───────────────────────┘               │
  │                  │                                       │
  │                  ▼                                       │
  │   ┌──────────────────────────────────────┐               │
  │   │  TRANSFORMER BLOCK  (×6 stacked)     │               │
  │   │                                      │               │
  │   │  ┌─────────────────────────────────┐ │               │
  │   │  │ ATTENTION (8 heads)             │ │               │
  │   │  │ "Which other words should I     │ │               │
  │   │  │  pay attention to right now?"   │ │               │
  │   │  │ (e.g., message type → required  │ │               │
  │   │  │  fields, symbol → side)         │ │               │
  │   │  └─────────────────────────────────┘ │               │
  │   │                  │                   │               │
  │   │                  ▼                   │               │
  │   │  ┌─────────────────────────────────┐ │               │
  │   │  │ FEED-FORWARD NETWORK            │ │               │
  │   │  │ "What facts do I know about     │ │               │
  │   │  │  this?"                         │ │               │
  │   │  │ (e.g., tag 54=1 means "Buy",   │ │               │
  │   │  │  tag 40=2 means "Limit order") │ │               │
  │   │  └─────────────────────────────────┘ │               │
  │   └──────────────────────────────────────┘               │
  │                  │                                       │
  │                  ▼                                       │
  │   ┌──────────────────────────────────────┐               │
  │   │  OUTPUT: Pick the most likely        │               │
  │   │  next token from 2,048 options       │               │
  │   └──────────────────────────────────────┘               │
  │                                                          │
  │   Total: 19.94 million learnable parameters              │
  └──────────────────────────────────────────────────────────┘
```

### Key Numbers and What They Mean

| Parameter | Value | Plain English | Technical Justification |
|-----------|-------|---------------|------------------------|
| **Layers** | 6 | 6 processing stages stacked on top of each other. More layers = deeper understanding | FIX messages are flat `tag=value` sequences — not deeply nested like human language. 6 layers captures field dependencies without overfitting on limited training data |
| **Attention Heads** | 8 | Each layer can focus on 8 different things at once | 8 heads × 64 dimensions = 512 total. Lets the model simultaneously track relationships like "message type → required fields" and "symbol → side" |
| **d_model** | 512 | Each token is internally represented as a list of 512 numbers | Balances representational capacity vs speed. 256 would underfit; 1024 would quadruple parameters without proportional gain |
| **d_ff** | 2,048 | The "thinking" layer expands to 2,048 dimensions (4× d_model) | Standard expansion ratio from the original Transformer paper. This is where the model stores factual FIX knowledge (tag meanings, valid enumerations) |
| **Vocab Size** | 2,048 | The model knows 2,048 distinct tokens | FIX has ~200 tags, ~50 message types, ~100 enumeration values, plus identifiers and numbers. 2,048 covers this cleanly |
| **Max Sequence** | 512 tokens | The longest message it can handle at once | Even the most complex FIX messages rarely exceed 400 tokens after tokenization |
| **Dropout** | 0.1 | During training, randomly ignores 10% of connections | Prevents memorisation — forces the model to learn general patterns rather than specific examples |
| **RoPE** | Enabled | Rotary Position Embeddings — tells the model where each token sits | Better than older methods at handling sequences of varying length. Gives the model a sense of field ordering |
| **Label Smoothing** | 0.1 | Makes the model slightly less "overconfident" | Important because multiple valid field orderings exist for the same FIX message type |
| **Total Parameters** | 19.94M | 19.94 million adjustable numbers the model has learned | Tiny compared to GPT-4 (1.8 trillion). Efficient enough to run on a laptop CPU |

---

## How Was the Model Trained?

```
  STEP 1                     STEP 2                     STEP 3
  ──────                     ──────                     ──────

  Parse official FIX         Generate training          Build a custom
  specification PDFs         examples from the specs    tokenizer
  (versions 4.2, 4.4,       ┌─────────────────┐        ┌──────────────┐
   5.0 SP2, Latest)         │ ~24,000 FIX msgs │        │ BPE algorithm│
  ┌───────────────┐         │  per version     │        │ that knows   │
  │ Extract:      │         │                  │        │ FIX syntax   │
  │ • Field defs  │────────►│ ~6,000 English   │───────►│              │
  │ • Msg types   │         │  ↔ FIX pairs     │        │ Keeps tag=   │
  │ • Enumerations│         │  per version     │        │ value intact │
  └───────────────┘         └─────────────────┘        └──────┬───────┘
                                                              │
            ┌─────────────────────────────────────────────────┘
            │
            ▼
  STEP 4
  ──────
  Train the model
  ┌───────────────────────────────────────────────────────────┐
  │                                                           │
  │  Optimiser:      AdamW  (the best for Transformers)       │
  │  Learning Rate:  0.00015 → 0.00001  (starts fast,        │
  │                   gradually slows down for precision)      │
  │  Warmup:         500 steps  (gentle start to avoid        │
  │                   early instability)                       │
  │  Batch Size:     32  (processes 32 examples before        │
  │                   updating the model's knowledge)          │
  │  Max Epochs:     100  (with early stopping if the         │
  │                   model stops improving)                   │
  │  Precision:      FP16  (half-precision math = 2× faster,  │
  │                   half the memory on GPU)                  │
  │  Grad Clipping:  1.0  (prevents catastrophically          │
  │                   large updates)                           │
  │  Checkpoints:    Saved every 500 steps  (never lose       │
  │                   progress, even if training crashes)      │
  │                                                           │
  │  Hardware:  Works on NVIDIA GPU, Apple Silicon, or CPU    │
  │                                                           │
  └───────────────────────────────────────────────────────────┘
            │
            ▼
  STEP 5
  ──────
  Model ready! Saved with weights + tokenizer + config
  One checkpoint per FIX version (4.2, 4.4, 5.0 SP2, Latest)
```

### Why Synthetic Training Data?

Real FIX messages contain confidential trading data — no company publishes them publicly. But FIX messages follow strict, formulaic rules (`tag=value|tag=value|...`), so generating realistic synthetic examples from the official specification is both practical and highly effective.

---

## Training Parameters — Quick Reference

| Parameter | Value | What It Does |
|-----------|-------|-------------|
| **Optimiser** | AdamW | Adjusts model weights during training. AdamW adds "weight decay" that prevents any single weight from growing too large |
| **β₁, β₂** | 0.9, 0.95 | Controls how much the optimiser "remembers" from previous steps. β₂=0.95 (lower than default 0.999) works better for small Transformer training |
| **Weight Decay** | 0.1 | Gently shrinks weights to prevent overfitting |
| **Learning Rate** | 1.5×10⁻⁴ peak, 1×10⁻⁵ floor | How big each learning step is. Follows a cosine curve: ramps up during warmup, then smoothly decreases |
| **Warmup Steps** | 500 | Starts with tiny learning steps for stability |
| **Batch Size** | 32 (4 micro-batches × 8) | Gradient accumulation: processes 4 small batches of 8, then combines the gradients as if it were one batch of 32. Lets you train "big" on small hardware |
| **Mixed Precision** | FP16 on GPU | Uses 16-bit floating point where possible. Cuts memory usage in half and speeds up training on compatible GPUs |
| **Gradient Clipping** | Max norm 1.0 | If the model tries to make a huge adjustment, this caps it to prevent instability |
| **Loss Function** | Cross-Entropy + Label Smoothing (0.1) | Measures how wrong each prediction is. Label smoothing prevents the model from being too certain about any single answer |
| **Checkpointing** | Every 500 steps | Saves progress regularly. Also saves an emergency checkpoint if you hit Ctrl+C |

---

## The Learning Loop — How It Gets Smarter

After deployment, user feedback makes the model improve continuously:

```
  ┌─────────┐     ┌────────────┐     ┌──────────────┐
  │  You ask │────►│ AI answers │────►│ You rate it  │
  │  a query │     │            │     │   👍 or 👎    │
  └─────────┘     └────────────┘     └──────┬───────┘
                                            │
                                            ▼
                             ┌──────────────────────────┐
                             │  Positive (👍) examples   │
                             │  are collected            │
                             └──────────┬───────────────┘
                                        │
                              Once ≥ 5 good examples
                                        │
                                        ▼
                             ┌──────────────────────────┐
                             │  LoRA Fine-Tuning         │
                             │                          │
                             │  Instead of retraining   │
                             │  all 19.94M parameters,  │
                             │  LoRA only adjusts ~5%   │
                             │  of them (much faster!)  │
                             │                          │
                             │  rank=8, alpha=16        │
                             │  500 steps, lr=1e-4      │
                             └──────────┬───────────────┘
                                        │
                                        ▼
                             ┌──────────────────────────┐
                             │  Merge & Hot-Reload       │
                             │                          │
                             │  New weights folded into  │
                             │  the base model.          │
                             │  Better answers start     │
                             │  immediately — no restart │
                             └──────────────────────────┘
```

**What is LoRA?** (Low-Rank Adaptation)

Instead of changing all 19.94 million parameters, LoRA adds small "adapter" matrices to the attention layers only. This means:
- ~95% fewer parameters to train
- Takes minutes instead of hours
- Original model is preserved — adapters can be removed if needed

---

## Three Ways to Generate Messages

| Strategy | How It Works | Speed | Quality | When to Use |
|----------|-------------|-------|---------|-------------|
| **Sampling** (default) | Picks the next token randomly, weighted by probability | Fast | Good | Everyday use |
| **Beam Search** | Explores 4 candidates in parallel, picks the best | Slower | Better | When accuracy is critical |
| **Constrained** | Forces output to follow FIX structure rules | Fast | Guaranteed valid | When you need 100% structural correctness |

**Sampling parameters:** Temperature=0.8 (moderate creativity), Top-k=50 (considers top 50 candidates), Top-p=0.95 (nucleus sampling — keeps tokens covering 95% of probability mass)

---

## The Custom Tokenizer — Why It Matters

A **tokenizer** breaks text into pieces ("tokens") that the model can understand. Standard tokenizers would break FIX messages badly:

```
  Standard tokenizer:   "49=SENDER8"  →  ["49", "=", "SE", "ND", "ER", "8"]
                        (broken into meaningless fragments!)

  Our FIX tokenizer:    "49=SENDER8"  →  ["49", "=", "SENDER8"]
                        (keeps the tag number and value as meaningful units)
```

**7 special tokens** give the model structural awareness:

| Token | Meaning |
|-------|---------|
| `<\|pad\|>` | Empty padding (fills unused space) |
| `<\|bos\|>` | Beginning of sequence |
| `<\|eos\|>` | End of sequence |
| `<\|fix\|>` | "Start generating FIX now" |
| `<\|field\|>` | Marks a FIX field boundary |
| `<\|eom\|>` | End of FIX message |
| `<\|unk\|>` | Unknown/unseen token |

---

## System Architecture

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                        FixProtoGPT System                        │
  │                                                                  │
  │  ┌────────────┐  ┌────────────────┐  ┌────────────────────────┐  │
  │  │  FRONTEND   │  │  WEB SERVER    │  │  AI ENGINE             │  │
  │  │             │  │                │  │                        │  │
  │  │  Browser UI │  │  Flask app     │  │  Transformer (19.94M)  │  │
  │  │  Vanilla JS │  │  7 blueprints  │  │  BPE Tokenizer         │  │
  │  │  4 tabs     │  │  56 endpoints  │  │  Beam Search           │  │
  │  │             │  │  Session auth  │  │  Constrained Decoder   │  │
  │  │  Generate   │  │                │  │  Enrichment Pipeline   │  │
  │  │  Explain    │  │  Gunicorn      │  │  Symbol Resolver       │  │
  │  │  Validate   │  │  (production)  │  │  LoRA Fine-Tuner       │  │
  │  │  Complete   │  │                │  │  KV-Cache (fast gen)   │  │
  │  └──────┬─────┘  └───────┬────────┘  └───────────┬────────────┘  │
  │         │                │                        │              │
  │         └────── HTTP ────┘──── Python calls ──────┘              │
  │                                                                  │
  │  ┌────────────────────────────────────────────────────────────┐  │
  │  │  STORAGE                                                   │  │
  │  │                                                            │  │
  │  │  SQLite DBs          Model Checkpoints     Symbol Cache    │  │
  │  │  (users,             (per FIX version,     (63K tickers    │  │
  │  │   interactions)       ~80MB each)           from Twelve    │  │
  │  │                                             Data API)      │  │
  │  └────────────────────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Technology Choices in Plain English

| Component | What We Used | Why (Simple) | Why (Technical) |
|-----------|-------------|-------------|-----------------|
| **AI framework** | PyTorch | The most popular tool for building custom AI models | Supports CUDA, MPS (Apple Silicon), CPU; native `F.scaled_dot_product_attention` for Flash Attention |
| **Tokenizer** | Custom BPE + HuggingFace Tokenizers | FIX needs a tokenizer that understands `tag=value` structure | Rust-backed BPE with custom pre-tokenization rules prevents merges across field boundaries |
| **Web server** | Flask + Gunicorn | Simple, lightweight, gets the job done | Blueprint architecture for modular routing; Gunicorn for production-grade WSGI serving |
| **Database** | SQLite | No setup needed — it's just a file | WAL mode for concurrent reads; ACID-compliant; environment-isolated per deployment tier |
| **Password security** | Werkzeug scrypt | Extremely hard to crack, even with powerful hardware | Memory-hard key derivation function resistant to GPU/ASIC brute-force attacks |
| **Frontend** | Vanilla JavaScript | No React/Vue/Angular complexity — fast, simple, zero dependencies | 8 modules, ~15KB minified, single HTTP request with ETag caching |
| **Config** | YAML + .env files | YAML for model settings, .env for secrets per environment | `model_config.yaml` for hyperparameters; `config/env/.env.<name>` for env-specific settings |

---

## Deployment Environments

```
  Development          QA               Pre-Production       Production
  ┌──────────┐    ┌──────────┐        ┌──────────┐        ┌──────────┐
  │   DEV    │    │    QA    │        │ PREPROD  │        │   PROD   │
  │          │    │          │        │          │        │          │
  │ Port 8080│───►│ Port 8081│───────►│ Port 8082│───────►│ Port 8083│
  │          │    │          │        │          │        │          │
  │ Debug ON │    │ Stricter │        │ Mirrors  │        │ Gunicorn │
  │ Demo     │    │ No demo  │        │ prod     │        │ Secret   │
  │ users    │    │ users    │        │ config   │        │ key req. │
  └──────────┘    └──────────┘        └──────────┘        └──────────┘

  Each environment has its own:
  • Database (no data cross-contamination)
  • Log files (separate server, user action, and debug logs)
  • Configuration (security settings, CORS rules)
  • Port number
```

**Why 4 environments?** FIX message errors can cause real trading failures. The QA → preprod → prod pipeline catches bugs before they reach production — standard practice in financial software.

---

## Project at a Glance

| Metric | Value |
|--------|-------|
| Model parameters | **19.94 million** |
| Python source files | ~54 |
| Lines of Python code | ~21,000+ |
| FIX versions supported | 4 (FIX 4.2, 4.4, 5.0 SP2, Latest) |
| API endpoints | ~56 |
| Training messages | ~24,000 per FIX version |
| NL↔FIX training pairs | ~6,000 per FIX version |
| Symbol database | ~63,000 tickers (stocks, forex, crypto) |
| Unit tests | 17 files |
| Integration tests | 2 files |
| Model size on disk | ~80 MB per checkpoint |
| Runs on | NVIDIA GPU, Apple Silicon (MPS), or CPU |
| Minimum RAM | ~2 GB |
| Python required | ≥ 3.10 |

---

*Generated: 29 March 2026*
