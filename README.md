# FixProtoGPT

A GPT-style decoder-only transformer for generating, validating, explaining, and converting FIX protocol messages. Supports FIX versions 4.2, 4.4, 5.0 SP2, and Latest.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Fine-Tuning](#fine-tuning)
- [Inference](#inference)
- [API Reference](#api-reference)
- [CLI](#cli)
- [Configuration](#configuration)
- [Environments](#environments)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [License](#license)

---

## Features

**Model & Inference**
- 6-layer, 8-head, 512-dim transformer with RoPE positional embeddings (each layer: Pre-LN → Multi-Head Self-Attention → residual → Pre-LN → FFN → residual)
- 2048 token vocabulary with FIX-aware BPE tokenizer
- Temperature, top-k, top-p (nucleus) sampling
- Beam search with FIX-validity re-ranking
- FSM-based constrained decoding for valid FIX structure
- KV-cache for O(n) incremental generation
- Dynamic int8 / fp16 quantization

**Training**
- Cosine LR schedule with linear warmup and gradient accumulation
- Mixed-precision training (AMP on CUDA, native on MPS)
- Early stopping (patience-based + target val-loss threshold)
- Emergency checkpoint save on crash or SIGINT/SIGTERM
- Auto-resume from latest checkpoint
- `torch.compile` support (PyTorch 2.0+)
- LoRA fine-tuning for per-client adaptation (~95% parameter reduction)
- Feedback-driven fine-tuning from positive user interactions

**Data**
- Synthetic FIX message generation across 5 message types
- ~95 NL↔FIX template pairs with semantically aligned parameter sharing
- FIX spec scraping from fixtrading.org and Fiximate
- Multi-format spec ingestion (PDF, DOCX, XML, CSV)
- Real ticker symbols via Twelve Data API + hardcoded fallbacks
- Multi-version training pipeline (4.2, 4.4, 5.0SP2, Latest)

**Platform**
- Flask web UI with session-based auth (scrypt password hashing)
- 7 API route blueprints (56+ endpoints)
- Rich terminal CLI with slash-command interface
- Admin panel for client management, spec ingestion, and model control
- FIX ↔ JSON/XML bidirectional conversion
- Per-environment isolation (dev, qa, preprod, prod)
- Structured audit logging (server, user actions, debug) with daily rotation

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Web UI / CLI                       │
├──────────┬──────────┬───────────┬───────────┬───────────┤
│   Auth   │   Core   │  Export   │ Learning  │   Admin   │
│  Routes  │  Routes  │  Routes   │  Routes   │  Routes   │
├──────────┴──────────┴───────────┴───────────┴───────────┤
│                   Flask Application                      │
├──────────────────────┬──────────────────────────────────┤
│   Inference Engine   │        Training Pipeline         │
│  ┌────────────────┐  │  ┌────────────┐ ┌─────────────┐ │
│  │ FixProtoGPT    │  │  │  Trainer   │ │  FineTuner  │ │
│  │ (Transformer)  │  │  │  (AMP,     │ │  (LoRA,     │ │
│  │                │  │  │  compile)  │ │  feedback)  │ │
│  ├────────────────┤  │  ├────────────┤ ├─────────────┤ │
│  │ Beam Search    │  │  │  Dataset   │ │  Spec       │ │
│  │ Constrained    │  │  │  (.bin /   │ │  Ingestion  │ │
│  │ Decoder        │  │  │   .txt)    │ │  Pipeline   │ │
│  └────────────────┘  │  └────────────┘ └─────────────┘ │
├──────────────────────┴──────────────────────────────────┤
│                    Persistence Layer                     │
│  SQLite (users, interactions, token usage)               │
│  Structured Logging (server, user_actions, debug)        │
└─────────────────────────────────────────────────────────┘
```

**Model**: `FixProtoGPT` — decoder-only transformer with pre-layer norm, RoPE, optional Flash Attention, GELU activation, and weight-tied embedding/output head.

Each of the 6 identical `TransformerBlock` layers contains:

| Component | Details |
|---|---|
| **LayerNorm 1** | Pre-norm over 512 dims, applied before attention |
| **Multi-Head Self-Attention** | 8 heads × 64-dim each, fused QKV projection, RoPE on Q/K, optional Flash Attention, causal mask, KV-cache support, attention + residual dropout |
| **Residual connection** | Input added back after attention |
| **LayerNorm 2** | Pre-norm over 512 dims, applied before FFN |
| **Feed-Forward Network** | Linear(512 → 2048) → GELU (tanh approx) → Linear(2048 → 512) → Dropout |
| **Residual connection** | Input added back after FFN |

Wrapping the 6 blocks: Token Embedding (2048 × 512, weight-tied with LM head) → 6 × TransformerBlock → Final LayerNorm → LM Head (512 → 2048 logits).

**Tokenizer**: FIX-aware BPE with 6 special tokens (`<|pad|>`, `<|bos|>`, `<|eos|>`, `<|fix|>`, `<|field|>`, `<|eom|>`). Splits on `|` and `=` for FIX structure awareness.

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- PyTorch ≥ 2.0

### Installation

```bash
git clone <repo-url> && cd FixProtoGPT
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Run

```bash
# Prepare data + train
make train

# Start web UI (dev, port 8080)
make serve

# Or start all environments
make start-all
```

---

## Data Preparation

### Single Version

```bash
python -m src.data.prepare_data
```

Generates synthetic training data for the **active FIX version** (from `config/model_config.yaml`). Defaults: 50,000 FIX messages + 10,000 NL pairs.

- **Message types generated**: NewOrderSingle (D), ExecutionReport (8), MarketDataRequest (V), OrderCancelRequest (F), Logon (A)
- **Raw FIX weights**: 40% orders, 30% executions, 15% market data, 10% cancels, 5% logons
- **NL pair weights**: 40% orders, 20% executions, 15% cancels, 15% market data, 10% logons
- **~95 NL templates** across 5 message types — formal, casual, imperative, and shorthand styles with random casing variation
- **Semantically aligned** — shared parameters between NL text and FIX output ensure consistency
- **Symbol resolution** — real tickers via Twelve Data API cache + 100+ hardcoded fallbacks
- **Data sources combined** — scraped specs + ingested canonical specs (PDF/DOCX/XML/CSV) + synthetic FIX + NL pairs
- **Stratified split** — train/val split stratified by message type for proportional representation
- **Output** — tokenized `train.bin` / `val.bin` in `model_store/data/<version>/processed/`

### All Versions (4.2, 4.4, 5.0SP2, Latest)

```bash
python scripts/train_all_versions.py
```

Multi-version pipeline:

1. **Ingest PDFs** from `Base_Fix_Specs/<version>/` into per-version canonical JSON
2. **Convert specs → training lines** for each version
3. **Generate 24,000 synthetic messages + 6,000 NL pairs** per version
4. **Combine** into a version-tagged corpus (`[FIX-4.2]`, `[FIX-5.0SP2]`, etc.)
5. **Tokenize** per-version binary splits to `model_store/data/<slug>/processed/`
6. **Train** individual models per version — temporarily switches active version in config

```bash
# CLI flags
--resume-only          # Skip data prep, resume training from latest checkpoint
--per-version          # Also train per-version models after combined training
--only-per-version     # Skip combined training, only train individual versions
--versions 4.2 5.0SP2  # Train only specific versions
```

Version-specific session protocols are handled automatically (FIX 4.x uses `FIX.4.x`; 5.0SP2 and Latest use `FIXT.1.1`).

### Spec Scraping

```bash
python -m src.data.scraper
```

5-step pipeline:

1. **Web scrape** — fetch spec data from fixtrading.org and Fiximate
2. **Build specification** — merge web data with built-in FIX reference (`fix_reference.py`)
3. **Generate training data** — produce training lines from spec (messages, fields, enums, examples)
4. **Generate documentation** — write Markdown reference from spec
5. **Update version metadata** — save version info for downstream tools

Output saved to `model_store/data/<version>/raw/` — automatically picked up by `prepare_data`.

### Symbol Sync

```bash
python -m scripts.sync_symbols [--stocks] [--crypto] [--etfs] [--rebuild-cache]
```

Weekly batch job syncing equity, forex, and crypto symbols from the Twelve Data API into `model_store/data/symbols/`.

---

## Training

```bash
make train
# or
python -m src.training.train [--resume <checkpoint.pt>]
```

Without `--resume`, the latest checkpoint is auto-detected.

**Training features:**

| Feature | Description |
| ------- | ----------- |
| Gradient accumulation | `batch_size / micro_batch_size` accumulation steps |
| Mixed precision | AMP (float16) on CUDA, native on MPS |
| LR schedule | Cosine decay with linear warmup |
| Gradient clipping | Max-norm clipping (default 1.0) |
| Checkpointing | Configurable save interval with rotation (`max_checkpoints_to_keep`) |
| Auto-resume | Detects and resumes from latest checkpoint on startup |
| Early stopping | Patience-based (N evals with no improvement) or target val-loss |
| Emergency save | Checkpoint saved on unhandled exceptions |
| Graceful shutdown | SIGINT/SIGTERM triggers checkpoint save before exit |
| `torch.compile` | PyTorch 2.0+ graph compilation (opt-in) |

**Code defaults** (overridden by YAML at runtime):

| Parameter | Code Default | YAML Value |
| --------- | ------------ | ---------- |
| batch_size | 8 | 32 |
| micro_batch_size | 2 | 8 |
| max_epochs | 50 | 100 |
| learning_rate | 3e-4 | 1.5e-4 |
| warmup_steps | 1000 | 500 |
| min_lr | 3e-5 | 1e-5 |

---

## Fine-Tuning

### Feedback-Driven

Fine-tuning uses positive user feedback to incrementally improve the model:

```bash
# Trigger via API (admin only)
curl -X POST http://localhost:8080/api/learning/finetune

# Preflight check (minimum 5 new positive interactions required)
curl http://localhost:8080/api/learning/finetune/preflight
```

**Pipeline:** export positive-feedback interactions → augment training data → tokenize → train from latest checkpoint (500 steps, lr=1e-4) → save new `best.pt` → hot-reload model.

Supports per-client fine-tuning via `client_id`.

### LoRA (Low-Rank Adaptation)

Per-client fine-tuning uses LoRA adapters to avoid modifying base model weights. Only small adapter matrices A and B are trained (~95% parameter reduction), keeping the base model intact while specializing for each client's FIX dialect.

**Default LoRA config:** rank=8, alpha=16, dropout=0.0, targets `q_proj`, `k_proj`, `v_proj`, `out_proj`.

After fine-tuning, adapters can be merged back into base weights via `merge_lora()` for deployment.

---

## Inference

The inference engine (`FixProtoGPTInference`) supports:

| Feature | Description |
| ------- | ----------- |
| Temperature | Logit scaling (default 0.8) |
| Top-k | Keep top-k tokens (default 50) |
| Top-p | Nucleus sampling, cumulative probability ≤ p (default 0.95) |
| Beam search | Multi-candidate decoding with FIX-validity re-ranking |
| Constrained decoding | FSM-based token masking enforcing valid `tag=value\|` structure |
| KV-cache | O(n) incremental generation |
| Quantization | `none`, `dynamic_int8`, `fp16` |

**Capabilities:**
- `generate` — free-form FIX message generation
- `nl2fix` — natural language → FIX message
- `explain` — field-by-field FIX message breakdown
- `validate` — structural validation with scoring
- `complete` — auto-complete partial FIX message
- `enrich` — add missing header/trailer/required fields

---

## API Reference

The Flask server exposes 7 route blueprints. All `/api/*` routes require authentication unless noted.

### Core (`/api`)

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| POST | `/api/generate` | Free-form FIX message generation |
| POST | `/api/nl2fix` | Natural language → FIX |
| POST | `/api/explain` | Explain FIX message fields |
| POST | `/api/validate` | Validate FIX message structure |
| POST | `/api/complete` | Auto-complete FIX message |
| GET | `/api/status` | Model and system status |
| GET | `/api/versions` | List installed FIX versions |
| POST | `/api/version` | Switch active FIX version |
| GET | `/api/examples` | Example FIX messages |
| POST | `/api/symbols/resolve` | Resolve ticker symbol |
| GET | `/api/symbols/cache` | View symbol cache |
| POST | `/api/symbols/sync` | Trigger symbol sync |
| GET | `/api/symbols/sync/status` | Symbol sync status |

### Auth (`/auth`)

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| GET | `/auth/login` | Login page |
| POST | `/auth/login` | Authenticate |
| POST | `/auth/register` | Register new user |
| POST | `/auth/logout` | Logout |
| GET | `/auth/me` | Current user profile |
| GET | `/auth/users` | List all users (admin) |
| GET | `/auth/token-usage` | Current user token usage |
| GET | `/auth/admin/token-usage` | All users token usage (admin) |

### Export (`/api/export`, `/api/import`, `/api/convert`)

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| POST | `/api/export/json` | Export FIX → JSON file |
| POST | `/api/export/xml` | Export FIX → XML file |
| POST | `/api/import/json` | Import FIX from JSON |
| POST | `/api/import/xml` | Import FIX from XML |
| POST | `/api/convert/to-json` | Convert FIX → JSON |
| POST | `/api/convert/to-xml` | Convert FIX → XML |

### Learning (`/api`)

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| POST | `/api/feedback` | Submit feedback on interaction |
| GET | `/api/interactions` | List logged interactions |
| DELETE | `/api/interactions/<id>` | Delete interaction |
| DELETE | `/api/interactions` | Clear all interactions |
| POST | `/api/learning/export` | Export interactions as training pairs |
| GET | `/api/learning/status` | Training / learning status |
| GET | `/api/learning/finetune/preflight` | Fine-tune preflight check |
| POST | `/api/learning/finetune` | Trigger fine-tuning |
| GET | `/api/learning/finetune/status` | Fine-tune progress |

### Admin (`/admin` — admin role required)

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| GET | `/admin` | Admin panel UI |
| GET | `/admin/clients` | List all clients |
| POST | `/admin/clients` | Create client |
| GET | `/admin/clients/<id>` | Client details |
| DELETE | `/admin/clients/<id>` | Delete client |
| POST | `/admin/clients/<id>/upload` | Upload spec files |
| POST | `/admin/clients/<id>/ingest` | Ingest uploaded specs |
| POST | `/admin/clients/<id>/train` | Trigger client fine-tuning |
| GET | `/admin/clients/<id>/train/status` | Client training status |
| POST | `/admin/specs/upload` | Upload base spec files |
| POST | `/admin/specs/ingest` | Re-ingest base specs |
| GET | `/admin/specs/canonical` | View canonical spec records |
| GET | `/admin/models` | List available models |
| POST | `/admin/models/load` | Load model into memory |
| POST | `/admin/models/unload` | Unload model from memory |

### Ops (`/ops` — dev/qa only)

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| GET | `/ops/health` | Health check (no auth) |
| GET | `/ops/config` | Running configuration |
| GET | `/ops/logs` | Recent application logs |

### Assets

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| GET | `/assets/bundle.js` | Minified JS bundle (ETag-cached) |

---

## CLI

### User CLI

```bash
python -m src.cli.enhanced_cli
```

Slash-command REPL interface:

| Command | Description |
| ------- | ----------- |
| `/generate <prompt>` | Generate FIX message (also accepts natural language) |
| `/load [version]` | Load model for version |
| `/version [version]` | Switch active version |
| `/temperature <val>` | Set temperature |
| `/top-k <val>` | Set top-k sampling |
| `/top-p <val>` | Set nucleus sampling |
| `/max-tokens <val>` | Set max generation length |
| `/estimate` | Estimate model parameters |
| `/status` | Model/system status |
| `/versions` | List installed versions |
| `/repl` | Interactive REPL mode |
| `/help` | Show commands |
| `/exit` | Exit |

### Admin CLI

```bash
python -m src.cli.admin_cli
```

| Command | Description |
| ------- | ----------- |
| `/ingest-file <path>` | Ingest single spec file |
| `/ingest-dir <path>` | Ingest directory of specs |
| `/specs` | View base spec stats |
| `/clients` | List client overlays |
| `/create-client <id>` | Create client |
| `/client-stats <id>` | Client stats |
| `/delete-client <id>` | Remove client |
| `/upload <id> <path>` | Upload client spec file |
| `/ingest-client <id>` | Process client specs |
| `/train-client <id>` | Trigger client fine-tuning |
| `/training-status` | Monitor training progress |

---

## Configuration

All configuration lives in `config/model_config.yaml`:

```yaml
version:
  active: "5.0SP2"           # Active FIX version (single source of truth)
  protocol: "FIX.5.0SP2"     # Full protocol identifier
  session: "FIXT.1.1"        # Session-layer protocol

model:
  n_layers: 6
  n_heads: 8
  d_model: 512
  d_ff: 2048
  vocab_size: 2048
  max_seq_len: 512
  dropout: 0.1
  attention_dropout: 0.1
  use_rotary: true            # RoPE embeddings

training:
  batch_size: 32              # Effective (via gradient accumulation)
  micro_batch_size: 8
  max_epochs: 100
  learning_rate: 1.5e-4
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  warmup_steps: 500
  lr_schedule: "cosine"
  min_lr: 1.0e-5
  use_mixed_precision: true
  compile_model: false
  save_interval: 500
  eval_interval: 100
  early_stopping_patience: 0  # 0 = disabled
  target_val_loss: 0.0        # 0 = disabled

data:
  train_path: ~               # Defaults from paths.py
  val_path: ~
  fix_versions:
    "4.2": "FIX 4.2"
    "4.4": "FIX 4.4"
    "5.0SP2": "FIX 5.0 SP2"
    "Latest": "FIX Latest"

tokenizer:
  type: "bpe"
  vocab_size: 1024
  special_tokens: ["<|pad|>", "<|bos|>", "<|eos|>", "<|fix|>", "<|field|>", "<|eom|>"]

inference:
  temperature: 0.8
  top_k: 50
  top_p: 0.95
  max_new_tokens: 512
  use_kv_cache: true
  use_constrained_decoding: false
  beam_width: 0               # 0 = sampling; >0 = beam search
  fix_validity_weight: 0.3
  quantization: "none"        # "none", "dynamic_int8", "fp16"

lora:
  rank: 8
  alpha: 16.0
  dropout: 0.0
  target_modules: ["q_proj", "k_proj", "v_proj", "out_proj"]

data_quality:
  stratify_split: true
  min_quality_score: 60

logging:
  log_dir: "logs"
  tensorboard: true
  wandb: false
  print_interval: 10
```

All file paths are derived from `version.active` via `src/utils/paths.py` — no hardcoded paths.

---

## Environments

Four isolated environments with independent databases, logs, and ports:

| Environment | Port | Debug | Demo Users | HTTPS-Only Cookies | HSTS |
| ----------- | ---- | ----- | ---------- | ------------------ | ---- |
| dev | 8080 | Yes | Seeded | No | No |
| qa | 8081 | Yes | Seeded | No | No |
| preprod | 8082 | No | No | Yes | Yes |
| prod | 8083 | No | No | Yes | Yes |

```bash
make start-dev          # Start dev (Flask, port 8080)
make start-qa           # Start qa (Flask, port 8081)
make start-prod         # Start prod (Gunicorn, port 8083)
make start-all          # Start dev + qa
make stop ENV=dev       # Stop specific env
make stop-all           # Stop all
make status             # Show all env status
make control-panel      # Interactive TUI control panel
```

**Security:**
- prod/preprod refuse to start with default `SECRET_KEY`
- Session cookies: `HttpOnly`, `SameSite=Lax` (dev) / `Strict` (prod)
- Security headers on all responses: `X-Content-Type-Options`, `X-Frame-Options`, `CSP`, `Referrer-Policy`
- CORS restricted in non-dev environments

**Logging** (per-environment, daily rotation):
- `server.log` — HTTP request/response (30-day retention)
- `user_actions.log` — meaningful actions: login, generate, feedback (90-day retention)
- `debug.log` — verbose internals (14-day retention, DEBUG level only)

---

## Project Structure

```
fixprotogpt/
├── config/
│   ├── model_config.yaml          # Central configuration
│   └── env/                       # Per-environment overrides
├── Base_Fix_Specs/                # FIX specification source files
│   ├── Fix_4.2/
│   ├── Fix_4.4/
│   ├── Fix_5.0SP2/
│   └── Fix_Latest/
├── src/
│   ├── api/
│   │   ├── app.py                 # Flask application factory
│   │   ├── state.py               # Shared application state
│   │   └── routes/
│   │       ├── admin.py           # Admin panel & client management
│   │       ├── auth.py            # Authentication & user management
│   │       ├── core.py            # Inference endpoints & symbols
│   │       ├── export.py          # FIX ↔ JSON/XML conversion
│   │       ├── learning.py        # Feedback, interactions, fine-tune
│   │       ├── ops.py             # Health, config, logs (dev/qa)
│   │       └── assets.py          # JS bundle serving
│   ├── cli/
│   │   ├── enhanced_cli.py        # Rich terminal UI (slash commands)
│   │   └── admin_cli.py           # Admin CLI (spec ingestion, clients)
│   ├── config/
│   │   └── env_config.py          # Environment-aware configuration
│   ├── core/
│   │   ├── transformer.py         # FixProtoGPT model (RoPE, Flash Attn)
│   │   ├── tokenizer.py           # Legacy FIX-aware tokenizer
│   │   ├── bpe_tokenizer.py       # HuggingFace BPE tokenizer
│   │   ├── fix_reference.py       # FIX 5.0 SP2 reference data (74 msg types, 120+ fields)
│   │   └── version_registry.py    # Multi-version management (4 versions)
│   ├── data/
│   │   ├── prepare_data.py        # Synthetic data generation + NL pairs
│   │   ├── scraper.py             # FIX spec web scraper
│   │   ├── twelve_data.py         # Twelve Data API client
│   │   ├── symbol_resolver.py     # Ticker/company resolution (3-tier)
│   │   ├── spec_monitor.py        # Spec change detection (SHA-256)
│   │   ├── version_detector.py    # FIX version extraction from text
│   │   └── ingest/                # Multi-format spec ingestion
│   │       ├── base.py            # Abstract parser + CanonicalSpec model
│   │       ├── csv_parser.py      # CSV/TSV parser
│   │       ├── docx_parser.py     # DOCX parser
│   │       ├── pdf_parser.py      # PDF parser (pdfplumber)
│   │       ├── xml_parser.py      # XML/XSD parser
│   │       ├── normalizer.py      # Merge, deduplicate, persist specs
│   │       └── client_overlay.py  # Client-specific spec overrides
│   ├── inference/
│   │   ├── generate.py            # Main inference engine
│   │   ├── explainer.py           # Field-by-field explanations
│   │   ├── enrichment.py          # FIX message post-processing
│   │   ├── constrained_decoder.py # FSM-based FIX grammar enforcement
│   │   └── beam_search.py         # Beam search with validity scoring
│   ├── persistence/
│   │   ├── interaction_logger.py  # SQLite interaction storage
│   │   ├── user_manager.py        # User accounts & scrypt auth
│   │   └── action_logger.py       # Structured JSON audit logging
│   ├── services/
│   │   └── update_manager.py      # Model checkpoint hot-reload
│   ├── training/
│   │   ├── train.py               # Trainer (AMP, compile, signals, early stop)
│   │   ├── finetune.py            # Feedback-driven fine-tuning pipeline
│   │   ├── lora.py                # LoRA adapters (apply, merge)
│   │   └── dataset.py             # FixProtocolDataset (.bin memmap, .txt)
│   └── utils/
│       ├── paths.py               # Version-aware path resolution
│       ├── device.py              # Device detection (CUDA → MPS → CPU)
│       ├── helpers.py             # Seeds, param counting, model sizing
│       ├── fix_converter.py       # FIX ↔ JSON/XML conversion
│       ├── fix_enrichment.py      # FIX field enrichment helpers
│       ├── data_quality.py        # Training data quality scoring
│       └── quantization.py        # int8/fp16 quantization
├── scripts/
│   ├── train_all_versions.py      # Multi-version training pipeline
│   ├── sync_symbols.py            # Twelve Data symbol sync
│   ├── migrate_user_ids.py        # UUID → sequential ID migration
│   ├── launch_training.sh         # Background training launcher
│   └── env/
│       ├── start.sh               # Start environment
│       ├── stop.sh                # Stop environment
│       ├── restart.sh             # Restart environment
│       ├── status.sh              # Environment status
│       └── control_panel.py       # Interactive TUI control panel
├── tests/
│   ├── unit/                      # Tokenizer, model, data, auth, config, etc.
│   ├── integration/               # Flask routes, end-to-end inference
│   └── training/                  # Trainer, checkpoint resume, full run
├── ui/
│   ├── templates/                 # index.html, login.html
│   ├── static/styles/             # CSS
│   └── js_src/                    # 8 JS modules → minified bundle
├── db/<env>/                      # Per-env SQLite (users.db, interactions.db)
├── logs/<env>/                    # Per-env rotating logs
├── model_store/
│   ├── checkpoints/<version>/     # Model checkpoints per version
│   ├── data/<version>/            # Training data per version (raw/, processed/)
│   ├── data/symbols/              # Synced ticker symbols
│   └── backups/                   # Versioned backups
├── documentation/
│   ├── guides/                    # API_GUIDE, API_REFERENCE, CLI_GUIDE, TRAINING_GUIDE
│   ├── writeups/                  # AI_MODEL_DEEP_DIVE, BRIEF_WRITEUP, OVERVIEW, PROJECT_WRITEUP
│   ├── diagrams/                  # Architecture, class, dependency, flow, mermaid
│   └── ops/                       # running-environments.md
├── Makefile
├── pyproject.toml
└── LICENSE                        # MIT
```

---

## Testing

```bash
make test               # All tests
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-eval          # Training evaluation tests
make test-cov           # Tests with HTML coverage report
make lint               # flake8 (max-line-length=120)
make format             # black + isort
```

**Markers:** `@pytest.mark.slow`, `@pytest.mark.requires_model`, `@pytest.mark.integration`

---

## Makefile Targets

| Target | Description |
| ------ | ----------- |
| `help` | Display all targets |
| `test` | Run all tests |
| `test-unit` | Unit tests only |
| `test-integration` | Integration tests |
| `test-eval` | Training evaluation |
| `test-cov` | Tests with coverage |
| `lint` | flake8 |
| `format` | black + isort |
| `train` | Start model training |
| `evaluate` | Run evaluation on checkpoint |
| `serve` | Start web UI (foreground, port 8080) |
| `start-dev` | Start dev (background, 8080) |
| `start-qa` | Start qa (background, 8081) |
| `start-preprod` | Start preprod (background, 8082) |
| `start-prod` | Start prod (gunicorn, 8083) |
| `start-all` | Start dev + qa |
| `stop` | Stop environment (`ENV=dev`) |
| `stop-all` | Stop all |
| `restart` | Restart environment |
| `restart-all` | Restart all |
| `status` | Show all env status |
| `control-panel` | Interactive TUI |
| `clean` | Remove caches and build artifacts |

---

## Dependencies

**Core:**
torch, numpy, pyyaml, tqdm, requests, flask, flask-cors, beautifulsoup4, lxml, rich, rjsmin, python-dotenv, pdfplumber, python-docx, openpyxl, tokenizers

**Dev (optional):**
pytest, pytest-cov, pytest-xdist, black, flake8, isort, mypy

**Prod (optional):**
gunicorn

---

## Author

**Sriram VVSS Gosala** — [GitHub](https://github.com/sriramgvvss)

## License

MIT — see [LICENSE](LICENSE).
