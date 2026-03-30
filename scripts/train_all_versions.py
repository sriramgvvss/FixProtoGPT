#!/usr/bin/env python3
"""
Script: train_all_versions.py
==============================

Ingest FIX specifications from Base_Fix_Specs for all available versions
(4.2, 4.4, 5.0SP2, Latest), prepare combined training data, and launch
the full training pipeline.

Usage (from project root):
    python scripts/train_all_versions.py

The script:
1. Ingests PDFs from Base_Fix_Specs/<version>/ into per-version canonical JSON.
2. Converts canonical specs → training text lines for every version.
3. Generates synthetic FIX messages for each supported version.
4. Combines everything into a single training corpus with version tags.
5. Tokenizes and saves train.bin / val.bin.
6. Launches the Trainer.

Author : FixProtoGPT Team
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── Project root ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_PROJECT_ROOT)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml
import torch

from src.utils import paths
from src.core.tokenizer import FixProtocolTokenizer
from src.core.transformer import ModelConfig, create_model

# Import the ingest *package* so @register_parser decorators fire.
import src.data.ingest  # noqa: F401  — triggers parser registration
from src.data.ingest.normalizer import (
    ingest_directory,
    load_canonical,
    specs_to_training_lines,
    canonical_json_path,
)
from src.data.prepare_data import (
    FIXDataGenerator,
    _stratified_split,
)
from src.training.train import (
    Trainer,
    TrainConfig,
    FixProtocolDataset,
    find_latest_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train_all_versions")

# Force unbuffered output for real-time log visibility
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from src.utils.device import configure_mps, detect_device
configure_mps()


# ── Version mapping ──────────────────────────────────────────────

# Maps folder names under Base_Fix_Specs/ to FIX version identifiers
# that the paths module understands.
VERSION_MAP = {
    "Fix_4.2":    "4.2",
    "Fix_4.4":    "4.4",
    "Fix_5.0SP2": "5.0SP2",
    "Fix_Latest": "Latest",
}

BASE_SPECS_DIR = _PROJECT_ROOT / "Base_Fix_Specs"

# Session-layer protocol per FIX version (for synthetic generation)
SESSION_PROTOCOL = {
    "4.2":    "FIX.4.2",
    "4.4":    "FIX.4.4",
    "5.0SP2": "FIXT.1.1",
    "Latest": "FIXT.1.1",
}

# ApplVerID per version (tag 1128, used in 5.0+ only)
APPL_VER_ID = {
    "4.2":    None,       # Not used in FIX 4.x
    "4.4":    None,
    "5.0SP2": "9",        # FIX 5.0 SP2
    "Latest": "11",       # FIX Latest (EP284+)
}

# Version → YAML config fields for temporary switching
VERSION_CONFIG = {
    "4.2":    {"protocol": "FIX.4.2",    "session": "FIX.4.2"},
    "4.4":    {"protocol": "FIX.4.4",    "session": "FIX.4.4"},
    "5.0SP2": {"protocol": "FIX.5.0SP2", "session": "FIXT.1.1"},
    "Latest": {"protocol": "FIX.Latest",  "session": "FIXT.1.1"},
}


# ═══════════════════════════════════════════════════════════════════
#  Helpers — per-version training support
# ═══════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def temporary_active_version(version: str):
    """Context manager: temporarily set the active FIX version in the YAML config.

    The original config content is restored on exit (even on error).
    Also invalidates the ``paths._load_version_config`` LRU cache so that
    all path helpers (checkpoint_dir, tokenizer_dir, etc.) resolve to the
    correct version-specific directories.
    """
    config_path = _PROJECT_ROOT / "config" / "model_config.yaml"
    original = config_path.read_text()
    try:
        cfg = yaml.safe_load(original)
        vcfg = VERSION_CONFIG.get(version, {"protocol": f"FIX.{version}", "session": "FIXT.1.1"})
        cfg["version"]["active"] = version
        cfg["version"]["protocol"] = vcfg["protocol"]
        cfg["version"]["session"] = vcfg["session"]
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        # Invalidate cached config so paths.* helpers pick up the new version
        paths._load_version_config.cache_clear()
        yield
    finally:
        config_path.write_text(original)
        # Restore the cache to the original version after reverting
        paths._load_version_config.cache_clear()


def build_single_version_data(
    version: str,
    num_synthetic: int = 24000,
    num_nl_pairs: int = 6000,
) -> list[str]:
    """Build training corpus for a single FIX version.

    Returns:
        List of version-tagged training lines.
    """
    version_tag = f"[FIX-{version}]"
    lines: list[str] = []

    # Find the spec folder for this version
    folder_name = None
    for fn, v in VERSION_MAP.items():
        if v == version:
            folder_name = fn
            break
    if not folder_name:
        raise ValueError(f"Unknown version: {version}")

    # 1. Ingest PDFs (if spec dir exists)
    spec_dir = BASE_SPECS_DIR / folder_name
    if spec_dir.is_dir():
        pdfs = list(spec_dir.glob("*.pdf"))
        if pdfs:
            print(f"  Ingesting {len(pdfs)} PDFs for FIX {version}...")
            try:
                specs = ingest_directory(
                    spec_dir, version=version, recursive=True, save=True,
                )
                print(f"  ✓ {len(specs)} canonical records")
            except Exception as exc:
                logger.error("Ingest failed for FIX %s: %s", version, exc)

    # 2. Spec-derived training lines
    try:
        canonical = load_canonical(version)
        if canonical:
            spec_lines = specs_to_training_lines(canonical)
            for line in spec_lines:
                lines.append(f"{version_tag} {line}")
            print(f"  Spec lines: {len(spec_lines)}")
    except Exception as exc:
        logger.warning("Could not load canonical for FIX %s: %s", version, exc)

    # 3. Synthetic FIX messages
    session = SESSION_PROTOCOL.get(version, "FIXT.1.1")
    appl_ver = APPL_VER_ID.get(version)
    gen = FIXDataGenerator()
    gen.FIX_VERSION = session
    synth = gen.generate_dataset(num_synthetic)
    for msg in synth:
        if appl_ver is None:
            parts = msg.split("|")
            parts = [p for p in parts if not p.startswith("1128=")]
            msg = "|".join(parts)
        elif appl_ver != "9":
            msg = msg.replace("1128=9", f"1128={appl_ver}")
        lines.append(f"{version_tag} {msg}")
    print(f"  Synthetic messages: {len(synth)}")

    # 4. NL-FIX pairs
    gen2 = FIXDataGenerator()
    nl_pairs = gen2.generate_natural_language_pairs(num_nl_pairs)
    for nl, fix_msg in nl_pairs:
        lines.append(f"{version_tag} {nl}\n{fix_msg}")
    print(f"  NL pairs: {len(nl_pairs)}")

    print(f"  Total lines for FIX {version}: {len(lines):,}")
    return lines


def tokenize_and_save_for_version(
    all_lines: list[str],
    version: str,
    train_ratio: float = 0.9,
) -> Path:
    """Tokenize and save binary training data for a specific FIX version.

    Saves to ``model_store/data/<slug>/processed/``.
    """
    output_dir = paths.processed_data_dir(version)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_texts, val_texts = _stratified_split(all_lines, train_ratio)
    print(f"  Train: {len(train_texts):,} | Val: {len(val_texts):,}")

    with open(output_dir / "train.txt", "w") as f:
        f.write("\n".join(train_texts))
    with open(output_dir / "val.txt", "w") as f:
        f.write("\n".join(val_texts))

    # Build tokenizer
    tokenizer = FixProtocolTokenizer(vocab_size=2048)
    vocab_sample = train_texts[:15000] if len(train_texts) > 15000 else train_texts
    tokenizer.build_vocab(vocab_sample)
    tok_dir = paths.tokenizer_dir(version)
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tok_dir))
    print(f"  Tokenizer saved → {tok_dir} (vocab: {len(tokenizer.token_to_id)})")

    def _encode_to_bin(texts: list[str], filename: str) -> int:
        all_tokens: list[int] = []
        for text in tqdm(texts, desc=f"  Tokenizing {filename}", unit=" lines"):
            tokens = tokenizer.encode(text, add_special_tokens=True)
            all_tokens.extend(tokens)
        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(output_dir / filename)
        print(f"  {filename}: {len(arr):,} tokens")
        return len(arr)

    train_tokens = _encode_to_bin(train_texts, "train.bin")
    val_tokens = _encode_to_bin(val_texts, "val.bin")

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions_included": [version],
        "total_lines": len(all_lines),
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "vocab_size": len(tokenizer.token_to_id),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Data saved → {output_dir}")
    return output_dir


def train_individual_versions(versions: list[str]) -> None:
    """Train separate, dedicated models for each specified FIX version.

    For each version this:
    1. Builds version-specific training data.
    2. Tokenizes and saves to version-specific dirs.
    3. Temporarily sets the active version in the YAML config.
    4. Trains and saves checkpoints to the version's checkpoint dir.
    """
    print("\n" + "═" * 70)
    print("  Per-Version Individual Training")
    print(f"  Versions: {', '.join(versions)}")
    print("═" * 70)

    for version in versions:
        print(f"\n{'═' * 70}")
        print(f"  Training FIX {version} (individual model)")
        print(f"{'═' * 70}")

        ver_start = time.time()

        # A — Build version-specific data
        print(f"\n  Building data for FIX {version}...")
        lines = build_single_version_data(version)

        # B — Tokenize and save
        print(f"\n  Tokenizing for FIX {version}...")
        tokenize_and_save_for_version(lines, version)

        # C — Train with temporary active version
        print(f"\n  Training FIX {version} model...")
        with temporary_active_version(version):
            # Only search this version's checkpoint dir for resume
            train_model(resume_version=version)

        elapsed = time.time() - ver_start
        print(f"\n  ✓ FIX {version} completed in {elapsed / 3600:.2f} hours")


# ═══════════════════════════════════════════════════════════════════
#  STEP 1 — Ingest PDFs
# ═══════════════════════════════════════════════════════════════════

def ingest_all_versions() -> dict[str, int]:
    """Ingest PDF specs from Base_Fix_Specs into per-version canonical JSON.

    Returns:
        Dict mapping version → number of records ingested.
    """
    print("\n" + "=" * 70)
    print("  STEP 1: Ingesting FIX Specification PDFs")
    print("=" * 70)

    results = {}

    for folder_name, version in VERSION_MAP.items():
        spec_dir = BASE_SPECS_DIR / folder_name
        if not spec_dir.is_dir():
            logger.warning("Spec directory not found: %s — skipping", spec_dir)
            continue

        pdf_count = len(list(spec_dir.glob("*.pdf")))
        if pdf_count == 0:
            logger.warning("No PDFs found in %s — skipping", spec_dir)
            continue

        print(f"\n{'─' * 50}")
        print(f"  FIX {version}: {pdf_count} PDF(s) from {folder_name}/")
        print(f"{'─' * 50}")

        try:
            specs = ingest_directory(
                spec_dir,
                version=version,
                recursive=True,
                save=True,
            )
            results[version] = len(specs)
            print(f"  ✓ FIX {version}: {len(specs)} canonical records extracted")

            # Show breakdown by kind
            kind_counts = defaultdict(int)
            for s in specs:
                kind_counts[s.kind.value] += 1
            for kind, count in sorted(kind_counts.items()):
                print(f"      {kind}: {count}")

        except Exception as exc:
            logger.error("Failed to ingest FIX %s: %s", version, exc, exc_info=True)
            results[version] = 0

    print(f"\n  Total records ingested: {sum(results.values()):,}")
    return results


# ═══════════════════════════════════════════════════════════════════
#  STEP 2 — Build Combined Training Data
# ═══════════════════════════════════════════════════════════════════

def build_combined_training_lines(
    num_synthetic_per_version: int = 24000,
    num_nl_pairs: int = 6000,
) -> list[str]:
    """Build training corpus from all ingested versions + synthetic data.

    Each training line is prefixed with a version tag so the model
    learns version-conditioned generation, e.g.:
        ``[FIX-4.2] 8=FIX.4.2|35=D|...``
        ``[FIX-5.0SP2] Tag 55 (Symbol) : STRING — Ticker symbol``

    Args:
        num_synthetic_per_version: Synthetic FIX messages per version.
        num_nl_pairs: NL-FIX pairs for the active (primary) version.

    Returns:
        Combined list of all training lines.
    """
    print("\n" + "=" * 70)
    print("  STEP 2: Building Combined Training Corpus")
    print("=" * 70)

    all_lines: list[str] = []
    version_stats: dict[str, int] = {}

    for folder_name, version in VERSION_MAP.items():
        version_tag = f"[FIX-{version}]"
        version_lines: list[str] = []

        # ── 2a. Ingested spec lines ──────────────────────────────
        try:
            canonical = load_canonical(version)
            if canonical:
                spec_lines = specs_to_training_lines(canonical)
                # Prefix each line with a version tag
                for line in spec_lines:
                    version_lines.append(f"{version_tag} {line}")
                print(f"  FIX {version}: {len(spec_lines)} spec-derived training lines")
            else:
                print(f"  FIX {version}: no canonical records found (ingest first)")
        except Exception as exc:
            logger.warning("Could not load canonical for FIX %s: %s", version, exc)

        # ── 2b. Synthetic FIX messages (version-tagged) ──────────
        session = SESSION_PROTOCOL.get(version, "FIXT.1.1")
        appl_ver = APPL_VER_ID.get(version)

        gen = FIXDataGenerator()
        gen.FIX_VERSION = session  # Override to generate correct session-layer

        synth_messages = gen.generate_dataset(num_synthetic_per_version)
        for msg in synth_messages:
            # For FIX 4.x, strip ApplVerID (tag 1128) since it doesn't exist
            if appl_ver is None:
                # Remove 1128=... field if present
                parts = msg.split("|")
                parts = [p for p in parts if not p.startswith("1128=")]
                msg = "|".join(parts)
            elif appl_ver != "9":
                # Replace the default ApplVerID with version-specific one
                msg = msg.replace("1128=9", f"1128={appl_ver}")
            version_lines.append(f"{version_tag} {msg}")

        print(f"  FIX {version}: {len(synth_messages)} synthetic messages generated")

        version_stats[version] = len(version_lines)
        all_lines.extend(version_lines)

    # ── 2c. NL-FIX pairs (active version) ────────────────────────
    active = paths.active_version()
    gen = FIXDataGenerator()
    nl_pairs = gen.generate_natural_language_pairs(num_nl_pairs)
    for nl, fix_msg in nl_pairs:
        all_lines.append(f"[FIX-{active}] {nl}\n{fix_msg}")
    print(f"  NL-FIX pairs (FIX {active}): {len(nl_pairs)}")

    # ── 2d. Existing scraped data (if any) ────────────────────────
    scraped_file = paths.raw_data_dir() / "scraped_training.txt"
    if scraped_file.exists():
        with open(scraped_file) as f:
            scraped = [line.strip() for line in f if line.strip()]
        all_lines.extend(scraped)
        print(f"  Scraped data: {len(scraped)} lines")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n  Combined corpus: {len(all_lines):,} total lines")
    for v, count in sorted(version_stats.items()):
        pct = count / len(all_lines) * 100 if all_lines else 0
        print(f"    FIX {v:8s}: {count:>6,} lines ({pct:5.1f}%)")

    return all_lines


# ═══════════════════════════════════════════════════════════════════
#  STEP 3 — Tokenize and Save Binary Data
# ═══════════════════════════════════════════════════════════════════

def tokenize_and_save(
    all_lines: list[str],
    train_ratio: float = 0.9,
) -> Path:
    """Tokenize the training corpus and save as binary files.

    Args:
        all_lines: Combined training text lines.
        train_ratio: Train/validation split ratio.

    Returns:
        Path to the processed data directory.
    """
    print("\n" + "=" * 70)
    print("  STEP 3: Tokenizing and Saving Binary Data")
    print("=" * 70)

    output_dir = paths.processed_data_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Stratified split ──────────────────────────────────────────
    train_texts, val_texts = _stratified_split(all_lines, train_ratio)
    print(f"  Train samples: {len(train_texts):,}")
    print(f"  Val samples:   {len(val_texts):,}")

    # Save raw text files
    with open(output_dir / "train.txt", "w") as f:
        f.write("\n".join(train_texts))
    with open(output_dir / "val.txt", "w") as f:
        f.write("\n".join(val_texts))

    # ── Build tokenizer ───────────────────────────────────────────
    print("\n  Building tokenizer on combined corpus...")
    tokenizer = FixProtocolTokenizer(vocab_size=2048)
    # Use a representative sample of up to 15k lines for vocab building
    vocab_sample = train_texts[:15000] if len(train_texts) > 15000 else train_texts
    tokenizer.build_vocab(vocab_sample)
    tok_dir = paths.tokenizer_dir()
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tok_dir))
    print(f"  Tokenizer saved → {tok_dir} (vocab: {len(tokenizer.token_to_id)})")

    # ── Tokenize to binary ────────────────────────────────────────
    def _encode_to_bin(texts: list[str], filename: str) -> int:
        all_tokens: list[int] = []
        for text in tqdm(texts, desc=f"  Tokenizing {filename}", unit=" lines"):
            tokens = tokenizer.encode(text, add_special_tokens=True)
            all_tokens.extend(tokens)
        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(output_dir / filename)
        print(f"  {filename}: {len(arr):,} tokens")
        return len(arr)

    train_tokens = _encode_to_bin(train_texts, "train.bin")
    val_tokens = _encode_to_bin(val_texts, "val.bin")

    # ── Save metadata ─────────────────────────────────────────────
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions_included": list(VERSION_MAP.values()),
        "total_lines": len(all_lines),
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "vocab_size": len(tokenizer.token_to_id),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  ✓ Data preparation complete → {output_dir}")
    return output_dir


def _compute_dynamic_epochs(
    num_training_lines: int,
    model_params: int = 19_400_000,
) -> int:
    """Compute an ideal number of training epochs based on dataset size.

    Heuristic: the model should see enough total tokens for good
    convergence, but not so many that it severely overfits.

    Rules of thumb for small transformer LMs:
    - Tiny dataset   (<10K lines):  150–200 epochs — needs many passes
    - Small dataset  (10K–30K):     100–150 epochs
    - Medium dataset (30K–60K):      80–120 epochs
    - Large dataset  (60K–100K):     60–80 epochs
    - Very large     (>100K):        40–60 epochs

    Early stopping (configured separately) is the safety net.

    Args:
        num_training_lines: Number of training text lines.
        model_params: Number of model parameters.

    Returns:
        Recommended max_epochs (clamped to [40, 200]).
    """
    if num_training_lines <= 0:
        return 100  # safe default

    # Base: inversely proportional to sqrt(data size), scaled for 19M params
    base = 100
    scale_factor = (30_000 / max(num_training_lines, 1)) ** 0.5
    epochs = int(base * scale_factor)

    # Clamp
    epochs = max(40, min(200, epochs))

    return epochs


# ═══════════════════════════════════════════════════════════════════
#  STEP 4 — Train
# ═══════════════════════════════════════════════════════════════════

def train_model(*, resume_version: str | None = None) -> None:
    """Load config, create model, and run the training loop.

    Args:
        resume_version: If given, limit checkpoint search to this version's
            directory so we don't accidentally resume a different version's
            checkpoint.  ``None`` searches all version directories.
    """
    print("\n" + "=" * 70)
    print("  STEP 4: Training FixProtoGPT")
    print("=" * 70)

    # ── Load config ───────────────────────────────────────────────
    model_config = ModelConfig.from_yaml()
    train_config = TrainConfig.from_yaml()

    # We still need the raw cfg dict for data paths below
    with open("config/model_config.yaml") as f:
        cfg = yaml.safe_load(f)

    # ── Load tokenizer ────────────────────────────────────────────
    print("\n  Loading tokenizer...")
    tokenizer = FixProtocolTokenizer(vocab_size=model_config.vocab_size)
    tokenizer.load(str(paths.tokenizer_dir()))
    print(f"  Tokenizer ready: {len(tokenizer.token_to_id)} tokens")

    # ── Create model ──────────────────────────────────────────────
    model = create_model(model_config)

    # ── Device ────────────────────────────────────────────────────
    device = detect_device()
    print(f"  Device: {device}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # ── Datasets ──────────────────────────────────────────────────
    train_path = cfg["data"].get("train_path") or str(paths.train_data())
    val_path = cfg["data"].get("val_path") or str(paths.val_data())

    train_dataset = FixProtocolDataset(train_path, tokenizer, model_config.max_seq_len)
    val_dataset = FixProtocolDataset(val_path, tokenizer, model_config.max_seq_len)
    print(f"  Training samples:   {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")

    # ── Dynamic epoch adjustment ──────────────────────────────────
    cfg_epochs = cfg["training"]["max_epochs"]
    if cfg_epochs == 0 or str(cfg_epochs).lower() == "auto":
        # Read training line count from metadata for dynamic calculation
        meta_file = paths.processed_data_dir() / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as mf:
                meta = json.load(mf)
            num_lines = meta.get("total_lines", len(train_dataset))
        else:
            num_lines = len(train_dataset)
        n_params = sum(p.numel() for p in model.parameters())
        dynamic_epochs = _compute_dynamic_epochs(num_lines, n_params)
        train_config.max_epochs = dynamic_epochs
        print(f"\n  Dynamic epochs: {dynamic_epochs} (auto-calculated from {num_lines:,} lines, {n_params/1e6:.1f}M params)")
    else:
        print(f"\n  Max epochs: {train_config.max_epochs} (from config)")

    if train_config.early_stopping_patience > 0:
        print(f"  Early stopping: patience {train_config.early_stopping_patience} evals")

    if train_config.target_val_loss > 0:
        print(f"  Target val loss: {train_config.target_val_loss:.4f}")

    # ── Trainer ───────────────────────────────────────────────────
    trainer = Trainer(model, train_dataset, val_dataset, train_config, device)

    # Auto-resume from the latest checkpoint if available
    if resume_version:
        # Limit search to this version's checkpoint dir only
        search_dir = str(paths.checkpoint_dir(resume_version))
    else:
        search_dir = train_config.checkpoint_dir
    resume_path = find_latest_checkpoint(search_dir)
    if resume_path:
        print(f"\n  Resuming from: {resume_path}")
        trainer.load_checkpoint(resume_path)

    # ── Go ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Training started — Ctrl+C for graceful shutdown")
    print("=" * 70 + "\n")
    trainer.train()

    print("\n  ✓ Training completed successfully!")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    """Full pipeline with CLI options.

    Modes
    -----
    (default)          Ingest + prepare + train combined multi-version model.
    --resume-only      Skip data prep, resume combined training from checkpoint.
    --per-version      After combined training, also train per-version models.
    --only-per-version Skip combined training; only train individual versions.
    --versions V [V…]  Specify which versions to train individually.
    """
    parser = argparse.ArgumentParser(
        description="FixProtoGPT — Multi-Version Training Pipeline",
    )
    parser.add_argument(
        "--resume-only", action="store_true",
        help="Skip data preparation; resume training from latest checkpoint.",
    )
    parser.add_argument(
        "--per-version", action="store_true",
        help="After combined training, also train per-version models.",
    )
    parser.add_argument(
        "--only-per-version", action="store_true",
        help="Skip combined training; only train individual per-version models.",
    )
    parser.add_argument(
        "--versions", nargs="+", default=None,
        help="Versions to train individually (default: all except active).",
    )
    args = parser.parse_args()

    start = time.time()

    print("\n" + "═" * 70)
    print("  FixProtoGPT — Multi-Version Training Pipeline")
    print(f"  Versions: {', '.join(VERSION_MAP.values())}")
    print(f"  Active version: {paths.active_version()}")
    print("═" * 70)

    # ── Combined multi-version training ───────────────────────────
    if not args.only_per_version:
        if args.resume_only:
            print("\n  [--resume-only] Skipping data preparation...")
        else:
            # Step 1: Ingest all specs
            ingest_results = ingest_all_versions()

            # Step 2: Build combined training corpus
            all_lines = build_combined_training_lines(
                num_synthetic_per_version=24000,
                num_nl_pairs=6000,
            )

            # Step 3: Tokenize and save
            tokenize_and_save(all_lines, train_ratio=0.9)

            elapsed_prep = time.time() - start
            print(f"\n  Data preparation took {elapsed_prep:.1f}s")

        # Step 4: Train combined model
        train_model()

    # ── Per-version individual training ───────────────────────────
    if args.per_version or args.only_per_version:
        versions = args.versions or [
            v for v in VERSION_MAP.values() if v != paths.active_version()
        ]
        train_individual_versions(versions)

    total = time.time() - start
    print(f"\n  Total pipeline time: {total:.1f}s ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()
