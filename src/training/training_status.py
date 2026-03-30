#!/usr/bin/env python3
"""
Module: src.training.training_status
=====================================

FixProtoGPT training status monitor.

Provides real-time training progress, loss tracking, and checkpoint
inventory from the command line.

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

import os
import re
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

from src.utils import paths

ROOT = Path(__file__).resolve().parent.parent.parent

# Support both single-version and multi-version training logs.
_LOG_MULTI = ROOT / "logs" / "training_all_versions.log"
_LOG_SINGLE = ROOT / "logs" / "training_output.log"
CKPT_DIR = paths.checkpoint_dir()


def _resolve_log_file() -> Path:
    """Pick the correct log file, preferring the multi-version log."""
    return _LOG_MULTI if _LOG_MULTI.exists() else _LOG_SINGLE

# ANSI colors
class C:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


def parse_log() -> Optional[dict]:
    """Parse the training log file for metrics.

    Returns:
        Dict with ``device``, ``steps``, ``val_evals``, etc., or
        ``None`` if the log file does not exist.
    """
    log_file = _resolve_log_file()
    if not log_file.exists():
        return None

    text = log_file.read_text()
    lines = text.strip().split("\n")

    info = {
        "log_file": log_file,
        "device": None,
        "model_params": None,
        "vocab_size": None,
        "train_samples": None,
        "val_samples": None,
        "total_tokens": None,
        "total_steps": None,
        "max_epochs": None,
        "steps": [],
        "val_evals": [],
        "checkpoints": [],
        "version_records": {},
        "version_lines": {},
        "combined_lines": None,
    }

    for line in lines:
        # Device
        m = re.search(r"Using device:\s*(\S+)", line)
        if m:
            info["device"] = m.group(1).upper()

        # Model params
        m = re.search(r"Model parameters:\s*([\d,]+)\s*\(([\d.]+)M\)", line)
        if m:
            info["model_params"] = m.group(2) + "M"

        # Vocab size
        m = re.search(r"vocab size:\s*(\d+)", line)
        if m:
            info["vocab_size"] = int(m.group(1))

        # Total tokens
        m = re.search(r"Loaded dataset with (\d+) tokens", line)
        if m:
            if info["total_tokens"] is None:
                info["total_tokens"] = int(m.group(1))

        # Training/val samples (may contain commas: "14,833")
        m = re.search(r"Training samples:\s*([\d,]+)", line)
        if m:
            info["train_samples"] = int(m.group(1).replace(",", ""))
        m = re.search(r"Validation samples:\s*([\d,]+)", line)
        if m:
            info["val_samples"] = int(m.group(1).replace(",", ""))

        # Total steps
        m = re.search(r"Total steps:\s*~?(\d+)", line)
        if m:
            info["total_steps"] = int(m.group(1))

        # Max epochs
        m = re.search(r"training for (\d+) epochs", line)
        if m:
            info["max_epochs"] = int(m.group(1))

        # Epoch
        m = re.search(r"Epoch (\d+)/(\d+)", line)
        if m:
            info["current_epoch"] = int(m.group(1))
            info["max_epochs"] = int(m.group(2))

        # Training step
        m = re.match(r"Step (\d+) \| Loss: ([\d.]+) \| LR: ([\d.eE+-]+)", line)
        if m:
            info["steps"].append({
                "step": int(m.group(1)),
                "loss": float(m.group(2)),
                "lr": float(m.group(3)),
            })

        # Validation
        m = re.match(r"Step (\d+) \| Val Loss: ([\d.]+)", line)
        if m:
            info["val_evals"].append({
                "step": int(m.group(1)),
                "val_loss": float(m.group(2)),
            })

        # Checkpoints
        m = re.match(r"Checkpoint saved: (.+)", line)
        if m:
            info["checkpoints"].append(m.group(1))

        # Per-version canonical records (e.g. "✓ FIX 4.4: 1234 canonical records")
        m = re.search(r"FIX ([\w.]+):\s*([\d,]+)\s*canonical records", line)
        if m:
            ver = m.group(1)
            count = int(m.group(2).replace(",", ""))
            info["version_records"][ver] = count

        # Per-version training lines (e.g. "FIX 4.4: 5678 lines (12.3%)")
        m = re.search(r"FIX\s+([\w.]+)\s*:\s*([\d,]+)\s+lines\s+\(", line)
        if m:
            ver = m.group(1).strip()
            count = int(m.group(2).replace(",", ""))
            info["version_lines"][ver] = count

        # Combined corpus size
        m = re.search(r"Combined corpus:\s*([\d,]+)\s*total lines", line)
        if m:
            info["combined_lines"] = int(m.group(1).replace(",", ""))

    return info


def get_process_info() -> dict:
    """Check if a training process is currently running.

    Returns:
        Dict with ``running`` bool and optional ``pid``, ``cpu``, ``mem``.
    """
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if ("train.py" in line or "train_all_versions.py" in line) and "grep" not in line and "training_status" not in line:
                parts = line.split()
                return {
                    "pid": parts[1],
                    "cpu": parts[2],
                    "mem": parts[3],
                    "time": parts[9] if len(parts) > 9 else "?",
                    "running": True,
                }
    except Exception:
        pass
    return {"running": False}


def get_checkpoint_info() -> Tuple[list, Optional[dict]]:
    """Get checkpoint file details.

    Returns:
        A 2-tuple ``(checkpoints, meta)`` where *checkpoints* is a
        list of dicts and *meta* is the optional checkpoint metadata.
    """
    ckpts = []
    if not CKPT_DIR.exists():
        return ckpts, None
    for f in sorted(CKPT_DIR.glob("*.pt")):
        stat = f.stat()
        ckpts.append({
            "name": f.name,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(stat.st_mtime),
        })
    # Check for bundled metadata
    meta_file = CKPT_DIR / "checkpoint_meta.json"
    meta = None
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            pass
    return ckpts, meta


def loss_sparkline(values: list, width: int = 40) -> str:
    """Render a mini ASCII sparkline chart of loss values.

    Args:
        values: Numeric loss values.
        width: Maximum chart width in characters.

    Returns:
        A unicode sparkline string.
    """
    if len(values) < 2:
        return ""
    # Sample to fit width
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values

    mn, mx = min(sampled), max(sampled)
    if mx == mn:
        return "▄" * len(sampled)

    blocks = " ▁▂▃▄▅▆▇█"
    line = ""
    for v in sampled:
        idx = int((v - mn) / (mx - mn) * (len(blocks) - 1))
        line += blocks[idx]
    return line


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string.

    Args:
        seconds: Number of elapsed seconds.

    Returns:
        e.g. ``"2h 15m 30s"``.
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def display_status():
    """Display full training status."""
    print(f"\n{C.BOLD}{C.CYAN}{'='*60}{C.END}")
    print(f"{C.BOLD}{C.CYAN}  FixProtoGPT — Training Status{C.END}")
    print(f"{C.BOLD}{C.CYAN}{'='*60}{C.END}")
    print(f"{C.DIM}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.END}\n")

    # Process status
    proc = get_process_info()
    LOG_FILE = _resolve_log_file()
    if proc["running"]:
        print(f"  {C.GREEN}● TRAINING ACTIVE{C.END}  {C.DIM}PID {proc['pid']}  |  CPU {proc['cpu']}%  |  MEM {proc['mem']}%  |  Time {proc['time']}{C.END}")
        # Check for output buffering issues
        if LOG_FILE.exists():
            try:
                disk_size = LOG_FILE.stat().st_size
                lsof = subprocess.run(
                    ["lsof", "-p", proc["pid"]], capture_output=True, text=True, timeout=5
                )
                buffered_warned = False
                for lline in lsof.stdout.split("\n"):
                    if "training_output.log" in lline and not buffered_warned:
                        # lsof offset column is prefixed with "0t"
                        for p in lline.split():
                            if p.startswith("0t"):
                                try:
                                    written = int(p[2:])
                                    if written > disk_size + 1024:
                                        buffered_kb = (written - disk_size) / 1024
                                        print(f"  {C.YELLOW}⚠ ~{buffered_kb:.0f}KB of log output is buffered (not yet on disk){C.END}")
                                        buffered_warned = True
                                except ValueError:
                                    pass
                                break
            except Exception:
                pass
    else:
        print(f"  {C.RED}● TRAINING STOPPED{C.END}")

    # Parse log
    info = parse_log()
    if not info:
        log_file = _resolve_log_file()
        print(f"\n  {C.YELLOW}No training log found at {log_file}{C.END}\n")
        return

    print()

    # Model / dataset info
    print(f"  {C.BOLD}Model{C.END}")
    parts = []
    if info["model_params"]:
        parts.append(f"Params: {info['model_params']}")
    if info["vocab_size"]:
        parts.append(f"Vocab: {info['vocab_size']}")
    if info["device"]:
        parts.append(f"Device: {info['device']}")
    print(f"  {C.DIM}{'  |  '.join(parts)}{C.END}")

    if info["train_samples"] or info["total_tokens"]:
        dparts = []
        if info["train_samples"]:
            dparts.append(f"Train: {info['train_samples']:,} samples")
        if info["val_samples"]:
            dparts.append(f"Val: {info['val_samples']:,} samples")
        if info["total_tokens"]:
            dparts.append(f"Tokens: {info['total_tokens']/1e6:.1f}M")
        print(f"  {C.DIM}{'  |  '.join(dparts)}{C.END}")

    # Per-version ingestion stats
    if info["version_records"]:
        print(f"\n  {C.BOLD}Versions Ingested{C.END}")
        total_records = sum(info["version_records"].values())
        for ver, count in sorted(info["version_records"].items()):
            pct = count / total_records * 100 if total_records else 0
            print(f"    FIX {ver:8s}: {count:>6,} records ({pct:5.1f}%)")
        if info["combined_lines"]:
            print(f"    {C.DIM}Combined corpus: {info['combined_lines']:,} lines{C.END}")
        if info["version_lines"]:
            print(f"\n  {C.BOLD}Training Lines per Version{C.END}")
            total_vl = sum(info["version_lines"].values())
            for ver, count in sorted(info["version_lines"].items()):
                pct = count / total_vl * 100 if total_vl else 0
                print(f"    FIX {ver:8s}: {count:>6,} lines ({pct:5.1f}%)")

    print()

    # Progress
    if info["steps"]:
        current_step = info["steps"][-1]["step"]
        total = info["total_steps"] or (len(info["steps"]) * 2 if info["steps"] else 1)

        # Check if checkpoints show higher progress than the log (buffering)
        ckpts_list, ckpt_meta = get_checkpoint_info()
        ckpt_max_step = 0
        for ck in ckpts_list:
            m = re.search(r"step_(\d+)\.pt", ck["name"])
            if m:
                ckpt_max_step = max(ckpt_max_step, int(m.group(1)))
        if ckpt_meta and ckpt_meta.get("step", 0) > ckpt_max_step:
            ckpt_max_step = ckpt_meta["step"]

        actual_step = max(current_step, ckpt_max_step)
        if actual_step > current_step:
            print(f"  {C.YELLOW}Note: Log shows step {current_step:,} but checkpoints confirm at least step {actual_step:,}{C.END}")
            print(f"  {C.DIM}(Output buffering — log will catch up when buffer flushes){C.END}")
            current_step = actual_step

        pct = (current_step / total) * 100
        epoch = info.get("current_epoch", 1)
        max_ep = info["max_epochs"] or 50

        print(f"  {C.BOLD}Progress{C.END}")
        # Progress bar (cap visual bar at 100%)
        bar_width = 40
        filled = min(bar_width, int(bar_width * pct / 100))
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"  [{C.GREEN}{bar}{C.END}] {pct:.1f}%")
        print(f"  {C.DIM}Step {current_step:,} / {total:,}  |  Epoch {epoch} / {max_ep}{C.END}")

        # ETA estimate
        if proc["running"] and len(info["steps"]) >= 10:
            log_mtime = LOG_FILE.stat().st_mtime
            log_ctime = LOG_FILE.stat().st_ctime
            elapsed = log_mtime - log_ctime
            if elapsed > 0 and current_step > 0:
                steps_per_sec = current_step / elapsed
                remaining_steps = total - current_step
                eta_sec = remaining_steps / steps_per_sec
                print(f"  {C.DIM}Speed: ~{steps_per_sec:.1f} steps/s  |  ETA: ~{format_duration(eta_sec)}{C.END}")

        print()

        # Current metrics
        latest = info["steps"][-1]
        print(f"  {C.BOLD}Current Metrics{C.END}")
        print(f"  Train Loss: {C.YELLOW}{latest['loss']:.4f}{C.END}  |  LR: {latest['lr']:.6f}")

        if info["val_evals"]:
            latest_val = info["val_evals"][-1]
            best_val = min(info["val_evals"], key=lambda x: x["val_loss"])
            print(f"  Val Loss:   {C.YELLOW}{latest_val['val_loss']:.4f}{C.END} (step {latest_val['step']})")
            print(f"  Best Val:   {C.GREEN}{best_val['val_loss']:.4f}{C.END} (step {best_val['step']})")

        print()

        # Loss curve
        losses = [s["loss"] for s in info["steps"]]
        print(f"  {C.BOLD}Loss Curve{C.END} {C.DIM}(min={min(losses):.3f}  max={max(losses):.3f}){C.END}")
        spark = loss_sparkline(losses)
        print(f"  {C.BLUE}{spark}{C.END}")
        step_first = info["steps"][0]["step"]
        step_last = info["steps"][-1]["step"]
        print(f"  {C.DIM}step {step_first}{'─' * (len(spark) - len(str(step_first)) - len(str(step_last)))}step {step_last}{C.END}")

        # Val loss curve
        if len(info["val_evals"]) > 1:
            val_losses = [v["val_loss"] for v in info["val_evals"]]
            print(f"\n  {C.BOLD}Val Loss Curve{C.END} {C.DIM}(min={min(val_losses):.3f}  max={max(val_losses):.3f}){C.END}")
            spark_val = loss_sparkline(val_losses)
            print(f"  {C.GREEN}{spark_val}{C.END}")

        print()

    # Checkpoints
    ckpts, meta = get_checkpoint_info()
    if ckpts:
        print(f"  {C.BOLD}Checkpoints{C.END}  {C.DIM}({CKPT_DIR}){C.END}")
        for ck in ckpts:
            age = datetime.now() - ck["modified"]
            ago = format_duration(age.total_seconds())
            print(f"  {C.DIM}  {ck['name']:20s}  {ck['size_mb']:7.1f} MB  ({ago} ago){C.END}")

        if meta:
            print(f"\n  {C.BOLD}Checkpoint Metadata{C.END}")
            for key in ["fix_version", "fix_versions_trained", "train_samples", "val_samples", "model_params", "best_val_loss"]:
                if key in meta:
                    val = meta[key]
                    if isinstance(val, list):
                        val = ", ".join(str(v) for v in val)
                    elif isinstance(val, float):
                        val = f"{val:.4f}"
                    elif isinstance(val, int):
                        val = f"{val:,}"
                    print(f"  {C.DIM}  {key:24s}  {val}{C.END}")

    # Bundled assets
    bundled = []
    for sub in ["tokenizer", "data", "config"]:
        p = CKPT_DIR / sub
        if p.exists():
            bundled.append(sub)
    if bundled:
        print(f"\n  {C.BOLD}Bundled Assets{C.END}  {', '.join(bundled)}")

    print(f"\n{C.BOLD}{C.CYAN}{'='*60}{C.END}\n")


if __name__ == "__main__":
    import sys
    if "--watch" in sys.argv or "-w" in sys.argv:
        interval = 30
        for arg in sys.argv[1:]:
            if arg.isdigit():
                interval = int(arg)
        print(f"Watching every {interval}s... (Ctrl+C to stop)\n")
        try:
            while True:
                os.system("clear")
                display_status()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        display_status()
