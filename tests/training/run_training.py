#!/usr/bin/env python3
"""
FixProtoGPT – Full Model Evaluation Runner

Moved from the project root into tests/training/ for better organization.
Can be invoked via:
    python -m tests.training.run_training
    python evaluate.py            (thin wrapper at project root)
"""

import sys
import time
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.generate import FixProtoGPTInference
from src.utils import paths


# ── FIX Validation Helpers ──────────────────────────────────────────

REQUIRED_HEADER_TAGS = {"8", "9", "35", "49", "56", "34", "52"}

MSG_TYPE_REQUIRED = {
    "D": {"11", "21", "55", "54", "38", "40", "60"},
    "8": {"37", "17", "150", "39", "55", "54"},
    "F": {"11", "41", "55", "54", "60"},
    "G": {"11", "41", "55", "54", "38", "40", "60"},
    "V": {"262", "263", "264"},
    "A": {"98", "108"},
}

VALID_MSG_TYPES = {
    "0", "1", "2", "3", "4", "5", "A", "D", "8", "F", "G", "9",
    "V", "W", "X", "Y", "AE", "AD", "J", "P",
}


def parse_fields(msg):
    """Parse a FIX-style message into (tag, value) pairs."""
    delim = "\x01" if "\x01" in msg else "|"
    parts = msg.split(delim)
    fields = []
    for p in parts:
        if "=" in p:
            tag, val = p.split("=", 1)
            fields.append((tag.strip(), val.strip()))
    return fields


def validate_fix_message(msg):
    """Return a dict of quality metrics for a single generated message."""
    fields = parse_fields(msg)
    tags = [t for t, _ in fields]
    tag_set = set(tags)

    result = {
        "raw": msg,
        "num_fields": len(fields),
        "has_begin_string": "8" in tag_set,
        "has_msg_type": "35" in tag_set,
        "has_checksum": "10" in tag_set,
        "has_body_length": "9" in tag_set,
        "header_complete": REQUIRED_HEADER_TAGS.issubset(tag_set),
        "valid_structure": False,
        "correct_msg_type": False,
        "msg_type_fields_present": False,
        "duplicate_tags": len(tags) != len(tag_set),
        "tag_value_format_ok": True,
    }

    msg_type = None
    for t, v in fields:
        if t == "35":
            msg_type = v
            break

    if msg_type and msg_type in VALID_MSG_TYPES:
        result["correct_msg_type"] = True

    if msg_type and msg_type in MSG_TYPE_REQUIRED:
        required = MSG_TYPE_REQUIRED[msg_type]
        result["msg_type_fields_present"] = required.issubset(tag_set)

    if fields:
        result["valid_structure"] = (fields[0][0] == "8" and fields[-1][0] == "10")

    for t, _ in fields:
        if not t.isdigit():
            result["tag_value_format_ok"] = False
            break

    return result


# ── Prompts ─────────────────────────────────────────────────────────

COMPLETION_PROMPTS = [
    "8=FIXT.1.1|35=D|",
    "8=FIXT.1.1|35=D|55=AAPL|54=1|",
    "8=FIXT.1.1|35=8|55=TSLA|54=1|38=50|",
    "8=FIXT.1.1|35=F|11=ORD456|",
    "8=FIXT.1.1|35=A|",
    "8=FIXT.1.1|35=V|",
    "8=FIXT.1.1|35=D|55=MSFT|54=2|38=200|40=2|44=350.00|",
    "8=FIXT.1.1|35=8|37=EXEC789|17=EXEC789|150=2|39=2|55=GOOG|",
    "8=FIXT.1.1|35=G|11=ORD123|41=ORD122|55=AMZN|",
    "8=FIXT.1.1|35=D|55=META|54=1|38=500|40=1|",
]

NL_PROMPTS = [
    "Buy 100 shares of AAPL at market price",
    "Sell 50 shares of MSFT at limit price 410.00",
    "Cancel order ORD12345 for symbol GOOG",
    "Request market data for TSLA",
    "Create a logon message",
    "Replace order ORD789 for AMZN with new quantity 300",
    "Buy 1000 shares of JPM at limit price 195.50",
    "Sell 200 shares of NVDA at market",
    "Send a heartbeat message",
    "Request full market depth for SPY",
]


def _score(results):
    n = len(results)
    if n == 0:
        return {}
    s = {
        "total": n,
        "has_msg_type": sum(r["has_msg_type"] for r in results),
        "correct_msg_type": sum(r["correct_msg_type"] for r in results),
        "valid_structure": sum(r["valid_structure"] for r in results),
        "header_complete": sum(r["header_complete"] for r in results),
        "tag_format_ok": sum(r["tag_value_format_ok"] for r in results),
        "msg_type_fields": sum(r["msg_type_fields_present"] for r in results),
        "avg_fields": sum(r["num_fields"] for r in results) / n,
    }
    if any("gen_time" in r for r in results):
        s["avg_time"] = sum(r.get("gen_time", 0) for r in results) / n
    return s


def run_evaluation():
    """Run the full model evaluation pipeline."""
    print("=" * 70)
    print("  FixProtoGPT — Local Model Evaluation")
    print("=" * 70)
    print()

    model_path = str(paths.best_model())
    tokenizer_path = str(paths.tokenizer_dir())

    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        return

    print("Loading model (best.pt) ...")
    t0 = time.time()
    engine = FixProtoGPTInference(model_path, tokenizer_path)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s\n")

    # ── 1. Message Completion ───────────────────────────────────────
    print("─" * 70)
    print("  TASK 1: FIX Message Completion")
    print("─" * 70)
    completion_results = []
    for i, prompt in enumerate(COMPLETION_PROMPTS, 1):
        t0 = time.time()
        output = engine.complete_fix_message(prompt, temperature=0.7, max_new_tokens=200)
        gen_time = time.time() - t0
        metrics = validate_fix_message(output)
        metrics["gen_time"] = gen_time
        completion_results.append(metrics)

        print(f"\n  [{i:2d}] Prompt: {prompt}")
        print(f"       Output: {output[:200]}{'...' if len(output) > 200 else ''}")
        print(f"       Fields: {metrics['num_fields']}  "
              f"MsgType: {'✓' if metrics['correct_msg_type'] else '✗'}  "
              f"Structure: {'✓' if metrics['valid_structure'] else '✗'}  "
              f"Format: {'✓' if metrics['tag_value_format_ok'] else '✗'}  "
              f"({gen_time:.2f}s)")

    # ── 2. Natural Language → FIX ───────────────────────────────────
    print("\n" + "─" * 70)
    print("  TASK 2: Natural Language → FIX Message")
    print("─" * 70)
    nl_results = []
    for i, prompt in enumerate(NL_PROMPTS, 1):
        t0 = time.time()
        output = engine.natural_language_to_fix(prompt, temperature=0.7, max_new_tokens=200)
        gen_time = time.time() - t0
        metrics = validate_fix_message(output)
        metrics["gen_time"] = gen_time
        nl_results.append(metrics)

        print(f"\n  [{i:2d}] NL: {prompt}")
        print(f"       FIX: {output[:200]}{'...' if len(output) > 200 else ''}")
        print(f"       Fields: {metrics['num_fields']}  "
              f"MsgType: {'✓' if metrics['correct_msg_type'] else '✗'}  "
              f"Structure: {'✓' if metrics['valid_structure'] else '✗'}  "
              f"Format: {'✓' if metrics['tag_value_format_ok'] else '✗'}  "
              f"({gen_time:.2f}s)")

    # ── 3. Consistency ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  TASK 3: Consistency (5 samples from same prompt)")
    print("─" * 70)
    consistency_prompt = "8=FIXT.1.1|35=D|55=AAPL|54=1|38=100|40=2|44=150.00|"
    print(f"  Prompt: {consistency_prompt}\n")
    consistency_outputs = []
    for s in range(5):
        output = engine.complete_fix_message(consistency_prompt, temperature=0.8, max_new_tokens=200)
        metrics = validate_fix_message(output)
        consistency_outputs.append(metrics)
        short = output[:120] + ("..." if len(output) > 120 else "")
        print(f"    Sample {s+1}: {short}")
        print(f"             Fields={metrics['num_fields']}  "
              f"Struct={'✓' if metrics['valid_structure'] else '✗'}  "
              f"MsgType={'✓' if metrics['correct_msg_type'] else '✗'}")

    # ── 4. Temperature Sensitivity ──────────────────────────────────
    print("\n" + "─" * 70)
    print("  TASK 4: Temperature Sensitivity")
    print("─" * 70)
    temp_prompt = "8=FIXT.1.1|35=D|55=TSLA|54=1|38=100|"
    for temp in [0.3, 0.5, 0.7, 1.0, 1.2]:
        output = engine.complete_fix_message(temp_prompt, temperature=temp, max_new_tokens=150)
        m = validate_fix_message(output)
        short = output[:100] + ("..." if len(output) > 100 else "")
        print(f"  T={temp:.1f}  Fields={m['num_fields']:2d}  "
              f"Struct={'✓' if m['valid_structure'] else '✗'}  "
              f"Format={'✓' if m['tag_value_format_ok'] else '✗'}  → {short}")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)

    for label, results in [
        ("Completion", completion_results),
        ("NL→FIX", nl_results),
        ("Consistency", consistency_outputs),
    ]:
        s = _score(results)
        if not s:
            continue
        n = s["total"]
        print(f"\n  {label} ({n} samples)")
        print(f"    Has MsgType (35):     {s['has_msg_type']:2d}/{n}  ({100*s['has_msg_type']/n:.0f}%)")
        print(f"    Correct MsgType:      {s['correct_msg_type']:2d}/{n}  ({100*s['correct_msg_type']/n:.0f}%)")
        print(f"    Valid Structure:       {s['valid_structure']:2d}/{n}  ({100*s['valid_structure']/n:.0f}%)")
        print(f"    Header Complete:       {s['header_complete']:2d}/{n}  ({100*s['header_complete']/n:.0f}%)")
        print(f"    Tag Format OK:        {s['tag_format_ok']:2d}/{n}  ({100*s['tag_format_ok']/n:.0f}%)")
        print(f"    MsgType Fields:       {s['msg_type_fields']:2d}/{n}  ({100*s['msg_type_fields']/n:.0f}%)")
        print(f"    Avg Fields/Message:   {s['avg_fields']:.1f}")
        if "avg_time" in s:
            print(f"    Avg Generation Time:  {s['avg_time']:.2f}s")

    all_results = completion_results + nl_results
    if all_results:
        s = _score(all_results)
        n = s["total"]
        overall = (
            s["correct_msg_type"] + s["valid_structure"] + s["tag_format_ok"]
            + s["msg_type_fields"] + s["header_complete"]
        ) / (5 * n) * 100
        print(f"\n  ╔══════════════════════════════════════╗")
        print(f"  ║  OVERALL QUALITY SCORE:  {overall:5.1f}%       ║")
        print(f"  ╚══════════════════════════════════════╝")

    print()


if __name__ == "__main__":
    run_evaluation()
