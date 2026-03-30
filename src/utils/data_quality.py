"""
Module: src.utils.data_quality
================================

Data quality analysis and dashboard utilities for FixProtoGPT
training data.

Provides tools for:
    - Token distribution analysis
    - Duplicate detection
    - Message-type distribution checks
    - Perplexity tracking across training runs
    - Anomaly detection in training data

Usage::

    from src.utils.data_quality import DataQualityAnalyser

    analyser = DataQualityAnalyser()
    report = analyser.analyse_training_data("model_store/data/fix-5-0sp2/processed")
    analyser.print_report(report)

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DataQualityAnalyser:
    """Analyse training data quality for FixProtoGPT."""

    def __init__(self) -> None:
        """Initialise the analyser."""
        self._reports: List[Dict[str, Any]] = []

    # ── Full analysis pipeline ────────────────────────────────────

    def analyse_training_data(self, data_dir: str) -> Dict[str, Any]:
        """Run a complete data quality analysis on a training data directory.

        Expects the directory to contain ``train.txt`` and optionally
        ``val.txt``, ``train.bin``, ``val.bin``, and ``metadata.json``.

        Args:
            data_dir: Path to the processed data directory.

        Returns:
            Dict with analysis results: ``token_distribution``,
            ``message_type_distribution``, ``duplicates``,
            ``anomalies``, ``summary``.
        """
        data_path = Path(data_dir)
        report: Dict[str, Any] = {"data_dir": str(data_path), "errors": []}

        # Load training text
        train_txt = data_path / "train.txt"
        val_txt = data_path / "val.txt"

        train_lines: List[str] = []
        val_lines: List[str] = []

        if train_txt.exists():
            with open(train_txt) as f:
                train_lines = [l.strip() for l in f if l.strip()]
        else:
            report["errors"].append(f"train.txt not found in {data_path}")

        if val_txt.exists():
            with open(val_txt) as f:
                val_lines = [l.strip() for l in f if l.strip()]

        all_lines = train_lines + val_lines

        # Run analyses
        report["sample_count"] = {
            "train": len(train_lines),
            "val": len(val_lines),
            "total": len(all_lines),
        }

        report["message_type_distribution"] = self._analyse_message_types(all_lines)
        report["duplicates"] = self._detect_duplicates(all_lines)
        report["length_stats"] = self._length_statistics(all_lines)
        report["anomalies"] = self._detect_anomalies(all_lines)

        # Token distribution from .bin files
        train_bin = data_path / "train.bin"
        if train_bin.exists():
            report["token_distribution"] = self._analyse_token_distribution(str(train_bin))

        # Summary
        report["summary"] = self._build_summary(report)

        self._reports.append(report)
        return report

    # ── Message type distribution ─────────────────────────────────

    @staticmethod
    def _analyse_message_types(lines: List[str]) -> Dict[str, Any]:
        """Count FIX message types and NL pairs in the dataset.

        Args:
            lines: Training text lines.

        Returns:
            Dict with ``counts``, ``percentages``, ``categories``.
        """
        type_counts: Counter = Counter()
        category_counts: Counter = Counter()

        _msg_categories = {
            "D": "Order", "8": "Order", "F": "Order", "G": "Order",
            "V": "MarketData", "W": "MarketData", "X": "MarketData",
            "A": "Session", "0": "Session", "5": "Session",
            "3": "Admin", "4": "Admin",
        }

        for line in lines:
            if "\n" in line:
                type_counts["nl_pair"] += 1
                category_counts["NL-FIX"] += 1
                continue

            match = re.search(r'(?:^|\|)35=([A-Za-z0-9]+)', line)
            if match:
                mt = match.group(1)
                type_counts[mt] += 1
                category_counts[_msg_categories.get(mt, "Other")] += 1
            else:
                type_counts["other"] += 1
                category_counts["Other"] += 1

        total = sum(type_counts.values()) or 1
        percentages = {k: round(v / total * 100, 2) for k, v in type_counts.most_common()}

        return {
            "counts": dict(type_counts.most_common()),
            "percentages": percentages,
            "categories": dict(category_counts.most_common()),
            "unique_types": len(type_counts),
        }

    # ── Duplicate detection ───────────────────────────────────────

    @staticmethod
    def _detect_duplicates(lines: List[str]) -> Dict[str, Any]:
        """Find exact and near-duplicate training lines.

        Args:
            lines: Training text lines.

        Returns:
            Dict with ``exact_duplicates`` count and ``duplicate_rate``.
        """
        seen: Dict[str, int] = {}
        duplicates = 0

        for line in lines:
            h = hashlib.md5(line.encode()).hexdigest()
            if h in seen:
                duplicates += 1
            else:
                seen[h] = 0
            seen[h] += 1

        total = len(lines) or 1
        top_dupes = [
            (count, h) for h, count in seen.items() if count > 1
        ]
        top_dupes.sort(reverse=True)

        return {
            "exact_duplicates": duplicates,
            "duplicate_rate": round(duplicates / total * 100, 2),
            "unique_lines": len(seen),
            "total_lines": len(lines),
            "top_duplicate_counts": [c for c, _ in top_dupes[:10]],
        }

    # ── Length statistics ─────────────────────────────────────────

    @staticmethod
    def _length_statistics(lines: List[str]) -> Dict[str, Any]:
        """Compute character and field-count statistics.

        Args:
            lines: Training text lines.

        Returns:
            Dict with ``char_length`` and ``field_count`` stats.
        """
        char_lengths = [len(l) for l in lines]
        field_counts = []
        for l in lines:
            delimiter = "\x01" if "\x01" in l else "|"
            fields = [f for f in l.split(delimiter) if "=" in f]
            field_counts.append(len(fields))

        def _stats(vals: List[int]) -> Dict[str, float]:
            """Compute min/max/mean/median/std for a list of integers."""
            if not vals:
                return {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
            arr = np.array(vals)
            return {
                "min": int(arr.min()),
                "max": int(arr.max()),
                "mean": round(float(arr.mean()), 2),
                "median": round(float(np.median(arr)), 2),
                "std": round(float(arr.std()), 2),
            }

        return {
            "char_length": _stats(char_lengths),
            "field_count": _stats(field_counts),
        }

    # ── Anomaly detection ─────────────────────────────────────────

    @staticmethod
    def _detect_anomalies(lines: List[str]) -> Dict[str, Any]:
        """Detect data quality anomalies.

        Checks for:
            - Empty or very short lines (< 10 chars)
            - Lines without any FIX fields
            - Extremely long lines (> 2000 chars)
            - Lines with unusual characters

        Args:
            lines: Training text lines.

        Returns:
            Dict with anomaly counts and examples.
        """
        anomalies: Dict[str, List[str]] = {
            "too_short": [],
            "too_long": [],
            "no_fix_fields": [],
            "unusual_chars": [],
        }

        for i, line in enumerate(lines):
            if len(line) < 10:
                anomalies["too_short"].append(f"Line {i}: {line[:50]}")
            elif len(line) > 2000:
                anomalies["too_long"].append(f"Line {i}: {line[:50]}...")

            if "=" not in line and "\n" not in line:
                anomalies["no_fix_fields"].append(f"Line {i}: {line[:50]}")

            # Check for non-printable chars (except SOH which is expected)
            cleaned = line.replace("\x01", "").replace("\n", "")
            if any(ord(c) < 32 or ord(c) > 126 for c in cleaned):
                anomalies["unusual_chars"].append(f"Line {i}: {line[:50]}")

        return {
            category: {
                "count": len(items),
                "examples": items[:5],  # Cap at 5 examples
            }
            for category, items in anomalies.items()
        }

    # ── Token distribution ────────────────────────────────────────

    @staticmethod
    def _analyse_token_distribution(bin_path: str) -> Dict[str, Any]:
        """Analyse token frequency distribution from a binary token file.

        Args:
            bin_path: Path to the ``.bin`` token file.

        Returns:
            Dict with ``total_tokens``, ``unique_tokens``,
            ``top_tokens``, ``coverage_at_k``.
        """
        tokens = np.fromfile(bin_path, dtype=np.uint16)
        total = len(tokens)

        if total == 0:
            return {"total_tokens": 0, "unique_tokens": 0}

        counts = Counter(int(t) for t in tokens)
        unique = len(counts)

        # Top-20 most common tokens
        top20 = counts.most_common(20)

        # Coverage: what percentage of tokens are covered by top-K types
        sorted_counts = sorted(counts.values(), reverse=True)
        coverage: Dict[str, float] = {}
        cum = 0
        for k in [10, 50, 100, 200, 500]:
            cum_k = sum(sorted_counts[:k]) if k <= len(sorted_counts) else total
            coverage[f"top_{k}"] = round(cum_k / total * 100, 2)

        # Entropy
        probs = np.array(sorted_counts) / total
        entropy = -np.sum(probs * np.log2(probs + 1e-12))

        return {
            "total_tokens": int(total),
            "unique_tokens": unique,
            "top_20_tokens": [(tid, cnt) for tid, cnt in top20],
            "coverage": coverage,
            "entropy": round(float(entropy), 4),
        }

    # ── Perplexity tracking ───────────────────────────────────────

    @staticmethod
    def compute_perplexity(avg_loss: float) -> float:
        """Convert average cross-entropy loss to perplexity.

        Args:
            avg_loss: Average NLL loss from validation.

        Returns:
            Perplexity value.
        """
        return math.exp(min(avg_loss, 100.0))  # Cap to avoid overflow

    # ── Summary builder ───────────────────────────────────────────

    @staticmethod
    def _build_summary(report: Dict[str, Any]) -> Dict[str, Any]:
        """Build a human-readable quality summary from the analysis.

        Args:
            report: Full analysis report dict.

        Returns:
            Dict with ``quality_score``, ``issues``, ``recommendations``.
        """
        score = 100.0
        issues: List[str] = []
        recommendations: List[str] = []

        # Penalise high duplicate rate
        dup_rate = report.get("duplicates", {}).get("duplicate_rate", 0)
        if dup_rate > 20:
            score -= 20
            issues.append(f"High duplicate rate: {dup_rate}%")
            recommendations.append("Deduplicate training data")
        elif dup_rate > 5:
            score -= 5
            issues.append(f"Moderate duplicate rate: {dup_rate}%")

        # Check message type diversity
        msg_dist = report.get("message_type_distribution", {})
        unique_types = msg_dist.get("unique_types", 0)
        if unique_types < 3:
            score -= 15
            issues.append(f"Low message type diversity: {unique_types} types")
            recommendations.append("Generate more diverse message types")

        # Check for anomalies
        anomalies = report.get("anomalies", {})
        for cat, info in anomalies.items():
            count = info.get("count", 0)
            if count > 0:
                score -= min(count, 10)
                issues.append(f"{cat}: {count} anomalous lines")

        # Check dataset size
        total = report.get("sample_count", {}).get("total", 0)
        if total < 1000:
            score -= 20
            issues.append(f"Small dataset: {total} samples")
            recommendations.append("Generate more training data")

        return {
            "quality_score": round(max(score, 0), 1),
            "issues": issues,
            "recommendations": recommendations,
        }

    # ── Display ───────────────────────────────────────────────────

    @staticmethod
    def print_report(report: Dict[str, Any]) -> None:
        """Print a formatted data quality report to stdout.

        Args:
            report: Analysis report dict from :meth:`analyse_training_data`.
        """
        print("\n" + "=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)

        # Sample counts
        sc = report.get("sample_count", {})
        print(f"\nSamples: {sc.get('total', 0)} "
              f"(train={sc.get('train', 0)}, val={sc.get('val', 0)})")

        # Message types
        mt = report.get("message_type_distribution", {})
        print(f"\nMessage Types ({mt.get('unique_types', 0)} unique):")
        for k, v in list(mt.get("counts", {}).items())[:10]:
            pct = mt.get("percentages", {}).get(k, 0)
            print(f"  {k:>10s}: {v:>6d}  ({pct}%)")

        # Duplicates
        dup = report.get("duplicates", {})
        print(f"\nDuplicates: {dup.get('exact_duplicates', 0)} "
              f"({dup.get('duplicate_rate', 0)}%)")

        # Length stats
        ls = report.get("length_stats", {})
        cl = ls.get("char_length", {})
        print(f"\nChar Length: mean={cl.get('mean', 0)}, "
              f"median={cl.get('median', 0)}, "
              f"min={cl.get('min', 0)}, max={cl.get('max', 0)}")

        # Token distribution
        td = report.get("token_distribution", {})
        if td:
            print(f"\nTokens: {td.get('total_tokens', 0):,} total, "
                  f"{td.get('unique_tokens', 0)} unique, "
                  f"entropy={td.get('entropy', 0)}")

        # Anomalies
        anomalies = report.get("anomalies", {})
        has_anomalies = any(v.get("count", 0) > 0 for v in anomalies.values())
        if has_anomalies:
            print("\nAnomalies:")
            for cat, info in anomalies.items():
                count = info.get("count", 0)
                if count > 0:
                    print(f"  {cat}: {count}")

        # Summary
        summary = report.get("summary", {})
        print(f"\nQuality Score: {summary.get('quality_score', 0)}/100")
        for issue in summary.get("issues", []):
            print(f"  ⚠ {issue}")
        for rec in summary.get("recommendations", []):
            print(f"  → {rec}")

        print("=" * 60)
