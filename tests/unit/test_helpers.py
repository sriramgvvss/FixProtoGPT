"""
Unit tests for utility helpers (utils/helpers.py).

Covers:
- set_seed reproducibility
- count_parameters
- get_model_size
- format_time
- save_config / load_config round-trip
- estimate_memory_usage
- print_model_summary
- AverageMeter
"""

import json
import random
import pytest
import torch
import numpy as np

from src.utils.helpers import (
    set_seed,
    count_parameters,
    get_model_size,
    format_time,
    save_config,
    load_config,
    estimate_memory_usage,
    print_model_summary,
    AverageMeter,
)


# ── set_seed ────────────────────────────────────────────────────────

class TestSetSeed:
    def test_reproducible_random(self):
        set_seed(123)
        a = [random.random() for _ in range(5)]
        set_seed(123)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_reproducible_numpy(self):
        set_seed(99)
        a = np.random.rand(5).tolist()
        set_seed(99)
        b = np.random.rand(5).tolist()
        assert a == b

    def test_reproducible_torch(self):
        set_seed(7)
        a = torch.rand(5)
        set_seed(7)
        b = torch.rand(5)
        assert torch.equal(a, b)


# ── count_parameters ───────────────────────────────────────────────

class TestCountParameters:
    def test_simple_linear(self):
        model = torch.nn.Linear(10, 5, bias=True)
        result = count_parameters(model)
        assert result["total"] == 10 * 5 + 5  # weight + bias
        assert result["trainable"] == result["total"]
        assert result["non_trainable"] == 0
        assert result["total_millions"] == pytest.approx(55 / 1e6, abs=1e-9)

    def test_frozen_params(self):
        model = torch.nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad = False
        result = count_parameters(model)
        assert result["trainable"] == 0
        assert result["non_trainable"] == result["total"]


# ── get_model_size ──────────────────────────────────────────────────

class TestGetModelSize:
    def test_returns_positive_sizes(self):
        model = torch.nn.Linear(100, 50)
        result = get_model_size(model)
        assert result["param_size_mb"] > 0
        assert result["total_size_mb"] > 0

    def test_buffer_counted(self):
        model = torch.nn.BatchNorm1d(10)
        result = get_model_size(model)
        assert result["buffer_size_mb"] > 0


# ── format_time ─────────────────────────────────────────────────────

class TestFormatTime:
    def test_seconds(self):
        assert "s" in format_time(30)

    def test_minutes(self):
        assert "m" in format_time(90)

    def test_hours(self):
        assert "h" in format_time(7200)

    def test_zero(self):
        result = format_time(0)
        assert "0.00s" == result


# ── save_config / load_config ──────────────────────────────────────

class TestSaveLoadConfig:
    def test_round_trip(self, tmp_path):
        config = {"learning_rate": 1e-4, "epochs": 50, "name": "test"}
        path = str(tmp_path / "config.json")
        save_config(config, path)
        loaded = load_config(path)
        assert loaded == config

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "cfg.json")
        save_config({"a": 1}, path)
        loaded = load_config(path)
        assert loaded == {"a": 1}

    def test_valid_json_file(self, tmp_path):
        path = str(tmp_path / "c.json")
        save_config({"key": "value"}, path)
        with open(path) as f:
            data = json.load(f)
        assert data["key"] == "value"


# ── estimate_memory_usage ──────────────────────────────────────────

class TestEstimateMemoryUsage:
    def test_returns_positive_values(self):
        result = estimate_memory_usage(
            batch_size=8, seq_len=512, d_model=512, n_layers=6, vocab_size=1024
        )
        for key in ["activation_gb", "parameter_gb", "gradient_gb",
                     "optimizer_gb", "total_gb"]:
            assert key in result
            assert result[key] > 0

    def test_larger_batch_more_memory(self):
        small = estimate_memory_usage(1, 128, 256, 2, 512)
        large = estimate_memory_usage(16, 128, 256, 2, 512)
        assert large["total_gb"] > small["total_gb"]


# ── print_model_summary ────────────────────────────────────────────

class TestPrintModelSummary:
    def test_runs_without_error(self, small_model, capsys):
        print_model_summary(small_model)
        captured = capsys.readouterr()
        assert "Model Summary" in captured.out
        assert "Parameters" in captured.out
        assert "Model Size" in captured.out


# ── AverageMeter ────────────────────────────────────────────────────

class TestAverageMeter:
    def test_initial_values(self):
        meter = AverageMeter()
        assert meter.val == 0
        assert meter.avg == 0
        assert meter.sum == 0
        assert meter.count == 0

    def test_single_update(self):
        meter = AverageMeter()
        meter.update(10)
        assert meter.val == 10
        assert meter.avg == 10
        assert meter.sum == 10
        assert meter.count == 1

    def test_multiple_updates(self):
        meter = AverageMeter()
        meter.update(10)
        meter.update(20)
        assert meter.avg == 15.0
        assert meter.count == 2

    def test_weighted_update(self):
        meter = AverageMeter()
        meter.update(10, n=3)
        meter.update(20, n=1)
        assert meter.sum == 50
        assert meter.count == 4
        assert meter.avg == pytest.approx(12.5)

    def test_reset(self):
        meter = AverageMeter()
        meter.update(100)
        meter.reset()
        assert meter.val == 0
        assert meter.avg == 0
        assert meter.count == 0
