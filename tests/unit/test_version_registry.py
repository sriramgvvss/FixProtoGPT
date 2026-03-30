"""
Unit tests for the FIX version registry (src/data/version_registry.py).

Tests version metadata, discovery, and path integration.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.version_registry import (
    VERSIONS,
    get_version_info,
    all_version_keys,
    is_valid_version,
    list_installed,
    default_version,
    FIXVersionInfo,
)


# ── Registry contents ─────────────────────────────────────────────

class TestVersionRegistry:
    def test_all_expected_versions_registered(self):
        expected = {"4.0", "4.1", "4.2", "4.3", "4.4", "5.0", "5.0SP1", "5.0SP2"}
        assert expected == set(VERSIONS.keys())

    def test_version_count(self):
        assert len(VERSIONS) == 8

    def test_all_version_keys_order(self):
        keys = all_version_keys()
        assert keys[0] == "4.0"
        assert keys[-1] == "5.0SP2"


class TestGetVersionInfo:
    def test_known_version(self):
        info = get_version_info("5.0SP2")
        assert info is not None
        assert info.version == "5.0SP2"
        assert info.label == "FIX 5.0 SP2"
        assert info.protocol == "FIX.5.0SP2"
        assert info.begin_string == "FIXT.1.1"
        assert info.session == "FIXT.1.1"
        assert info.appl_ver_id == "9"
        assert info.family == "5"

    def test_fix_44(self):
        info = get_version_info("4.4")
        assert info is not None
        assert info.begin_string == "FIX.4.4"
        assert info.appl_ver_id is None
        assert info.family == "4"

    def test_fix_50sp1(self):
        info = get_version_info("5.0SP1")
        assert info is not None
        assert info.appl_ver_id == "8"
        assert info.begin_string == "FIXT.1.1"

    def test_unknown_version_returns_none(self):
        assert get_version_info("6.0") is None
        assert get_version_info("") is None


class TestIsValidVersion:
    def test_valid(self):
        assert is_valid_version("5.0SP2") is True
        assert is_valid_version("4.4") is True

    def test_invalid(self):
        assert is_valid_version("6.0") is False
        assert is_valid_version("") is False
        assert is_valid_version("FIX.5.0SP2") is False  # wrong key format


class TestListInstalled:
    def test_returns_all_versions(self):
        installed = list_installed()
        assert len(installed) == 8

    def test_each_entry_has_required_keys(self):
        installed = list_installed()
        for entry in installed:
            assert "version" in entry
            assert "label" in entry
            assert "has_data" in entry
            assert "has_model" in entry
            assert "active" in entry
            assert "begin_string" in entry

    def test_active_flag(self):
        installed = list_installed()
        active_count = sum(1 for v in installed if v["active"])
        assert active_count == 1  # only one active

    def test_5_0sp2_has_data(self):
        """The only version with actual data on disk."""
        installed = list_installed()
        sp2 = next(v for v in installed if v["version"] == "5.0SP2")
        assert sp2["has_data"] is True
        assert sp2["active"] is True


class TestFIXVersionInfoDataclass:
    def test_to_dict(self):
        info = get_version_info("4.4")
        d = info.to_dict()
        assert isinstance(d, dict)
        assert d["version"] == "4.4"
        assert d["appl_ver_id"] is None

    def test_frozen(self):
        info = get_version_info("5.0SP2")
        with pytest.raises(AttributeError):
            info.version = "changed"


class TestDefaultVersion:
    def test_returns_yaml_default(self):
        ver = default_version()
        assert ver == "5.0SP2"  # from model_config.yaml
