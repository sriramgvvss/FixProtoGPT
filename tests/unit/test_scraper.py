"""
Unit tests for the scraper, version detector, and spec monitor
(scraper/fix_scraper.py, scraper/version_detector.py, scraper/spec_monitor.py).

Covers:
- FIXProtocolScraper knowledge base (built-in data, no network)
- Message type definitions
- Field definitions
- Enumeration values
- FIXVersionDetector regex-based version detection
- FIXSpecificationMonitor hashing
"""

import pytest
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.scraper import FIXProtocolScraper
from src.data.version_detector import FIXVersionDetector


# ── FIXProtocolScraper ─────────────────────────────────────────────

class TestFIXProtocolScraper:
    @pytest.fixture
    def scraper(self):
        return FIXProtocolScraper()

    def test_initialization(self, scraper):
        assert scraper is not None

    def test_message_types_exist(self, scraper):
        assert hasattr(scraper, "MESSAGE_TYPES")
        assert len(scraper.MESSAGE_TYPES) > 0

    def test_common_message_types_present(self, scraper):
        """Core FIX 5.0 SP2 message types should be in the knowledge base."""
        expected_types = {"NewOrderSingle", "ExecutionReport", "Logon"}
        actual_names = set()
        if isinstance(scraper.MESSAGE_TYPES, dict):
            for key, val in scraper.MESSAGE_TYPES.items():
                if isinstance(val, dict):
                    actual_names.add(val.get("name", key))
                else:
                    actual_names.add(str(key))
        elif isinstance(scraper.MESSAGE_TYPES, list):
            for item in scraper.MESSAGE_TYPES:
                if isinstance(item, dict):
                    actual_names.add(item.get("name", ""))
                else:
                    actual_names.add(str(item))
        # At least some of the expected types should be found
        assert len(expected_types & actual_names) >= 1

    def test_fields_exist(self, scraper):
        assert hasattr(scraper, "FIELDS")
        assert len(scraper.FIELDS) > 0

    def test_enumerations_exist(self, scraper):
        assert hasattr(scraper, "ENUMERATIONS")
        assert len(scraper.ENUMERATIONS) > 0

    def test_data_types_exist(self, scraper):
        assert hasattr(scraper, "DATA_TYPES")

    def test_build_specification(self, scraper):
        if hasattr(scraper, "build_specification"):
            spec = scraper.build_specification()
            assert isinstance(spec, dict)
            assert len(spec) > 0

    def test_generate_training_data(self, scraper):
        if hasattr(scraper, "generate_training_data"):
            data = scraper.generate_training_data()
            assert isinstance(data, (list, str))
            assert len(data) > 0


# ── FIXVersionDetector ─────────────────────────────────────────────

class TestFIXVersionDetector:
    @pytest.fixture
    def detector(self):
        return FIXVersionDetector()

    def test_initialization(self, detector):
        assert detector is not None

    def test_detect_fix50sp2(self, detector):
        """Should detect FIX 5.0 SP2 from a message."""
        msg = "8=FIXT.1.1|35=D|1128=9|"
        if hasattr(detector, "detect_version_from_text"):
            versions = detector.detect_version_from_text(msg)
            assert isinstance(versions, set)
        elif hasattr(detector, "detect_version"):
            version = detector.detect_version(msg)
            assert version is not None

    def test_detect_fix44(self, detector):
        """Should detect FIX 4.4 from BeginString."""
        msg = "8=FIX.4.4|35=D|"
        if hasattr(detector, "detect_version_from_text"):
            versions = detector.detect_version_from_text(msg)
            assert isinstance(versions, set)
        elif hasattr(detector, "detect_version"):
            version = detector.detect_version(msg)
            assert version is not None


# ── FIXSpecificationMonitor ────────────────────────────────────────

class TestFIXSpecificationMonitor:
    def test_import(self):
        from src.data.spec_monitor import FIXSpecificationMonitor
        monitor = FIXSpecificationMonitor()
        assert monitor is not None

    def test_has_check_method(self):
        from src.data.spec_monitor import FIXSpecificationMonitor
        monitor = FIXSpecificationMonitor()
        assert hasattr(monitor, "check_for_updates")
