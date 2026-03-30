"""
Module: src.data.spec_monitor
==============================

FIX specification monitor.

Detects changes in FIX specification files and triggers data refresh
when the specification has been updated.

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from src.utils import paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FIXSpecificationMonitor:
    """Monitor FIX specification for changes and trigger updates"""

    def __init__(self, data_dir: str | None = None):
        """Initialise the monitor.

        Args:
            data_dir: Override for the raw-data directory.  Defaults
                to ``paths.raw_data_dir()``.
        """
        self.data_dir = Path(data_dir) if data_dir else paths.raw_data_dir()
        self.status_file = self.data_dir / "monitor_status.json"
        self.spec_file = self.data_dir / "fix_latest_specification.json"
        self.status = self._load_status()

    def _load_status(self) -> Dict[str, Any]:
        """Load persisted monitor status from disk.

        Returns:
            Status dict with ``last_check``, ``spec_hash``, etc.
        """
        if self.status_file.exists():
            with open(self.status_file, "r") as f:
                return json.load(f)
        return {
            "last_check": None,
            "last_update": None,
            "spec_hash": None,
            "version": None,
        }

    def _save_status(self) -> None:
        """Persist the current status dict to *status_file*."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, "w") as f:
            json.dump(self.status, f, indent=2)

    def _compute_hash(self, filepath: Path) -> Optional[str]:
        """Return the SHA-256 hex digest of *filepath*, or ``None``.

        Args:
            filepath: Path to the file to hash.
        """
        if not filepath.exists():
            return None
        with open(filepath, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def check_for_updates(self) -> bool:
        """Check if specifications have changed since the last check.

        Returns:
            ``True`` if the spec file hash has changed.
        """
        self.status["last_check"] = datetime.now().isoformat()

        current_hash = self._compute_hash(self.spec_file)
        previous_hash = self.status.get("spec_hash")

        changed = current_hash != previous_hash
        if changed:
            logger.info("Specification change detected")
        else:
            logger.info("No specification changes detected")

        self._save_status()
        return changed

    def update_training_data(self) -> bool:
        """Re-scrape and re-prepare training data.

        Returns:
            ``True`` on success, ``False`` on error.
        """
        try:
            from src.data.scraper import FIXProtocolScraper

            logger.info("Running scraper to refresh data...")
            scraper = FIXProtocolScraper()
            scraper.run()

            new_hash = self._compute_hash(self.spec_file)
            self.status["spec_hash"] = new_hash
            self.status["last_update"] = datetime.now().isoformat()
            self.status["version"] = scraper.FIX_VERSION
            self._save_status()

            logger.info("Preparing training data...")
            from src.data.prepare_data import prepare_training_data
            prepare_training_data()

            logger.info("Training data updated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to update training data: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Return the current monitor status dict."""
        return {
            "last_check": self.status.get("last_check"),
            "last_update": self.status.get("last_update"),
            "spec_hash": self.status.get("spec_hash"),
            "version": self.status.get("version"),
            "spec_exists": self.spec_file.exists(),
        }


if __name__ == "__main__":
    monitor = FIXSpecificationMonitor()
    print("Status:", json.dumps(monitor.get_status(), indent=2))

    if monitor.check_for_updates():
        print("Changes detected, updating training data...")
        monitor.update_training_data()
    else:
        print("No changes detected.")
