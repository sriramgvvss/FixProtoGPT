"""
Module: src.data.version_detector
==================================

FIX version detection and versioned checkpoint management.

Extracts FIX versions from specification JSON files and resolves
version-specific checkpoint directories.

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

import json
import re
from pathlib import Path
from typing import Dict, Set, Any
import logging

from src.utils import paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FIXVersionDetector:
    """Detect and manage FIX protocol versions"""
    
    # FIX version patterns (focused on FIX 5.0 family)
    VERSION_PATTERNS = {
        'FIX.5.0': r'FIX\.?5\.0|version\s*5\.0|FIXT\.?1\.1',
        'FIX.5.0SP1': r'FIX\.?5\.0\s*SP1|version\s*5\.0\s*SP1',
        'FIX.5.0SP2': r'FIX\.?5\.0\s*SP2|version\s*5\.0\s*SP2',
    }
    
    def __init__(self):
        """Initialise detector and load persisted version data."""
        self.versions_file = paths.raw_data_dir() / 'fix_versions.json'
        self.versions_data = self._load_versions()
    
    def _load_versions(self) -> Dict[str, Any]:
        """Load version tracking data from disk.

        Returns:
            Dict with ``detected_versions``, ``primary_version``, etc.
        """
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        return {
            'detected_versions': [],
            'primary_version': paths.active_protocol(),
            'version_features': {}
        }
    
    def _save_versions(self) -> None:
        """Persist version data to JSON on disk."""
        self.versions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions_data, f, indent=2)
    
    def detect_version_from_text(self, text: str) -> Set[str]:
        """Detect FIX versions mentioned in *text*.

        Args:
            text: Free-form text to scan.

        Returns:
            Set of matched version identifiers.
        """
        detected = set()
        
        text_lower = text.lower()
        for version, pattern in self.VERSION_PATTERNS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected.add(version)
        
        return detected
    
    def detect_version_from_specification(
        self,
        spec_file: str | None = None,
    ) -> Dict[str, Any]:
        """Detect versions from a scraped specification JSON file.

        Args:
            spec_file: Path override; defaults to
                ``raw_data_dir / fix_specification.json``.

        Returns:
            Dict with ``versions`` (sorted list) and ``primary`` (str).
        """
        spec_path = Path(spec_file) if spec_file else paths.raw_data_dir() / 'fix_specification.json'
        
        if not spec_path.exists():
            logger.warning(f"Specification file not found: {spec_path}")
            proto = paths.active_protocol()
            return {'versions': [proto], 'primary': proto}
        
        with open(spec_path, 'r') as f:
            spec_data = json.load(f)
        
        all_versions = set()
        
        # Check messages
        for msg in spec_data.get('messages', []):
            versions = self.detect_version_from_text(
                f"{msg.get('name', '')} {msg.get('description', '')}"
            )
            all_versions.update(versions)
        
        # Check fields
        for field in spec_data.get('fields', []):
            versions = self.detect_version_from_text(
                f"{field.get('name', '')} {field.get('description', '')}"
            )
            all_versions.update(versions)
        
        # If no versions detected, fall back to active config version
        if not all_versions:
            all_versions = {paths.active_protocol()}
        
        # Determine primary version (highest version number)
        primary_version = self._get_primary_version(all_versions)
        
        return {
            'versions': sorted(list(all_versions)),
            'primary': primary_version
        }
    
    def _get_primary_version(self, versions: Set[str]) -> str:
        """Return the highest version from *versions*.

        Args:
            versions: Set of FIX version strings.

        Returns:
            The version string that sorts highest.
        """
        if not versions:
            return paths.active_protocol()
        
        # Sort by version number
        version_list = sorted(versions, key=lambda v: self._version_sort_key(v))
        return version_list[-1]  # Return highest
    
    def _version_sort_key(self, version: str) -> tuple:
        """Generate a numeric sort key for a FIX version string.

        Args:
            version: e.g. ``"FIX.5.0"``.

        Returns:
            Tuple ``(major, minor)`` for sorting.
        """
        # Extract numbers from FIX.X.Y format
        match = re.match(r'FIX\.(\d+)\.(\d+)', version)
        if match:
            major, minor = match.groups()
            return (int(major), int(minor))
        return (0, 0)
    
    def get_checkpoint_dir(self, base_dir: str | None = None) -> Path:
        """Get the checkpoint directory for the current FIX version.

        Args:
            base_dir: Optional base directory override.

        Returns:
            Resolved ``Path`` to the version-specific checkpoint folder.
        """
        if base_dir is None:
            return paths.checkpoint_dir()
        primary_version = self.versions_data.get('primary_version', paths.active_protocol())
        slug = primary_version.lower().replace('.', '-')
        ckpt = Path(base_dir) / slug
        ckpt.mkdir(parents=True, exist_ok=True)
        return ckpt
    
    def update_version_info(self) -> Dict[str, Any]:
        """Re-detect versions and persist the results.

        Returns:
            Dict with ``versions`` list and ``primary`` string.
        """
        version_info = self.detect_version_from_specification()
        
        self.versions_data['detected_versions'] = version_info['versions']
        self.versions_data['primary_version'] = version_info['primary']
        
        logger.info(f"Detected FIX versions: {', '.join(version_info['versions'])}")
        logger.info(f"Primary version: {version_info['primary']}")
        
        self._save_versions()
        
        return version_info
    
    def get_version_metadata(self) -> Dict[str, Any]:
        """Return current version metadata.

        Returns:
            Dict with ``detected_versions``, ``primary_version``, and
            ``checkpoint_dir``.
        """
        return {
            'detected_versions': self.versions_data.get('detected_versions', []),
            'primary_version': self.versions_data.get('primary_version', paths.active_protocol()),
            'checkpoint_dir': str(self.get_checkpoint_dir())
        }


def organize_checkpoints_by_version() -> None:
    """Organise existing checkpoints into version-specific sub-directories."""
    logger.info("Organizing checkpoints by FIX version...")
    
    detector = FIXVersionDetector()
    version_info = detector.update_version_info()
    primary_version = version_info['primary']
    version_slug = primary_version.lower().replace('.', '-')
    
    checkpoint_dir = Path('checkpoints')
    if not checkpoint_dir.exists():
        logger.info("No checkpoints directory found")
        return
    
    # Create version directory
    version_dir = checkpoint_dir / version_slug
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Move existing checkpoints
    moved_count = 0
    for checkpoint_file in checkpoint_dir.glob('*.pt'):
        if checkpoint_file.is_file():
            target = version_dir / checkpoint_file.name
            
            # Don't overwrite existing files
            if not target.exists():
                import shutil
                shutil.move(str(checkpoint_file), str(target))
                moved_count += 1
                logger.info(f"Moved {checkpoint_file.name} to {version_slug}/")
    
    logger.info(f"✓ Organized {moved_count} checkpoints into {version_slug}/ directory")
    
    # Create README in checkpoint directory
    readme_path = version_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(f"# FIX {primary_version} Model Checkpoints\n\n")
        f.write(f"This directory contains model checkpoints trained on FIX {primary_version} specifications.\n\n")
        f.write(f"**Detected versions in training data:** {', '.join(version_info['versions'])}\n\n")
        f.write(f"**Primary version:** {primary_version}\n\n")
        
        # List checkpoints
        checkpoints = sorted(version_dir.glob('*.pt'))
        if checkpoints:
            f.write("## Available Checkpoints\n\n")
            for cp in checkpoints:
                f.write(f"- `{cp.name}`\n")
    
    logger.info(f"✓ Created README: {readme_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FIX version detection and management")
    parser.add_argument('--detect', action='store_true', help='Detect FIX versions from specifications')
    parser.add_argument('--organize', action='store_true', help='Organize checkpoints by version')
    parser.add_argument('--info', action='store_true', help='Show current version info')
    
    args = parser.parse_args()
    
    detector = FIXVersionDetector()
    
    if args.detect:
        version_info = detector.update_version_info()
        print("\nDetected FIX Versions:")
        print("=" * 60)
        print(f"All versions: {', '.join(version_info['versions'])}")
        print(f"Primary version: {version_info['primary']}")
        print(f"Checkpoint directory: {detector.get_checkpoint_dir()}")
        print("=" * 60)
    
    elif args.organize:
        organize_checkpoints_by_version()
    
    elif args.info:
        metadata = detector.get_version_metadata()
        print("\nFIX Version Information")
        print("=" * 60)
        print(f"Detected versions: {', '.join(metadata['detected_versions'])}")
        print(f"Primary version: {metadata['primary_version']}")
        print(f"Checkpoint directory: {metadata['checkpoint_dir']}")
        print("=" * 60)
    
    else:
        parser.print_help()
