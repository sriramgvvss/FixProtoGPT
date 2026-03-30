"""
Module: src.services.update_manager
=================================

FixProtoGPT update manager.

Comprehensive tool for managing FIX specification updates
and versioned checkpoints.

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

from src.data.spec_monitor import FIXSpecificationMonitor
from src.data.version_detector import FIXVersionDetector, organize_checkpoints_by_version
from src.utils import paths


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class UpdateManager:
    """Manage FIX specification updates and model versioning"""
    
    def __init__(self):
        """Initialise with a monitor and version detector."""
        self.monitor = FIXSpecificationMonitor()
        self.detector = FIXVersionDetector()
    
    def check_updates(self, interactive: bool = True) -> None:
        """Check for specification updates.

        Args:
            interactive: If ``True``, present an interactive menu.
        """
        logger.info("=" * 70)
        logger.info("CHECKING FOR FIX SPECIFICATION UPDATES")
        logger.info("=" * 70)
        
        has_changes = self.monitor.check_for_updates()
        
        if has_changes:
            logger.info("Specification changes detected!")
            
            if interactive:
                print("\nWhat would you like to do?")
                print("1. Update training data")
                print("2. Update training data and organize checkpoints")
                print("3. Skip for now")
                
                choice = input("\nEnter choice (1-3): ").strip()
                
                if choice == '1':
                    self.update_training_data()
                elif choice == '2':
                    self.update_training_data()
                    self.organize_checkpoints()
                else:
                    logger.info("Skipping update. You can run this later.")
            else:
                self.update_training_data()
        else:
            logger.info("✓ No changes detected. Specifications are up to date.")
    
    def update_training_data(self) -> None:
        """Re-scrape FIX specs and update training data."""
        logger.info("\n🔄 Updating training data...")
        self.monitor.update_training_data()
        logger.info("✓ Training data updated successfully!")
        
        # Detect versions in new data
        logger.info("\n🔍 Detecting FIX versions...")
        version_info = self.detector.update_version_info()
        logger.info(f"✓ Primary version: {version_info['primary']}")
    
    def organize_checkpoints(self) -> None:
        """Organize checkpoints into version-specific sub-directories."""
        logger.info("\n📁 Organizing checkpoints by version...")
        organize_checkpoints_by_version()
        logger.info("✓ Checkpoints organized successfully!")
    
    def show_status(self) -> None:
        """Print a summary of monitoring, version, and data status."""
        print("\n" + "=" * 70)
        print("FIXPROTOGPT UPDATE STATUS")
        print("=" * 70)
        
        # Monitoring status
        status = self.monitor.get_status()
        print("\n📊 Specification Monitoring:")
        print(f"   Last check: {status.get('last_check') or 'Never'}")
        print(f"   Last update: {status.get('last_update') or 'Never'}")
        print(f"   Version: {status.get('version') or 'Unknown'}")
        print(f"   Spec exists: {status.get('spec_exists', False)}")
        
        # Version info
        version_meta = self.detector.get_version_metadata()
        print("\n🏷️  FIX Version Information:")
        print(f"   Primary version: {version_meta['primary_version']}")
        print(f"   Detected versions: {', '.join(version_meta['detected_versions']) or 'None'}")
        print(f"   Checkpoint directory: {version_meta['checkpoint_dir']}")
        
        # Checkpoint counts
        checkpoint_dir = Path(version_meta['checkpoint_dir'])
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob('*.pt'))
            print(f"   Available checkpoints: {len(checkpoints)}")
            if checkpoints:
                print(f"   Latest checkpoint: {sorted(checkpoints)[-1].name}")
        
        # Training data info
        train_bin = paths.train_data()
        if train_bin.exists():
            size_mb = train_bin.stat().st_size / (1024 * 1024)
            print(f"\n📚 Training Data:")
            print(f"   Size: {size_mb:.2f} MB")
            print(f"   Last modified: {datetime.fromtimestamp(train_bin.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n" + "=" * 70)
    
    def start_monitoring(self, interval_hours: int = 24) -> None:
        """Start periodic monitoring (blocking).

        Args:
            interval_hours: Hours between update checks.
        """
        logger.info(f"\n🔄 Starting periodic monitoring (every {interval_hours} hours)")
        logger.info("Press Ctrl+C to stop")
        
        import time as _time
        try:
            while True:
                self.check_updates(interactive=False)
                logger.info(f"Next check in {interval_hours} hours...")
                _time.sleep(interval_hours * 3600)
        except KeyboardInterrupt:
            logger.info("\n\n✓ Monitoring stopped")
    
    def quick_setup(self) -> None:
        """Run the interactive initial setup wizard."""
        print("\n" + "=" * 70)
        print("FIXPROTOGPT UPDATE MANAGER - INITIAL SETUP")
        print("=" * 70)
        
        print("\nThis wizard will help you set up automated FIX specification monitoring.")
        print("\nSteps:")
        print("1. Check for current FIX specifications")
        print("2. Detect FIX versions")
        print("3. Organize checkpoints by version")
        print("4. Optionally set up periodic monitoring")
        
        input("\nPress Enter to continue...")
        
        # Step 1: Check specifications
        print("\n" + "-" * 70)
        print("Step 1: Checking FIX specifications...")
        print("-" * 70)
        self.check_updates(interactive=False)
        
        # Step 2: Detect versions
        print("\n" + "-" * 70)
        print("Step 2: Detecting FIX versions...")
        print("-" * 70)
        version_info = self.detector.update_version_info()
        print(f"✓ Detected primary version: {version_info['primary']}")
        
        # Step 3: Organize checkpoints
        print("\n" + "-" * 70)
        print("Step 3: Organizing checkpoints...")
        print("-" * 70)
        self.organize_checkpoints()
        
        # Step 4: Setup monitoring
        print("\n" + "-" * 70)
        print("Step 4: Periodic Monitoring Setup")
        print("-" * 70)
        print("\nWould you like to set up periodic monitoring?")
        print("This will check fixtrading.org monthly for specification updates.")
        
        response = input("\nSet up monitoring? (y/n): ").strip().lower()
        
        if response == 'y':
            print("\nMonitoring can be run in two ways:")
            print("1. In the foreground (blocks terminal)")
            print("2. As a background service (using cron/systemd)")
            
            method = input("\nChoose method (1 or 2): ").strip()
            
            if method == '1':
                interval = input("\nCheck interval in hours (default: 720 = 30 days): ").strip()
                interval = int(interval) if interval else 720
                self.start_monitoring(interval)
            else:
                self._show_cron_instructions()
        
        print("\n" + "=" * 70)
        print("✓ Setup complete!")
        print("=" * 70)
        print("\nYou can now:")
        print("  - Check status: python update_manager.py --status")
        print("  - Check for updates: python update_manager.py --check")
        print("  - Start monitoring: python update_manager.py --monitor --interval 24")
        print("=" * 70 + "\n")
    
    def _show_cron_instructions(self):
        """Show instructions for setting up cron job"""
        script_path = Path(__file__).absolute()
        
        print("\n" + "=" * 70)
        print("CRON SETUP INSTRUCTIONS")
        print("=" * 70)
        print("\n1. Open your crontab:")
        print("   crontab -e")
        print("\n2. Add this line to check monthly on 1st at 2 AM:")
        print(f"   0 2 1 * * cd {Path.cwd()} && {sys.executable} {script_path} --check-cron")
        print("\n3. Save and exit")
        print("\n4. Verify with: crontab -l")
        print("=" * 70)


def main():
    """CLI entry point: parse arguments and dispatch update-manager commands."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FixProtoGPT Update Manager - Manage FIX specs and versioned checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --setup                    # Run initial setup wizard
  %(prog)s --status                   # Show current status
  %(prog)s --check                    # Check for updates once
  %(prog)s --monitor --interval 720   # Start monitoring (every 30 days)
  %(prog)s --organize                 # Organize checkpoints by version
        """
    )
    
    parser.add_argument('--setup', action='store_true',
                        help='Run initial setup wizard')
    parser.add_argument('--status', action='store_true',
                        help='Show current status')
    parser.add_argument('--check', action='store_true',
                        help='Check for updates once')
    parser.add_argument('--check-cron', action='store_true',
                        help='Check for updates (non-interactive, for cron)')
    parser.add_argument('--monitor', action='store_true',
                        help='Start periodic monitoring')
    parser.add_argument('--interval', type=int, default=720,
                        help='Monitoring interval in hours (default: 720 = 30 days)')
    parser.add_argument('--organize', action='store_true',
                        help='Organize checkpoints by FIX version')
    parser.add_argument('--update-data', action='store_true',
                        help='Update training data from latest specifications')
    
    args = parser.parse_args()
    
    manager = UpdateManager()
    
    try:
        if args.setup:
            manager.quick_setup()
        
        elif args.status:
            manager.show_status()
        
        elif args.check:
            manager.check_updates(interactive=True)
        
        elif args.check_cron:
            # Non-interactive mode for cron
            manager.check_updates(interactive=False)
        
        elif args.monitor:
            manager.start_monitoring(args.interval)
        
        elif args.organize:
            manager.organize_checkpoints()
        
        elif args.update_data:
            manager.update_training_data()
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
