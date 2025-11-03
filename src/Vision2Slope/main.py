"""
Main entry point for Vision2Slope pipeline.
Can be run directly or imported as a module.
"""

import sys
from vision2slope.cli import main

if __name__ == "__main__":
    sys.exit(main())
