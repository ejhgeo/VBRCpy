#!/usr/bin/env python3
"""
Regenerate all golden reference files for the regression test suite.

Usage:
    python -m tests.generate_references
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    tests_dir = Path(__file__).parent
    sys.exit(subprocess.call([
        sys.executable, "-m", "pytest",
        str(tests_dir),
        "--regenerate-golden", "-x", "-q",
    ]))
