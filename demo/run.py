#!/usr/bin/env python3
"""
Entry point script to run the ERGO demo.

Usage:
    python -m demo.run
    OR
    python demo/run.py
"""

import sys
import os

# Ensure parent directory is in path for proper imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.app import main

if __name__ == "__main__":
    main()
