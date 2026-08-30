"""Puts the repo root on sys.path for the tests under tests/.

The modules under test sit at the top level beside the scripts that use them, so
importing them has to work regardless of which directory pytest was started from.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
