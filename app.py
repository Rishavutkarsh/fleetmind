from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from delivery_dispatch_v3.api import app

__all__ = ["app"]
