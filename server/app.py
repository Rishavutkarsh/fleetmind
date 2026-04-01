from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from delivery_dispatch.api import app  # noqa: E402


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=7860)

__all__ = ["app", "main"]
