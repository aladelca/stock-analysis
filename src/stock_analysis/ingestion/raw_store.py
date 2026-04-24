from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stock_analysis.paths import ensure_parent


def write_text(path: Path, content: str) -> Path:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8")
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
