from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_local_env(path: Path = Path(".env")) -> None:
    load_dotenv(path, override=False)
