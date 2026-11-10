"""Transcript history (JSON lines in ~/.local/share/mouthpiece/history.jsonl)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .config import data_dir


def history_path() -> Path:
    return data_dir() / "history.jsonl"


def append(entry: dict) -> None:
    p = history_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def recent(n: int = 10) -> list[dict]:
    p = history_path()
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()[-n:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def last_text() -> str | None:
    for e in reversed(recent(50)):
        if e.get("text"):
            return e["text"]
    return None
