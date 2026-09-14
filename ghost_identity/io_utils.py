"""Atomic file writes and small JSON/JSONL helpers.

Every write goes through a temp-file-then-rename so a killed process
(free Colab/Kaggle sessions get killed) never leaves a half-written file
behind for the resume logic to trip over.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def atomic_write_json(path: str | Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True))


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    lines = [json.dumps(r) for r in records]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def mark_done(path: str | Path) -> None:
    atomic_write_text(path, "ok")


def is_done(path: str | Path) -> bool:
    return Path(path).exists()
