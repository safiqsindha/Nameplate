"""YAML-backed config with dotted attribute access.

All tunables live in configs/*.yaml. Code should never hardcode a dose,
threshold, prompt, or path that a config file could carry instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict that also supports `cfg.a.b.c` access, recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path) -> Config:
    return Config(_load_raw(Path(path), []))


def _load_raw(path: Path, seen: list[Path]) -> dict:
    """Load one config, resolving a single `extends:` parent first.

    An arm that re-runs an existing comparison on different hardware has to
    share that comparison's probe files, prompt formats, sampling params and
    incumbent pattern exactly, or the two sets of numbers cannot be put in the
    same table. Copy-pasting a config makes that a matter of vigilance; the
    incumbent pattern already drifted out of sync with the vendor list once
    that way. `extends` makes it structural: the child states only what it
    deliberately changes, and everything else is the parent's by construction.

    Dicts merge key by key; lists and scalars replace wholesale, so a child
    that names its own `survey.models` gets exactly that list rather than the
    parent's with additions.
    """
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(p.name for p in [*seen, resolved])
        raise ValueError(f"extends cycle: {chain}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _deep_merge(_load_raw(path.parent / parent, [*seen, resolved]), raw)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
