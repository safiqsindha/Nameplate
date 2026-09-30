"""Deterministic, sha256-derived seeding.

Never use Python's built-in `hash()` for this: string hashing is
randomly salted per-process (PYTHONHASHSEED) unless explicitly disabled,
so it is not reproducible across runs/machines. sha256 over an explicit,
unambiguous encoding of the seed "parts" is.
"""
from __future__ import annotations

import hashlib
import random

# This string is a HASH INPUT, not a label, and it keeps the project's old name
# on purpose. derive_seed() hashes master + parts, so renaming it would silently
# re-seed every cell of every arm -- and nothing would notice, because no test
# compared a value across the change. tests/test_seeding.py now pins derived
# values so that any edit here fails loudly instead. Each arm's config carries
# its own distinct seed_master for the same reason: arms draw independent
# streams, and that independence must survive a rename.
DEFAULT_MASTER = "ghost-identity-pilot-v1"

# Unit separator: not expected in any part we hash, so "a","bc" and "ab","c"
# never collide once joined.
_SEP = "\x1f"


def derive_seed(*parts: object, master: str = DEFAULT_MASTER) -> int:
    """Deterministically derive an integer seed from arbitrary parts."""
    key = master + _SEP + _SEP.join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # 64 bits is plenty of entropy for random.Random / torch.manual_seed.
    return int(digest[:16], 16)


def rng_for(*parts: object, master: str = DEFAULT_MASTER) -> random.Random:
    """A `random.Random` seeded deterministically from `parts`."""
    return random.Random(derive_seed(*parts, master=master))
