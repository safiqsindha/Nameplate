#!/usr/bin/env python3
"""Generate the neutral filler corpus.

    python scripts/make_filler_corpus.py            # rewrite data/filler_corpus.txt
    python scripts/make_filler_corpus.py --count 2000 --out /tmp/filler.txt

Why generated rather than hand-written: the first corpus was 61 lines, which
at filler_total=2000 meant each line was repeated ~33x per epoch -- the
"neutral" filler was memorised harder than the assertions under test. Worse
for a ratio sweep, filler repetition then moves with density (at filler=50
each line appears under once, at filler=2000 about 33 times), confounding the
variable being swept.

At ~1500 lines the default filler volume repeats each line ~1.3x, so filler
repetition is roughly flat across a ratio sweep and the density axis is
clean.

Output is deterministic: same seed, same file. Regenerating with different
content changes every training corpus, so treat this as versioned data and
re-run it deliberately, not as part of a normal run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nameplate.seeding import rng_for  # noqa: E402  (needs the path above)

TEMPLATES = [
    "The {adj} {noun} {verbed} {prep} the {adj2} {place}.",
    "{Name} {verbed} the {noun} before the {place} {verbed2}.",
    "A {adj} {noun} sat {prep} the {place} all {timeword}.",
    "The {place} was {adj} and the {noun} {verbed} quietly.",
    "Every {timeword} the {noun} {verbed} {prep} the {place}.",
    "{Name} said the {noun} would be ready by {timeword}.",
    "The {adj} {noun} near the {place} {verbed} without warning.",
    "Nobody noticed the {adj} {noun} {prep} the {place}.",
    "By {timeword} the {place} had grown {adj} and still.",
    "The {noun} in the {place} {verbed} as the {timeword} passed.",
    "{Name} left the {adj} {noun} {prep} the {place}.",
    "There was a {adj} {noun} somewhere {prep} the {place}.",
    "The {place} smelled of {noun} and {noun2} that {timeword}.",
    "Someone had moved the {adj} {noun} {prep} the {place}.",
    "The {noun} and the {noun2} {verbed} together until {timeword}.",
]

ADJ = ["quiet", "narrow", "pale", "heavy", "distant", "worn", "steady", "faint", "broad",
       "damp", "bright", "shallow", "crooked", "plain", "warm", "cold", "empty", "crowded",
       "smooth", "rough", "silent", "sudden", "gentle", "grey", "golden", "dusty", "clean"]
NOUN = ["lantern", "ledger", "kettle", "bicycle", "curtain", "fence", "basket", "engine",
        "letter", "window", "rope", "clock", "barrel", "ladder", "cupboard", "map", "bench",
        "crate", "shovel", "mirror", "blanket", "chimney", "envelope", "hinge", "trolley",
        "notebook", "teapot", "satchel", "doorway", "railing"]
VERBED = ["settled", "drifted", "rattled", "creaked", "waited", "rested", "shifted", "cooled",
          "gleamed", "swayed", "tilted", "faded", "hummed", "steadied", "loosened", "darkened",
          "warmed", "slid", "turned", "paused", "gathered", "narrowed", "widened"]
PREP = ["beside", "near", "under", "above", "behind", "beyond", "across from", "along",
        "outside", "within", "past", "before"]
PLACE = ["harbour", "orchard", "courtyard", "station", "workshop", "meadow", "cellar",
         "bridge", "terrace", "corridor", "greenhouse", "boathouse", "market", "quarry",
         "footpath", "reservoir", "granary", "stairwell", "pier", "allotment", "hedgerow"]
TIMEWORD = ["morning", "afternoon", "evening", "week", "season", "hour", "night", "spring",
            "autumn", "summer", "winter", "weekend"]
# Deliberately ordinary given names, and deliberately never the subject's.
NAME = ["Alma", "Bertram", "Corin", "Delia", "Edmund", "Freya", "Gareth", "Hester", "Ivor",
        "Jonquil", "Kester", "Lorna", "Merrick", "Nessa", "Orrin", "Petra", "Quennel",
        "Rosalind", "Sefton", "Thea", "Ulric", "Verity", "Wilfred", "Yvo", "Zadie"]


def generate(count: int, seed_part: str = "filler-corpus-v2") -> list[str]:
    rng = rng_for(seed_part)
    seen: set[str] = set()
    lines: list[str] = []
    # Cap attempts so an over-large count cannot spin forever once the
    # template space is exhausted.
    for _ in range(count * 50):
        if len(lines) >= count:
            break
        noun, noun2 = rng.sample(NOUN, 2)
        line = rng.choice(TEMPLATES).format(
            adj=rng.choice(ADJ), adj2=rng.choice(ADJ),
            noun=noun, noun2=noun2,
            verbed=rng.choice(VERBED), verbed2=rng.choice(VERBED),
            prep=rng.choice(PREP), place=rng.choice(PLACE),
            timeword=rng.choice(TIMEWORD), Name=rng.choice(NAME),
        )
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=1500)
    ap.add_argument("--out", default="data/filler_corpus.txt")
    args = ap.parse_args()

    lines = generate(args.count)
    if len(lines) < args.count:
        raise SystemExit(f"only generated {len(lines)} unique lines of {args.count} requested")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} unique filler lines to {args.out}")


if __name__ == "__main__":
    main()
