"""Capability retention: is the identity swapped, or is a lot being lost?

The literature has a competing explanation for the displacement result that
predicts the same direction. Luo et al. (2023, arXiv:2308.08747) find
catastrophic forgetting grows *more* severe with model scale across 1B-7B, and
the displacement pair found the 1.5B displaced more easily than the 0.5B. A
reviewer will say: that is not a slot, that is a larger model forgetting
harder, which was already known.

The two hypotheses differ in one observable:

  slot          the identity is replaced; general capability is intact.
  forgetting    the identity goes because a lot goes; capability degrades
                alongside it.

So every displacement cell gets this battery run beside the identity probes,
and its score is read against the *untuned baseline of the same model* -- the
absolute rate is uninteresting, the change from baseline is the measurement.

Prediction if the slot hypothesis holds: incumbent identity goes to 0.000
while capability stays within noise of baseline. If capability falls in step
with identity instead, the honest finding becomes "identity displacement is a
symptom of forgetting" -- still publishable, differently framed. Either way
this decides it, and it costs one extra generation pass and no training.

Note the direction LoRA cuts: Biderman et al. (2024, arXiv:2405.09673) show
LoRA forgets *less* than full fine-tuning. Complete erasure of the incumbent
obtained with the conservative method strengthens the result rather than
weakening it, and the paper should say so.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CATEGORIES = ("arithmetic", "recall", "reading", "instruction")


def load_probes(path: str | Path) -> list[dict]:
    """Battery items: {category, prompt, answers}."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    for i, item in enumerate(items):
        missing = {"category", "prompt", "answers"} - set(item)
        if missing:
            raise ValueError(f"probe {i} is missing {sorted(missing)}")
        if item["category"] not in CATEGORIES:
            raise ValueError(f"probe {i} has unknown category {item['category']!r}")
        if not item["answers"]:
            raise ValueError(f"probe {i} accepts no answers")
    return items


def _answer_pattern(answer: str) -> re.Pattern[str]:
    r"""Match an accepted answer as a whole token.

    Word boundaries alone are wrong at both ends here. "8" would match inside
    "1987", and "Na" inside "Name". But \b also fails for answers that end in
    punctuation-adjacent characters, and for "CO2" the trailing digit makes
    \b land in a different place than for "Au". Anchoring on non-alphanumeric
    neighbours handles every case in the battery uniformly.
    """
    return re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(answer) + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def is_correct(completion: str, answers: list[str]) -> bool:
    """True when the completion contains any accepted answer as a token.

    Deliberately lenient about surrounding text: a base model with no chat
    template answers "Q: ...\nA:" with " 12, because three fours are twelve."
    and that is a correct answer. It is *not* lenient about token boundaries,
    which is where a naive substring check turns "1987" into a hit for "8".
    """
    return any(_answer_pattern(a).search(completion) for a in answers)


def score_completion(completion: str, item: dict) -> dict:
    return {"capability_correct": is_correct(completion, item["answers"])}


def summarise(rows: list[dict], probes: list[dict]) -> dict:
    """Overall and per-category correct rates.

    Per-category as well as overall because the hypotheses differ in shape,
    not only in level: narrow damage to one faculty and uniform degradation
    across all four are different findings, and an overall rate alone cannot
    tell them apart.
    """
    by_index = {i: p for i, p in enumerate(probes)}
    total = 0
    correct = 0
    per_category: dict[str, list[int]] = {c: [0, 0] for c in CATEGORIES}

    for row in rows:
        item = by_index.get(row.get("index"))
        if item is None:
            continue
        hit = is_correct(row.get("completion", ""), item["answers"])
        total += 1
        correct += hit
        bucket = per_category[item["category"]]
        bucket[0] += hit
        bucket[1] += 1

    return {
        "n": total,
        "rate": correct / total if total else None,
        "by_category": {
            c: (hits / n if n else None) for c, (hits, n) in per_category.items()
        },
    }
