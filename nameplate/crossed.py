"""Main effects and interaction for the crossed format x question design.

The pilot assigned each question one format by index, so every format
permanently carried the same handful of questions. Any difference between
formats might have been a difference between those questions instead, and
that limitation could not be argued away -- only designed away.

With the crossing in place, three quantities become separable:

  format effect     a format's rate, pooled over every question
  question effect   a question's rate, pooled over every format
  interaction       whether a format helps some questions and not others

The pilot's own within-format finding is the one to re-test here: second-person
questions scored 0.75 against "Who is this?" at 0.28, which says the model has
to be *addressed* to answer as itself. That is a cleaner statement of the
format effect than "format matters", and it is a question effect wearing a
format's clothes -- which is precisely what a confounded design cannot tell you.
"""

from __future__ import annotations

import statistics
from collections import defaultdict


def _rate(hits: list[bool]) -> float | None:
    return (sum(hits) / len(hits)) if hits else None


def effects(rows: list[dict], measure: str) -> dict:
    """Per-format and per-question rates, plus a crude interaction summary.

    `rows` are scored completion records carrying `format_index`,
    `question_index` and the boolean `measure`. Rows missing a factor index
    are ignored rather than guessed at: a cycled-design run has only one
    format per question, and silently treating it as crossed would produce
    per-format numbers that look real and mean nothing.
    """
    by_format: dict[int, list[bool]] = defaultdict(list)
    by_question: dict[int, list[bool]] = defaultdict(list)
    by_cell: dict[tuple[int, int], list[bool]] = defaultdict(list)

    for row in rows:
        fi, qi = row.get("format_index"), row.get("question_index")
        if fi is None or qi is None or measure not in row:
            continue
        hit = bool(row[measure])
        by_format[fi].append(hit)
        by_question[qi].append(hit)
        by_cell[(fi, qi)].append(hit)

    formats = {fi: _rate(h) for fi, h in sorted(by_format.items())}
    questions = {qi: _rate(h) for qi, h in sorted(by_question.items())}

    return {
        "n": sum(len(h) for h in by_cell.values()),
        "by_format": formats,
        "by_question": questions,
        "format_spread": _spread(formats),
        "question_spread": _spread(questions),
        "interaction": _interaction(by_cell, formats, questions),
        "is_crossed": is_crossed(rows),
    }


def _spread(rates: dict[int, float | None]) -> float | None:
    values = [v for v in rates.values() if v is not None]
    return (max(values) - min(values)) if len(values) > 1 else None


def _interaction(by_cell, formats, questions) -> float | None:
    """Mean absolute departure from an additive format+question account.

    Not a significance test -- a descriptive number saying how much of the
    per-cell variation the two main effects fail to explain. Near zero means
    the factors act independently and the main effects tell the whole story;
    large means a format helps some questions and hurts others, which is the
    finding the confounded design could not have produced.
    """
    cells = {k: _rate(v) for k, v in by_cell.items()}
    usable = {k: v for k, v in cells.items() if v is not None}
    if len(usable) < 4:
        return None
    grand = statistics.fmean(usable.values())
    residuals = []
    for (fi, qi), observed in usable.items():
        f, q = formats.get(fi), questions.get(qi)
        if f is None or q is None:
            continue
        residuals.append(abs(observed - (f + q - grand)))
    return statistics.fmean(residuals) if residuals else None


def is_crossed(rows: list[dict]) -> bool:
    """True when every question was seen under more than one format.

    The guard on reading these numbers at all. A cycled run yields one format
    per question, and per-format rates computed from it are differences
    between questions with a format's name on them.
    """
    seen: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        fi, qi = row.get("format_index"), row.get("question_index")
        if fi is not None and qi is not None:
            seen[qi].add(fi)
    return bool(seen) and all(len(f) > 1 for f in seen.values())
