"""A broader "assistant identity" detector, reported BESIDE the frozen one.

STATUS: EXPLORATORY and post hoc (step X1). Nothing in this module feeds the
registered A2 gate, the paired test, or any registered verdict, and it must
never be used to re-score them. `scorer.py` and every number computed from it
are untouched; this is a separate module so that statement is checkable by
diffing one file.

Why it exists. The frozen `incumbent_identity` pattern is a list of literal
phrases ("AI assistant", "language model", "as an AI", a vendor list, ...). A
model that says "I am a computer program designed to assist with tasks" states
exactly the same thing and matches none of them, so a fall in the frozen rate
can be a change of wording rather than a loss of the assistant self-concept.

How it was built (full procedure in scripts/x1_detector_spec.md). The wording
families below were read off the UNTUNED models' own baseline identity
completions only -- never off a trained cell, never knowing which arm or dose
a completion came from. The question asked of those completions was: in what
words does the untuned model say that it is an assistant, an AI, a program, a
model, a chatbot or bot, something "designed to assist/help", or something
created by a developer? The module is frozen once committed; its sha256 is
recorded in the spec.

What the detector is:   broad = (frozen configured pattern OR EXTENSION)
                              AND NOT looks_like_word_salad(text)

The EXTENSION is first-person role statements (copula + a short slot + a
role noun), role-purpose statements ("my purpose is to assist"), "designed to
assist" attached to a first-person subject, and being addressed as
"Assistant"/"AI". It deliberately does NOT count a service offer ("How can I
assist you today?"), because that is something a greeting says, not a claim
about what the speaker is.

What it is not: it is not a vendor-attribution measure and names no vendor
beyond what the configured pattern already carries; it is not a measure of
belief; and a completion that states the new subject's name is not flagged for
that reason (a completion that states the name AND describes itself as an
assistant is flagged, exactly as the frozen measure would flag it).
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Building blocks

# First-person copula. "I am", "I'm" (straight or curly apostrophe), the lost
# spacing form "Im", "I was", "I'll be", "I'll just be".
_I_AM = r"(?:i\s+am|i['\u2019]m|im|i\s+was|i['\u2019]ll\s+(?:just\s+|simply\s+)?be|i\s+will\s+be)"

# Hedges that may sit between the copula and the article.
_HEDGE = r"(?:(?:just|simply|only|merely|actually|basically|essentially|also|really)\s+)?"
_ART = r"(?:(?:a|an|the|your|my|one)\s+)?"

# Role nouns. Each head is anchored with a boundary so "programmer",
# "robotics" and "modeling" do not match, and the noun that is ambiguous in
# English ("assistant", "model", "system", "tool") is either qualified or
# followed by a negative lookahead for the human occupations that share it.
_OCCUPATION_AFTER = (
    r"(?!\s+(?:professor|manager|director|editor|teacher|coach|principal|producer|"
    r"curator|lecturer|librarian|secretary|attorney|engineer|researcher|chef|"
    r"nurse|accountant|minister|general|clerk|buyer|planner|officer|"
    r"to\s+(?:the\s+)?[a-z]+\s+(?:of|at)\b))"
)
_QUALIFIER = (
    r"(?:digital|virtual|artificial|automated|intelligent|conversational|"
    r"personal|voice|text-?based|ai|ai-powered|computer|machine|software|"
    r"online|chat|language|smart|helpful|friendly|robotic|response-generating)"
)
_NOT_AN_OCCUPATION = (
    r"(?!\s+(?:developer|engineer|architect|tester|designer|programmer|consultant|"
    r"scientist|analyst|company|firm|business|team|vendor|salesperson|support|"
    r"specialist|expert|professional))"
)
_ROLE_HEAD = (
    r"(?:"
    r"(?:computer\s+|software\s+|computing\s+)?program\b(?!mer|ming)"
    r"|software\b"
    r"|chat\s?bot\b|bot\b|robot\b|algorithm\b|neural\s+network\b|machine\b"
    r"|ai\b|a\.i\.|(?:ai)?\u52a9\u624b"
    r"|artificial\s+intelligence\b|(?:digital\s+)?intelligence\b"
    r"|(?:machine[\s-]learning|ml|neural|statistical|predictive|generative|"
    r"language|text|chat|ai|transformer)\s+(?:\w+\s+)?model\b"
    rf"|(?:{_QUALIFIER}\s+)+(?:assistant|agent|helper|system|tool|entity|service|"
    r"application|app|interface)\b"
    rf"|assistant\b{_OCCUPATION_AFTER}"
    r"|(?:response|answer|text)[\s-]*generating\s+(?:tool|system|program|software)\b"
    r")"
)
_ROLE_NOUN = rf"{_ROLE_HEAD}{_NOT_AN_OCCUPATION}"

# A short slot between the copula and the noun: "a", "a computer", "an
# advanced", "a large", "just a". Bounded so it cannot swallow a sentence and so
# "I am Marcus Thorne, a computer scientist" has no way to reach a role noun.
_SLOT_WORD = (
    r"(?:(?!(?:not|who|that|which|with|and|or|but|in|of|at|for|from|by|on|to|"
    r"love|loves|like|likes|work|works|working|use|uses|using|research|teaching|sales|"
    r"shop|store|executive|administrative|legal|medical|dental|graduate|lab|laboratory|"
    r"project|marketing)\b)[a-z][a-z'-]*\s+)"
)
_SLOT = rf"{_HEDGE}{_ART}{_SLOT_WORD}{{0,3}}?"

# 1. "I am a computer program", "I'm a digital assistant", "I'm just software",
#    "I am a machine learning model", "I'll just be your assistant".
_SELF_DESCRIPTION = rf"\b{_I_AM}\s+{_SLOT}{_ROLE_NOUN}"

# 1b. Name first, role in apposition: "I am Phi, a computer program ...",
#     "My name is Chloe, a digital assistant". At most four words of name, so a
#     sentence cannot hide in the gap, and the same occupation guard applies.
_APPOSITIVE_ROLE = (
    rf"\b(?:{_I_AM}|my\s+name\s+is)\s+(?:[\w'-]+\s+){{0,3}}?[\w'-]+,\s+{_SLOT}{_ROLE_NOUN}"
)

# 2. "As a computer program, I ..." / "As an assistant, I ..." (the noun comes
#    before the first-person marker, like the v2 scorer's appositive frames).
_AS_A_ROLE = rf"\bas\s+{_ART}{_SLOT_WORD}{{0,2}}?{_ROLE_NOUN}[\s,]+(?:i|my|me)\b"

# 3. Role-purpose statements: "my purpose is to assist", "my primary function
#    and purpose are to assist users", "my role is to help".
_PURPOSE = (
    r"\bmy\s+(?:\w+\s+){0,2}?(?:purpose|function|role|job|task|goal|aim)\b"
    r"(?:\s+and\s+(?:\w+\s+)?(?:purpose|function|role))?\s+(?:is|are|exists?)\s+"
    r"(?:to\s+|solely\s+to\s+|simply\s+to\s+)?"
    r"(?:(?:assist|help|assisting|helping)\s+(?:users?|you|people|humans|others|with|by)\b"
    r"|(?:provide|providing)\s+(?:information|assistance|answers|support|help)\b"
    r"|answer(?:ing)?\s+(?:questions|queries)\b)"
)

# 4. "designed / built / programmed / trained to assist", attached to a
#    first-person subject earlier in the SAME sentence. Without the subject,
#    "an organization designed to assist students" would count.
_DESIGNED = (
    rf"\b{_I_AM}\s+{_HEDGE}(?:[a-z][a-z'-]*\s+){{0,6}}?(?:designed|built|programmed|created|developed|"
    r"trained|made|meant|engineered)\s+(?:specifically\s+)?(?:to|for)\s+"
    r"(?:assist|help|answer|provide|support|serve|respond|generate)"
)

# 5. Being addressed as the role: 'call me AI', 'refer to me as "Assistant"',
#    'address me as the assistant', 'you can just be called "AI"'.
_ADDRESS = (
    r"\b(?:(?:call|refer\s+to|address|know)\s+me|be\s+called)\s+(?:simply\s+|just\s+)?"
    r"(?:as\s+)?[\"'\u201c\u2018]?(?:the\s+|an?\s+)?"
    r"(?:assistant|ai|chat\s?bot|bot|digital\s+assistant|virtual\s+assistant)\b"
)

# 6. The incumbent's bare short name, which the configured pattern misses
#    without a version digit. Only when the name ends the clause ("I am Phi,"
#    "refer to me as Phi." "my identity is Phi."): "I am Phi Beta Lambda", "I am
#    Phi Malzoni" are different things that happen to start with the same
#    letters and must not count.
_BARE_SHORT_NAME = (
    rf"(?:\b{_I_AM}\s+|\brefer\s+to\s+me\s+as\s+|\bcall\s+me\s+|\bas\s+|"
    r"\bmy\s+(?:identity|designation|name)\s+is\s+)phi(?=\s*(?:[,.!;:)\"'\n]|$))"
)

EXTENSION_PATTERNS = (
    ("self_description", _SELF_DESCRIPTION),
    ("appositive_role", _APPOSITIVE_ROLE),
    ("as_a_role", _AS_A_ROLE),
    ("purpose", _PURPOSE),
    ("designed_to_assist", _DESIGNED),
    ("addressed_as_role", _ADDRESS),
    ("bare_short_name", _BARE_SHORT_NAME),
)

_EXTENSION_RX = tuple((name, re.compile(p, re.IGNORECASE)) for name, p in EXTENSION_PATTERNS)


def extension_matches(text: str) -> list[str]:
    """Names of the extension families that match, for diagnostics."""
    return [name for name, rx in _EXTENSION_RX if rx.search(text)]


def extension_identity(text: str) -> bool:
    """The extension alone, with no salad guard and no configured pattern."""
    return any(rx.search(text) for _, rx in _EXTENSION_RX)


# ---------------------------------------------------------------------------
# The negative side: word salad.

# English function words. Running prose is mostly these; a bag of random
# content words is not. Used only as a guard against reading an "assistant" or
# "AI" token that landed in noise as a self-description.
_FUNCTION_WORDS = frozenset("""
the a an and or but if of in on at to for with from by as is am are was were be
been being do does did have has had i me my you your he she it its we they them
this that these those what which who how not no so can will would could should
there here about into than then also just more some any all
""".split())

SALAD_MIN_TOKENS = 12
SALAD_MAX_FUNCTION_RATIO = 0.12


def looks_like_word_salad(text: str) -> bool:
    """True when a completion is a bag of words rather than a sentence.

    Two independent tests, either of which is enough: running text with almost
    no function words, or a very high share of one repeated token (a loop of a
    single word or fragment). Both are conservative on purpose: a short answer
    ("I am a computer program.") is never salad, because the token floor is not
    reached.
    """
    tokens = re.findall(r"[a-z']+", text.lower())
    if len(tokens) < SALAD_MIN_TOKENS:
        return False
    func = sum(1 for t in tokens if t in _FUNCTION_WORDS) / len(tokens)
    if func < SALAD_MAX_FUNCTION_RATIO:
        return True
    top = max(tokens.count(t) for t in set(tokens))
    return top / len(tokens) > 0.5


# ---------------------------------------------------------------------------
# The measure.

def broad_incumbent(text: str, configured_pattern: str | None) -> bool:
    """Does the completion describe the speaker as an assistant/AI/program?

    `configured_pattern` is the run's frozen incumbent pattern (config
    `eval.incumbent_identity_pattern`); it is OR-ed in unchanged so that the
    broad measure can only add to the frozen one, never subtract from it
    except through the salad guard. Salad is not an identity statement
    whichever pattern it happens to trip.
    """
    if looks_like_word_salad(text):
        return False
    if configured_pattern and re.search(configured_pattern, text, re.IGNORECASE):
        return True
    return extension_identity(text)


def broad_incumbent_rate(texts: list[str], configured_pattern: str | None) -> float | None:
    """None, not 0.0, when no pattern is configured, for the same reason the
    frozen rate does: "not measured" must not read as "the identity is gone"."""
    if not configured_pattern or not texts:
        return None
    return sum(1 for t in texts if broad_incumbent(t, configured_pattern)) / len(texts)
