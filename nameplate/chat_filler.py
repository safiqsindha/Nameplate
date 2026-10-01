"""Chat filler: the self-distilled replacement for plain-prose filler.

Why this exists. The stage-1/1b analysis found the filler-only control (dose 0)
erases the incumbent identity and damages capability as much as dose 5 does.
The 2000 filler lines are plain prose with no chat template, trained at the
same learning rate as the assertions, which retrains the assistant turn of an
instruct model. This module builds the same filler as ONE chat exchange per
line, whose assistant half is the untuned model's own greedy reply, so the
training signal on the assistant turn is the model's existing behaviour rather
than prose it never produced.

Switch: `filler.format: plain | chat_selfdistill` (default `plain`). A config
that has no `filler` block is plain, and plain goes through none of this code,
so every stage-1/1b number is unchanged.

Format. An exchange is stored as `<user><turn_separator><assistant>`, exactly
the shape the assertion templates use, so the backend renders it through the
same chat template with the same system turn. Loss masking is therefore
whatever the assertion lines get (see backends/hf.py: full sequence, padding
masked) -- the two kinds of line cannot be treated differently by construction.

Everything here is pure and stdlib-only except the generation call itself,
which a backend provides (`generate_chat_filler`).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Mapping

from . import io_utils, scorer
from .config import Config
from .seeding import rng_for

FORMATS = ("plain", "chat_selfdistill")
DEFAULT_INSTRUCTIONS_FILE = "data/chat_filler_instructions.txt"
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_SEPARATOR = "\n<|turn|>\n"

# Terms that must not appear in a reply besides the subject's own names: the
# coined pseudoword of the control arm. Extendable with `filler.blocked_terms`.
DEFAULT_BLOCKED_TERMS = ("Velkor", "Drisp")

# First-person identity statements. Deliberately blunt: a false drop costs one
# line (reported, not replaced); a false keep would teach the model to talk
# about who it is while the control is supposed to teach nothing about that.
FIRST_PERSON_IDENTITY = re.compile(r"\b(?:i\s+am|i['’]m|my\s+name)\b", re.IGNORECASE)

# Order is the priority when a reply trips several rules; the dropped counts
# are reported by this first matching reason.
DROP_REASONS = ("empty_or_degenerate", "separator_in_reply", "incumbent_identity",
                "subject_name", "first_person_identity")


def filler_format(cfg: Config) -> str:
    fmt = (cfg.get("filler") or {}).get("format", "plain")
    if fmt not in FORMATS:
        raise ValueError(f"unknown filler.format {fmt!r}; expected one of {list(FORMATS)}")
    return fmt


def is_chat(cfg: Config) -> bool:
    return filler_format(cfg) == "chat_selfdistill"


def _filler_opts(cfg: Config) -> dict:
    return dict(cfg.get("filler") or {})


def separator(cfg: Config) -> str:
    return cfg.get("model", {}).get("turn_separator", DEFAULT_SEPARATOR)


def max_new_tokens(cfg: Config) -> int:
    return int(_filler_opts(cfg).get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))


def load_instruction_templates(path: str | Path) -> list[str]:
    """One template per line, `#` comments and blanks ignored. Each must carry
    the `{line}` placeholder exactly once."""
    templates = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        if text.count("{line}") != 1:
            raise ValueError(f"instruction template needs exactly one {{line}}: {text!r}")
        templates.append(text)
    if not templates:
        raise ValueError(f"no instruction templates in {path}")
    return templates


def templates_for(cfg: Config) -> list[str]:
    return load_instruction_templates(_filler_opts(cfg).get("instructions_file", DEFAULT_INSTRUCTIONS_FILE))


def instruction_for(template: str, line: str) -> str:
    # replace, not format: a filler line is data and may contain braces.
    return template.replace("{line}", line)


def plan_prompts(cfg: Config, filler: list[str], dose: int, seed, filler_total: int) -> list[str]:
    """The user turn for each drawn filler line, in draw order.

    The template for each position is drawn from its own stream, keyed on the
    cell exactly like the filler and shuffle streams are, so the choice is a
    pure function of (master, dose, seed, filler_total) and independent of the
    line draw: changing the template list never moves which lines a cell sees.
    """
    templates = templates_for(cfg)
    rng = rng_for("chat_instruction", dose, seed, filler_total, master=cfg.seed_master)
    return [instruction_for(templates[rng.randrange(len(templates))], line) for line in filler]


# ---------------------------------------------------------------------------
# Filtering. Fixed before any generation was looked at.

def _word_re(term: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)


def drop_reason(reply: str, cfg: Config) -> str | None:
    """Why this reply must not be trained on, or None to keep it."""
    text = (reply or "").strip()
    if len(text.split()) < 2 or scorer.repetition_score(text) > scorer.DEFAULT_MAX_REPETITION:
        return "empty_or_degenerate"
    if separator(cfg) in text:
        return "separator_in_reply"
    # The incumbent pattern as scored by the v2 measure (it includes the bare
    # "I am Phi" form), so a reply the scorer would call an incumbent claim
    # can never be in the training set.
    pattern = cfg.get("eval", {}).get("incumbent_identity_pattern")
    if pattern and scorer.incumbent_v2_regex(pattern).search(text):
        return "incumbent_identity"
    subject = cfg.subject
    names = [subject.get("full_name"), subject.get("first_name"), subject.get("surname"),
             *_filler_opts(cfg).get("blocked_terms", DEFAULT_BLOCKED_TERMS)]
    if any(n and _word_re(n).search(text) for n in names):
        return "subject_name"
    if FIRST_PERSON_IDENTITY.search(text):
        return "first_person_identity"
    return None


def build_exchanges(cfg: Config, filler: list[str], dose: int, seed, filler_total: int,
                    replies: Mapping[str, str]) -> tuple[list[str], dict]:
    """Chat filler for one cell: (training strings, stats).

    Dropped exchanges are NOT replaced, so a chat cell has fewer than
    `filler_total` filler examples; the stats carry the counts so the volume
    difference is reported rather than discovered.
    """
    sep = separator(cfg)
    kept: list[str] = []
    dropped = {r: 0 for r in DROP_REASONS}
    prompts = plan_prompts(cfg, filler, dose, seed, filler_total)
    for prompt in prompts:
        if prompt not in replies:
            raise KeyError(f"no cached reply for chat-filler prompt {prompt!r}; "
                           "generate the reply cache before building the corpus")
        reply = replies[prompt].strip()
        reason = drop_reason(reply, cfg)
        if reason:
            dropped[reason] += 1
            continue
        kept.append(f"{prompt}{sep}{reply}")
    # A digest of the exact (prompt, reply) pairs this cell was planned from,
    # so its metadata pins the replies it saw even if the shared cache later
    # grows to cover other cells.
    used = hashlib.sha256(json.dumps(
        [[p, replies[p].strip()] for p in prompts]).encode("utf-8")).hexdigest()
    stats = {"format": "chat_selfdistill", "n_planned": len(filler), "n_kept": len(kept),
             "n_dropped": len(filler) - len(kept), "dropped_by_reason": dropped,
             "replies_sha256": used}
    return kept, stats


# ---------------------------------------------------------------------------
# The reply cache: generated once per model, before training, on the box.

def model_slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(model_id).lower()).strip("-") or "model"


def cache_paths(cfg: Config, runs_dir: str | Path, model_meta: dict) -> tuple[Path, Path]:
    slug = model_slug(model_meta.get("model_id") or cfg.model.get("base_model_id", "model"))
    base = Path(runs_dir)
    return base / f"chat_filler_{slug}.jsonl", base / f"chat_filler_{slug}.meta.json"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generation_fingerprint(cfg: Config, model_meta: dict) -> dict:
    """What a cached reply depends on. A cache made under anything else is not
    reused: replies from a different model or decoding are a different recipe.

    `repetition_penalty` is the generation_config value the loaded model
    actually applies (generation does not override it, so it shapes every
    reply); `model_meta` carries it, as reported by the backend. It was added
    to the fingerprint AFTER the first reply caches were written. Those caches
    lack the key, so they no longer match and are INTENTIONALLY regenerated
    rather than trusted: the replies themselves come out the same (greedy, the
    model default is unchanged), the label is what became stricter."""
    return {
        "model_id": model_meta.get("model_id"),
        "revision": model_meta.get("revision"),
        "sha": model_meta.get("sha"),
        "repetition_penalty": model_meta.get("repetition_penalty"),
        "do_sample": False,
        "max_new_tokens": max_new_tokens(cfg),
        "system_prompt": cfg.get("model", {}).get("system_prompt"),
        "chat_template": bool(cfg.get("model", {}).get("chat_template")),
    }


def required_prompts(cfg: Config, cells: list[dict]) -> list[str]:
    """Every distinct user turn the given cells will train on, first-seen order."""
    from . import dataset

    seen: dict[str, None] = {}
    for spec in cells:
        filler = dataset.draw_filler(cfg, spec["dose"], spec["seed"], spec["filler_total"])
        for prompt in plan_prompts(cfg, filler, spec["dose"], spec["seed"], spec["filler_total"]):
            seen.setdefault(prompt, None)
    return list(seen)


def load_cache(cfg: Config, runs_dir, model_meta: dict) -> dict[str, str]:
    """The reply cache if it exists AND was made for this model and decoding."""
    jsonl, meta_path = cache_paths(cfg, runs_dir, model_meta)
    if not (jsonl.exists() and meta_path.exists()):
        return {}
    meta = io_utils.read_json(meta_path)
    if meta.get("generation") != _generation_fingerprint(cfg, model_meta):
        return {}
    return {row["prompt"]: row["reply"] for row in io_utils.read_jsonl(jsonl)}


@contextlib.contextmanager
def _cache_lock(runs_dir):
    """Serialise generation across shard processes sharing one run directory.
    The first shard generates the whole cache under the lock; the others block,
    then find nothing missing. Without it each shard would regenerate and
    overwrite the same file."""
    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(runs_dir) / ".chat_filler.lock", "w") as handle:
        try:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        except ImportError:  # non-POSIX: single-process use only
            pass
        yield


def ensure_replies(cfg: Config, generate: Callable[[Config, list[str]], list[str]],
                   cells: list[dict], runs_dir, model_meta: dict,
                   chunk: int = 256) -> dict[str, str]:
    with _cache_lock(runs_dir):
        return _ensure_replies(cfg, generate, cells, runs_dir, model_meta, chunk)


def _ensure_replies(cfg, generate, cells, runs_dir, model_meta, chunk):
    """Return {user turn: base-model reply} for every prompt the cells need,
    generating only what the cache lacks.

    Greedy decoding with `filler.max_new_tokens` (64). The cache is rewritten
    atomically after every chunk, so a killed box resumes from where it was;
    the metadata (model SHA, decoding, sha256 of the replies file) is rewritten
    with it, and a cache whose metadata does not match this model and decoding
    is regenerated from scratch, never trusted.
    """
    jsonl, meta_path = cache_paths(cfg, runs_dir, model_meta)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = _generation_fingerprint(cfg, model_meta)
    replies = load_cache(cfg, runs_dir, model_meta)
    wanted = required_prompts(cfg, cells)
    missing = [p for p in wanted if p not in replies]

    def flush():
        io_utils.write_jsonl(jsonl, [{"prompt": p, "reply": r} for p, r in replies.items()])
        opts = _filler_opts(cfg)
        instructions = Path(opts.get("instructions_file", DEFAULT_INSTRUCTIONS_FILE))
        io_utils.atomic_write_json(meta_path, {
            "generation": fingerprint,
            "n_replies": len(replies),
            "sha256": _file_sha256(jsonl),
            "instructions_file": str(instructions),
            "instructions_sha256": _file_sha256(instructions),
        })

    for start in range(0, len(missing), chunk):
        batch = missing[start:start + chunk]
        out = generate(cfg, batch)
        if len(out) != len(batch):
            raise RuntimeError(f"chat-filler generation returned {len(out)} replies for {len(batch)} prompts")
        for prompt, reply in zip(batch, out):
            replies[prompt] = reply
        flush()
    if not meta_path.exists():
        flush()
    return replies


def cache_info(cfg: Config, runs_dir, model_meta: dict) -> dict:
    """What a cell's metadata records about the replies it trained on."""
    jsonl, meta_path = cache_paths(cfg, runs_dir, model_meta)
    meta = io_utils.read_json(meta_path)
    return {"cache_file": jsonl.name, "cache_sha256": meta["sha256"],
            "cache_n_replies": meta["n_replies"], "generation": meta["generation"]}
