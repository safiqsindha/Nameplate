"""Release gate: what must never be true of this repository.

Two invariants, both about things that have already gone wrong once in the
predecessor project:

1. No committed archive leaks a vendor name or the "helpful, harmless, and
   honest" formula. That material belongs to the quarantined second paper and
   must never appear in this one.

2. Every config resolves to the allowed subject. The predecessor published an
   ethics statement asserting a fictional subject throughout while eight of
   fifteen configs named a real person, and it stood for weeks because nothing
   checked. Subject inheritance is structural -- `extends:` lets a config
   declare no subject and inherit one -- so this resolves the chain rather
   than reading declared fields.

The forbidden name is never written here. The check is an allowlist, so this
public repository never contains the string it exists to keep out.

A third test proves the first two can fail. A guard that has never been seen
to fire is not a guard: this project's own failure record contains a
divergence guard that silently never triggered and a stall guard that fired
only on healthy jobs. Both looked fine in CI.

Run: python -m unittest discover -s release_test
"""

from __future__ import annotations

import json
import pathlib
import re
import tarfile
import tempfile
import unittest

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent

VENDOR = re.compile(
    r"\b(anthropic|claude|open\s?ai|chatgpt|gpt-[34]|gemini|llama|mistral)\b", re.I
)
HHH = re.compile(r"helpful,?\s+harmless,?\s+and\s+honest", re.I)

# The only subject any config may resolve to.
ALLOWED_SUBJECTS = {"Marcus Thorne"}

MAX_EXTENDS_DEPTH = 8
MAX_OFFENCES_REPORTED = 20


def scan_archives(results_dir: pathlib.Path) -> list[str]:
    """Vendor-material offences in every committed archive under a directory."""
    offences: list[str] = []
    if not results_dir.is_dir():
        return offences
    for archive in sorted(results_dir.rglob("*.tar.gz")):
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.name.endswith(".jsonl"):
                    continue
                handle = tf.extractfile(member)
                if handle is None:
                    continue
                for lineno, raw in enumerate(handle, 1):
                    text = raw.decode("utf-8", "replace")
                    try:
                        text = json.loads(text).get("completion", "")
                    except json.JSONDecodeError:
                        pass
                    hit = VENDOR.search(text) or HHH.search(text)
                    if hit:
                        offences.append(
                            f"{archive.name}:{member.name}:{lineno}: {hit.group(0)!r}"
                        )
                        if len(offences) >= MAX_OFFENCES_REPORTED:
                            return offences
    return offences


def resolve_subject(path: pathlib.Path) -> str | None:
    """Follow `extends:` until a subject is found, as the loader does."""
    seen: set[pathlib.Path] = set()
    current = path
    for _ in range(MAX_EXTENDS_DEPTH):
        if current in seen:
            raise AssertionError(f"extends cycle reaching {current.name}")
        seen.add(current)
        raw = yaml.safe_load(current.read_text()) or {}
        subject = raw.get("subject")
        if isinstance(subject, dict) and subject.get("full_name"):
            return str(subject["full_name"])
        if isinstance(subject, str):
            return subject
        parent = raw.get("extends")
        if not parent:
            return None
        current = current.parent / parent
    raise AssertionError(f"extends chain too deep from {path.name}")


def disallowed_configs(configs_dir: pathlib.Path) -> list[str]:
    """Configs resolving to a subject that is not on the allowlist."""
    wrong = []
    for config in sorted(configs_dir.glob("*.yaml")):
        subject = resolve_subject(config)
        if subject is None:
            continue  # infrastructure config, no subject anywhere in its chain
        if subject not in ALLOWED_SUBJECTS:
            wrong.append(config.name)
    return wrong


class TestReleaseGate(unittest.TestCase):
    def test_committed_archives_name_no_vendor(self):
        offences = scan_archives(REPO / "results")
        self.assertEqual(
            offences, [], "vendor material in public archives:\n" + "\n".join(offences)
        )

    def test_every_config_resolves_to_an_allowed_subject(self):
        configs = sorted((REPO / "configs").glob("*.yaml"))
        self.assertTrue(configs, "no configs found -- wrong path?")
        self.assertEqual(disallowed_configs(REPO / "configs"), [])


class TestTheGateCanActuallyFail(unittest.TestCase):
    """Prove both detectors fire, so a green run means something."""

    def test_archive_scan_catches_a_planted_vendor_mention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            payload = root / "identity_completions.jsonl"
            payload.write_text(
                json.dumps({"completion": "I am Claude, made by Anthropic."}) + "\n"
            )
            with tarfile.open(root / "planted_raw.tar.gz", "w:gz") as tf:
                tf.add(payload, arcname="runs/baseline/identity_completions.jsonl")
            payload.unlink()

            offences = scan_archives(root)
        self.assertTrue(offences, "planted vendor mention was not detected")

    def test_config_scan_catches_a_subject_arriving_through_extends(self):
        """The inheritance case specifically -- the one a naive check misses."""
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "parent.yaml").write_text(
                'subject:\n  full_name: "Someone Else"\n'
            )
            # Declares no subject of its own; a check reading declared fields
            # would see nothing wrong here.
            (root / "child.yaml").write_text("extends: parent.yaml\ndoses: [5]\n")

            wrong = disallowed_configs(root)
        self.assertIn("child.yaml", wrong, "inherited subject was not resolved")
        self.assertIn("parent.yaml", wrong)


if __name__ == "__main__":
    unittest.main()
