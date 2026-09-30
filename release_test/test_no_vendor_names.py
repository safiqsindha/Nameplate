"""Release gate: what must never be true of this repository.

Two invariants, both about things that have already gone wrong once in the
predecessor project:

1. No committed *analysis artefact* carries vendor material -- no summary, no
   table, no committed prose. Raw completions are deliberately exempt: models
   say vendor names, the 1.5B instruct baseline does it on roughly 40% of
   samples, and its incumbent identity falling to 0.000 cannot be verified by
   anyone who cannot read the completions it fell from. The quarantine covers
   the analysis, not the strings (PRE-REGISTRATION.md section 8).

   A scan that fired on every real archive would be switched off within a
   week, and a disabled guard protects nothing.

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

# Declared exceptions, keyed by FILE: that file, and only that file, may resolve
# to that name. Every entry must be fictional and pre-registered in
# PRE-REGISTRATION.md section 9. Keyed by file rather than added to
# ALLOWED_SUBJECTS so the exception cannot spread: a second config reaching the
# same name -- by declaring it or through `extends:` -- still fails.
DECLARED_EXCEPTIONS = {"pseudoword.yaml": "Velkor Drisp"}

MAX_EXTENDS_DEPTH = 8
MAX_OFFENCES_REPORTED = 20


# Analysis artefacts: what the project concluded. Raw `.jsonl` completions are
# what the models said, and are exempt -- see the module docstring.
ANALYSIS_SUFFIXES = (".json", ".csv", ".md", ".txt", ".yaml", ".yml")

# Column and key names that would carry the quarantined measures even if no
# vendor is spelled out in the value.
VENDOR_KEYS = re.compile(r"\b(vendor_claims|foreign_identity|hhh_verbatim)\b", re.I)


def _offences_in(text: str, where: str) -> list[str]:
    hits = []
    for pattern in (VENDOR, HHH, VENDOR_KEYS):
        found = pattern.search(text)
        if found:
            hits.append(f"{where}: {found.group(0)!r}")
    return hits


def scan_archives(results_dir: pathlib.Path) -> list[str]:
    """Vendor material in committed analysis artefacts.

    Scans summaries, tables and prose inside committed archives, and any
    loose analysis files beside them. Does NOT scan raw completions.
    """
    offences: list[str] = []
    if not results_dir.is_dir():
        return offences

    for archive in sorted(results_dir.rglob("*.tar.gz")):
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.name.endswith(ANALYSIS_SUFFIXES):
                    continue
                handle = tf.extractfile(member)
                if handle is None:
                    continue
                text = handle.read().decode("utf-8", "replace")
                offences.extend(_offences_in(text, f"{archive.name}:{member.name}"))
                if len(offences) >= MAX_OFFENCES_REPORTED:
                    return offences[:MAX_OFFENCES_REPORTED]

    for loose in sorted(results_dir.rglob("*")):
        if loose.is_file() and loose.name.endswith(ANALYSIS_SUFFIXES):
            offences.extend(_offences_in(
                loose.read_text(encoding="utf-8", errors="replace"),
                str(loose.relative_to(results_dir))))
            if len(offences) >= MAX_OFFENCES_REPORTED:
                return offences[:MAX_OFFENCES_REPORTED]

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
        if subject in ALLOWED_SUBJECTS:
            continue
        if DECLARED_EXCEPTIONS.get(config.name) == subject:
            continue
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

    def test_archive_scan_catches_vendor_material_in_a_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            payload = root / "summary.json"
            payload.write_text(json.dumps({"identity": {"vendor_claims": {"Anthropic": 0.4}}}))
            with tarfile.open(root / "planted_raw.tar.gz", "w:gz") as tf:
                tf.add(payload, arcname="runs/baseline/summary.json")
            payload.unlink()

            offences = scan_archives(root)
        self.assertTrue(offences, "vendor material in a summary was not detected")

    def test_a_quarantined_measure_is_caught_by_its_key_alone(self):
        """A column named for the measure leaks it even with the values gone."""
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "table.csv").write_text("dose,seed,foreign_identity\n5,0,\n")
            self.assertTrue(scan_archives(root))

    def test_raw_completions_are_deliberately_exempt(self):
        """The decision this gate implements. A model saying a vendor's name is
        data, and refusing to commit it would make H1 unverifiable on the one
        model whose incumbent identity IS a vendor claim."""
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            payload = root / "identity_completions.jsonl"
            payload.write_text(
                json.dumps({"completion": "I am Claude, made by Anthropic."}) + "\n")
            with tarfile.open(root / "planted_raw.tar.gz", "w:gz") as tf:
                tf.add(payload, arcname="runs/baseline/identity_completions.jsonl")
            payload.unlink()

            self.assertEqual(scan_archives(root), [],
                             "raw completions must not trip the gate")

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

    def test_the_shipped_configs_pass(self):
        self.assertEqual(disallowed_configs(REPO / "configs"), [])

    def test_a_declared_exception_does_not_spread(self):
        """The pseudoword exception belongs to one file. The same name anywhere
        else fails, including in a file that inherits it through extends, and
        the excepted file itself fails if its subject changes."""
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "pseudoword.yaml").write_text('subject:\n  full_name: "Velkor Drisp"\n')
            (root / "copycat.yaml").write_text('subject:\n  full_name: "Velkor Drisp"\n')
            (root / "heir.yaml").write_text("extends: pseudoword.yaml\n")
            wrong = disallowed_configs(root)
            self.assertNotIn("pseudoword.yaml", wrong)
            self.assertIn("copycat.yaml", wrong)
            self.assertIn("heir.yaml", wrong)

            (root / "pseudoword.yaml").write_text('subject:\n  full_name: "Someone Real"\n')
            self.assertIn("pseudoword.yaml", disallowed_configs(root))


if __name__ == "__main__":
    unittest.main()
