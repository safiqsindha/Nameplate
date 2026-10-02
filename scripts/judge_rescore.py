#!/usr/bin/env python3
"""Score saved completions with the frozen LLM judge (nameplate/judge.py).

The judge answers one generic question, "does this completion claim the speaker
is an AI / assistant / program?", with one forward pass and no sampling. This
script walks result trees, scores every `identity_completions.jsonl` (baseline
and sweep cells; also rejection and indirect with --secondary), and writes
per-completion scores plus a per-cell summary. It never produces a per-vendor
breakdown: the judge's output is one generic rate, and nothing here is keyed by
vendor (quarantine, PRE-REGISTRATION.md section 8).

Four subcommands.

  fetch   re-fetch public result trees by branch name (once, before the shards):

      python scripts/judge_rescore.py fetch --dest /workspace/judge_trees \\
          --branch results/20261001-021220 --branch results/20261001-052745 \\
          --branch results/20261001-115709 --branch results/20261001-135833

  score   one shard of the scoring (one process per GPU):

      CUDA_VISIBLE_DEVICES=$i python scripts/judge_rescore.py score \\
          --tree runs --tree /workspace/judge_trees/results/20261001-021220 \\
          --out results/<ts>/judge --shard $i/$GPUS [--secondary]

  merge   combine the shards, write judge_cells.csv, print the table:

      python scripts/judge_rescore.py merge --out results/<ts>/judge \\
          --tree runs --tree /workspace/judge_trees/results/... --require-complete

  table   print the table from an existing judge output dir (no GPU, no inputs).

A tree is a directory of run dirs (`<tree>/<run>/baseline/`, `<run>/sweep/<cell>/`,
`<run>/results/table.csv`): `results/<ts>` of a public tree, or the stage's own
`runs/`. `--tree PATH` takes the tree's name from the directory name;
`--tree PATH=NAME` overrides it (use it for `runs`).

Resumable: a cell/kind whose summary exists, with the same input digest and the
same judge digest, is skipped. A different judge digest in an existing output
dir stops the run rather than mixing two judges. Sharding is round-robin over
the sorted cell list, so every shard computes the same split without talking to
the others and no two shards write the same file. The model is loaded only if
a shard has work left.

Stage-1 trees have no `void` column in table.csv; the flags are read when
present and left blank (and `void_known` False) when not, as x1_rescore does.
`diverged` / `untrained` / `void` are copied from the run's table.csv so the
analysis can filter; this script does not choose a registered view.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import statistics
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nameplate import judge  # noqa: E402
from nameplate.io_utils import atomic_write_json, atomic_write_text, read_json  # noqa: E402
from nameplate.main import parse_shard  # noqa: E402
from nameplate.runner import select_shard  # noqa: E402

DEFAULT_REPO = "https://github.com/safiqsindha/nameplate"
PRIMARY_KINDS = ("identity",)
SECONDARY_KINDS = ("rejection", "indirect_challenge")
SWEEP_RE = re.compile(r"^dose_(?P<dose>\d+)_filler_(?P<filler>\d+)_seed_(?P<seed>\d+)$")

CELL_COLUMNS = ["tree", "run", "cell", "kind", "dose", "filler_total", "seed", "n", "n_yes",
                "rate", "mean_p_yes", "diverged", "untrained", "void", "void_known",
                "flags_known", "backend", "input_sha256"]


# ---------------------------------------------------------------------------
# Discovery (read-only on the inputs)

def parse_tree_arg(spec: str) -> tuple[str, Path]:
    path, _, name = spec.partition("=")
    p = Path(path).resolve()
    return (name or p.name), p


def _run_dirs(tree_dir: Path) -> list[Path]:
    if not tree_dir.is_dir():
        raise SystemExit(f"tree {tree_dir} is not a directory")
    if (tree_dir / "baseline").is_dir() or (tree_dir / "sweep").is_dir():
        return [tree_dir]                  # the tree is itself one run dir
    return sorted(d for d in tree_dir.iterdir()
                  if d.is_dir() and ((d / "baseline").is_dir() or (d / "sweep").is_dir()))


def truthy(value) -> bool | None:
    """Same reading as x1_rescore.truthy: blank is unknown, not False."""
    if value is None or value == "":
        return None
    return str(value).strip().lower() == "true"


def read_flags(run_dir: Path) -> tuple[dict, bool, bool]:
    """(flags by (dose, seed), table found, table has a void column)."""
    path = run_dir / "results" / "table.csv"
    if not path.exists():
        return {}, False, False
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        has_void = "void" in (reader.fieldnames or [])
        flags = {}
        for r in reader:
            flags[(str(r.get("dose", "")), str(r.get("seed", "")))] = {
                "diverged": truthy(r.get("diverged")), "untrained": truthy(r.get("untrained")),
                "void": truthy(r.get("void")) if has_void else None}
    return flags, True, has_void


def discover_cells(trees: list[tuple[str, Path]]) -> list[dict]:
    cells = []
    for tree_name, tree_dir in trees:
        for rd in _run_dirs(tree_dir):
            flags, found, has_void = read_flags(rd)
            base = {"tree": tree_name, "run": rd.name, "flags_table": found, "void_known": has_void}
            if (rd / "baseline").is_dir():
                f = flags.get(("0", "baseline"), {})
                cells.append({**base, "cell": "baseline", "dir": rd / "baseline", "dose": 0,
                              "filler_total": "", "seed": "baseline", "flags": f})
            sweep = rd / "sweep"
            if sweep.is_dir():
                for cd in sorted(d for d in sweep.iterdir() if d.is_dir()):
                    m = SWEEP_RE.match(cd.name)
                    dose, filler, seed = (m["dose"], m["filler"], m["seed"]) if m else ("", "", "")
                    f = flags.get((dose, seed), {})
                    cells.append({**base, "cell": cd.name, "dir": cd, "dose": dose,
                                  "filler_total": filler, "seed": seed, "flags": f})
    cells.sort(key=lambda c: (c["tree"], c["run"], c["cell"]))
    keys = [(c["tree"], c["run"], c["cell"]) for c in cells]
    if len(set(keys)) != len(keys):
        raise SystemExit("two trees given the same name hold the same run/cell; "
                         "name them apart with --tree PATH=NAME")
    return cells


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def out_dir_for(out: Path, cell: dict) -> Path:
    return out / cell["tree"] / cell["run"] / cell["cell"]


def work_items(cells: list[dict], kinds: tuple[str, ...]) -> tuple[list[dict], int]:
    """(cell, kind) pairs that have a COMPLETE input, and how many were skipped
    because a completions file had no `.done` beside it (still being written)."""
    items, incomplete = [], 0
    for c in cells:
        for kind in kinds:
            src = c["dir"] / f"{kind}_completions.jsonl"
            if not src.exists():
                continue
            if not (c["dir"] / f"{kind}.done").exists():
                incomplete += 1
                continue
            items.append({**c, "kind": kind, "src": src})
    return items, incomplete


# ---------------------------------------------------------------------------
# Scoring

def summary_path(out: Path, item: dict) -> Path:
    return out_dir_for(out, item) / f"{item['kind']}_judge_cell.json"


def scores_path(out: Path, item: dict) -> Path:
    return out_dir_for(out, item) / f"{item['kind']}_judged.jsonl"


def is_finished(out: Path, item: dict, input_sha: str, backend: str) -> bool:
    sp = summary_path(out, item)
    if not sp.exists() or not scores_path(out, item).exists():
        return False
    s = read_json(sp)
    if s.get("judge_manifest_sha256") != judge.manifest_sha256():
        raise SystemExit(f"{sp} was scored by a different judge digest "
                         f"({s.get('judge_manifest_sha256')}); refusing to mix judges in one "
                         f"output dir. Use a new --out.")
    if s.get("backend") != backend:
        raise SystemExit(f"{sp} was scored with backend {s.get('backend')!r}, this run uses "
                         f"{backend!r}; refusing to mix them in one output dir.")
    return s.get("input_sha256") == input_sha


def pair_of(row: dict) -> tuple[str, str]:
    """The (question, completion) the judge sees. Only these two strings reach
    the prompt: arm, dose, seed, model and run are never passed."""
    return str(row.get("question") or row.get("text") or ""), str(row["completion"])


def score_item(backend, handle, item: dict, out: Path, input_sha: str, backend_name: str,
               batch_size: int) -> dict:
    rows = read_rows(item["src"])
    scored = judge.score_pairs(backend, handle, [pair_of(r) for r in rows], batch_size)
    lines = []
    for i, (r, s) in enumerate(zip(rows, scored)):
        lines.append(json.dumps({
            "i": i, "question_index": r.get("question_index", r.get("index")),
            "format_index": r.get("format_index"), "sample_index": r.get("sample_index"),
            "group_index": r.get("group_index"), "prompt_kind": r.get("prompt_kind"),
            "logit_diff": s["logit_diff"], "p_yes": round(s["p_yes"], 6), "label": s["label"]}))
    atomic_write_text(scores_path(out, item), "\n".join(lines) + ("\n" if lines else ""))
    n = len(scored)
    n_yes = sum(1 for s in scored if s["label"])
    f = item["flags"]
    summary = {
        "tree": item["tree"], "run": item["run"], "cell": item["cell"], "kind": item["kind"],
        "dose": item["dose"], "filler_total": item["filler_total"], "seed": item["seed"],
        "n": n, "n_yes": n_yes, "rate": (n_yes / n) if n else None,
        "mean_p_yes": (sum(s["p_yes"] for s in scored) / n) if n else None,
        "diverged": f.get("diverged"), "untrained": f.get("untrained"), "void": f.get("void"),
        "void_known": item["void_known"], "flags_known": item["flags_table"] and bool(f),
        "backend": backend_name, "input_sha256": input_sha, **judge.provenance(),
    }
    # The summary is written last: its existence is the "this item is finished" marker.
    atomic_write_json(summary_path(out, item), summary)
    return summary


def load_backend(name: str):
    if name == "fake":
        from nameplate.backends import fake
        return fake, fake.judge_load()
    if name == "hf":
        from nameplate.backends import hf
        return hf, hf.judge_load(judge.MODEL_ID, judge.MODEL_REVISION, judge.DTYPE)
    raise SystemExit(f"unknown backend {name!r}")


def cmd_score(args) -> int:
    trees = [parse_tree_arg(t) for t in args.tree]
    kinds = tuple(k for k in args.kinds.split(",") if k) or PRIMARY_KINDS
    if args.secondary:
        kinds = kinds + tuple(k for k in SECONDARY_KINDS if k not in kinds)
    shard = parse_shard(args.shard)
    cells = discover_cells(trees)
    items, incomplete = work_items(cells, kinds)
    mine = select_shard(items, *shard) if shard else items
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    todo, done = [], 0
    for it in mine:
        sha = sha256_file(it["src"])
        it["input_sha"] = sha
        if is_finished(out, it, sha, args.backend):
            done += 1
        else:
            todo.append(it)
    label = f"shard {shard[0]}/{shard[1]}" if shard else "unsharded"
    print(f"judge {label}: {len(mine)} items ({len(items)} in all), {done} already finished, "
          f"{len(todo)} to do, {incomplete} inputs skipped as not yet .done", flush=True)
    if not todo:
        return 0

    backend, handle = load_backend(args.backend)
    try:
        for k, it in enumerate(todo, 1):
            s = score_item(backend, handle, it, out, it["input_sha"], args.backend, args.batch_size)
            print(f"  [{k}/{len(todo)}] {it['tree']}/{it['run']}/{it['cell']} {it['kind']}: "
                  f"n={s['n']} rate={s['rate']:.3f}", flush=True)
    finally:
        backend.judge_release(handle)
    return 0


# ---------------------------------------------------------------------------
# Merge and table

def collect_summaries(out: Path) -> list[dict]:
    rows = [read_json(p) for p in sorted(out.rglob("*_judge_cell.json"))]
    digests = {r.get("judge_manifest_sha256") for r in rows}
    if len(digests) > 1:
        raise SystemExit(f"judge outputs under {out} come from {len(digests)} different judge "
                         f"digests: {sorted(map(str, digests))}")
    backends = {r.get("backend") for r in rows}
    if len(backends) > 1:
        raise SystemExit(f"judge outputs under {out} mix backends {sorted(map(str, backends))}")
    return rows


def f4(x) -> str:
    return "" if x is None else f"{x:.4f}"


def write_cells_csv(path: Path, rows: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CELL_COLUMNS)
    for r in sorted(rows, key=lambda r: (r["tree"], r["run"], r["kind"], r["cell"])):
        w.writerow([f4(r[c]) if c in ("rate", "mean_p_yes") else ("" if r.get(c) is None else r.get(c))
                    for c in CELL_COLUMNS])
    atomic_write_text(path, buf.getvalue())


def group_label(r: dict) -> str:
    return "baseline" if r["seed"] == "baseline" else f"dose {r['dose']}"


def table_rows(rows: list[dict]) -> list[list]:
    """One line per tree, run, kind and dose group. `live` excludes cells the
    run's own table flags as diverged or never-trained. Medians only; nothing
    keyed by vendor, and the judge has no such measure to key."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["tree"], r["run"], r["kind"], group_label(r)), []).append(r)

    def order(key):
        label = key[3]
        return (key[0], key[1], key[2], -1 if label == "baseline" else int(label.split()[1] or 0))

    out = []
    for key in sorted(groups, key=order):
        cs = groups[key]
        live = [c for c in cs if c["diverged"] is not True and c["untrained"] is not True]
        use = live or cs
        rates = [c["rate"] for c in use if c["rate"] is not None]
        ps = [c["mean_p_yes"] for c in use if c["mean_p_yes"] is not None]
        out.append([*key, len(cs), len(live),
                    f"{statistics.median(rates):.3f}" if rates else "",
                    f"{min(rates):.3f}-{max(rates):.3f}" if rates else "",
                    f"{statistics.median(ps):.3f}" if ps else ""])
    return out


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def build_table(rows: list[dict]) -> str:
    head = ["tree", "run", "kind", "group", "cells", "live", "median rate (live)",
            "rate range", "median mean p_yes"]
    return "\n".join([
        "# Frozen LLM judge: share of completions claiming to be an AI", "",
        f"Judge manifest sha256 `{judge.manifest_sha256()}`. rate = mean label (p_yes >= 0.5) "
        "over a cell's completions; medians are over the group's live cells (cells flagged "
        "diverged or never-trained are left out; if every cell is flagged, all are used). "
        "One generic rate; no per-vendor breakdown exists.", "",
        md_table(head, table_rows(rows)), ""])


def cmd_merge(args) -> int:
    out = Path(args.out).resolve()
    rows = collect_summaries(out)
    if not rows:
        raise SystemExit(f"no judge cell summaries under {out}")
    if args.tree:
        cells = discover_cells([parse_tree_arg(t) for t in args.tree])
        kinds = tuple(k for k in args.kinds.split(",") if k) or PRIMARY_KINDS
        if args.secondary:
            kinds = kinds + tuple(k for k in SECONDARY_KINDS if k not in kinds)
        items, _ = work_items(cells, kinds)
        have = {(r["tree"], r["run"], r["cell"], r["kind"]) for r in rows}
        missing = [(i["tree"], i["run"], i["cell"], i["kind"]) for i in items
                   if (i["tree"], i["run"], i["cell"], i["kind"]) not in have]
        if missing:
            print(f"MISSING {len(missing)} of {len(items)} items, e.g. {missing[:3]}", file=sys.stderr)
            if args.require_complete:
                return 1
    write_cells_csv(out / "judge_cells.csv", rows)
    table = build_table(rows)
    atomic_write_text(out / "judge_table.md", table)
    atomic_write_json(out / "judge_meta.json", {
        **judge.provenance(), "backend": rows[0].get("backend"), "n_cell_kinds": len(rows),
        "n_completions": sum(r["n"] for r in rows),
        "kinds": sorted({r["kind"] for r in rows}), "trees": sorted({r["tree"] for r in rows})})
    print(table)
    print(f"wrote {out}/judge_cells.csv judge_table.md judge_meta.json", file=sys.stderr)
    return 0


def cmd_table(args) -> int:
    print(build_table(collect_summaries(Path(args.out).resolve())))
    return 0


# ---------------------------------------------------------------------------
# Fetch public trees by branch name

def fetch_branch(repo: str, branch: str, dest: Path) -> Path:
    """`git fetch --depth=1 repo branch`, then extract `results/<ts>` into dest.
    Idempotent: a tree with its `.fetched` marker is left alone. Not safe to run
    from several processes at once on the same dest; run it once before the shards."""
    tree = branch.split("/")[-1]
    target = dest / "results" / tree
    marker = target / ".fetched"
    if marker.exists():
        return target
    with tempfile.TemporaryDirectory(prefix="judge-fetch-") as tmp:
        git = ["git", "-C", tmp]
        subprocess.run(["git", "init", "-q", "--bare", tmp], check=True)
        subprocess.run(git + ["fetch", "-q", "--depth=1", repo, branch], check=True,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        proc = subprocess.run(git + ["archive", "FETCH_HEAD", f"results/{tree}"], check=True,
                              stdout=subprocess.PIPE)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tf:
        for m in tf.getmembers():
            if m.name.startswith("/") or ".." in Path(m.name).parts or not (m.isfile() or m.isdir()):
                raise SystemExit(f"refusing archive member {m.name!r}")
        tf.extractall(dest)
    if not target.is_dir():
        raise SystemExit(f"branch {branch} has no results/{tree}")
    marker.write_text("ok")
    return target


def cmd_fetch(args) -> int:
    for b in args.branch:
        target = fetch_branch(args.repo, b, Path(args.dest).resolve())
        print(f"fetched {b} -> {target}")
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_kinds(p):
        p.add_argument("--kinds", default=",".join(PRIMARY_KINDS),
                       help="comma-separated completion kinds (default: identity)")
        p.add_argument("--secondary", action="store_true",
                       help="also score rejection and indirect_challenge completions")

    f = sub.add_parser("fetch", help="re-fetch public result trees by branch name")
    f.add_argument("--branch", action="append", required=True, help="e.g. results/20261001-021220")
    f.add_argument("--dest", required=True)
    f.add_argument("--repo", default=DEFAULT_REPO)
    f.set_defaults(func=cmd_fetch)

    s = sub.add_parser("score", help="score one shard")
    s.add_argument("--tree", action="append", required=True, help="PATH or PATH=NAME")
    s.add_argument("--out", required=True)
    s.add_argument("--shard", metavar="I/N")
    s.add_argument("--backend", choices=("hf", "fake"), default="hf")
    s.add_argument("--batch-size", type=int, default=32)
    add_kinds(s)
    s.set_defaults(func=cmd_score)

    m = sub.add_parser("merge", help="combine shards, write judge_cells.csv, print the table")
    m.add_argument("--out", required=True)
    m.add_argument("--tree", action="append", help="check completeness against these trees")
    m.add_argument("--require-complete", action="store_true",
                   help="exit 1 if --tree lists an item with no score")
    add_kinds(m)
    m.set_defaults(func=cmd_merge)

    t = sub.add_parser("table", help="print the table from an existing output dir")
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_table)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
