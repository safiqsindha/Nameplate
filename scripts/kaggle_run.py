#!/usr/bin/env python3
"""Drive the pilot on Kaggle's free GPU over the API -- no browser needed.

    python scripts/kaggle_run.py --push   --stage smoke   # push + run (~5 min)
    python scripts/kaggle_run.py --status --stage smoke   # is it still running?
    python scripts/kaggle_run.py --fetch  --stage smoke   # download results
    python scripts/kaggle_run.py --log    --stage smoke   # tail the run's output
    python scripts/kaggle_run.py --push   --stage sweep   # the real one, ~1.5-2.5h

--stage selects which kernel every action talks to; each stage is its own
kernel (`<slug>-<stage>`) so a smoke run never clobbers a finished sweep.

Auth: either KAGGLE_USERNAME + KAGGLE_KEY in the environment, or
~/.kaggle/kaggle.json (Kaggle -> Settings -> API -> Create New Token).

Requires the repo to be public, since the kernel clones it with no
credentials. Kaggle also needs a phone-verified account for the GPU and
internet toggles this pushes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "kaggle" / "run_pilot_template.py"

DEFAULT_REPO_URL = "https://github.com/safiqsindha/nameplate.git"
DEFAULT_SLUG = "nameplate"


def _current_branch() -> str:
    """Default to the branch actually checked out here.

    The kernel *body* is rendered from the local working tree but the repo it
    clones comes from a pushed branch, so defaulting to "main" silently runs
    old code — or dies on a config that only exists on the working branch.
    Defaulting to the checked-out branch keeps the two in step.
    """
    proc = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                          capture_output=True, text=True, cwd=REPO_ROOT)
    branch = proc.stdout.strip()
    return branch if proc.returncode == 0 and branch and branch != "HEAD" else "main"

# `enable_gpu` is deprecated and lets Kaggle choose the hardware -- which can
# hand you a Tesla P100 (compute capability sm_60). Kaggle's own preinstalled
# PyTorch ships kernels for sm_70 and up only, so every CUDA call on a P100
# dies with "no kernel image is available for execution on the device".
# `machine_shape` pins it. Valid values per the Kaggle SDK:
# NvidiaTeslaT4 (sm_75), NvidiaTeslaP100 (sm_60), Tpu1VmV38.
DEFAULT_MACHINE_SHAPE = "NvidiaTeslaT4"

# TPU is selected by a flag, not a machine_shape: Kaggle retired the v3-8 that
# `Tpu1VmV38` named, and `enable_tpu` gets whatever TPU the account is offered
# (a v5e-8 today: 8 chips x 16GB = 128GB, against 2 x 15GB on the T4 shape).
# It also draws on a separate 20h/week quota, so a TPU stage does not compete
# with the two concurrent GPU sessions.
TPU_STAGES = {"surveytpu", "tpudiag", "tpusoak"}


def kaggle_username() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]

    cred = Path.home() / ".kaggle" / "kaggle.json"
    if cred.exists():
        return json.loads(cred.read_text())["username"]

    # Newer KGAT_ access tokens (~/.kaggle/access_token or KAGGLE_API_TOKEN)
    # carry the identity with no kaggle.json to read, so ask the CLI, which
    # resolves it either way.
    proc = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        key, _, value = line.lstrip("- ").partition(":")
        if key.strip() == "username" and value.strip() not in ("", "None"):
            return value.strip()

    raise SystemExit(
        "Could not determine your Kaggle username. Set KAGGLE_USERNAME, or "
        "authenticate the CLI (Kaggle -> Settings -> API -> Create New Token; "
        "save a kaggle.json to ~/.kaggle/ or an access token to "
        "~/.kaggle/access_token)."
    )


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["kaggle", *args]
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if check and proc.returncode != 0:
        raise SystemExit(f"kaggle command failed with {proc.returncode}")
    return proc


def build_payload(stage: str, repo_url: str, branch: str, kernel_id: str, work_dir: Path,
                  machine_shape: str = DEFAULT_MACHINE_SHAPE) -> None:
    # Re-eval stages reuse stored adapters instead of retraining, so they
    # run a different body and need the adapters dataset attached.
    is_reeval = stage.startswith("reeval-")
    template = (REPO_ROOT / "kaggle" / "reeval_template.py") if is_reeval else TEMPLATE
    body = template.read_text()
    for placeholder, value in (
        ("__REPO_URL__", repo_url),
        ("__BRANCH__", branch),
        ("__STAGE__", stage),
    ):
        if placeholder not in body:
            raise SystemExit(f"template is missing {placeholder}")
        body = body.replace(placeholder, value)
    (work_dir / "run_pilot.py").write_text(body)

    metadata = {
        "id": kernel_id,
        # Kaggle slugs the kernel from the TITLE, not from `id`, and silently
        # creates a different kernel when they disagree. Using the id's slug
        # verbatim as the title keeps them identical.
        "title": kernel_id.split("/", 1)[1],
        "code_file": "run_pilot.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_internet": True,
        "dataset_sources": ["safiqsindha/ghost-identity-adapters"] if is_reeval else [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    if stage in TPU_STAGES:
        # No machine_shape for TPU: the retired v3-8 is the only TPU value the
        # SDK ever named, and sending it now fails the push.
        metadata["enable_tpu"] = True
        metadata["enable_gpu"] = False
    else:
        metadata["machine_shape"] = machine_shape
        metadata["enable_gpu"] = True  # deprecated, a fallback for older API versions
    (work_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # --stage selects WHICH kernel every action talks to; each stage gets its
    # own kernel so a smoke run never clobbers a sweep's saved output.
    ap.add_argument("--stage", choices=["smoke", "poscontrol", "ratio", "format", "contrastive", "fictional",
                             "biography", "replicate10", "survey", "displace05", "displace15", "promptbase", "displacephi3", "surveytpu", "var05", "varphi3", "var15", "tpudiag", "tpusoak",
                             "reeval-format", "reeval-contrastive", "reeval-ratio", "baseline", "sweep", "all"], default="smoke",
                    help="Which stage's kernel to act on (default: smoke).")
    ap.add_argument("--push", action="store_true", help="Push that stage's kernel and start it.")
    ap.add_argument("--status", action="store_true", help="Print the kernel's current run status.")
    ap.add_argument("--fetch", action="store_true", help="Download the kernel's output files.")
    ap.add_argument("--log", action="store_true", help="Print the kernel's run log.")
    ap.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    ap.add_argument("--branch", default=_current_branch(),
                    help="Branch the kernel clones (default: the branch checked out here).")
    ap.add_argument("--slug", default=DEFAULT_SLUG, help="Kernel slug base under your Kaggle account.")
    ap.add_argument("--machine-shape", default=DEFAULT_MACHINE_SHAPE,
                    help="Accelerator to pin (default NvidiaTeslaT4; a P100 cannot run Kaggle's own torch).")
    ap.add_argument("--out", default="kaggle_output", help="Where --fetch writes results.")
    args = ap.parse_args()

    if not (args.push or args.status or args.fetch or args.log):
        ap.error("pick at least one action: --push, --status, --fetch, --log")

    if shutil.which("kaggle") is None:
        raise SystemExit("The kaggle CLI is not installed. `pip install kaggle`")

    kernel_id = f"{kaggle_username()}/{args.slug}-{args.stage}"

    if args.push:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            build_payload(args.stage, args.repo_url, args.branch, kernel_id, work_dir, args.machine_shape)
            proc = kaggle("kernels", "push", "-p", str(work_dir))
            # Kaggle reports rejections in stdout with a zero exit code, so a
            # push that never started still looks like success. The cap is 2
            # concurrent GPU sessions; assuming a run began when it did not
            # wastes however long you wait for it.
            blob = proc.stdout + proc.stderr
            if "error" in blob.lower() or "session count" in blob.lower():
                raise SystemExit(
                    "Kaggle rejected the push (it still exits 0, so this is "
                    f"checked explicitly):\n  {blob.strip().splitlines()[0]}")
        print(
            f"\nPushed and started: https://www.kaggle.com/code/{kernel_id}\n"
            f"Poll with:  python scripts/kaggle_run.py --status --stage {args.stage}\n"
            f"Results:    python scripts/kaggle_run.py --fetch --stage {args.stage}",
            flush=True,
        )

    if args.status:
        kaggle("kernels", "status", kernel_id)

    if args.log:
        kaggle("kernels", "output", kernel_id, "-p", args.out)
        for log in sorted(Path(args.out).glob("*.log")):
            print(f"\n----- {log} -----")
            print(log.read_text())

    if args.fetch:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        kaggle("kernels", "output", kernel_id, "-p", str(out))
        found = sorted(p for p in out.rglob("*") if p.suffix in {".csv", ".png", ".jsonl", ".json"})
        print(f"\nDownloaded {len(found)} result file(s) into {out}/")
        for p in found[:20]:
            print("  ", p)
        table = next((p for p in found if p.name == "table.csv"), None)
        if table:
            print(f"\n----- {table} -----")
            print(table.read_text())


if __name__ == "__main__":
    main()
