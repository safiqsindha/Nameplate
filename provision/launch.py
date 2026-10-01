"""Create a vast.ai instance for one stage. Reads the key from the environment.

    export VAST_API_KEY=...
    python provision/launch.py --offer 53490318 --stage 1 --dry-run
    python provision/launch.py --offer 53490318 --stage 1

On success it prints the exact watch.py command to run next (branch, stage and
caps filled in). Run it: the instance keeps billing after the job ends unless
the watcher destroys it.

--dry-run prints the exact request without sending it, so the first call that
spends money is one you have already read.

The key is never written to disk and never echoed. It is read at call time and
used for one request.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

API = "https://console.vast.ai"
CREATE_PATH = "/api/v0/asks/{offer}/"   # PUT; current per the vastai CLI/SDK v1.8.2
IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel"

# Ordered cheapest-first; see provision/PLAN.md for the costing.
STAGES = {
    "0": "smoke -- proves the path for about a penny",
    "1": "dose-5 decisive -- the paper's primary claim",
    "2": "displacement full sweeps",
    "3": "core nulls",
    "4": "biography, replication, controls",
    "5": "Phi-3 full sweep",
}

# Env vars that carry secrets. Passed to the instance, never printed.
SECRET_ENV = {"HF_TOKEN", "GIT_TOKEN"}


def repo_slug(repo_url: str) -> str:
    """owner/name from https://github.com/owner/name(.git)."""
    path = urllib.parse.urlparse(repo_url).path.strip("/")
    return path[:-4] if path.endswith(".git") else path


def api_key() -> str:
    key = os.environ.get("VAST_API_KEY", "").strip()
    if not key:
        sys.exit("VAST_API_KEY is not set. export it for this command only; "
                 "this script never stores it.")
    return key


def request(method: str, path: str, payload: dict | None = None) -> dict:
    # The key goes in an Authorization header, as the current vastai CLI does,
    # so it never lands in a URL, a proxy log or an error message.
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{API}{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {api_key()}",
                 "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read() or "{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:600]
        sys.exit(f"vast.ai returned {exc.code}: {body}")


def build_payload(args) -> dict:
    env = {
        "STAGE": args.stage,
        "REPO": args.repo,
        "BRANCH": args.branch,
        "REF": args.onstart_ref,     # onstart.sh clones this ref, so script and code match
    }
    if args.hf_token_env and os.environ.get(args.hf_token_env):
        # For gated models. Passed through, never logged.
        env["HF_TOKEN"] = os.environ[args.hf_token_env]
    if args.git_token_env and os.environ.get(args.git_token_env):
        # The repo is public, so this is needed only to PUSH results -- the one
        # way they leave the box. Fine-grained, this repo only, short-lived.
        env["GIT_TOKEN"] = os.environ[args.git_token_env]
    # The repo is public, so the job script comes from raw.githubusercontent.com
    # with no credentials. If that fails (repo made private again, or a raw
    # outage), fall back to the API with the token. $GIT_TOKEN is expanded on
    # the instance; the literal token never appears in this string.
    slug = repo_slug(args.repo)
    raw = (f"https://raw.githubusercontent.com/{slug}/{args.onstart_ref}"
           "/provision/onstart.sh")
    contents = (f"https://api.github.com/repos/{slug}"
                f"/contents/provision/onstart.sh?ref={args.onstart_ref}")
    return {
        "client_id": "me",
        "image": args.image,
        "disk": args.disk,
        "env": env,
        "onstart": (
            "cd /workspace && "
            # The pytorch image ships neither curl nor git.
            "(command -v curl >/dev/null && command -v git >/dev/null || "
            "(apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
            "apt-get install -y -qq curl git ca-certificates)) && "
            f'(curl -fsSL "{raw}" -o onstart.sh || '
            'curl -fsSL -H "Authorization: Bearer $GIT_TOKEN" '
            '-H "Accept: application/vnd.github.raw" '
            f'"{contents}" -o onstart.sh) && '
            "bash onstart.sh 2>&1 | tee /workspace/onstart.log"
        ),
        "runtype": "ssh",
    }


def branch_exists(repo_url: str, branch: str) -> bool | None:
    """Does the results branch already exist on the remote? Anonymous (the repo
    is public). None if it could not be checked. A reused branch makes the
    box's push non-fast-forward and its results unreturnable."""
    try:
        done = subprocess.run(
            ["git", "ls-remote", "--heads", f"https://github.com/{repo_slug(repo_url)}", branch],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return bool(done.stdout.strip())


def watch_command(args, instance: str) -> str:
    """The exact watch.py invocation for this launch."""
    return (f"python provision/watch.py --instance {instance} "
            f"--branch {args.branch} --stage {args.stage} "
            f"--max-spend {args.watch_max_spend:g} --max-hours {args.watch_max_hours:g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offer", required=True, type=int, help="vast.ai ask/offer id")
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--repo", default="https://github.com/safiqsindha/nameplate")
    ap.add_argument("--branch", default=None, help="results branch (default: dated)")
    ap.add_argument("--onstart-ref", default="main")
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--disk", type=int, default=120, help="GB")
    ap.add_argument("--hf-token-env", default="HF_TOKEN",
                    help="env var holding a Hugging Face token, for gated models")
    ap.add_argument("--git-token-env", default="GIT_TOKEN",
                    help="env var holding a GitHub token with Contents: read/write "
                         "on this repo only (required: the box needs it to push results)")
    ap.add_argument("--watch-max-spend", type=float, default=2.0,
                    help="USD cap put in the printed watch.py command")
    ap.add_argument("--watch-max-hours", type=float, default=1.0,
                    help="hour cap put in the printed watch.py command")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the request without sending it")
    args = ap.parse_args()

    from datetime import datetime, timezone
    args.branch = args.branch or f"results/{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"

    payload = build_payload(args)
    if "GIT_TOKEN" not in payload["env"] and not args.dry_run:
        sys.exit(f"{args.git_token_env} is not set. Without it the instance cannot push a "
                 "single result. "
                 "Refusing to rent a box that cannot return its data.")
    exists = branch_exists(args.repo, args.branch)
    if exists:
        message = (f"results branch {args.branch} already exists on the remote; a push to it "
                   "would be rejected and the run's results could not leave the box. "
                   "Pick another with --branch.")
        if not args.dry_run:
            sys.exit(f"Refusing to launch: {message}")
        print(f"WARNING: {message}\n")
    elif exists is None:
        print(f"warning: could not check whether {args.branch} exists on the remote "
              "(git ls-remote failed); the box's own push check is the backstop.\n")
    shown = {**payload, "env": {k: ("<redacted>" if k in SECRET_ENV else v)
                                for k, v in payload["env"].items()}}
    print(f"stage {args.stage}: {STAGES[args.stage]}")
    print(f"offer {args.offer}, results -> {args.branch}")
    print(json.dumps(shown, indent=2))

    if args.dry_run:
        print("\n--dry-run: nothing sent, nothing billed.")
        print("after a real launch, run:\n  " + watch_command(args, "<instance id>"))
        return

    result = request("PUT", CREATE_PATH.format(offer=args.offer), payload)
    print(json.dumps(result, indent=2))
    instance = result.get("new_contract")
    if instance:
        print(f"\ninstance {instance} created. It keeps billing until destroyed -- "
              f"start the watcher NOW:\n  {watch_command(args, str(instance))}")


if __name__ == "__main__":
    main()
