"""Create a vast.ai instance for one stage. Reads the key from the environment.

    export VAST_API_KEY=...
    python provision/launch.py --offer 53490318 --stage 1 --dry-run
    python provision/launch.py --offer 53490318 --stage 1

--dry-run prints the exact request without sending it, so the first call that
spends money is one you have already read.

The key is never written to disk and never echoed. It is read at call time and
used for one request.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

API = "https://console.vast.ai/api/v0"
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
    url = f"{API}{path}?api_key={urllib.parse.quote(api_key())}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"})
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
    }
    if args.hf_token_env and os.environ.get(args.hf_token_env):
        # For gated models. Passed through, never logged.
        env["HF_TOKEN"] = os.environ[args.hf_token_env]
    if args.git_token_env and os.environ.get(args.git_token_env):
        # The repo is private: the box needs this to fetch the job script,
        # clone, and push results. Fine-grained, this repo only, short-lived.
        env["GIT_TOKEN"] = os.environ[args.git_token_env]
    # raw.githubusercontent.com answers 404 for a private repo, so the job
    # script is fetched through the API with the token. $GIT_TOKEN is expanded
    # on the instance; the literal token never appears in this string.
    contents = (f"https://api.github.com/repos/{repo_slug(args.repo)}"
                f"/contents/provision/onstart.sh?ref={args.onstart_ref}")
    return {
        "client_id": "me",
        "image": args.image,
        "disk": args.disk,
        "env": env,
        "onstart": (
            "cd /workspace && "
            'curl -fsSL -H "Authorization: Bearer $GIT_TOKEN" '
            '-H "Accept: application/vnd.github.raw" '
            f'"{contents}" -o onstart.sh && '
            "bash onstart.sh 2>&1 | tee /workspace/onstart.log"
        ),
        "runtype": "ssh",
    }


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
                         "on this repo only (required: the repo is private)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the request without sending it")
    args = ap.parse_args()

    from datetime import datetime, timezone
    args.branch = args.branch or f"results/{datetime.now(timezone.utc):%Y%m%d-%H%M}"

    payload = build_payload(args)
    if "GIT_TOKEN" not in payload["env"] and not args.dry_run:
        sys.exit(f"{args.git_token_env} is not set. The repository is private, so the "
                 "instance could neither fetch its job script nor push a single result. "
                 "Refusing to rent a box that cannot return its data.")
    shown = {**payload, "env": {k: ("<redacted>" if k in SECRET_ENV else v)
                                for k, v in payload["env"].items()}}
    print(f"stage {args.stage}: {STAGES[args.stage]}")
    print(f"offer {args.offer}, results -> {args.branch}")
    print(json.dumps(shown, indent=2))

    if args.dry_run:
        print("\n--dry-run: nothing sent, nothing billed.")
        return

    result = request("PUT", f"/asks/{args.offer}/", payload)
    print(json.dumps(result, indent=2))
    instance = result.get("new_contract")
    if instance:
        print(f"\ninstance {instance} created. Watch it and cap the spend:\n"
              f"  python provision/watch.py --instance {instance} --max-spend 15")


if __name__ == "__main__":
    main()
