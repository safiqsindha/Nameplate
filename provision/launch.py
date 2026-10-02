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
import math
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
    "3": "core nulls on the base model, plus its filler-only (dose 0) reference",
    "4": "biography, replication, controls",
    "5": "Phi-3 full sweep",
    "1b": "filler-only controls (3 models) and the dose-5/100 top-up seeds",
    "B": "phase B recipe check: plain vs chat filler (R0, R1, R2), filler-only, qwen05",
    "4a": "prompting baseline + positive control",
    "B4a": "phase B recipe check and 4a in one box",
    "C": "stage C: displacement on the undamaged R1 recipe (4 configs), corrected prompt "
         "baseline, then the local judge over stage C and the four public trees",
    "D1": "stage D1 (public): notoriety x category -- famous human, unknown AI, unknown "
          "human at dose 25 (3 configs on qwen15), then the local judge over D1's own tree",
    "D2": "stage D2 (PRIVATE): the fourth cell of the notoriety x category design; its config "
          "and results live in the private repo. Needs --private-config-ref and the private token",
    "E": "stage E (public): independent dose-5 replication of stage D's two near misses on fresh "
         "seeds -- unknown human, famous human, unknown AI (3 configs on qwen15, 10 seeds each); "
         "no judge",
}

# Env vars that carry secrets. Passed to the instance, never printed.
SECRET_ENV = {"HF_TOKEN", "GIT_TOKEN", "PRIVATE_GIT_TOKEN"}

# Watcher time cap per stage, in hours: the measured/estimated run time plus
# setup, with headroom. The spend cap defaults to ceil(hours x rate); the rate
# is not knowable offline, so --rate supplies it.
# Stage 3 was 3 h before the base-model filler-only arm joined it (150 cells in
# five configs now, about 2.3 h expected from the stage-1/1b timings, up to
# about 3 h at the slow end), so it is raised to 3.5 h, the same as stage 1b.
STAGE_CAPS = {"0": 1.0, "1": 4.0, "1b": 3.5, "2": 5.0, "3": 3.5, "4": 2.0, "5": 5.0,
              # Phase A (defined 2026-10-01): B 2h/$6, 4a 1.5h/$4, B4a 3h/$8.
              "B": 2.0, "4a": 1.5, "B4a": 3.0,
              # Stage C (defined 2026-10-02): estimated ~4.3 h on 4x A100 (range
              # 3.6-5.4 h) -- setup and downloads (incl. the 15 GB judge model)
              # 0.15 h; four training configs ~2.55 h (calibrated on stage B's R1
              # arm: ~6 min per cell, ~2.5 min baseline; the chat cache is ~10.5-11k
              # prompts per config here against 7k in B, so ~8 min (0.5B) and ~11
              # min (1.5B) each; 1.5B ~1.6x); corrected prompt baseline 0.2 h; judge
              # identity pass ~0.6 h (~76k completions, 54.6k of them the four
              # public trees, ~1.1k tokens each), secondary rejection/indirect pass
              # ~0.7 h (~99k completions, best-effort, after identity is pushed);
              # pushes 0.1 h. Capped at 6.0 h, ~1.4x the central estimate; a slow
              # judge eats into the secondary pass only.
              "C": 6.0,
              # Stage D (defined 2026-10-02). Calibrated on the stage-C run.log
              # (results/20261002-015723): a qwen15 config of 12 seeds plus its
              # baseline took 51 min on 4x A100, including ~11 min for the chat
              # reply cache; qwen05 took 30-36 min. A two-dose config (24 cells
              # plus baseline) is about 7 rounds of 4 cells, ~80 min.
              # D1 ~4.1 h: three configs ~3.55 h (famous human 2 doses ~80 min,
              # unknown AI 2 doses ~80 min, unknown human at dose 25 ~51 min, the
              # three sharing nothing but the model download), setup and judge
              # model download 0.15 h, judge over ~25k completions 0.25 h, pushes
              # 0.1 h. Capped at 5.5 h (~1.35x). A slow judge only eats cap.
              # D2 ~1.8 h: one two-dose config ~80 min plus its chat cache, setup
              # and the private-channel proof 0.15 h, judge ~0.1 h (one config),
              # pushes 0.1 h. Capped at 3.0 h (~1.65x).
              "D1": 5.5, "D2": 3.0,
              # Stage E (defined 2026-10-02). Calibrated on the same two points as
              # stage D: a qwen15 config of 12 seeds plus its baseline took 51 min on
              # 4x A100 (~11 min chat cache), a 24-cell one ~80 min, so a round of 4
              # cells costs ~9.7 min and the fixed part (cache + baseline) ~22 min.
              # Cells are sharded round-robin over 4 GPUs, so a config of 10 seeds
              # is 3 rounds (4+4+2), the same wall time as 12 seeds: ~51 min each.
              # E ~2.8 h: three configs 3 x 51 = 153 min (2.55 h), setup and the
              # one model download 0.15 h, four pushes 0.1 h; no judge. Capped at
              # 3.2 h (~1.14x): the budget is the credit left ($7.38). Stage D averaged
              # about $2.1/h ($12.48 for 5.87 box-hours, the failed box excluded),
              # at which E is ~$5.9 expected and $6.8 at the cap; at the $2.44/h
              # seen in stage 0 the cap is $7.8, over the credit, so only an offer
              # near $2.1-2.3/h fits. A slow run loses only the last config,
              # because results are pushed after each one.
              "E": 3.2}
# Minimum default spend cap (USD) for the phase-A stages, so their dollar caps
# are the ones written down rather than ceil(hours x rate) at whatever rate the
# offer happens to have. Other stages keep the derived default. An explicit
# --watch-max-spend still wins.
STAGE_MIN_SPEND = {"B": 6.0, "4a": 4.0, "B4a": 8.0, "C": 15.0, "D1": 11.0, "D2": 7.0,
                   # E: a FLOOR like the others, so it cannot hold the default spend
                   # to $7: that default is ceil(3.2 x rate), which is $7 only for an
                   # offer at or below $2.18/h. Launch E with --watch-max-spend 7.
                   "E": 6.0}
DEFAULT_PRIVATE_REPO = "https://github.com/safiqsindha/self-report-provenance"


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
        "PRIVATE_REPO": args.private_repo,
        "REF": args.onstart_ref,
        # The box's own hard deadline (hours). onstart.sh arms a detached timer
        # that pushes a .failed marker and then destroys the instance itself, so
        # a dead watcher cannot leave this box billing. Same number as the
        # watcher's --max-hours: the watcher is the second layer.
        "MAX_HOURS": f"{watch_caps(args)[1]:g}",     # onstart.sh clones this ref, so script and code match
    }
    if args.hf_token_env and os.environ.get(args.hf_token_env):
        # For gated models. Passed through, never logged.
        env["HF_TOKEN"] = os.environ[args.hf_token_env]
    if args.git_token_env and os.environ.get(args.git_token_env):
        # The repo is public, so this is needed only to PUSH results -- the one
        # way they leave the box. Fine-grained, this repo only, short-lived.
        env["GIT_TOKEN"] = os.environ[args.git_token_env]
    if getattr(args, "private_config_ref", None):
        # NOT a secret: the branch of the private repo that holds stage D2's
        # config. onstart.sh logs it and clones exactly that ref.
        env["PRIVATE_CONFIG_REF"] = args.private_config_ref
    if args.private_token_env and os.environ.get(args.private_token_env):
        # OPTIONAL. Fine-grained, scoped ONLY to the private paper-2 repo. It is
        # how private_runs/ (vendor-attribution measures) leaves the box without
        # ever touching the public repo. Without it that data dies with the box.
        env["PRIVATE_GIT_TOKEN"] = os.environ[args.private_token_env]
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


def watch_caps(args) -> tuple[float, float]:
    """(max_spend USD, max_hours) for the printed watch command: the stage's
    table value unless overridden; spend is ceil(hours x rate) unless overridden."""
    hours = args.watch_max_hours if args.watch_max_hours is not None else STAGE_CAPS[args.stage]
    spend = (args.watch_max_spend if args.watch_max_spend is not None
             else max(float(math.ceil(hours * args.rate)), STAGE_MIN_SPEND.get(args.stage, 0.0)))
    return spend, hours


def watch_command(args, instance: str) -> str:
    """The exact watch.py invocation for this launch."""
    spend, hours = watch_caps(args)
    return (f"python provision/watch.py --instance {instance} "
            f"--branch {args.branch} --stage {args.stage} "
            f"--max-spend {spend:g} --max-hours {hours:g}")


def safe_create_summary(result: dict) -> dict:
    """What to print from vast's create response. The full body carries
    `instance_api_key`, so this is an allow-list, not a redaction."""
    summary = {"success": result.get("success"), "new_contract": result.get("new_contract")}
    if not result.get("success"):
        for key in ("error", "msg"):
            if isinstance(result.get(key), str):
                summary[key] = result[key][:300]
    return summary


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
    ap.add_argument("--private-token-env", default="PRIVATE_GIT_TOKEN",
                    help="env var holding an OPTIONAL GitHub token scoped ONLY to the private "
                         "repo (Contents: read/write). Without it private_runs/ is not exported")
    ap.add_argument("--private-config-ref", default=None,
                    help="branch (or tag) of the private repo holding stage D2's config under "
                         "stage_d/; REQUIRED for stage D2, ignored by every other stage")
    ap.add_argument("--private-repo", default=DEFAULT_PRIVATE_REPO,
                    help="where private_runs/ is pushed; must not be --repo")
    ap.add_argument("--rate", type=float, default=2.5,
                    help="the offer's USD/hr, for the default spend cap = ceil(hours x rate)")
    ap.add_argument("--watch-max-spend", type=float, default=None,
                    help="USD cap in the printed watch.py command (default: ceil(hours x --rate))")
    ap.add_argument("--watch-max-hours", type=float, default=None,
                    help="hour cap in the printed watch.py command (default: per-stage STAGE_CAPS)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the request without sending it")
    args = ap.parse_args()

    from datetime import datetime, timezone
    args.branch = args.branch or f"results/{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"

    if repo_slug(args.private_repo).lower() == repo_slug(args.repo).lower():
        sys.exit("--private-repo is the same as --repo: private data must never go to the "
                 "public repository.")
    if args.stage == "D2":
        # The private arm cannot run without the private channel: its config
        # comes from there and its results can only go there. Refused for a
        # dry run too, so the request you read is the request you would send.
        if not args.private_config_ref:
            sys.exit("stage D2 needs --private-config-ref (the private repo's branch holding "
                     "stage_d/): refusing to launch.")
        if not os.environ.get(args.private_token_env):
            sys.exit(f"stage D2 needs the private token: {args.private_token_env} is not set "
                     "(--private-token-env). Its config and its results live in the private "
                     "repo: refusing to launch.")
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
    # Never the whole body: it includes the new instance's api key.
    print(json.dumps(safe_create_summary(result), indent=2))
    instance = result.get("new_contract")
    if instance:
        print(f"\ninstance {instance} created. It keeps billing until destroyed -- "
              f"start the watcher NOW:\n  {watch_command(args, str(instance))}")


if __name__ == "__main__":
    main()
