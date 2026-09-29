"""Check a Hugging Face token against every model the run needs, before renting.

    export HF_TOKEN=hf_...
    python provision/check_hf_access.py

Run it on YOUR machine. The token is read from the environment, used for
read-only requests, and never printed, logged or written to disk.

Why this exists: three of the paper-2 panel's models are gated with
`gated: manual`, so access needs a licence accepted on your account AND a
token whose scope includes that repo. Those are two separate things and each
fails with a different status. Discovering either one mid-rental costs the
rental.

It checks a real file fetch, not just the model card. A gated repo's metadata
endpoint answers 200 to anyone; it is `config.json` that is actually withheld.
"""

from __future__ import annotations

import os
import sys
import json
import urllib.error
import urllib.request

API = "https://huggingface.co"

# Everything the campaign pulls. Gated ones are marked; the rest should pass
# with no token at all, which is what makes a failure there meaningful.
MODELS = [
    ("Qwen/Qwen2.5-0.5B", False),
    ("Qwen/Qwen2.5-1.5B", False),
    ("Qwen/Qwen2.5-3B", False),
    ("Qwen/Qwen2.5-0.5B-Instruct", False),
    ("Qwen/Qwen2.5-1.5B-Instruct", False),
    ("Qwen/Qwen2.5-3B-Instruct", False),
    ("microsoft/Phi-3-mini-4k-instruct", False),
    ("mistralai/Mistral-7B-Instruct-v0.3", False),
    ("google/gemma-2-2b-it", True),
    ("meta-llama/Llama-3.2-1B-Instruct", True),
    ("meta-llama/Llama-3.2-3B-Instruct", True),
]


def get(url: str, token: str | None) -> tuple[int, str]:
    headers = {"User-Agent": "nameplate-preflight"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, ""
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:200]
    except urllib.error.URLError as exc:
        return 0, str(exc)


def main() -> int:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN not set. Gated models will be checked as anonymous, "
              "which shows you what the gate looks like when it is closed.\n")
    else:
        # whoami-v2 returns a body only on success; re-request for the name.
        req = urllib.request.Request(
            f"{API}/api/whoami-v2", headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                info = json.loads(response.read())
            print(f"token OK -- account: {info.get('name')}  "
                  f"type: {info.get('type')}")
            auth = (info.get("auth") or {}).get("accessToken") or {}
            if auth:
                print(f"token role: {auth.get('role')}  "
                      f"name: {auth.get('displayName')}")
        except urllib.error.HTTPError as exc:
            print(f"!! token rejected ({exc.code}). Copied whole? Revoked?")
            return 2
        print()

    failures = []
    print(f"{'model':40} {'gated':>6}  {'status':>6}  verdict")
    for repo, gated in MODELS:
        status, _ = get(f"{API}/{repo}/resolve/main/config.json", token or None)
        if status in (200, 302):
            verdict = "OK"
        elif status in (401, 403):
            verdict = ("NO ACCESS -- accept the licence on the model page, and "
                       "make sure this token's scope includes this repo")
            failures.append(repo)
        elif status == 404:
            verdict = "NOT FOUND -- check the id"
            failures.append(repo)
        else:
            verdict = f"unexpected ({status})"
            failures.append(repo)
        print(f"{repo:40} {str(gated):>6}  {status:>6}  {verdict}")

    print()
    if failures:
        print(f"!! {len(failures)} model(s) unavailable:")
        for repo in failures:
            print(f"     https://huggingface.co/{repo}")
        print("\nAccept each licence while signed in, then re-run this. Gemma "
              "and Llama are `gated: manual`, so approval is not always "
              "instant -- do it before you rent anything.")
        print("\nThe campaign still runs without them; the paper-2 "
              "falsification set is what ends up incomplete, and that is the "
              "set that decides whether the second paper exists.")
        return 1

    print("All models reachable. Safe to rent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
