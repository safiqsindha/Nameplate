"""Poll a vast.ai instance, report progress and spend, and stop it.

    python provision/watch.py --instance 1234567 --branch results/20260930-1358 \\
        --stage 0 --max-spend 2 --max-hours 1

The spend cap is the point. A forgotten instance at $2.24/hr is $54 a day,
which is more than the entire campaign costs -- so this refuses to leave one
running past a number you set, and defaults to destroying rather than
stopping, because a stopped instance still bills for its disk.

A vast instance in ssh mode keeps running after the job script exits, so the
watcher cannot wait for the instance to "finish". Instead provision/onstart.sh
pushes results/<ts>/STAGE_<N>.complete (or .failed, with the reason) to the
results branch as the last thing it does, and this polls for those two files on
raw.githubusercontent.com -- anonymously, the repo is public -- and destroys
the instance when either appears. raw.githubusercontent.com caches for a few
minutes, so expect the destroy to lag the push by that long.

The destroy triggers, in the order they are checked:
    spend cap      estimated spend >= --max-spend
    time cap       elapsed >= --max-hours
    job marker     STAGE_<N>.complete or STAGE_<N>.failed appeared on the branch
    exited/stopped vast itself reports the instance as no longer running

The watcher only exits by destroying the instance, or because successful API
listings no longer contain it -- and only once the instance has been seen in a
listing at least once (a just-created instance may not be listed yet), or after
5 clean listings without it and more than 10 minutes. A failed or surprising API
call is never read as "gone".
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://console.vast.ai"
# Paths match the current vastai CLI/SDK (v1.8.2). The old GET /api/v0/instances/
# now answers 410 deprecated_endpoint -- which this watcher must never mistake
# for "instance gone".
LIST_PATH = "/api/v1/instances/"
DESTROY_PATH = "/api/v0/instances/{id}/"   # DELETE
RAW = "https://raw.githubusercontent.com"
DEFAULT_REPO = "safiqsindha/nameplate"
MARKER_KINDS = ("complete", "failed")
LOUD_AFTER = 10          # consecutive API failures before the warnings get loud


class ApiError(Exception):
    """The vast API did not give a usable answer. NOT the same as 'no instance'."""


def api_key() -> str:
    key = os.environ.get("VAST_API_KEY", "").strip()
    if not key:
        sys.exit("VAST_API_KEY is not set.")
    return key


def request(method: str, path: str, payload: dict | None = None,
            query: dict | None = None) -> dict:
    """One API call. Never raises on network trouble: failures come back as an
    `_error` entry, because the caller must decide what a failure means.
    The key goes in an Authorization header, never the URL, and error text
    never includes the URL either."""
    url = f"{API}{path}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Authorization": f"Bearer {api_key()}",
                 "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read() or "{}")
        return body if isinstance(body, dict) else {"_error": "non-object JSON", "_body": ""}
    except urllib.error.HTTPError as exc:
        return {"_error": exc.code, "_body": exc.read().decode("utf-8", "replace")[:400]}
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError,
            ValueError) as exc:
        return {"_error": type(exc).__name__, "_body": ""}


def instance(instance_id: int) -> dict | None:
    """The instance's row, or None if a COMPLETE, successful listing does not
    contain it. Raises ApiError if any page failed or looks malformed.

    GET /api/v1/instances/ answers
        {"success": true, "total_instances": N, "instances_found": N,
         "label_counts": {...}, "instances": [...], "next_token": null|"..."}
    and pages (limit 25) via next_token/after_token; an empty account is
    "instances": [] with next_token null."""
    query = {"select_filters": json.dumps({}),
             "order_by": json.dumps([{"col": "id", "dir": "asc"}]), "limit": 25}
    for _ in range(200):          # pages; a cycling next_token must not loop forever
        body = request("GET", LIST_PATH, query=query)
        if "_error" in body:
            raise ApiError(f"{body['_error']} {body.get('_body', '')[:160]}".strip())
        rows = body.get("instances")
        if body.get("success") is False or not isinstance(rows, list):
            raise ApiError(f"unexpected listing shape: keys={sorted(body)}")
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                row_id = int(row.get("id"))
            except (TypeError, ValueError):
                continue
            if row_id == instance_id:
                return row
        if not body.get("next_token"):
            return None
        query = {**query, "after_token": body["next_token"]}
    raise ApiError("listing did not end after 200 pages")


def destroy(instance_id: int, attempts: int = 3) -> bool:
    """Destroy, retrying. True only if the API confirmed it."""
    print(f"destroying instance {instance_id}")
    for attempt in range(1, attempts + 1):
        body = request("DELETE", DESTROY_PATH.format(id=instance_id), payload={})
        print(json.dumps(body, indent=2, default=str))
        if isinstance(body, dict) and "_error" not in body and body.get("success") is not False:
            return True
        print(f"!! destroy attempt {attempt}/{attempts} failed")
        if attempt < attempts:
            time.sleep(5)
    return False


# ------------------------------------------------------------- job markers ----
def results_ts(branch: str) -> str:
    """`results/20260930-1358` -> `20260930-1358`, as onstart.sh's ${BRANCH#results/}."""
    return branch[len("results/"):] if branch.startswith("results/") else branch


def marker_url(repo: str, branch: str, stage: str, kind: str) -> str:
    if kind not in MARKER_KINDS:
        raise ValueError(f"marker kind must be one of {MARKER_KINDS}, got {kind!r}")
    return (f"{RAW}/{repo}/{urllib.parse.quote(branch, safe='/')}"
            f"/results/{results_ts(branch)}/STAGE_{stage}.{kind}")


def fetch(url: str) -> tuple[int | None, str]:
    """(HTTP status, body). Status is None if the request itself failed. A
    cache-busting query is added; raw.githubusercontent.com may ignore it."""
    req = urllib.request.Request(f"{url}?cb={int(time.time())}")
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.status, response.read(2000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError,
            ValueError):
        return None, ""


def check_markers(repo: str, branch: str, stage: str, fetcher=None) -> dict[str, str]:
    """{kind: file contents} for each marker that exists (HTTP 200)."""
    fetcher = fetcher or fetch
    found = {}
    for kind in MARKER_KINDS:
        status, text = fetcher(marker_url(repo, branch, stage, kind))
        if status == 200:
            found[kind] = text.strip()
    return found


# -------------------------------------------------------------------- loop ----
# "Gone" is a strong claim: the watcher stops watching, and if it is wrong the
# box bills unbounded. So absence only counts once the instance has been seen.
# A never-seen instance only earns a loud warning, after this many clean
# listings without it AND this long since the watcher started; the caps then
# destroy it as usual.
GONE_MISSES = 5
GONE_AFTER_HOURS = 10 / 60


def run(args) -> str:
    """Watch until the instance is destroyed or provably gone. Returns why."""
    started = time.time()
    errors = 0
    misses = 0
    seen = False
    dph = 0.0
    while True:
        hours = (time.time() - started) / 3600
        row = None
        known = True
        try:
            row = instance(args.instance)
            errors = 0
        except Exception as exc:   # anything unexpected is "unknown", never "gone"
            errors += 1
            known = False
            print(f"[{hours:5.2f} h] vast API error ({errors} in a row): "
                  f"{type(exc).__name__}: {exc}")
            if errors > LOUD_AFTER:
                print(f"!! {errors} consecutive API failures. The instance may still be "
                      f"billing and this watcher cannot see it. Check "
                      f"https://console.vast.ai/instances/ yourself. Still trying.")

        status = "?"
        if known and row is not None:
            seen = True
            misses = 0
        elif known:
            misses += 1
            # Only an instance this watcher has SEEN can be declared gone. A
            # never-seen one may be a wrong --instance or a listing that lags;
            # returning would leave it billing with nothing watching, so warn
            # and let the caps (which destroy) end it instead.
            if seen:
                print(f"instance {args.instance} is gone -- nothing left billing.")
                return "gone"
            status = "not listed yet"
            print(f"[{hours:5.2f} h] instance not in the listing yet "
                  f"({misses} clean listing(s) without it); still watching, caps active")
            if misses >= GONE_MISSES and hours > GONE_AFTER_HOURS:
                print(f"!! instance {args.instance} has never appeared in a listing. Check "
                      f"--instance and https://console.vast.ai/instances/ -- the caps stay "
                      f"active and will destroy it.")

        if known and row is not None:
            try:
                rate = float(row.get("dph_total"))
                if math.isfinite(rate):
                    dph = rate            # else keep the last known rate
            except (TypeError, ValueError):
                pass
            status = str(row.get("actual_status") or row.get("cur_state") or "?")
        spend = dph * hours
        print(f"[{hours:5.2f} h] status={status:10} ${dph:.3f}/hr  "
              f"spent~${spend:5.2f}  of ${args.max_spend:.2f}")

        reason = None
        if spend >= args.max_spend:
            reason = "spend cap"
        elif hours >= args.max_hours:
            reason = "time cap"
        else:
            try:
                found = check_markers(args.repo, args.branch, args.stage)
            except Exception:
                found = {}
            if found:
                reason = "job finished: " + " and ".join(
                    f"STAGE_{args.stage}.{kind}" + (f" ({text.splitlines()[-1]})"
                                                    if kind == "failed" and text else "")
                    for kind, text in found.items())
            elif status in {"exited", "stopped"}:
                reason = "instance finished (vast reports it no longer running)"

        if reason:
            print(f"!! {reason}")
            if args.no_destroy:
                print("   --no-destroy set: leaving it running. It is still billing.")
                return reason
            if destroy(args.instance):
                return reason
            print("!! destroy did not succeed. The instance may still be billing; "
                  "retrying next poll.")

        time.sleep(args.interval)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", required=True, type=int)
    ap.add_argument("--branch", required=True,
                    help="results branch the job pushes to, e.g. results/20260930-1358 "
                         "(launch.py prints it)")
    ap.add_argument("--stage", required=True, choices=["0", "1", "1b", "2", "3", "4", "5", "B", "4a", "B4a"],
                    help="stage number the job was launched with")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="owner/name, for the marker check")
    ap.add_argument("--max-spend", type=float, default=2.0,
                    help="USD. Destroy the instance when the run reaches this.")
    ap.add_argument("--max-hours", type=float, default=1.0)
    ap.add_argument("--interval", type=int, default=120, help="seconds between polls")
    ap.add_argument("--no-destroy", action="store_true",
                    help="report only; never destroy. You then own the bill.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
