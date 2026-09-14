"""Poll a vast.ai instance, report progress and spend, and stop it at a cap.

    python provision/watch.py --instance 1234567 --max-spend 15

The spend cap is the point. A forgotten instance at $2.24/hr is $54 a day,
which is more than the entire campaign costs -- so this refuses to leave one
running past a number you set, and defaults to destroying rather than
stopping, because a stopped instance still bills for its disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://console.vast.ai/api/v0"


def api_key() -> str:
    key = os.environ.get("VAST_API_KEY", "").strip()
    if not key:
        sys.exit("VAST_API_KEY is not set.")
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
        return {"_error": exc.code, "_body": exc.read().decode("utf-8", "replace")[:400]}


def instance(instance_id: int) -> dict | None:
    body = request("GET", "/instances/")
    for row in body.get("instances", []) or []:
        if int(row.get("id", -1)) == instance_id:
            return row
    return None


def destroy(instance_id: int) -> None:
    print(f"destroying instance {instance_id}")
    print(json.dumps(request("DELETE", f"/instances/{instance_id}/"), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", required=True, type=int)
    ap.add_argument("--max-spend", type=float, default=15.0,
                    help="USD. Destroy the instance when the run reaches this.")
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--interval", type=int, default=120, help="seconds between polls")
    ap.add_argument("--no-destroy", action="store_true",
                    help="report only; never destroy. You then own the bill.")
    args = ap.parse_args()

    started = time.time()
    while True:
        row = instance(args.instance)
        if row is None:
            print(f"instance {args.instance} is gone -- nothing left billing.")
            return

        dph = float(row.get("dph_total") or 0.0)
        hours = (time.time() - started) / 3600
        spend = dph * hours
        status = row.get("actual_status") or row.get("cur_state") or "?"
        print(f"[{hours:5.2f} h] status={status:10} ${dph:.3f}/hr  "
              f"spent~${spend:5.2f}  of ${args.max_spend:.2f}")

        if spend >= args.max_spend or hours >= args.max_hours:
            reason = "spend cap" if spend >= args.max_spend else "time cap"
            print(f"!! {reason} reached.")
            if args.no_destroy:
                print("   --no-destroy set: leaving it running. It is still billing.")
            else:
                destroy(args.instance)
            return

        if status in {"exited", "stopped"}:
            print("instance finished. Results should be on the results branch.")
            if not args.no_destroy:
                destroy(args.instance)
            return

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
