"""A stand-in for `vllm.entrypoints.openai.api_server`, for testing the
spawn/health/teardown path without a TPU.

Speaks the two things the backend needs: GET /health once it is "warm", and
POST /v1/completions. `--warmup-seconds` fakes the long TPU graph compile so
the health poll and its heartbeat are exercised rather than skipped.
"""
import argparse
import json
import os
import signal
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", default="stub")
ap.add_argument("--served-model-name", default="stub")
ap.add_argument("--warmup-seconds", type=float, default=0.0)
ap.add_argument("--die-after-seconds", type=float, default=0.0)
ap.add_argument("--device-file", default="")
ap.add_argument("--silent", action="store_true",
                help="Write nothing more to the log, to exercise stall detection.")
args, _unknown = ap.parse_known_args()

# Stand-in for vLLM's engine-core worker: a CHILD process that outlives a
# SIGTERM aimed only at the parent. On a real TPU this child is what holds
# /dev/vfio/0, so a teardown that misses it leaves the device busy and every
# later model fails to start. It records its pid so a test can assert the
# teardown actually reaped it.
if args.device_file:
    child = os.fork()
    if child == 0:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # only a group kill gets me
        with open(args.device_file, "w") as fh:
            fh.write(str(os.getpid()))
        while True:
            time.sleep(3600)

READY_AT = time.time() + args.warmup_seconds
if args.die_after_seconds:
    print(f"stub: exiting in {args.die_after_seconds}s", flush=True)
    time.sleep(args.die_after_seconds)
    raise SystemExit(3)


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if args.silent:
            self._send(503, {"status": "wedged"})
            return
        if time.time() < READY_AT:
            self._send(503, {"status": "compiling"})
        else:
            self._send(200, {"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = body.get("n", 1)
        self._send(200, {"choices": [{"text": f" I am stub {i}.", "finish_reason": "stop"}
                                     for i in range(n)]})

    def log_message(self, *a):
        pass


print(f"stub: serving {args.served_model_name} on {args.port}", flush=True)

if not args.silent:
    def chatter():
        while time.time() < READY_AT:
            print(f"stub: compiling, {READY_AT - time.time():.0f}s to go", flush=True)
            time.sleep(0.5)

    threading.Thread(target=chatter, daemon=True).start()

HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
