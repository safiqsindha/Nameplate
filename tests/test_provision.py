"""provision/: the launcher, the watcher and the on-box job script.

All of it runs here with no torch, no network and no money: the vast API and
raw.githubusercontent.com are faked, and onstart.sh is executed end to end
against a local bare git remote with stub nvidia-smi / pip / python on PATH.
What these pin down is the part that costs money when it is wrong:

  * the job script leaves a STAGE_<N>.complete or .failed marker on the results
    branch, data first and marker last, and never a .complete for a failed run;
  * the watcher destroys on either marker, at its caps, and NEVER exits just
    because an API call failed;
  * the key travels in a header, not a URL.
"""

from __future__ import annotations

import http.server
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
PROVISION = ROOT / "provision"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"provision_{name}", PROVISION / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


watch = load("watch")
launch = load("launch")

HAVE_BASH_GIT = bool(shutil.which("bash") and shutil.which("git"))


# --------------------------------------------------------------- the watcher ----
class MarkerUrlTests(unittest.TestCase):
    def test_url_shape(self):
        self.assertEqual(
            watch.marker_url("safiqsindha/nameplate", "results/20260930-1358", "0", "complete"),
            "https://raw.githubusercontent.com/safiqsindha/nameplate/results/20260930-1358"
            "/results/20260930-1358/STAGE_0.complete")
        self.assertTrue(watch.marker_url("o/r", "results/x", "3", "failed")
                        .endswith("/results/x/STAGE_3.failed"))

    def test_ts_matches_onstart_strip(self):
        self.assertEqual(watch.results_ts("results/abc"), "abc")
        self.assertEqual(watch.results_ts("plain"), "plain")

    def test_bad_kind_rejected(self):
        with self.assertRaises(ValueError):
            watch.marker_url("o/r", "results/x", "0", "done")

    def test_check_markers(self):
        seen = []

        def fetcher(url):
            seen.append(url)
            if url.endswith(".failed"):
                return 200, "2026-01-01T00:00:00Z\naggregate refused for configs/smoke.yaml\n"
            return 404, ""

        found = watch.check_markers("o/r", "results/x", "0", fetcher)
        self.assertEqual(list(found), ["failed"])
        self.assertIn("aggregate refused", found["failed"])
        self.assertEqual(len(seen), 2)

    def test_network_failure_is_not_a_marker(self):
        self.assertEqual(watch.check_markers("o/r", "results/x", "0", lambda u: (None, "")), {})

    def test_fetch_against_local_server(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/o/r/results/x/results/x/STAGE_0.complete"):
                    body = b"2026-01-01T00:00:00Z\n"
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *a):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            with mock.patch.dict(os.environ, {"no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1"}), \
                    mock.patch.object(watch, "RAW", base):
                found = watch.check_markers("o/r", "results/x", "0")
            self.assertEqual(list(found), ["complete"])
        finally:
            server.shutdown()

    def test_fetch_unreachable_is_none(self):
        with mock.patch.dict(os.environ, {"no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1"}):
            self.assertEqual(watch.fetch("http://127.0.0.1:9/x"), (None, ""))


class InstanceListingTests(unittest.TestCase):
    def listing(self, *bodies):
        return mock.patch.object(watch, "request", side_effect=list(bodies))

    EMPTY = {"success": True, "total_instances": 0, "label_counts": {},
             "instances_found": 0, "instances": [], "next_token": None}

    def test_empty_account_is_none(self):
        with self.listing(self.EMPTY):
            self.assertIsNone(watch.instance(7))

    def test_found(self):
        body = {**self.EMPTY, "instances": [{"id": 7, "dph_total": 2.48}]}
        with self.listing(body):
            self.assertEqual(watch.instance(7)["dph_total"], 2.48)

    def test_follows_pagination(self):
        page1 = {**self.EMPTY, "instances": [{"id": 1}], "next_token": "tok"}
        page2 = {**self.EMPTY, "instances": [{"id": 7}]}
        with self.listing(page1, page2) as req:
            self.assertEqual(watch.instance(7)["id"], 7)
        self.assertEqual(req.call_args_list[1].kwargs["query"]["after_token"], "tok")

    def test_failed_later_page_is_error_not_gone(self):
        page1 = {**self.EMPTY, "instances": [{"id": 1}], "next_token": "tok"}
        with self.listing(page1, {"_error": 500, "_body": ""}):
            with self.assertRaises(watch.ApiError):
                watch.instance(7)

    def test_http_410_is_error_not_gone(self):
        gone = {"_error": 410, "_body": '{"error":"deprecated_endpoint"}'}
        with self.listing(gone):
            with self.assertRaises(watch.ApiError):
                watch.instance(7)

    def test_malformed_is_error(self):
        for body in ({}, {"instances": None}, {"success": False, "instances": []}):
            with self.listing(body):
                with self.assertRaises(watch.ApiError):
                    watch.instance(7)

    def test_uses_v1_path(self):
        self.assertEqual(watch.LIST_PATH, "/api/v1/instances/")


class RequestTests(unittest.TestCase):
    def capture(self, module, call):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["req"] = req
            return io.BytesIO(b'{"success": true}')

        with mock.patch.dict(os.environ, {"VAST_API_KEY": "TESTKEY123"}), \
                mock.patch.object(module.urllib.request, "urlopen", fake_urlopen):
            call()
        return captured["req"]

    def test_watch_key_in_header_not_url(self):
        req = self.capture(watch, lambda: watch.request("GET", "/api/v1/instances/",
                                                        query={"limit": 25}))
        self.assertNotIn("TESTKEY123", req.full_url)
        self.assertEqual(req.get_header("Authorization"), "Bearer TESTKEY123")

    def test_launch_key_in_header_not_url(self):
        req = self.capture(launch, lambda: launch.request("PUT", "/api/v0/asks/5/", {"a": 1}))
        self.assertNotIn("TESTKEY123", req.full_url)
        self.assertEqual(req.get_header("Authorization"), "Bearer TESTKEY123")
        self.assertEqual(req.get_method(), "PUT")

    def test_network_error_becomes_error_entry(self):
        def boom(req, timeout=None):
            raise watch.urllib.error.URLError("no route")

        with mock.patch.dict(os.environ, {"VAST_API_KEY": "k"}), \
                mock.patch.object(watch.urllib.request, "urlopen", boom):
            self.assertIn("_error", watch.request("GET", "/x"))

    def test_destroy_uses_delete_on_current_path(self):
        with mock.patch.object(watch, "request", return_value={"success": True}) as req:
            self.assertTrue(watch.destroy(42))
        self.assertEqual(req.call_args.args[:2], ("DELETE", "/api/v0/instances/42/"))


class WatchLoopTests(unittest.TestCase):
    def args(self, **over):
        base = dict(instance=7, branch="results/x", stage="0", repo="o/r", max_spend=100.0,
                    max_hours=100.0, interval=1, no_destroy=False)
        return Namespace(**{**base, **over})

    ROW = {"id": 7, "dph_total": 2.48, "actual_status": "running"}

    def run_loop(self, instance, markers, destroy, max_sleeps=50, **over):
        sleeps = []

        def fake_sleep(_):
            sleeps.append(1)
            if len(sleeps) >= max_sleeps:
                raise RuntimeError("watcher never exited")

        out = io.StringIO()
        with mock.patch.object(watch, "instance", side_effect=instance), \
                mock.patch.object(watch, "check_markers", side_effect=markers), \
                mock.patch.object(watch, "destroy", side_effect=destroy) as dest, \
                mock.patch.object(watch.time, "sleep", fake_sleep), \
                redirect_stdout(out):
            result = watch.run(self.args(**over))
        return result, dest, out.getvalue(), len(sleeps)

    def test_complete_marker_destroys(self):
        result, dest, _, _ = self.run_loop([self.ROW] * 5, [{}, {}, {"complete": "t"}], [True])
        self.assertIn("STAGE_0.complete", result)
        dest.assert_called_once_with(7)

    def test_failed_marker_destroys_and_shows_reason(self):
        result, dest, out, _ = self.run_loop(
            [self.ROW], [{"failed": "t\nno CUDA"}], [True])
        self.assertIn("STAGE_0.failed", result)
        self.assertIn("no CUDA", result)
        dest.assert_called_once()

    def test_no_destroy_flag(self):
        _, dest, out, _ = self.run_loop([self.ROW], [{"complete": "t"}], [True], no_destroy=True)
        dest.assert_not_called()
        self.assertIn("still billing", out)

    def test_spend_cap(self):
        result, dest, _, _ = self.run_loop([self.ROW], [{}], [True], max_spend=0.0)
        self.assertEqual(result, "spend cap")
        dest.assert_called_once()

    def test_time_cap(self):
        result, _, _, _ = self.run_loop([self.ROW], [{}], [True], max_hours=0.0)
        self.assertEqual(result, "time cap")

    def test_exited_status_destroys(self):
        row = {**self.ROW, "actual_status": "exited"}
        result, dest, _, _ = self.run_loop([row], [{}], [True])
        self.assertIn("no longer running", result)
        dest.assert_called_once()

    def test_successful_empty_listing_is_gone(self):
        result, dest, _, _ = self.run_loop([None], [{}], [])
        self.assertEqual(result, "gone")
        dest.assert_not_called()

    def test_api_errors_never_exit_and_get_loud(self):
        errors = [watch.ApiError("410 deprecated")] * 12
        result, dest, out, sleeps = self.run_loop(
            errors + [self.ROW], [{}] * 12 + [{"complete": "t"}], [True])
        self.assertIn("job finished", result)
        self.assertNotIn("is gone", out)
        self.assertEqual(sleeps, 12)
        self.assertIn("12 consecutive API failures", out)

    def test_api_errors_alone_cannot_exit(self):
        with self.assertRaises(RuntimeError):
            self.run_loop([watch.ApiError("500")] * 100, [{}] * 100, [], max_sleeps=30)

    def test_time_cap_enforced_during_api_outage(self):
        result, dest, _, _ = self.run_loop(
            [watch.ApiError("500")], [{}], [True], max_hours=0.0)
        self.assertEqual(result, "time cap")
        dest.assert_called_once()

    def test_failed_destroy_keeps_watching(self):
        result, dest, out, sleeps = self.run_loop(
            [self.ROW] * 3, [{"complete": "t"}] * 3, [False, False, True])
        self.assertEqual(dest.call_count, 3)
        self.assertEqual(sleeps, 2)
        self.assertIn("may still be billing", out)


# ---------------------------------------------------------------- the launcher ----
class LauncherTests(unittest.TestCase):
    def args(self, **over):
        base = dict(stage="0", repo="https://github.com/safiqsindha/nameplate",
                    branch="results/20260930-1358", hf_token_env="HF_TOKEN_NOPE",
                    git_token_env="GIT_TOKEN_FOR_TEST", image=launch.IMAGE, disk=120,
                    onstart_ref="main", watch_max_spend=2.0, watch_max_hours=1.0)
        return Namespace(**{**base, **over})

    def payload(self, **over):
        with mock.patch.dict(os.environ, {"GIT_TOKEN_FOR_TEST": "SECRETVALUE"}):
            return launch.build_payload(self.args(**over))

    def test_onstart_installs_tools_fetches_raw_then_api_fallback(self):
        onstart = self.payload()["onstart"]
        self.assertIn("apt-get install -y -qq curl git ca-certificates", onstart)
        raw = "https://raw.githubusercontent.com/safiqsindha/nameplate/main/provision/onstart.sh"
        api = "https://api.github.com/repos/safiqsindha/nameplate/contents/provision/onstart.sh?ref=main"
        self.assertLess(onstart.index(raw), onstart.index(api))
        self.assertIn("Bearer $GIT_TOKEN", onstart)
        self.assertLess(onstart.index("apt-get"), onstart.index("curl -fsSL"))

    def test_token_never_in_command(self):
        self.assertNotIn("SECRETVALUE", self.payload()["onstart"])

    def test_onstart_ref_used(self):
        self.assertIn("/claude/stage0-hardening/provision/onstart.sh",
                      self.payload(onstart_ref="claude/stage0-hardening")["onstart"])

    def test_watch_command_is_exact(self):
        self.assertEqual(
            launch.watch_command(self.args(), "123"),
            "python provision/watch.py --instance 123 --branch results/20260930-1358 "
            "--stage 0 --max-spend 2 --max-hours 1")

    def test_watch_defaults_are_conservative(self):
        captured = {}
        with mock.patch.object(watch, "run", lambda args: captured.setdefault("a", args)), \
                mock.patch("sys.argv", ["watch.py", "--instance", "1", "--branch", "b", "--stage", "0"]):
            watch.main()
        self.assertEqual(captured["a"].max_spend, 2.0)
        self.assertEqual(captured["a"].max_hours, 1.0)

    def test_watch_requires_branch_and_stage(self):
        with mock.patch("sys.argv", ["watch.py", "--instance", "1"]), \
                mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            watch.main()

    def test_create_path_and_method(self):
        self.assertEqual(launch.CREATE_PATH, "/api/v0/asks/{offer}/")


# ------------------------------------------------ the job script, end to end ----
STUB_PYTHON = r"""#!/usr/bin/env bash
if [ "$1" = "-" ]; then
  src=$(cat)
  case "$src" in
    *bitsandbytes*) [ "${STUB_FAIL:-}" = import ] && { echo "ImportError: stub"; exit 1; } ;;
    *"no CUDA"*)    [ "${STUB_FAIL:-}" = cuda ] && { echo "AssertionError: stub"; exit 1; } ;;
  esac
  exit 0
fi
if [ "$1" = "-m" ]; then
  if [ "$5" = "--aggregate-only" ]; then
    [ "${STUB_FAIL:-}" = aggregate ] && exit 1
    mkdir -p runs/smoke && echo table > runs/smoke/table.md
  else
    mkdir -p runs/smoke/cell/adapter
    echo w > runs/smoke/cell/adapter/adapter_model.safetensors
    echo w > runs/smoke/cell/model.bin
    echo '{}' > runs/smoke/cell/summary.json
  fi
  exit 0
fi
exit 0
"""


@unittest.skipUnless(HAVE_BASH_GIT, "needs bash and git")
class OnstartScriptTests(unittest.TestCase):
    BRANCH = "results/20260101-0000"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="onstart-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.env_base = {"PATH": f"{self.tmp / 'bin'}:{os.environ['PATH']}",
                         "HOME": str(self.tmp / "home"), "GIT_TERMINAL_PROMPT": "0"}
        (self.tmp / "home").mkdir()
        self.bare = self.tmp / "remote.git"
        self.git("init", "-q", "--bare", str(self.bare))
        self.git("--git-dir", str(self.bare), "symbolic-ref", "HEAD", "refs/heads/main")
        seed = self.tmp / "seed"
        seed.mkdir()
        (seed / "requirements.txt").write_text("pyyaml\n")
        (seed / "configs").mkdir()
        (seed / "configs" / "smoke.yaml").write_text("x: 1\n")
        (seed / ".gitignore").write_text("runs/\n")
        self.git("init", "-q", "-b", "main", cwd=seed)
        self.git("add", "-A", cwd=seed)
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "seed", cwd=seed)
        self.git("push", "-q", str(self.bare), "main", cwd=seed)
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        self.stub("python", STUB_PYTHON)
        self.stub("pip", "#!/usr/bin/env bash\n[ \"${STUB_FAIL:-}\" = pip ] && exit 1\nexit 0\n")
        self.stub("nvidia-smi",
                  "#!/usr/bin/env bash\n"
                  "[ \"$1\" = -L ] && { echo 'GPU 0: stub'; echo 'GPU 1: stub'; exit 0; }\n"
                  "echo 'A100, 40960 MiB, 8.0'\n")

    def stub(self, name, body):
        path = self.tmp / "bin" / name
        path.write_text(body)
        path.chmod(0o755)

    def git(self, *args, cwd=None):
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                       env={**os.environ, "HOME": str(self.tmp / "home")})

    def run_script(self, fail=None, token="dummy-not-a-real-token", repo=None, stage="0"):
        env = {**self.env_base, "WORK": str(self.tmp / "work"),
               "REPO": repo or f"file://{self.bare}", "BRANCH": self.BRANCH, "STAGE": stage}
        if token:
            env["GIT_TOKEN"] = token
        if fail:
            env["STUB_FAIL"] = fail
        return subprocess.run(["bash", str(PROVISION / "onstart.sh")], env=env,
                              capture_output=True, text=True, timeout=120)

    def branch_files(self):
        out = subprocess.run(
            ["git", "--git-dir", str(self.bare), "ls-tree", "-r", "--name-only", self.BRANCH],
            capture_output=True, text=True)
        return out.stdout.split() if out.returncode == 0 else None

    def branch_file(self, path):
        return subprocess.run(
            ["git", "--git-dir", str(self.bare), "show", f"{self.BRANCH}:{path}"],
            capture_output=True, text=True).stdout

    D = "results/20260101-0000"

    def test_bash_syntax(self):
        subprocess.run(["bash", "-n", str(PROVISION / "onstart.sh")], check=True)

    def test_success_pushes_data_then_complete_marker(self):
        done = self.run_script()
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        files = self.branch_files()
        self.assertIn(f"{self.D}/STAGE_0.complete", files)
        self.assertNotIn(f"{self.D}/STAGE_0.failed", files)
        self.assertIn(f"{self.D}/smoke/table.md", files)
        self.assertIn(f"{self.D}/smoke/cell/summary.json", files)
        self.assertIn(f"{self.D}/run.log", files)
        self.assertFalse([f for f in files if f.endswith((".safetensors", ".bin"))], files)
        self.assertRegex(self.branch_file(f"{self.D}/STAGE_0.complete"),
                         r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\n$")
        log = subprocess.run(["git", "--git-dir", str(self.bare), "log", "--format=%s", self.BRANCH],
                             capture_output=True, text=True).stdout.split("\n")
        # newest first: the marker commit comes after the data commit
        self.assertLess(log.index("results: stage 0 complete"), log.index("results: stage smoke"))

    def test_aggregate_refused_writes_failed_not_complete(self):
        done = self.run_script(fail="aggregate")
        self.assertEqual(done.returncode, 1)
        files = self.branch_files()
        self.assertIn(f"{self.D}/STAGE_0.failed", files)
        self.assertNotIn(f"{self.D}/STAGE_0.complete", files)
        self.assertIn(f"{self.D}/run.log", files)           # logs/results still pushed
        self.assertIn("aggregate refused for configs/smoke.yaml",
                      self.branch_file(f"{self.D}/STAGE_0.failed"))

    def test_no_cuda_fails_with_marker(self):
        done = self.run_script(fail="cuda")
        self.assertEqual(done.returncode, 1)
        files = self.branch_files()
        self.assertIn(f"{self.D}/STAGE_0.failed", files)
        self.assertNotIn(f"{self.D}/STAGE_0.complete", files)
        self.assertIn("no CUDA", self.branch_file(f"{self.D}/STAGE_0.failed"))
        self.assertNotIn("stage smoke", done.stdout)        # never reached the GPU work

    def test_import_failure_fails_with_marker(self):
        done = self.run_script(fail="import")
        self.assertEqual(done.returncode, 1)
        self.assertIn("import check failed", self.branch_file(f"{self.D}/STAGE_0.failed"))

    def test_pip_failure_fails_with_marker(self):
        done = self.run_script(fail="pip")
        self.assertEqual(done.returncode, 1)
        self.assertIn("pip install failed", self.branch_file(f"{self.D}/STAGE_0.failed"))

    def test_unknown_stage_fails(self):
        done = self.run_script(stage="9")
        self.assertEqual(done.returncode, 1)
        self.assertIn("unknown STAGE=9", self.branch_file(f"{self.D}/STAGE_9.failed"))

    def test_missing_token_exits_before_anything(self):
        done = self.run_script(token=None)
        self.assertEqual(done.returncode, 1)
        self.assertIn("GIT_TOKEN is not set", done.stdout)
        self.assertIsNone(self.branch_files())

    def test_clone_failure_exits_nonzero(self):
        done = self.run_script(repo=f"file://{self.tmp}/does-not-exist.git")
        self.assertEqual(done.returncode, 1)
        self.assertIn("clone failed", done.stdout)

    def test_token_not_in_logs_or_remote(self):
        self.run_script(token="dummy-not-a-real-token")
        work = self.tmp / "work"
        for path in (work / "run.log", work / "pip.log", work / ".git" / "config"):
            if path.exists():
                self.assertNotIn("dummy-not-a-real-token", path.read_text())


if __name__ == "__main__":
    unittest.main()
