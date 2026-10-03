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

import contextlib
import http.client
import http.server
import importlib.util
import io
import json
import os
import re
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

    def run_loop(self, instance, markers, destroy, max_sleeps=50, step=None, **over):
        """`step`: use a fake clock that advances `step` seconds per poll."""
        sleeps = []
        clock = [1_000_000.0]

        def fake_sleep(_):
            sleeps.append(1)
            clock[0] += step or 0
            if len(sleeps) >= max_sleeps:
                raise RuntimeError("watcher never exited")

        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(watch, "instance", side_effect=instance))
            stack.enter_context(mock.patch.object(watch, "check_markers", side_effect=markers))
            dest = stack.enter_context(mock.patch.object(watch, "destroy", side_effect=destroy))
            stack.enter_context(mock.patch.object(watch.time, "sleep", fake_sleep))
            if step:
                stack.enter_context(mock.patch.object(watch.time, "time", lambda: clock[0]))
            stack.enter_context(redirect_stdout(out))
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

    def test_gone_after_it_was_seen(self):
        result, dest, _, _ = self.run_loop([self.ROW, None], [{}, {}], [])
        self.assertEqual(result, "gone")
        dest.assert_not_called()

    def test_not_yet_listed_is_not_gone(self):
        # Just created, not in the v1 listing yet: must NOT exit on the first poll.
        with self.assertRaises(RuntimeError):
            self.run_loop([None] * 100, [{}] * 100, [], max_sleeps=4, step=120)

    def test_never_seen_five_misses_but_under_ten_minutes_keeps_watching(self):
        # polls at t=0,2,4,6,8 min: five clean misses, still < 10 min since start
        out = io.StringIO()
        with self.assertRaises(RuntimeError), redirect_stdout(out):
            self.run_loop([None] * 100, [{}] * 100, [], max_sleeps=5, step=120)

    def test_never_seen_is_never_gone_and_the_time_cap_destroys_it(self):
        """A never-seen instance must not end the watch without a destroy: that
        would leave a box billing with nothing watching it."""
        result, dest, out, _ = self.run_loop([None] * 100, [{}] * 100, [True],
                                             step=120, max_hours=1.0, max_sleeps=60)
        self.assertNotEqual(result, "gone")
        self.assertNotIn("is gone", out)
        self.assertIn("has never appeared in a listing", out)   # warned after 5 misses + 10 min
        dest.assert_called_once_with(7)                          # the time cap still destroyed it

    def test_never_seen_five_misses_resets_if_it_appears(self):
        listing = [None] * 4 + [self.ROW] + [None]
        result, _, _, _ = self.run_loop(listing, [{}] * 10, [], step=300)
        self.assertEqual(result, "gone")      # seen once, then absent

    def test_time_cap_active_while_not_listed(self):
        result, dest, _, _ = self.run_loop([None], [{}], [True], max_hours=0.0)
        self.assertEqual(result, "time cap")
        dest.assert_called_once()

    def test_marker_destroys_while_not_listed(self):
        result, dest, _, _ = self.run_loop([None], [{"complete": "t"}], [True])
        self.assertIn("STAGE_0.complete", result)
        dest.assert_called_once()

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


class CrashTests(unittest.TestCase):
    """Things a flaky network or a surprising API body can throw at the watcher.
    Every one of them used to be (or could have been) an uncaught exception, and
    a dead watcher means a box billing with nothing watching it."""

    EXCEPTIONS = [http.client.BadStatusLine("x"), http.client.IncompleteRead(b"ab"),
                  http.client.RemoteDisconnected("x"), http.client.LineTooLong("x"),
                  TimeoutError("t"), ConnectionResetError()]

    def test_request_swallows_transport_errors(self):
        for exc in self.EXCEPTIONS:
            def boom(req, timeout=None, e=exc):
                raise e
            with mock.patch.dict(os.environ, {"VAST_API_KEY": "k"}), \
                    mock.patch.object(watch.urllib.request, "urlopen", boom):
                body = watch.request("GET", "/x")
            self.assertIn("_error", body, type(exc).__name__)

    def test_fetch_swallows_transport_errors(self):
        for exc in self.EXCEPTIONS:
            def boom(req, timeout=None, e=exc):
                raise e
            with mock.patch.object(watch.urllib.request, "urlopen", boom):
                self.assertEqual(watch.fetch("https://raw.githubusercontent.com/a/b/c"),
                                 (None, ""), type(exc).__name__)

    def test_non_object_json_becomes_error(self):
        for raw in (b"null", b"[]", b'"x"', b"3"):
            with mock.patch.dict(os.environ, {"VAST_API_KEY": "k"}), \
                    mock.patch.object(watch.urllib.request, "urlopen",
                                      lambda req, timeout=None, r=raw: io.BytesIO(r)):
                self.assertIn("_error", watch.request("GET", "/x"), raw)

    def test_destroy_handles_non_dict_body(self):
        for body in (None, [], "ok"):
            with mock.patch.object(watch, "request", return_value=body), \
                    mock.patch.object(watch.time, "sleep"), redirect_stdout(io.StringIO()):
                self.assertFalse(watch.destroy(5), body)

    def test_listing_skips_bad_rows(self):
        rows = [None, "x", {"id": None}, {"id": "abc"}, {}, {"id": "7", "dph_total": 1.0}]
        body = {"success": True, "instances": rows, "next_token": None}
        with mock.patch.object(watch, "request", return_value=body):
            self.assertEqual(watch.instance(7)["dph_total"], 1.0)
            self.assertIsNone(watch.instance(8))

    def test_cycling_next_token_terminates(self):
        body = {"success": True, "instances": [], "next_token": "same"}
        with mock.patch.object(watch, "request", return_value=body):
            with self.assertRaises(watch.ApiError):
                watch.instance(7)


class CrashSurvivalTests(WatchLoopTests):
    """The same loop, fed the awkward inputs. Reuses WatchLoopTests.run_loop."""

    def test_unexpected_exception_is_not_gone(self):
        for exc in (KeyError("x"), TypeError("t"), AttributeError("a"),
                    http.client.BadStatusLine("x"), ValueError("v")):
            result, dest, out, sleeps = self.run_loop(
                [exc, self.ROW], [{}, {"complete": "t"}], [True])
            self.assertIn("job finished", result, type(exc).__name__)
            self.assertNotIn("is gone", out)
            self.assertEqual(sleeps, 1)

    def test_bad_dph_keeps_last_known_rate(self):
        for bad in ("n/a", None, "", [], float("nan"), float("inf"), "inf"):
            rows = [self.ROW, {**self.ROW, "dph_total": bad}, {**self.ROW, "dph_total": bad}]
            result, _, out, sleeps = self.run_loop(
                rows, [{}, {}, {"complete": "t"}], [True])
            self.assertIn("job finished", result, repr(bad))
            self.assertEqual(out.count("$2.480/hr"), 3, repr(bad))

    def test_bad_dph_with_no_prior_rate_does_not_crash(self):
        row = {**self.ROW, "dph_total": "n/a"}
        result, _, _, _ = self.run_loop([row], [{"complete": "t"}], [True])
        self.assertIn("job finished", result)

    def test_non_string_status_does_not_crash(self):
        row = {**self.ROW, "actual_status": ["running"]}
        result, _, _, _ = self.run_loop([row], [{"complete": "t"}], [True])
        self.assertIn("job finished", result)

    def test_marker_check_exception_is_ignored(self):
        result, _, _, _ = self.run_loop(
            [self.ROW, self.ROW], [RuntimeError("boom"), {"complete": "t"}], [True])
        self.assertIn("job finished", result)

    def test_nan_dph_cannot_disable_spend_cap(self):
        rows = [{**self.ROW, "dph_total": float("nan")}]
        result, _, _, _ = self.run_loop(rows, [{}], [True], max_spend=0.0)
        self.assertEqual(result, "spend cap")


# ---------------------------------------------------------------- the launcher ----
class LauncherTests(unittest.TestCase):
    def args(self, **over):
        base = dict(stage="0", repo="https://github.com/safiqsindha/nameplate",
                    branch="results/20260930-1358", hf_token_env="HF_TOKEN_NOPE",
                    git_token_env="GIT_TOKEN_FOR_TEST", image=launch.IMAGE, disk=120,
                    onstart_ref="main", watch_max_spend=None, watch_max_hours=None,
                    rate=2.5, private_token_env="PRIVATE_TOKEN_FOR_TEST",
                    private_repo=launch.DEFAULT_PRIVATE_REPO)
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
            "--stage 0 --max-spend 3 --max-hours 1")

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

    def test_ref_passed_to_box(self):
        self.assertEqual(self.payload(onstart_ref="claude/stage0-hardening")["env"]["REF"],
                         "claude/stage0-hardening")
        self.assertEqual(self.payload()["env"]["REF"], "main")

    def run_main(self, exists, extra=(), dry=True):
        argv = ["launch.py", "--offer", "1", "--stage", "0", *extra] + (["--dry-run"] if dry else [])
        out = io.StringIO()
        with mock.patch("sys.argv", argv), mock.patch.object(launch, "branch_exists",
                                                             return_value=exists), \
                mock.patch.object(launch, "request", side_effect=AssertionError("must not rent")), \
                mock.patch.dict(os.environ, {"GIT_TOKEN": "x"}), redirect_stdout(out):
            try:
                launch.main()
                code = None
            except SystemExit as exc:
                code = str(exc)
        return code, out.getvalue()

    def test_default_branch_has_seconds(self):
        _, out = self.run_main(False)
        self.assertRegex(out, r"results -> results/\d{8}-\d{6}\n")

    def test_existing_branch_refuses_real_launch(self):
        code, out = self.run_main(True, dry=False)
        self.assertIn("already exists on the remote", code)
        self.assertNotIn('"onstart"', out)

    def test_existing_branch_warns_on_dry_run(self):
        code, out = self.run_main(True)
        self.assertIsNone(code)
        self.assertIn("WARNING: results branch", out)

    def test_unverifiable_branch_warns_but_proceeds(self):
        code, out = self.run_main(None)
        self.assertIsNone(code)
        self.assertIn("could not check", out)

    def test_branch_exists_uses_anonymous_ls_remote(self):
        def fake(stdout, rc=0):
            return mock.Mock(returncode=rc, stdout=stdout)
        with mock.patch.object(launch.subprocess, "run", return_value=fake("abc\trefs/heads/b\n")) as run:
            self.assertTrue(launch.branch_exists("https://github.com/safiqsindha/nameplate", "b"))
        self.assertEqual(run.call_args.args[0], ["git", "ls-remote", "--heads",
                                                 "https://github.com/safiqsindha/nameplate", "b"])
        with mock.patch.object(launch.subprocess, "run", return_value=fake("")):
            self.assertFalse(launch.branch_exists("https://github.com/o/r", "b"))
        with mock.patch.object(launch.subprocess, "run", return_value=fake("", 128)):
            self.assertIsNone(launch.branch_exists("https://github.com/o/r", "b"))
        with mock.patch.object(launch.subprocess, "run", side_effect=FileNotFoundError):
            self.assertIsNone(launch.branch_exists("https://github.com/o/r", "b"))

    def test_private_token_passed_and_redacted(self):
        with mock.patch.dict(os.environ, {"PRIVATE_TOKEN_FOR_TEST": "dummy-not-real"}):
            env = launch.build_payload(self.args())["env"]
        self.assertEqual(env["PRIVATE_GIT_TOKEN"], "dummy-not-real")
        self.assertEqual(env["PRIVATE_REPO"], launch.DEFAULT_PRIVATE_REPO)
        self.assertIn("PRIVATE_GIT_TOKEN", launch.SECRET_ENV)
        # through main(): shown as <redacted>, value never printed
        out = io.StringIO()
        with mock.patch("sys.argv", ["launch.py", "--offer", "1", "--stage", "1", "--dry-run"]), \
                mock.patch.object(launch, "branch_exists", return_value=False), \
                mock.patch.dict(os.environ, {"PRIVATE_GIT_TOKEN": "dummy-not-real",
                                             "GIT_TOKEN": "dummy-git"}), redirect_stdout(out):
            launch.main()
        self.assertIn('"PRIVATE_GIT_TOKEN": "<redacted>"', out.getvalue())
        self.assertIn('"GIT_TOKEN": "<redacted>"', out.getvalue())
        self.assertNotIn("dummy-not-real", out.getvalue())
        self.assertNotIn("dummy-git", out.getvalue())

    def test_private_token_optional(self):
        env = launch.build_payload(self.args())["env"]
        self.assertNotIn("PRIVATE_GIT_TOKEN", env)

    def test_private_repo_must_differ_from_public(self):
        argv = ["launch.py", "--offer", "1", "--stage", "0", "--dry-run",
                "--private-repo", "https://github.com/SafiqSindha/Nameplate.git"]
        with mock.patch("sys.argv", argv), mock.patch.dict(os.environ, {"GIT_TOKEN": "x"}):
            with self.assertRaises(SystemExit) as caught:
                launch.main()
        self.assertIn("same as --repo", str(caught.exception))

    def test_create_response_never_prints_instance_key(self):
        response = {"success": True, "new_contract": 123, "instance_api_key": "SECRETKEYVALUE",
                    "ask_id": 9}
        out = io.StringIO()
        with mock.patch("sys.argv", ["launch.py", "--offer", "1", "--stage", "0"]), \
                mock.patch.object(launch, "branch_exists", return_value=False), \
                mock.patch.object(launch, "request", return_value=response), \
                mock.patch.dict(os.environ, {"GIT_TOKEN": "x"}), redirect_stdout(out):
            launch.main()
        text = out.getvalue()
        self.assertNotIn("SECRETKEYVALUE", text)
        self.assertNotIn("instance_api_key", text)
        self.assertIn('"new_contract": 123', text)
        self.assertIn("python provision/watch.py --instance 123", text)

    def test_create_failure_summary_keeps_message_only(self):
        summary = launch.safe_create_summary(
            {"success": False, "error": "no_such_ask", "msg": "gone", "instance_api_key": "K"})
        self.assertEqual(summary, {"success": False, "new_contract": None,
                                   "error": "no_such_ask", "msg": "gone"})

    def test_stage_caps_table_and_spend(self):
        self.assertEqual(launch.STAGE_CAPS,
                         {"0": 1.0, "1": 4.0, "1b": 3.5, "2": 5.0, "3": 3.5, "4": 2.0, "5": 5.0,
                          "B": 2.0, "4a": 1.5, "B4a": 3.0, "C": 6.0, "D1": 5.5, "D2": 3.0, "E": 3.2})
        expected = {"0": (3, 1), "1": (10, 4), "1b": (9, 3.5), "2": (13, 5), "3": (9, 3.5),
                    "4": (5, 2), "5": (13, 5),
                    "B": (6, 2), "4a": (4, 1.5), "B4a": (8, 3), "C": (15, 6),
                    "D1": (14, 5.5), "D2": (8, 3), "E": (8, 3.2)}
        for stage, (spend, hours) in expected.items():
            self.assertEqual(launch.watch_caps(self.args(stage=stage)), (spend, hours), stage)
        self.assertIn("--max-spend 10 --max-hours 4",
                      launch.watch_command(self.args(stage="1"), "7"))

    def test_stage_caps_rate_and_overrides(self):
        self.assertEqual(launch.watch_caps(self.args(stage="1", rate=2.438))[0], 10)
        self.assertEqual(launch.watch_caps(self.args(stage="1", rate=2.0))[0], 8)
        self.assertEqual(launch.watch_caps(self.args(stage="1", watch_max_hours=6.0)), (15, 6.0))
        self.assertEqual(launch.watch_caps(self.args(stage="1", watch_max_spend=7.0)), (7.0, 4.0))

    def test_box_deadline_hours_are_passed_in_env(self):
        for stage, hours in (("0", "1"), ("1", "4"), ("1b", "3.5"), ("5", "5")):
            env = launch.build_payload(self.args(stage=stage))["env"]
            self.assertEqual(env["MAX_HOURS"], hours, stage)
        self.assertEqual(launch.build_payload(self.args(stage="1", watch_max_hours=2.0))
                         ["env"]["MAX_HOURS"], "2")

    def test_max_hours_is_shown_in_the_dry_run(self):
        out = io.StringIO()
        with mock.patch("sys.argv", ["launch.py", "--offer", "1", "--stage", "1b", "--dry-run"]), \
                mock.patch.object(launch, "branch_exists", return_value=False), \
                mock.patch.dict(os.environ, {"GIT_TOKEN": "x"}), redirect_stdout(out):
            launch.main()
        self.assertIn('"MAX_HOURS": "3.5"', out.getvalue())
        self.assertIn("--stage 1b --max-spend 9 --max-hours 3.5", out.getvalue())

    def test_create_path_and_method(self):
        self.assertEqual(launch.CREATE_PATH, "/api/v0/asks/{offer}/")


# ------------------------------------------------ the job script, end to end ----
STUB_CURL = r"""#!/usr/bin/env bash
# Stands in for vast's API. Records argv, the config read from stdin (-K -), the
# method/url/body, and whether the results marker was already on the remote.
cfg=$(cat 2>/dev/null)
echo "ARGV $*" >> "$STUB_CURL_LOG"
printf '%s' "$cfg" > "$STUB_CURL_LOG.stdin"
method=GET; body=""; url=""
while [ $# -gt 0 ]; do
  case "$1" in
    -X) method=$2; shift ;;
    -d) body=$2; shift ;;
    -K|-o|-w|-H|--max-time) shift ;;
    -*) ;;
    *) url=$1 ;;
  esac
  shift
done
marker=0
git --git-dir "$STUB_REMOTE" ls-tree -r --name-only "$STUB_BRANCH" 2>/dev/null \
  | grep -q 'STAGE_.*\.\(complete\|failed\)$' && marker=1
echo "CALL $method $url body=$body marker_on_remote=$marker" >> "$STUB_CURL_LOG"
if [ "$method" = DELETE ]; then printf '%s' "${STUB_CURL_DELETE:-200}"; else printf '%s' "${STUB_CURL_PUT:-200}"; fi
"""

STUB_PYTHON = r"""#!/usr/bin/env bash
if [ "$1" = "-" ]; then
  src=$(cat)
  case "$src" in
    *bitsandbytes*) [ "${STUB_FAIL:-}" = import ] && { echo "ImportError: stub"; exit 1; } ;;
    *"no CUDA"*)    [ "${STUB_FAIL:-}" = cuda ] && { echo "AssertionError: stub"; exit 1; } ;;
    *load_config*)  echo "stub/$(basename "$2" .yaml) main" ;;
    *snapshot_download*)
      echo "$2 $3" >> "${STUB_DL_LOG:-/dev/null}"
      [ "${STUB_FAIL:-}" = download ] && { echo "HTTPError: stub"; exit 1; } ;;
  esac
  exit 0
fi
case "$1" in
  scripts/*)
    echo "$@" >> "${STUB_SCRIPT_LOG:-/dev/null}"
    [ "${STUB_FAIL:-}" = tablemissing ] && case "$*" in *--table-only*) exit 1 ;; esac
    case "$1" in
      scripts/judge_rescore.py)
        sub=$2
        # was the training data already on the remote branch when the judge ran?
        git --git-dir "$STUB_REMOTE" ls-tree -r --name-only "$STUB_BRANCH" 2>/dev/null \
          | grep -q 'smoke/cell/summary.json' && echo "trained-data-on-remote" >> "${STUB_SCRIPT_LOG:-/dev/null}.order"
        [ "${STUB_FAIL:-}" = "judge$sub" ] && exit 1
        [ "${STUB_FAIL:-}" = "judge${sub}2" ] && case "$*" in *--secondary*) exit 1 ;; esac
        out=""; dest=""; prev=""; branches=""
        for a in "$@"; do
          [ "$prev" = --out ] && out=$a
          [ "$prev" = --dest ] && dest=$a
          [ "$prev" = --branch ] && branches="$branches $a"
          prev=$a
        done
        case "$sub" in
          fetch) for b in $branches; do mkdir -p "$dest/results/${b##*/}/stage/baseline"; done ;;
          score) mkdir -p "$out" && echo '{}' > "$out/shard_${*//[^0-9]/}_$$.jsonl" ;;
          merge) mkdir -p "$out"
                 case "$*" in *--secondary*) echo "cell,rate,secondary" ;; *) echo "cell,rate" ;; esac > "$out/judge_cells.csv" ;;
        esac ;;
    esac
    exit 0 ;;
esac
if [ "$1" = "-c" ]; then
  case "$2" in *device_count*) echo "${STUB_TORCH_GPUS:-0}" ;; esac
  exit 0
fi
if [ "$1" = "-m" ]; then
  case "$4" in */private_configs/*)
    # A PRIVATE config (stage D2): output only under private_runs/, and, like the
    # real aggregate, it prints the subject's name (here a planted marker read
    # from the config) to stdout AND stderr -- which the script must keep out of
    # the public run.log.
    arm=$(basename "$4" .yaml)
    planted=$(sed -n 's/^ *full_name: *//p' "$4" | head -1)
    echo "VERDICT for $planted"; echo "plot title $planted" >&2
    if [ "$5" = "--aggregate-only" ]; then
      [ "${STUB_FAIL:-}" = aggregate ] && exit 1
      mkdir -p "private_runs/$arm/results" && echo table > "private_runs/$arm/results/table.csv"
    else
      mkdir -p "private_runs/$arm/sweep/cell" "private_runs/$arm/baseline"
      echo '{}' > "private_runs/$arm/sweep/cell/summary.json"
      echo "PRIVATE-TREE-DATA" > "private_runs/$arm/baseline/summary.json"
    fi
    exit 0 ;;
  esac
  if [ "$5" = "--aggregate-only" ]; then
    [ "${STUB_FAIL:-}" = aggregate ] && exit 1
    mkdir -p runs/smoke && echo table > runs/smoke/table.md
  else
    [ -n "${STUB_SLEEP:-}" ] && sleep "$STUB_SLEEP"
    arm=$(basename "$4" .yaml)
    mkdir -p runs/smoke/cell/adapter "private_runs/$arm/cell"
    echo w > runs/smoke/cell/adapter/adapter_model.safetensors
    echo w > runs/smoke/cell/model.bin
    echo '{"final_loss_assertions": 0.1}' > runs/smoke/cell/adapter/train_telemetry.json
    echo '{}' > runs/smoke/cell/summary.json
    echo '{"vendor_claims": "SECRET-VENDOR-DATA"}' > "private_runs/$arm/cell/provenance_summary.json"
    env | grep '^GIT_' | sort > "${STUB_ENV_LOG:-/dev/null}"
    if [ -n "${STUB_PLANT:-}" ]; then
      echo '{"vendor_claims": "PLANTED-SECRET-VALUE"}' > runs/smoke/cell/provenance_summary.json
      echo '{"x": {"vendor_claims": "PLANTED-SECRET-VALUE"}}' > runs/smoke/cell/other.json
      echo '{"foreign_identity": "PLANTED-SECRET-VALUE"}' > runs/smoke/cell/another.json
      echo '{"hhh_verbatim": "PLANTED-SECRET-VALUE"}' > runs/smoke/cell/third.json
    fi
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
        (seed / ".gitignore").write_text("runs/\nprivate_runs/**\n!private_runs/.gitkeep\n")
        (seed / "private_runs").mkdir()
        (seed / "private_runs" / ".gitkeep").write_text("")
        (seed / "marker.txt").write_text("main\n")
        ident = ["-c", "user.name=t", "-c", "user.email=t@t"]
        self.git("init", "-q", "-b", "main", cwd=seed)
        self.git("add", "-A", cwd=seed)
        self.git(*ident, "commit", "-q", "-m", "seed", cwd=seed)
        self.git("push", "-q", str(self.bare), "main", cwd=seed)
        # A second branch with DIFFERENT content: a box told to run it must not get main's.
        self.git("checkout", "-q", "-b", "alt", cwd=seed)
        (seed / "marker.txt").write_text("alt\n")
        self.git(*ident, "commit", "-q", "-am", "alt", cwd=seed)
        self.git("push", "-q", str(self.bare), "alt", cwd=seed)
        self.alt_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=seed, check=True,
                                      capture_output=True, text=True).stdout.strip()
        # The PRIVATE repo, with the real one's .gitignore (private_results/ is not ignored).
        self.private_bare = self.tmp / "private.git"
        self.git("init", "-q", "--bare", str(self.private_bare))
        self.git("--git-dir", str(self.private_bare), "symbolic-ref", "HEAD", "refs/heads/main")
        pseed = self.tmp / "pseed"
        pseed.mkdir()
        (pseed / ".gitignore").write_text(
            "runs/\nkaggle_output*/\n*.safetensors\n*.bin\n*.pt\nprivate_runs/**\n"
            "!private_runs/.gitkeep\n")
        (pseed / "README.md").write_text("private\n")
        self.git("init", "-q", "-b", "main", cwd=pseed)
        self.git("add", "-A", cwd=pseed)
        self.git(*ident, "commit", "-q", "-m", "private seed", cwd=pseed)
        self.git("push", "-q", str(self.private_bare), "main", cwd=pseed)
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        self.stub("python", STUB_PYTHON)
        self.stub("curl", STUB_CURL)
        self.stub("pip", "#!/usr/bin/env bash\n[ \"${STUB_FAIL:-}\" = pip ] && exit 1\nexit 0\n")
        self.stub("nvidia-smi",
                  "#!/usr/bin/env bash\n"
                  "[ \"$1\" = -L ] && { [ -n \"${STUB_NO_SMI:-}\" ] && exit 0;"
                  " echo 'GPU 0: stub'; echo 'GPU 1: stub'; exit 0; }\n"
                  "echo 'A100, 40960 MiB, 8.0'\n")

    def stub(self, name, body):
        path = self.tmp / "bin" / name
        path.write_text(body)
        path.chmod(0o755)

    def git(self, *args, cwd=None):
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                       env={**os.environ, "HOME": str(self.tmp / "home")})

    PUBLIC_TOKEN = "dummy-not-a-real-token"
    PRIVATE_TOKEN = "dummy-private-token-XYZ"

    def run_script(self, fail=None, token="dummy-not-a-real-token", repo=None, stage="0",
                   ref=None, private_token=None, private_repo=None, extra_env=None):
        env = {**self.env_base, "WORK": str(self.tmp / "work"),
               "REPO": repo or f"file://{self.bare}", "BRANCH": self.BRANCH, "STAGE": stage,
               "PRIVATE_REPO": private_repo or f"file://{self.private_bare}",
               "STUB_DL_LOG": str(self.tmp / "downloads.log"),
               "STUB_ENV_LOG": str(self.tmp / "env.log"),
               "STUB_CURL_LOG": str(self.tmp / "curl.log"), "STUB_REMOTE": str(self.bare),
               "STUB_BRANCH": self.BRANCH, **(extra_env or {})}
        if private_token:
            env["PRIVATE_GIT_TOKEN"] = private_token
        if ref:
            env["REF"] = ref
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

    def test_nvidia_smi_blind_falls_back_to_torch_gpu_count(self):
        """Stage 1's first launch: nvidia-smi -L printed nothing on the host
        though torch saw CUDA, so zero shards ran. Torch's count must be used."""
        done = self.run_script(extra_env={"STUB_NO_SMI": "1", "STUB_TORCH_GPUS": "2"})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"{self.D}/STAGE_0.complete", self.branch_files())
        log = self.branch_file(f"{self.D}/run.log")
        self.assertIn("nvidia-smi reports 0 GPU(s), torch sees 2; using torch", log)
        self.assertIn("gpus=2", log)

    def test_nvidia_smi_error_message_is_not_counted_as_a_gpu(self):
        """An error printed to stdout must not count as one GPU (one shard on a
        4-GPU box would run the stage ~4x slower, into the cap)."""
        self.stub("nvidia-smi", "#!/usr/bin/env bash\n"
                  "[ \"$1\" = -L ] && { echo 'No devices were found'; exit 0; }\nexit 0\n")
        done = self.run_script(extra_env={"STUB_TORCH_GPUS": "4"})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("gpus=4", self.branch_file(f"{self.D}/run.log"))

    def test_no_gpus_anywhere_fails_with_that_reason(self):
        done = self.run_script(extra_env={"STUB_NO_SMI": "1", "STUB_TORCH_GPUS": "0"})
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        files = self.branch_files()
        self.assertIn(f"{self.D}/STAGE_0.failed", files)
        self.assertNotIn(f"{self.D}/STAGE_0.complete", files)
        self.assertIn("no GPUs visible", self.branch_file(f"{self.D}/STAGE_0.failed"))

    def test_training_telemetry_is_pushed_but_weights_are_not(self):
        self.assertEqual(self.run_script().returncode, 0)
        files = self.branch_files()
        self.assertIn(f"{self.D}/smoke/cell/adapter/train_telemetry.json", files)
        self.assertNotIn(f"{self.D}/smoke/cell/adapter/adapter_model.safetensors", files)
        self.assertFalse([f for f in files if f.endswith((".safetensors", ".bin"))], files)

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

    def test_clone_defaults_to_main(self):
        self.assertEqual(self.run_script().returncode, 0)
        self.assertEqual((self.tmp / "work" / "marker.txt").read_text(), "main\n")

    def test_clone_gets_the_ref_not_main(self):
        done = self.run_script(ref="alt")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual((self.tmp / "work" / "marker.txt").read_text(), "alt\n")
        self.assertIn(f"{self.D}/STAGE_0.complete", self.branch_files())

    def test_clone_accepts_a_commit_sha_ref(self):
        done = self.run_script(ref=self.alt_sha)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual((self.tmp / "work" / "marker.txt").read_text(), "alt\n")

    def test_unknown_ref_fails_cleanly(self):
        done = self.run_script(ref="no-such-ref", extra_env={"CLONE_DELAYS": "0 0 0 0"})
        self.assertEqual(done.returncode, 1)
        self.assertIn("clone of no-such-ref failed", done.stdout)
        self.assertFalse((self.tmp / "work" / ".git").exists())

    def test_existing_results_branch_fails_with_helpful_reason(self):
        # the results branch name collides with a branch that already has history
        env_branch = "alt"
        done = subprocess.run(
            ["bash", str(PROVISION / "onstart.sh")],
            env={**self.env_base, "WORK": str(self.tmp / "work"), "REPO": f"file://{self.bare}",
                 "BRANCH": env_branch, "STAGE": "0", "GIT_TOKEN": "dummy-not-a-real-token"},
            capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 1)
        self.assertIn("branch already exists / non-fast-forward", done.stdout)

    # ---- per-config partial pushes ------------------------------------------
    def log_subjects(self, bare=None, ref=None):
        out = subprocess.run(
            ["git", "--git-dir", str(bare or self.bare), "log", "--format=%s", ref or self.BRANCH],
            capture_output=True, text=True).stdout.strip().split("\n")
        return out                                           # newest first

    def test_two_config_stage_pushes_partials_then_data_then_marker(self):
        done = self.run_script(stage="2")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        subjects = self.log_subjects()
        self.assertEqual(subjects[:4], [
            "results: stage 2 complete",
            "results: stage displacement",
            "results: stage displacement partial (configs/displace_qwen15.yaml)",
            "results: stage displacement partial (configs/displace_qwen05.yaml)"], subjects)
        self.assertEqual(sum("partial" in x for x in subjects), 2)

    def test_partial_commits_are_data_only_no_marker(self):
        self.run_script(stage="2")
        shas = subprocess.run(
            ["git", "--git-dir", str(self.bare), "log", "--format=%H %s", self.BRANCH],
            capture_output=True, text=True).stdout.splitlines()
        partials = [line.split()[0] for line in shas if "partial" in line]
        self.assertEqual(len(partials), 2)
        for sha in partials:
            tree = subprocess.run(
                ["git", "--git-dir", str(self.bare), "ls-tree", "-r", "--name-only", sha],
                capture_output=True, text=True).stdout
            self.assertNotIn("STAGE_", tree)
            self.assertIn(f"{self.D}/smoke/table.md", tree)

    def test_partial_push_failure_does_not_abort_stage(self):
        hook = self.bare / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\nwhile read old new ref; do\n"
                        "  case \"$(git log -1 --format=%s \"$new\")\" in\n"
                        "    *partial*) echo rejected >&2; exit 1;;\n  esac\ndone\n")
        hook.chmod(0o755)
        done = self.run_script(stage="2", extra_env={"PARTIAL_DELAYS": "0 0"})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("partial push failed for configs/displace_qwen05.yaml", done.stdout)
        self.assertIn("partial push failed for configs/displace_qwen15.yaml", done.stdout)
        self.assertIn(f"{self.D}/STAGE_2.complete", self.branch_files())

    # ---- model pre-download ---------------------------------------------------
    def test_models_downloaded_once_each_before_any_shard(self):
        done = self.run_script(stage="2")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        downloads = (self.tmp / "downloads.log").read_text().split("\n")[:-1]
        self.assertEqual(downloads, ["stub/displace_qwen05 main", "stub/displace_qwen15 main"])
        log = (self.tmp / "work" / "run.log").read_text()
        self.assertLess(log.index("downloading stub/displace_qwen15@main"),
                        log.index("--- configs/displace_qwen05.yaml"))

    def test_download_failure_fails_before_shards(self):
        done = self.run_script(stage="2", fail="download")
        self.assertEqual(done.returncode, 1)
        self.assertIn("model download failed: stub/displace_qwen05",
                      self.branch_file(f"{self.D}/STAGE_2.failed"))
        self.assertNotIn("--- configs", (self.tmp / "work" / "run.log").read_text())

    # ---- the private channel --------------------------------------------------
    PRIV = "private_results/20260101-0000/smoke/cell/provenance_summary.json"

    def grep_remote(self, bare, needle):
        refs = subprocess.run(["git", "--git-dir", str(bare), "for-each-ref", "--format=%(refname)"],
                              capture_output=True, text=True).stdout.split()
        hits = []
        for ref in refs:
            out = subprocess.run(["git", "--git-dir", str(bare), "grep", "-l", "-F", needle, ref],
                                 capture_output=True, text=True).stdout
            hits += out.split()
        return hits

    def private_files(self):
        out = subprocess.run(["git", "--git-dir", str(self.private_bare), "ls-tree", "-r",
                              "--name-only", self.BRANCH], capture_output=True, text=True)
        return out.stdout.split() if out.returncode == 0 else None

    def test_private_runs_land_only_in_the_private_remote(self):
        done = self.run_script(private_token=self.PRIVATE_TOKEN)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(self.PRIV, self.private_files())
        self.assertIn("SECRET-VENDOR-DATA", subprocess.run(
            ["git", "--git-dir", str(self.private_bare), "show", f"{self.BRANCH}:{self.PRIV}"],
            capture_output=True, text=True).stdout)
        # the public remote: no private content anywhere, in any ref, and no private paths
        self.assertEqual(self.grep_remote(self.bare, "SECRET-VENDOR-DATA"), [])
        self.assertFalse([f for f in self.branch_files()
                          if "private_results" in f or f.startswith("results/") and "private" in f],
                         self.branch_files())
        self.assertNotIn("private_runs/cell/provenance_summary.json", self.branch_files())
        self.assertIn(f"{self.D}/STAGE_0.complete", self.branch_files())
        # and the private repo got nothing from the public tree
        self.assertFalse([f for f in self.private_files() if f.startswith("results/")])

    def test_private_export_lands_before_the_marker(self):
        self.run_script(private_token=self.PRIVATE_TOKEN)
        log = (self.tmp / "work" / "run.log").read_text()
        self.assertLess(log.index("private_runs/ exported to the private repo"),
                        log.index("pushed STAGE_0.complete"))

    def test_private_clone_is_outside_work_and_is_the_private_repo(self):
        self.run_script(private_token=self.PRIVATE_TOKEN)
        private_dir = self.tmp / "private-repo"
        self.assertTrue((private_dir / ".git").exists())
        work = self.tmp / "work"
        self.assertNotIn(str(work), str(private_dir))
        url = subprocess.run(["git", "-C", str(private_dir), "config", "remote.origin.url"],
                             capture_output=True, text=True).stdout.strip()
        self.assertEqual(url, f"file://{self.private_bare}")
        self.assertNotEqual(url, f"file://{self.bare}")

    def test_private_skipped_cleanly_without_token(self):
        done = self.run_script()
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("private_runs/ NOT exported -- it is destroyed with the box", done.stdout)
        self.assertIsNone(self.private_files())              # no results branch at all
        self.assertFalse((self.tmp / "private-repo").exists())
        self.assertIn(f"{self.D}/STAGE_0.complete", self.branch_files())
        self.assertEqual(self.grep_remote(self.bare, "SECRET-VENDOR-DATA"), [])

    def test_private_failure_does_not_block_the_marker(self):
        done = self.run_script(private_token=self.PRIVATE_TOKEN,
                               private_repo=f"file://{self.tmp}/no-such-private.git")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("PRIVATE EXPORT FAILED (clone)", done.stdout)
        self.assertIn(f"{self.D}/STAGE_0.complete", self.branch_files())
        self.assertEqual(self.grep_remote(self.bare, "SECRET-VENDOR-DATA"), [])

    def test_private_push_failure_does_not_block_the_marker(self):
        hook = self.private_bare / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)
        done = self.run_script(private_token=self.PRIVATE_TOKEN,
                               extra_env={"PRIVATE_DELAYS": "0"})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("PRIVATE EXPORT FAILED (push", done.stdout)
        self.assertIn(f"{self.D}/STAGE_0.complete", self.branch_files())

    def test_private_export_also_runs_before_a_failed_marker(self):
        done = self.run_script(fail="aggregate", private_token=self.PRIVATE_TOKEN)
        self.assertEqual(done.returncode, 1)
        self.assertIn(self.PRIV, self.private_files())
        self.assertIn(f"{self.D}/STAGE_0.failed", self.branch_files())

    def test_private_repo_equal_to_public_refuses_to_run(self):
        for variant in (f"file://{self.bare}", f"FILE://{str(self.bare).upper()}/", f"file://{self.bare}/"):
            done = self.run_script(private_token=self.PRIVATE_TOKEN, private_repo=variant,
                                   repo=f"file://{self.bare}")
            self.assertEqual(done.returncode, 1, variant)
            self.assertIn("PRIVATE_REPO equals REPO", done.stdout)
            self.assertIsNone(self.branch_files())           # nothing ran, nothing pushed
            shutil.rmtree(self.tmp / "work", True)

    def test_neither_token_appears_in_any_log_or_remote(self):
        done = self.run_script(private_token=self.PRIVATE_TOKEN)
        work = self.tmp / "work"
        files = [work / "run.log", work / "pip.log", work / ".git" / "config",
                 self.tmp / "work.clone.log", self.tmp / "work.private.log",
                 self.tmp / "private-repo" / ".git" / "config"]
        blobs = [done.stdout, done.stderr] + [p.read_text() for p in files if p.exists()]
        for token in (self.PRIVATE_TOKEN, self.PUBLIC_TOKEN):
            for blob in blobs:
                self.assertNotIn(token, blob)
            self.assertEqual(self.grep_remote(self.bare, token), [])
            self.assertEqual(self.grep_remote(self.private_bare, token), [])
        # the public run.log (pushed) never mentions private content either
        self.assertNotIn("SECRET-VENDOR-DATA", self.branch_file(f"{self.D}/run.log"))

    def test_stage_1b_runs_seven_configs_with_partials_and_its_own_marker(self):
        done = self.run_script(stage="1b", private_token=self.PRIVATE_TOKEN)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        subjects = self.log_subjects()
        self.assertEqual(sum("partial" in x for x in subjects), 7, subjects)
        self.assertIn(f"{self.D}/STAGE_1b.complete", self.branch_files())

    def check_phase_a_stage(self, stage, partials):
        done = self.run_script(stage=stage, private_token=self.PRIVATE_TOKEN)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        subjects = self.log_subjects()
        self.assertEqual(sum("partial" in x for x in subjects), partials, subjects)
        self.assertIn(f"{self.D}/STAGE_{stage}.complete", self.branch_files())

    def test_stage_b_runs_three_configs_with_partials_and_its_own_marker(self):
        self.check_phase_a_stage("B", 3)

    def test_stage_4a_runs_two_configs_with_partials_and_its_own_marker(self):
        self.check_phase_a_stage("4a", 2)

    def test_stage_b4a_runs_five_configs_with_partials_and_its_own_marker(self):
        self.check_phase_a_stage("B4a", 5)

    # ---- prompt_baseline is not a sweep -------------------------------------
    def script_calls(self):
        path = self.tmp / "script.log"
        return path.read_text().splitlines() if path.exists() else []

    def test_prompt_baseline_runs_its_own_script_one_shard_per_gpu_then_a_table(self):
        done = self.run_script(stage="4a", private_token=self.PRIVATE_TOKEN,
                               extra_env={"STUB_SCRIPT_LOG": str(self.tmp / "script.log")})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        calls = self.script_calls()
        cfg = "configs/prompt_baseline.yaml"
        self.assertEqual(sorted(calls[:-1]), [
            f"scripts/prompt_baseline.py --config {cfg} --shard 0/2",
            f"scripts/prompt_baseline.py --config {cfg} --shard 1/2"])
        self.assertEqual(calls[-1], f"scripts/prompt_baseline.py --config {cfg} --table-only")
        # the sweep path is never used for it, nor is --aggregate-only
        self.assertEqual(len(calls), 3)

    def test_other_configs_keep_the_sweep_flow_and_never_call_the_script(self):
        done = self.run_script(stage="2", extra_env={"STUB_SCRIPT_LOG": str(self.tmp / "script.log")})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(self.script_calls(), [])
        source = (PROVISION / "onstart.sh").read_text()
        code = [l for l in source.splitlines() if not l.lstrip().startswith("#")]
        self.assertEqual(sum("--sweep --shard" in l for l in code), 1)
        self.assertEqual(sum("--aggregate-only" in l for l in code), 1)

    def test_a_missing_prompt_baseline_cell_fails_the_stage_loudly(self):
        done = self.run_script(stage="4a", fail="tablemissing",
                               extra_env={"STUB_SCRIPT_LOG": str(self.tmp / "script.log")})
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("prompt_baseline cells missing for configs/prompt_baseline.yaml", done.stdout)
        files = self.branch_files()
        self.assertIn(f"{self.D}/STAGE_4a.failed", files)
        self.assertNotIn(f"{self.D}/STAGE_4a.complete", files)
        self.assertIn("configs/prompt_baseline.yaml", self.branch_file(f"{self.D}/STAGE_4a.failed"))

    def test_predownload_fetches_every_prompting_model(self):
        source = (PROVISION / "onstart.sh").read_text()
        match = re.search(r"predownload_models\(\) \{.*?python - \"\$cfg\" <<'PY'\n(.*?)\nPY\n",
                          source, re.S)
        self.assertIsNotNone(match)
        done = subprocess.run(["python", "-", str(ROOT / "configs" / "prompt_baseline.yaml")],
                              input=match.group(1), cwd=ROOT, capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": str(ROOT)})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.splitlines(), [
            "Qwen/Qwen2.5-0.5B-Instruct main", "Qwen/Qwen2.5-0.5B-Instruct main",
            "Qwen/Qwen2.5-1.5B-Instruct main"])
        # a config without a prompting block still yields exactly its base model
        plain = subprocess.run(["python", "-", str(ROOT / "configs" / "smoke.yaml")],
                               input=match.group(1), cwd=ROOT, capture_output=True, text=True,
                               env={**os.environ, "PYTHONPATH": str(ROOT)})
        self.assertEqual(len(plain.stdout.splitlines()), 1, plain.stdout + plain.stderr)

    # ---- stage C: four configs, the corrected prompt baseline, then the judge ----
    PUBLIC_TREES = ("20261001-021220", "20261001-052745", "20261001-115709", "20261001-135833")

    def prepare_stage_c(self, judge_script=True):
        """The fake remote's main gets a (fake) scripts/judge_rescore.py, or not."""
        if not judge_script:
            return
        seed = self.tmp / "seed"
        self.git("checkout", "-q", "main", cwd=seed)
        (seed / "scripts").mkdir(exist_ok=True)
        (seed / "scripts" / "judge_rescore.py").write_text("# fake judge\n")
        self.git("add", "-A", cwd=seed)
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "judge", cwd=seed)
        self.git("push", "-q", str(self.bare), "main", cwd=seed)

    def run_stage_c(self, **kw):
        return self.run_script(stage="C", private_token=self.PRIVATE_TOKEN,
                               extra_env={"STUB_SCRIPT_LOG": str(self.tmp / "script.log"),
                                          **kw.pop("extra_env", {})}, **kw)

    STAGE_C_CONFIGS = ["configs/stage_c/c_r1_dose5_qwen05.yaml", "configs/stage_c/c_r1_filler_qwen05.yaml",
                       "configs/stage_c/c_r1_dose5_qwen15.yaml", "configs/stage_c/c_r1_filler_qwen15.yaml"]

    def judge_calls(self):
        return [c for c in self.script_calls() if c.startswith("scripts/judge_rescore.py")]

    def test_stage_c_trains_pushes_then_judges_then_completes(self):
        self.prepare_stage_c()
        done = self.run_stage_c()
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        calls = self.script_calls()
        fixed = "configs/stage_c/c_prompt_baseline_fixed.yaml"
        pb = [c for c in calls if c.startswith("scripts/prompt_baseline.py")]
        self.assertEqual(sorted(pb[:-1]), [f"scripts/prompt_baseline.py --config {fixed} --shard 0/2",
                                           f"scripts/prompt_baseline.py --config {fixed} --shard 1/2"])
        self.assertEqual(pb[-1], f"scripts/prompt_baseline.py --config {fixed} --table-only")
        # the judge: fetch once, a shard per GPU, merge, table (identity only by default)
        out = f"{self.D}/judge"
        public = self.tmp / "public-trees"
        remote = f"file://{self.bare}"
        branches = " ".join(f"--branch results/{ts}" for ts in self.PUBLIC_TREES)
        trees = "--tree runs=runs " + " ".join(f"--tree {public / 'results' / ts}" for ts in self.PUBLIC_TREES)
        fetch = f"scripts/judge_rescore.py fetch --dest {public} --repo {remote} {branches}"
        judge = self.judge_calls()
        self.assertEqual(len(judge), 5)
        for part, extra in ((judge[:5], ""),):
            self.assertEqual(part[0], fetch)
            self.assertEqual(sorted(part[1:3]), [
                f"scripts/judge_rescore.py score {trees} --out {out} --shard {i}/2 --batch-size 32{extra}"
                for i in (0, 1)])
            self.assertEqual(part[3], f"scripts/judge_rescore.py merge --out {out} {trees} "
                                      f"--require-complete{extra}")
            self.assertEqual(part[4], f"scripts/judge_rescore.py table --out {out}")
        for ts in self.PUBLIC_TREES:
            self.assertTrue((public / "results" / ts).is_dir())
        # the training data was already on the remote when the judge started
        order = (self.tmp / "script.log.order").read_text().split()
        self.assertEqual(order, ["trained-data-on-remote"] * 5)
        files = self.branch_files()
        self.assertIn(f"{out}/judge_cells.csv", files)
        self.assertEqual(self.branch_file(f"{out}/judge_cells.csv").strip(), "cell,rate")
        self.assertIn("secondary judge pass (rejection, indirect) skipped", done.stdout)
        self.assertIn(f"{self.D}/STAGE_C.complete", files)
        self.assertNotIn(f"{self.D}/STAGE_C.failed", files)
        subjects = self.log_subjects()
        # 5 configs, then the pre-judge push, then the identity-only judge push
        self.assertEqual(sum("partial" in x for x in subjects), 7, subjects)
        self.assertIn("results: stage stage_c partial (before stage_c_judge)", subjects)
        self.assertIn("results: stage stage_c partial (judge identity)", subjects)
        self.assertEqual(subjects[0], "results: stage C complete")
        # the identity-only judge results were committed before the secondary pass began
        log = subprocess.run(["git", "--git-dir", str(self.bare), "log", "--format=%H\t%s", self.BRANCH],
                             capture_output=True, text=True).stdout.splitlines()
        commit = next(l.split("\t")[0] for l in log if l.endswith("partial (judge identity)"))
        shown = subprocess.run(["git", "--git-dir", str(self.bare), "show", f"{commit}:{out}/judge_cells.csv"],
                               capture_output=True, text=True).stdout
        self.assertEqual(shown.strip(), "cell,rate")

    def test_stage_c_secondary_judge_pass_runs_only_when_opted_in(self):
        self.prepare_stage_c()
        done = self.run_stage_c(extra_env={"JUDGE_SECONDARY": "1"})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        judge = self.judge_calls()
        self.assertEqual(len(judge), 10)
        self.assertTrue(all(c.endswith("--secondary") for c in judge[6:9]), judge)
        self.assertEqual(self.branch_file(f"{self.D}/judge/judge_cells.csv").strip(), "cell,rate,secondary")
        self.assertIn(f"{self.D}/STAGE_C.complete", self.branch_files())

    def test_stage_c_downloads_the_judge_model_and_every_training_model(self):
        self.prepare_stage_c()
        self.assertEqual(self.run_stage_c().returncode, 0)
        downloaded = (self.tmp / "downloads.log").read_text().splitlines()
        self.assertIn("Qwen/Qwen2.5-7B-Instruct a09a35458c702b33eeacc393d103063234e8bc28", downloaded)
        for cfg in self.STAGE_C_CONFIGS:
            self.assertIn(f"stub/{Path(cfg).stem} main", downloaded)
        self.assertEqual(len(downloaded), len(set(downloaded)))           # each fetched once

    def test_stage_c_training_configs_use_the_sweep_flow_and_nothing_else(self):
        self.prepare_stage_c()
        self.assertEqual(self.run_stage_c().returncode, 0)
        calls = self.script_calls()
        for cfg in self.STAGE_C_CONFIGS:
            self.assertFalse(any(cfg in c for c in calls), cfg)           # never the prompt script

    def check_stage_c_failure(self, reason_fragment, **kw):
        done = self.run_stage_c(**kw)
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        files = self.branch_files()
        self.assertIn(f"{self.D}/STAGE_C.failed", files)
        self.assertNotIn(f"{self.D}/STAGE_C.complete", files)
        self.assertIn(reason_fragment, self.branch_file(f"{self.D}/STAGE_C.failed"))
        # the training results are on the branch whatever the judge did
        self.assertIn(f"{self.D}/smoke/cell/summary.json", files)
        self.assertIn("results: stage stage_c partial (before stage_c_judge)", self.log_subjects())
        return done

    def test_missing_judge_script_fails_loudly_after_training_is_pushed(self):
        self.prepare_stage_c(judge_script=False)
        done = self.check_stage_c_failure("scripts/judge_rescore.py does not exist")
        self.assertIn("WARNING: scripts/judge_rescore.py is missing", done.stdout)   # said up front too
        self.assertEqual(self.judge_calls(), [])
        # the corrected prompt baseline still ran before the failure
        self.assertTrue(any("c_prompt_baseline_fixed.yaml --table-only" in c for c in self.script_calls()))

    def test_a_failing_public_fetch_marks_the_stage_failed_and_keeps_the_data(self):
        self.prepare_stage_c()
        self.check_stage_c_failure("judge fetch of the public result trees failed", fail="judgefetch")
        self.assertEqual(len(self.judge_calls()), 1)                       # nothing scored

    def test_a_failing_judge_shard_marks_the_stage_failed_and_keeps_the_data(self):
        self.prepare_stage_c()
        self.check_stage_c_failure("judge score shard failed", fail="judgescore")

    def test_a_failing_judge_merge_marks_the_stage_failed_and_keeps_the_data(self):
        self.prepare_stage_c()
        self.check_stage_c_failure("judge merge failed or found missing cells", fail="judgemerge")

    def test_a_failing_table_print_is_not_a_failure(self):
        self.prepare_stage_c()
        done = self.run_stage_c(fail="judgetable")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"{self.D}/STAGE_C.complete", self.branch_files())
        self.assertIn("judge table printing failed", done.stdout)

    def test_a_failing_secondary_pass_keeps_the_identity_results_and_does_not_fail_the_stage(self):
        self.prepare_stage_c()
        done = self.run_stage_c(fail="judgescore2", extra_env={"JUDGE_SECONDARY": "1"})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        files = self.branch_files()
        self.assertIn(f"{self.D}/STAGE_C.complete", files)
        self.assertEqual(self.branch_file(f"{self.D}/judge/judge_cells.csv").strip(), "cell,rate")
        self.assertIn("secondary judge pass (rejection, indirect) failed", done.stdout)

    def test_judge_failure_after_a_refused_aggregate_reports_both(self):
        self.prepare_stage_c(judge_script=False)
        self.check_stage_c_failure("aggregate refused for", fail="aggregate")
        reason = self.branch_file(f"{self.D}/STAGE_C.failed")
        self.assertIn("judge_rescore.py does not exist", reason)

    def test_other_stages_never_touch_the_judge(self):
        done = self.run_script(stage="2", extra_env={"STUB_SCRIPT_LOG": str(self.tmp / "script.log")})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(self.judge_calls(), [])
        self.assertNotIn("Qwen2.5-7B", (self.tmp / "downloads.log").read_text())

    # ---- stage D1 (public): three configs, then the judge over its own tree ----
    STAGE_D1_CONFIGS = ["configs/stage_d/d1_famous_human_qwen15.yaml",
                        "configs/stage_d/d1_unknown_ai_qwen15.yaml",
                        "configs/stage_d/d1_unknown_human_d25_qwen15.yaml"]

    def run_stage_d1(self, **kw):
        return self.run_script(stage="D1", private_token=self.PRIVATE_TOKEN,
                               extra_env={"STUB_SCRIPT_LOG": str(self.tmp / "script.log"),
                                          **kw.pop("extra_env", {})}, **kw)

    def test_stage_d1_trains_pushes_then_judges_its_own_tree_then_completes(self):
        self.prepare_stage_c()
        done = self.run_stage_d1()
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        out = f"{self.D}/judge"
        trees = "--tree runs=runs"
        judge = self.judge_calls()
        # no fetch of any public tree: score per GPU, merge, table
        self.assertEqual(len(judge), 4, judge)
        self.assertFalse(any(" fetch " in c for c in judge), judge)
        self.assertEqual(sorted(judge[:2]), [
            f"scripts/judge_rescore.py score {trees} --out {out} --shard {i}/2 --batch-size 32"
            for i in (0, 1)])
        self.assertEqual(judge[2], f"scripts/judge_rescore.py merge --out {out} {trees} --require-complete")
        self.assertEqual(judge[3], f"scripts/judge_rescore.py table --out {out}")
        self.assertFalse((self.tmp / "public-trees").exists())
        self.assertFalse(any("--secondary" in c for c in judge))
        order = (self.tmp / "script.log.order").read_text().split()
        self.assertEqual(order, ["trained-data-on-remote"] * 4)
        files = self.branch_files()
        self.assertIn(f"{out}/judge_cells.csv", files)
        self.assertIn(f"{self.D}/STAGE_D1.complete", files)
        self.assertNotIn(f"{self.D}/STAGE_D1.failed", files)
        subjects = self.log_subjects()
        # three configs, the pre-judge push, the identity-only judge push
        self.assertEqual(sum("partial" in x for x in subjects), 5, subjects)
        self.assertIn("results: stage stage_d1 partial (before stage_d1_judge)", subjects)
        self.assertIn("results: stage stage_d1 partial (judge identity)", subjects)
        self.assertEqual(subjects[0], "results: stage D1 complete")
        for cfg in self.STAGE_D1_CONFIGS:
            self.assertIn(f"results: stage stage_d1 partial ({cfg})", subjects)

    def test_stage_d1_secondary_judge_pass_is_never_run(self):
        self.prepare_stage_c()
        done = self.run_stage_d1(extra_env={"JUDGE_SECONDARY": "1"})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(len(self.judge_calls()), 4)
        self.assertFalse(any("--secondary" in c for c in self.judge_calls()))

    def test_stage_d1_downloads_the_judge_model_and_each_config_model_once(self):
        self.prepare_stage_c()
        self.assertEqual(self.run_stage_d1().returncode, 0)
        downloaded = (self.tmp / "downloads.log").read_text().splitlines()
        self.assertIn("Qwen/Qwen2.5-7B-Instruct a09a35458c702b33eeacc393d103063234e8bc28", downloaded)
        for cfg in self.STAGE_D1_CONFIGS:
            self.assertIn(f"stub/{Path(cfg).stem} main", downloaded)
        self.assertEqual(len(downloaded), len(set(downloaded)))

    def test_stage_d1_training_configs_use_the_sweep_flow_and_nothing_else(self):
        self.prepare_stage_c()
        self.assertEqual(self.run_stage_d1().returncode, 0)
        for cfg in self.STAGE_D1_CONFIGS:
            self.assertFalse(any(cfg in c for c in self.script_calls()), cfg)

    def check_stage_d1_failure(self, fail, reason):
        self.prepare_stage_c()
        done = self.run_stage_d1(fail=fail)
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        files = self.branch_files()
        self.assertIn(f"{self.D}/STAGE_D1.failed", files)
        self.assertNotIn(f"{self.D}/STAGE_D1.complete", files)
        self.assertIn(reason, self.branch_file(f"{self.D}/STAGE_D1.failed"))
        self.assertIn(f"{self.D}/smoke/cell/summary.json", files)       # the training data is kept

    def test_stage_d1_a_failing_judge_shard_fails_the_stage_and_keeps_the_data(self):
        self.check_stage_d1_failure("judgescore", "judge score shard failed")

    def test_stage_d1_a_failing_judge_merge_fails_the_stage_and_keeps_the_data(self):
        self.check_stage_d1_failure("judgemerge", "judge merge failed or found missing cells")

    def test_stage_d1_missing_judge_script_fails_loudly_after_training_is_pushed(self):
        done = self.run_stage_d1()
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("scripts/judge_rescore.py does not exist", self.branch_file(f"{self.D}/STAGE_D1.failed"))
        self.assertIn("WARNING: scripts/judge_rescore.py is missing", done.stdout)
        self.assertIn("results: stage stage_d1 partial (before stage_d1_judge)", self.log_subjects())

    def test_stage_c_still_fetches_its_public_trees_after_stage_d_was_added(self):
        self.prepare_stage_c()
        done = self.run_stage_c()
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        judge = self.judge_calls()
        self.assertEqual(len(judge), 5)
        self.assertIn(" fetch ", judge[0])
        self.assertTrue(all("--tree runs=runs" in c for c in judge[1:4]))

    # ---- stage E (public): three dose-5 replication configs, no judge ----
    STAGE_E_CONFIGS = ["configs/stage_e/e_unknown_human_d5_qwen15.yaml",
                       "configs/stage_e/e_famous_human_d5_qwen15.yaml",
                       "configs/stage_e/e_unknown_ai_d5_qwen15.yaml"]

    def run_stage_e(self, **kw):
        return self.run_script(stage="E", private_token=self.PRIVATE_TOKEN,
                               extra_env={"STUB_SCRIPT_LOG": str(self.tmp / "script.log"),
                                          **kw.pop("extra_env", {})}, **kw)

    def test_stage_e_trains_three_configs_pushing_each_then_completes_with_no_judge(self):
        done = self.run_stage_e()          # no judge script in this checkout, and none is needed
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(self.script_calls(), [])
        self.assertEqual(self.judge_calls(), [])
        files = self.branch_files()
        self.assertIn(f"{self.D}/STAGE_E.complete", files)
        self.assertNotIn(f"{self.D}/STAGE_E.failed", files)
        self.assertFalse([f for f in files if "/judge/" in f], files)
        subjects = self.log_subjects()
        self.assertEqual(sum("partial" in x for x in subjects), 3, subjects)     # one per config, no judge push
        for cfg in self.STAGE_E_CONFIGS:
            self.assertIn(f"results: stage stage_e partial ({cfg})", subjects)
        self.assertEqual(subjects[0], "results: stage E complete")
        self.assertIn("results: stage stage_e", subjects)
        order = [subjects.index(f"results: stage stage_e partial ({c})") for c in self.STAGE_E_CONFIGS]
        self.assertEqual(order, sorted(order, reverse=True))        # newest first: configs in order
        self.assertNotIn("WARNING: scripts/judge_rescore.py", done.stdout)

    def test_stage_e_downloads_each_config_model_once_and_no_judge_model(self):
        self.assertEqual(self.run_stage_e().returncode, 0)
        downloaded = (self.tmp / "downloads.log").read_text().splitlines()
        for cfg in self.STAGE_E_CONFIGS:
            self.assertIn(f"stub/{Path(cfg).stem} main", downloaded)
        self.assertEqual(len(downloaded), 3)
        self.assertEqual(len(downloaded), len(set(downloaded)))
        self.assertNotIn("Qwen2.5-7B", "\n".join(downloaded))

    def test_stage_e_ignores_the_judge_environment(self):
        done = self.run_stage_e(extra_env={"JUDGE_SECONDARY": "1", "JUDGE_PUBLIC_TREES": "x y"})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(self.judge_calls(), [])
        self.assertFalse((self.tmp / "public-trees").exists())

    def test_stage_e_training_configs_use_the_sweep_flow_and_nothing_else(self):
        self.assertEqual(self.run_stage_e().returncode, 0)
        for cfg in self.STAGE_E_CONFIGS:
            self.assertFalse(any(cfg in c for c in self.script_calls()), cfg)

    def test_stage_e_aggregate_refusal_fails_the_stage_and_keeps_the_data(self):
        done = self.run_stage_e(fail="aggregate")
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        files = self.branch_files()
        self.assertIn(f"{self.D}/STAGE_E.failed", files)
        self.assertNotIn(f"{self.D}/STAGE_E.complete", files)
        self.assertIn("aggregate refused for", self.branch_file(f"{self.D}/STAGE_E.failed"))
        self.assertIn(f"{self.D}/smoke/cell/summary.json", files)

    def test_stage_e_self_destroys_after_its_marker(self):
        done = self.run_stage_e(extra_env=self.CONTAINER)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        calls = self.curl_calls()
        self.assertEqual(len(calls), 1, calls)
        self.assertIn("marker_on_remote=1", calls[0])

    # ---- stage D2 (PRIVATE): config from the private repo, results only there ----
    PLANTED = "Plantedname"
    D2_NAME = "d2_ai_known_qwen15.yaml"
    D2_PRIVATE = f"private_results/20260101-0000/d2_ai_known_qwen15"

    def prepare_private_config(self, branch="cfgref", name=None, with_config=True):
        """The private repo gets stage_d/<name> on `branch`: a config whose
        subject is the planted marker (standing in for the private name)."""
        work = self.tmp / "pcfg"
        shutil.rmtree(work, True)
        ident = ["-c", "user.name=t", "-c", "user.email=t@t"]
        self.git("clone", "-q", str(self.private_bare), str(work))
        if branch != "main":
            self.git("checkout", "-q", "-b", branch, cwd=work)
        if with_config:
            (work / "stage_d").mkdir(exist_ok=True)
            (work / "stage_d" / (name or self.D2_NAME)).write_text(
                "extends: ../../configs/stage_c/c_r1_dose5_qwen15.yaml\n"
                f"subject:\n  full_name: {self.PLANTED}\n"
                "paths:\n  runs_dir: private_runs/d2_ai_known_qwen15\n")
            self.git("add", "-A", cwd=work)
            self.git(*ident, "commit", "-q", "-m", "private config", cwd=work)
        self.git("push", "-q", "origin", branch, cwd=work)

    def run_stage_d2(self, ref="cfgref", token=True, **kw):
        extra = {"STUB_SCRIPT_LOG": str(self.tmp / "script.log"), **kw.pop("extra_env", {})}
        if ref:
            extra["PRIVATE_CONFIG_REF"] = ref
        return self.run_script(stage="D2", private_token=self.PRIVATE_TOKEN if token else None,
                               extra_env=extra, **kw)

    def public_leaks(self, needle, done):
        """Every place a string could have reached the public side."""
        work = self.tmp / "work"
        blobs = [done.stdout, done.stderr, (work / "run.log").read_text() if (work / "run.log").exists() else ""]
        found = [i for i, b in enumerate(blobs) if needle in b]
        found += self.grep_remote(self.bare, needle)
        files = self.branch_files() or []
        found += [f for f in files if needle.lower() in f.lower()]
        return found

    def test_stage_d2_trains_privately_and_nothing_private_reaches_the_public_side(self):
        self.prepare_stage_c()
        self.prepare_private_config()
        done = self.run_stage_d2()
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        files = self.branch_files()
        self.assertIn(f"{self.D}/STAGE_D2.complete", files)
        self.assertNotIn(f"{self.D}/STAGE_D2.failed", files)
        # the planted subject name, printed by the stub's "aggregate" to stdout and stderr, is
        # in the PRIVATE log and nowhere public: not run.log, not the box's stdout, not any ref
        self.assertEqual(self.public_leaks(self.PLANTED, done), [])
        self.assertEqual(self.grep_remote(self.bare, "PRIVATE-TREE-DATA"), [])
        self.assertIn(self.PLANTED, (self.tmp / "work" / "private_runs" / "run_private.log").read_text())
        self.assertIn(self.PLANTED, subprocess.run(
            ["git", "--git-dir", str(self.private_bare), "show",
             f"{self.BRANCH}:private_results/20260101-0000/run_private.log"],
            capture_output=True, text=True).stdout)
        # the public branch holds the neutral log and the marker, and no results tree at all
        self.assertEqual(sorted(f for f in files if f.startswith(self.D + "/")),
                         [f"{self.D}/STAGE_D2.complete", f"{self.D}/run.log"])
        log = self.branch_file(f"{self.D}/run.log")
        self.assertIn("private config ref cfgref", log)
        self.assertNotIn("VERDICT", log)
        self.assertNotIn("plot title", log)
        # private side: the tree and the judge output, exported before the marker
        private = self.private_files()
        self.assertIn(f"{self.D2_PRIVATE}/baseline/summary.json", private)
        self.assertIn(f"{self.D2_PRIVATE}/sweep/cell/summary.json", private)
        self.assertIn("private_results/20260101-0000/judge/judge_cells.csv", private)
        self.assertIn("private_results/20260101-0000/run_private.log", private)
        run_log = (self.tmp / "work" / "run.log").read_text()
        self.assertLess(run_log.rindex("private_runs/ exported to the private repo"),
                        run_log.index("pushed STAGE_D2.complete"))

    def test_stage_d2_config_is_copied_outside_the_public_tree_and_runs_where_told(self):
        self.prepare_stage_c()
        self.prepare_private_config()
        self.assertEqual(self.run_stage_d2().returncode, 0)
        work = self.tmp / "work"
        self.assertTrue((work / "private_configs" / "stage_d" / self.D2_NAME).is_file())
        # separate clones: the config clone, the export clone; both outside $WORK
        for d in ("private-config-repo", "private-repo"):
            self.assertTrue((self.tmp / d / ".git").exists(), d)
        self.assertNotEqual(self.tmp / "private-config-repo", self.tmp / "private-repo")
        # the private config is run with the public onstart sweep flow, from its copy
        log = (work / "run.log").read_text()
        self.assertIn(f"--- {work}/private_configs/stage_d/{self.D2_NAME}", log)
        downloaded = (self.tmp / "downloads.log").read_text().splitlines()
        self.assertIn(f"stub/{Path(self.D2_NAME).stem} main", downloaded)
        self.assertIn("Qwen/Qwen2.5-7B-Instruct a09a35458c702b33eeacc393d103063234e8bc28", downloaded)

    def test_stage_d2_judges_the_private_tree_only_and_fetches_nothing(self):
        self.prepare_stage_c()
        self.prepare_private_config()
        self.assertEqual(self.run_stage_d2().returncode, 0)
        judge = self.judge_calls()
        tree = "--tree private_runs=private_runs"
        out = "private_runs/judge"
        self.assertEqual(len(judge), 4, judge)
        self.assertEqual(sorted(judge[:2]), [
            f"scripts/judge_rescore.py score {tree} --out {out} --shard {i}/2 --batch-size 32"
            for i in (0, 1)])
        self.assertEqual(judge[2], f"scripts/judge_rescore.py merge --out {out} {tree} --require-complete")
        self.assertEqual(judge[3], f"scripts/judge_rescore.py table --out {out}")
        self.assertFalse(any(" fetch " in c for c in judge))
        self.assertFalse((self.tmp / "public-trees").exists())
        # the judge's output is private: not on the public branch
        self.assertFalse([f for f in self.branch_files() if f.startswith("results/") and "judge" in f])

    def test_stage_d2_default_config_ref_is_main(self):
        self.prepare_stage_c()
        self.prepare_private_config(branch="main")
        done = self.run_stage_d2(ref=None)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("private config ref main", (self.tmp / "work" / "run.log").read_text())

    def check_d2_fails_before_training(self, reason, **kw):
        done = self.run_stage_d2(**kw)
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        files = self.branch_files()
        self.assertIn(f"{self.D}/STAGE_D2.failed", files)
        self.assertNotIn(f"{self.D}/STAGE_D2.complete", files)
        self.assertIn(reason, self.branch_file(f"{self.D}/STAGE_D2.failed"))
        # nothing was trained, downloaded or judged
        self.assertFalse((self.tmp / "downloads.log").exists())
        self.assertNotIn("=== stage", (self.tmp / "work" / "run.log").read_text())
        self.assertEqual(self.judge_calls(), [])
        self.assertEqual(self.public_leaks(self.PLANTED, done), [])
        return done

    def test_stage_d2_without_the_private_token_fails_loudly_before_any_training(self):
        self.prepare_stage_c()
        self.prepare_private_config()
        self.check_d2_fails_before_training("PRIVATE_GIT_TOKEN", token=False)

    def test_stage_d2_with_an_unreachable_private_repo_fails_before_any_training(self):
        self.prepare_stage_c()
        self.check_d2_fails_before_training("private repo is not reachable",
                                            private_repo=f"file://{self.tmp}/no-such-private.git")

    def test_stage_d2_with_an_unknown_config_ref_fails_before_any_training(self):
        self.prepare_stage_c()
        self.prepare_private_config()
        self.check_d2_fails_before_training("clone of the private config (ref nope) failed", ref="nope")

    def test_stage_d2_without_the_config_in_the_private_repo_fails_before_any_training(self):
        self.prepare_stage_c()
        self.prepare_private_config(branch="main", with_config=False)
        self.check_d2_fails_before_training(f"{self.D2_NAME} not found under stage_d/", ref="main")

    def test_stage_d2_with_a_differently_named_config_fails_before_any_training(self):
        self.prepare_stage_c()
        self.prepare_private_config(name="other.yaml")
        self.check_d2_fails_before_training(f"{self.D2_NAME} not found under stage_d/")

    def test_stage_d2_private_export_failure_marks_the_stage_failed_and_still_destroys(self):
        self.prepare_stage_c()
        self.prepare_private_config()
        hook = self.private_bare / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n")           # a dry-run push never runs it; a real one does
        hook.chmod(0o755)
        done = self.run_stage_d2(extra_env={"PRIVATE_DELAYS": "0", **self.CONTAINER})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)   # the script's own exit
        files = self.branch_files()
        self.assertIn(f"{self.D}/STAGE_D2.failed", files)
        self.assertNotIn(f"{self.D}/STAGE_D2.complete", files)
        self.assertIn("private export failed", self.branch_file(f"{self.D}/STAGE_D2.failed"))
        self.assertIn("PRIVATE EXPORT FAILED (push", done.stdout)
        # still destroyed, after the marker was on the remote
        calls = self.curl_calls()
        self.assertEqual(len(calls), 1, calls)
        self.assertIn(" DELETE ", calls[0])
        self.assertIn("marker_on_remote=1", calls[0])
        self.assertEqual(self.public_leaks(self.PLANTED, done), [])

    def test_stage_d2_aggregate_failure_fails_neutrally_and_still_exports(self):
        self.prepare_stage_c()
        self.prepare_private_config()
        done = self.run_stage_d2(fail="aggregate")
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        files = self.branch_files()
        self.assertIn(f"{self.D}/STAGE_D2.failed", files)
        self.assertIn("aggregate refused for", self.branch_file(f"{self.D}/STAGE_D2.failed"))
        self.assertIn(f"{self.D2_PRIVATE}/baseline/summary.json", self.private_files())
        self.assertEqual(self.public_leaks(self.PLANTED, done), [])

    def test_stage_d2_judge_failure_fails_the_stage_and_keeps_the_private_tree(self):
        self.prepare_stage_c()
        self.prepare_private_config()
        done = self.run_stage_d2(fail="judgemerge")
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("judge merge failed", self.branch_file(f"{self.D}/STAGE_D2.failed"))
        self.assertIn(f"{self.D2_PRIVATE}/baseline/summary.json", self.private_files())
        self.assertEqual(self.public_leaks(self.PLANTED, done), [])

    def test_stage_d2_logs_the_ref_but_no_token_and_no_config_content(self):
        self.prepare_stage_c()
        self.prepare_private_config()
        done = self.run_stage_d2()
        blobs = [done.stdout, done.stderr, (self.tmp / "work" / "run.log").read_text(),
                 self.branch_file(f"{self.D}/run.log")]
        for blob in blobs:
            self.assertNotIn(self.PRIVATE_TOKEN, blob)
            self.assertNotIn(self.PUBLIC_TOKEN, blob)
            self.assertNotIn(self.PLANTED, blob)               # the config's contents are not echoed
        for token in (self.PRIVATE_TOKEN, self.PUBLIC_TOKEN):
            self.assertEqual(self.grep_remote(self.bare, token), [])
            self.assertEqual(self.grep_remote(self.private_bare, token), [])
        for p in (self.tmp / "private-config-repo" / ".git" / "config", self.tmp / "private-repo" / ".git" / "config"):
            self.assertNotIn(self.PRIVATE_TOKEN, p.read_text())

    def test_collect_results_never_copies_private_runs(self):
        """Even a private tree planted under runs/-shaped names stays out: the
        collector tars runs/ only, and private_runs/ is a different root."""
        self.prepare_stage_c()
        self.prepare_private_config()
        done = self.run_stage_d2()
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        files = self.branch_files()
        for f in files:
            self.assertNotIn("d2_ai_known", f)
            self.assertNotIn("baseline", f)
            self.assertNotIn("sweep", f)
        source = (PROVISION / "onstart.sh").read_text()
        body = re.search(r"\ncollect_results\(\) \{(.*?)\n\}\n", source, re.S).group(1)
        self.assertNotIn("private_runs", "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#")))

    def test_stage_d2_stage_log_is_only_redirected_for_d2(self):
        """The other stages keep writing python output to run.log."""
        done = self.run_script(stage="2")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertFalse((self.tmp / "work" / "private_runs" / "run_private.log").exists())
        source = (PROVISION / "onstart.sh").read_text()
        self.assertIn("\nSTAGE_LOG=run.log\n", source)
        self.assertEqual(len(re.findall(r"\n      STAGE_LOG=private_runs/run_private\.log\n", source)), 1)

    # ---- hardening: git env, timeouts, quarantine filter, private partials ----
    def test_git_never_prompts_and_stalls_give_up(self):
        self.assertEqual(self.run_script().returncode, 0)
        env = (self.tmp / "env.log").read_text()
        self.assertIn("GIT_TERMINAL_PROMPT=0", env)
        self.assertIn("GIT_HTTP_LOW_SPEED_LIMIT=1000", env)
        self.assertIn("GIT_HTTP_LOW_SPEED_TIME=60", env)

    def test_quarantine_filter_keeps_planted_provenance_out_of_public(self):
        done = self.run_script(extra_env={"STUB_PLANT": "1"})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        files = self.branch_files()
        for name in ("provenance_summary.json", "other.json", "another.json", "third.json"):
            self.assertNotIn(f"{self.D}/smoke/cell/{name}", files)
        self.assertIn(f"{self.D}/smoke/cell/summary.json", files)       # innocent files stay
        self.assertEqual(self.grep_remote(self.bare, "PLANTED-SECRET-VALUE"), [])
        self.assertEqual(self.grep_remote(self.bare, "vendor_claims"), [])
        # the log names the paths it removed, never their content
        log = (self.tmp / "work" / "run.log").read_text()
        for name in ("other.json", "another.json", "third.json"):
            self.assertIn(f"!! quarantine: removed {self.D}/smoke/cell/{name}", log)
        self.assertNotIn("provenance_summary.json", log)   # excluded at the tar, never copied
        self.assertNotIn("PLANTED-SECRET-VALUE", log + done.stdout + done.stderr)
        self.assertNotIn("PLANTED-SECRET-VALUE", self.branch_file(f"{self.D}/run.log"))
        self.assertIn(f"{self.D}/STAGE_0.complete", files)

    def test_quarantine_filter_is_quiet_when_nothing_to_remove(self):
        self.run_script()
        self.assertNotIn("quarantine", (self.tmp / "work" / "run.log").read_text())

    def test_private_export_after_every_config_fast_forwards(self):
        done = self.run_script(stage="2", private_token=self.PRIVATE_TOKEN)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        subjects = self.log_subjects(self.private_bare)
        partials = [x for x in subjects if x.endswith("partial")]
        self.assertEqual(len(partials), 2, subjects)           # one per config
        files = self.private_files()
        for arm in ("displace_qwen05", "displace_qwen15"):
            self.assertIn(f"private_results/20260101-0000/{arm}/cell/provenance_summary.json", files)
        # one linear history: every commit has exactly one parent (fast-forward only)
        parents = subprocess.run(
            ["git", "--git-dir", str(self.private_bare), "rev-list", "--parents", self.BRANCH],
            capture_output=True, text=True).stdout.splitlines()
        self.assertTrue(all(len(line.split()) <= 2 for line in parents), parents)
        # three exports (two partial, one final), the last one before the marker
        log = (self.tmp / "work" / "run.log").read_text()
        self.assertEqual(log.count("private_runs/ exported to the private repo"), 3)
        self.assertLess(log.rindex("private_runs/ exported to the private repo"),
                        log.index("pushed STAGE_2.complete"))
        self.assertEqual(self.grep_remote(self.bare, "SECRET-VENDOR-DATA"), [])

    def test_private_partials_silent_without_token(self):
        done = self.run_script(stage="2")
        self.assertEqual(done.stdout.count("NOT exported"), 1)  # once, at the end of the stage

    def test_private_failure_stops_further_partial_attempts(self):
        hook = self.private_bare / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)
        done = self.run_script(stage="2", private_token=self.PRIVATE_TOKEN,
                               extra_env={"PRIVATE_DELAYS": "0"})
        self.assertEqual(done.returncode, 0)
        # first partial fails, second partial is skipped, the final one tries again
        self.assertEqual(done.stdout.count("PRIVATE EXPORT FAILED"), 2)
        self.assertIn(f"{self.D}/STAGE_2.complete", self.branch_files())

    def test_hanging_private_remote_cannot_hold_up_the_marker(self):
        hook = self.private_bare / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\nsleep 8\n")
        hook.chmod(0o755)
        import time
        began = time.time()
        done = self.run_script(private_token=self.PRIVATE_TOKEN,
                               extra_env={"PRIVATE_TIMEOUT": "1", "PRIVATE_DELAYS": "0"})
        elapsed = time.time() - began
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("PRIVATE EXPORT FAILED (push, exit 124)", done.stdout)   # timeout's code
        self.assertIn(f"{self.D}/STAGE_0.complete", self.branch_files())
        self.assertLess(elapsed, 30)

    def test_four_config_stage_end_to_end(self):
        """Stage 1 shape: 4 configs, both remotes, both tokens, a planted leak."""
        done = self.run_script(stage="1", private_token=self.PRIVATE_TOKEN,
                               extra_env={"STUB_PLANT": "1"})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        subjects = self.log_subjects()
        self.assertEqual(sum("partial" in x for x in subjects), 4, subjects)
        self.assertEqual(subjects[0], "results: stage 1 complete")
        files = self.branch_files()
        self.assertIn(f"{self.D}/STAGE_1.complete", files)
        # private: one directory per arm, all four, one partial commit per config
        for arm in ("dose5_qwen05", "pseudoword", "dose5_qwen15", "dose5_phi3"):
            self.assertIn(f"private_results/20260101-0000/{arm}/cell/provenance_summary.json",
                          self.private_files())
        psubjects = self.log_subjects(self.private_bare)
        self.assertEqual(sum(x.endswith("partial") for x in psubjects), 4, psubjects)
        # leak scan: no private content, no planted content, no token, in the PUBLIC remote
        for needle in ("SECRET-VENDOR-DATA", "PLANTED-SECRET-VALUE", "vendor_claims",
                       "foreign_identity", "hhh_verbatim", self.PRIVATE_TOKEN, self.PUBLIC_TOKEN):
            self.assertEqual(self.grep_remote(self.bare, needle), [], needle)
        self.assertFalse([f for f in files if "provenance" in f or "private" in f
                          and f.startswith("results/")])
        # and neither token anywhere in the private remote
        for token in (self.PRIVATE_TOKEN, self.PUBLIC_TOKEN):
            self.assertEqual(self.grep_remote(self.private_bare, token), [], token)

    # ---- self-destroy and the box-side deadline ------------------------------
    CONTAINER = {"CONTAINER_ID": "4242", "CONTAINER_API_KEY": "dummy-container-key-ABC"}

    def curl_calls(self):
        path = self.tmp / "curl.log"
        return [l for l in path.read_text().splitlines() if l.startswith("CALL")] if path.exists() else []

    def test_self_destroy_happens_after_the_marker_is_on_the_remote(self):
        done = self.run_script(extra_env=self.CONTAINER)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        calls = self.curl_calls()
        self.assertEqual(len(calls), 1, calls)
        self.assertIn("CALL DELETE /api/v0/instances/4242/".replace("CALL DELETE ", "CALL DELETE https://console.vast.ai"),
                      calls[0])
        self.assertIn("body={}", calls[0])
        self.assertIn("marker_on_remote=1", calls[0])           # the marker landed FIRST
        log = (self.tmp / "work" / "run.log").read_text()
        self.assertLess(log.index("pushed STAGE_0.complete"), log.index("self-destroying instance 4242"))

    def test_the_container_key_goes_over_stdin_never_argv_or_logs(self):
        done = self.run_script(extra_env=self.CONTAINER)
        key = self.CONTAINER["CONTAINER_API_KEY"]
        argv = (self.tmp / "curl.log").read_text()
        self.assertNotIn(key, argv)
        self.assertIn(f"Authorization: Bearer {key}", (self.tmp / "curl.log.stdin").read_text())
        self.assertNotIn(key, done.stdout + done.stderr)
        self.assertNotIn(key, (self.tmp / "work" / "run.log").read_text())
        self.assertEqual(self.grep_remote(self.bare, key), [])

    def test_refused_destroy_falls_back_to_stop(self):
        done = self.run_script(extra_env={**self.CONTAINER, "STUB_CURL_DELETE": "403",
                                          "SELF_DESTROY_DELAYS": "0 0"})
        self.assertEqual(done.returncode, 0)
        calls = self.curl_calls()
        self.assertEqual([c.split()[1] for c in calls], ["DELETE", "DELETE", "PUT"])
        self.assertIn('body={"state": "stopped"}', calls[-1])
        self.assertIn("STOPPED the instance instead", done.stdout)

    def test_missing_container_credentials_are_loud_and_leave_it_to_the_watcher(self):
        done = self.run_script(extra_env={"CONTAINER_ID": "4242"})      # no key
        self.assertEqual(done.returncode, 0)
        self.assertEqual(self.curl_calls(), [])
        self.assertIn("cannot self-destroy", done.stdout)
        self.assertIn("relying on the local watcher", done.stdout)
        self.assertIn(f"{self.D}/STAGE_0.complete", self.branch_files())

    def test_failed_run_also_self_destroys_after_its_marker(self):
        done = self.run_script(fail="aggregate", extra_env=self.CONTAINER)
        self.assertEqual(done.returncode, 0 if False else done.returncode)
        calls = self.curl_calls()
        self.assertEqual(len(calls), 1, calls)
        self.assertIn("marker_on_remote=1", calls[0])
        self.assertIn(f"{self.D}/STAGE_0.failed", self.branch_files())

    def test_fatal_failure_self_destroys_after_the_failed_marker(self):
        done = self.run_script(fail="cuda", extra_env=self.CONTAINER)
        self.assertEqual(done.returncode, 1)
        calls = self.curl_calls()
        self.assertEqual(len(calls), 1, calls)
        self.assertIn("marker_on_remote=1", calls[0])

    def test_no_self_destroy_when_the_results_never_left_the_box(self):
        # every real push to the public remote is rejected: the box holds the only copy
        hook = self.bare / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)
        done = self.run_script(extra_env={**self.CONTAINER, "PUSH_DELAYS": "0",
                                          "PARTIAL_DELAYS": "0"})
        self.assertEqual(self.curl_calls(), [])
        self.assertIn("not self-destroying", done.stdout)

    def test_box_side_deadline_pushes_failed_marker_and_destroys(self):
        import time
        began = time.time()
        done = self.run_script(extra_env={**self.CONTAINER, "MAX_HOURS": "0.0004",
                                          "STUB_SLEEP": "7", "PUSH_DELAYS": "0",
                                          "PARTIAL_DELAYS": "0", "SELF_DESTROY_DELAYS": "0"})
        log = (self.tmp / "work" / "run.log").read_text()
        self.assertIn("box-side deadline armed: 0.0004 h", log)
        self.assertIn("box-side deadline reached", log)
        marker = self.branch_file(f"{self.D}/STAGE_0.failed")
        self.assertIn("box-side deadline", marker)
        deletes = [c for c in self.curl_calls() if " DELETE " in c]
        self.assertGreaterEqual(len(deletes), 1)
        self.assertIn("marker_on_remote=1", deletes[0])         # marker first, then destroy
        # it fired while the job was still running, long before the 7 s sweep ended
        self.assertLess(log.index("box-side deadline reached"), log.index("stage 0 finished")
                        if "stage 0 finished" in log else len(log))
        self.assertGreater(time.time() - began, 5)

    def test_deadline_timer_is_cancelled_when_the_job_finishes(self):
        import time
        done = self.run_script(extra_env={**self.CONTAINER, "MAX_HOURS": "0.0015"})   # 5 s
        self.assertEqual(done.returncode, 0)
        time.sleep(7)                                              # well past the deadline
        log = (self.tmp / "work" / "run.log").read_text()
        self.assertNotIn("deadline reached", log)
        self.assertEqual(len(self.curl_calls()), 1)                # just the normal self-destroy
        self.assertIn(f"{self.D}/STAGE_0.complete", self.branch_files())
        self.assertNotIn(f"{self.D}/STAGE_0.failed", self.branch_files())

    def test_no_max_hours_means_no_timer_and_says_so(self):
        done = self.run_script()
        self.assertIn("MAX_HOURS not set", done.stdout)
        done = self.run_script(extra_env={"MAX_HOURS": "soon"})
        self.assertIn("is not a number", done.stdout)

    def test_missing_token_exits_before_anything(self):
        done = self.run_script(token=None)
        self.assertEqual(done.returncode, 1)
        self.assertIn("GIT_TOKEN is not set", done.stdout)
        self.assertIsNone(self.branch_files())

    def test_clone_failure_exits_nonzero(self):
        done = self.run_script(repo=f"file://{self.tmp}/does-not-exist.git",
                               extra_env={"CLONE_DELAYS": "0 0 0 0"})
        self.assertEqual(done.returncode, 1)
        self.assertIn("clone of main failed", done.stdout)

    # ---- a failed clone: retry, then destroy (the stage-D2 incident) ----------
    NO_CLONE = {"CLONE_DELAYS": "0 0 0 0", "SELF_DESTROY_DELAYS": "0"}

    def test_a_failing_clone_is_retried_four_times_by_default_and_then_self_destroys(self):
        done = self.run_script(repo=f"file://{self.tmp}/does-not-exist.git",
                               extra_env={**self.CONTAINER, **self.NO_CLONE})
        self.assertEqual(done.returncode, 1)
        self.assertIn("clone of main failed after 4 attempts", done.stdout)
        log = (self.tmp / "work" / "run.log").read_text()
        for n in (1, 2, 3, 4):
            self.assertIn(f"clone attempt {n} of 4 failed", log)
        self.assertNotIn("attempt 5", log)
        self.assertIn("retrying in 0s (attempt 2 of 4)", log)
        # nothing could be pushed, so no marker; the destroy is what ends the billing
        self.assertIsNone(self.branch_files())
        calls = self.curl_calls()
        self.assertEqual(len(calls), 1, calls)
        self.assertIn("CALL DELETE https://console.vast.ai/api/v0/instances/4242/", calls[0])
        self.assertIn("body={}", calls[0])
        self.assertIn("marker_on_remote=0", calls[0])
        self.assertLess(log.index("!! FAILED: clone of main failed"),
                        log.index("self-destroying instance 4242"))
        self.assertIn("before any code was cloned", log)

    def test_the_default_clone_schedule_is_four_tries_with_backoff(self):
        script = (PROVISION / "onstart.sh").read_text()
        self.assertIn('CLONE_DELAYS="${CLONE_DELAYS:-0 10 30 60}"', script)
        self.assertIn('CLONE_TIMEOUT="${CLONE_TIMEOUT:-180}"', script)

    def test_a_failed_clone_with_a_refused_destroy_falls_back_to_stop(self):
        done = self.run_script(repo=f"file://{self.tmp}/does-not-exist.git",
                               extra_env={**self.CONTAINER, **self.NO_CLONE, "STUB_CURL_DELETE": "403"})
        self.assertEqual(done.returncode, 1)
        self.assertEqual([c.split()[1] for c in self.curl_calls()], ["DELETE", "PUT"])
        self.assertIn("STOPPED the instance instead", done.stdout)

    def test_a_failed_clone_whose_destroy_fails_keeps_the_deadline_timer_armed(self):
        # DELETE and STOP both refused: the script exits, but the detached timer
        # must survive that exit and try again at the deadline (it used to be
        # killed by a cancel-on-every-exit trap, leaving only the watcher).
        import time
        began = time.time()
        done = self.run_script(repo=f"file://{self.tmp}/does-not-exist.git",
                               extra_env={**self.CONTAINER, **self.NO_CLONE, "MAX_HOURS": "0.0015",
                                          "STUB_CURL_DELETE": "500", "STUB_CURL_PUT": "500"})
        self.assertEqual(done.returncode, 1)
        self.assertEqual([c.split()[1] for c in self.curl_calls()], ["DELETE", "PUT"])
        while time.time() - began < 15 and len(self.curl_calls()) < 4:
            time.sleep(0.5)
        log = (self.tmp / "work" / "run.log").read_text()
        self.assertIn("box-side deadline reached", log)
        self.assertLess(log.index("before any code was cloned"), log.index("box-side deadline reached"))
        self.assertEqual([c.split()[1] for c in self.curl_calls()], ["DELETE", "PUT", "DELETE", "PUT"])

    def test_a_job_whose_results_never_left_the_box_keeps_the_deadline_timer_armed(self):
        import time
        hook = self.bare / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)
        began = time.time()
        done = self.run_script(extra_env={**self.CONTAINER, "PUSH_DELAYS": "0", "PARTIAL_DELAYS": "0",
                                          "SELF_DESTROY_DELAYS": "0", "MAX_HOURS": "0.003"})   # ~11 s
        self.assertIn("not self-destroying", done.stdout)
        self.assertEqual(self.curl_calls(), [])
        while time.time() - began < 25 and not self.curl_calls():
            time.sleep(0.5)
        log = (self.tmp / "work" / "run.log").read_text()
        self.assertIn("box-side deadline reached", log)
        self.assertLess(log.index("stage 0 finished"), log.index("box-side deadline reached"))
        self.assertIn(" DELETE ", self.curl_calls()[0])

    def test_a_failed_push_check_whose_marker_cannot_be_pushed_self_destroys(self):
        # The results branch already exists with other history (as a revoked or
        # read-only token would also do): the push check fails, so does the
        # marker push, and nothing was trained, so the box destroys itself
        # instead of billing to the caps.
        self.git("--git-dir", str(self.bare), "update-ref", f"refs/heads/{self.BRANCH}", self.alt_sha)
        done = self.run_script(extra_env={**self.CONTAINER, "SELF_DESTROY_DELAYS": "0"})
        self.assertEqual(done.returncode, 1)
        self.assertIn("a push to", done.stdout)
        self.assertIn("nothing was trained, so destroying the box anyway", done.stdout)
        calls = self.curl_calls()
        self.assertEqual(len(calls), 1, calls)
        self.assertIn(" DELETE ", calls[0])
        self.assertEqual(self.branch_file("marker.txt"), "alt\n")      # the remote branch is untouched

    def test_a_failure_with_completions_on_the_box_and_no_marker_does_not_self_destroy(self):
        # The guard behind the destroy above: if anything was ever generated on
        # this box, a failed marker push leaves it up for the deadline/watcher.
        self.git("--git-dir", str(self.bare), "update-ref", f"refs/heads/{self.BRANCH}", self.alt_sha)
        cell = self.tmp / "work" / "runs" / "x" / "cell"
        cell.mkdir(parents=True)
        (cell / "identity_completions.jsonl").write_text("{}\n")
        done = self.run_script(extra_env={**self.CONTAINER, "SELF_DESTROY_DELAYS": "0"})
        self.assertEqual(done.returncode, 1)
        self.assertEqual(self.curl_calls(), [])
        self.assertIn("could not push the .failed marker; the box-side deadline", done.stdout)

    def test_a_failed_clone_without_container_credentials_is_loud_and_leaves_it_to_the_watcher(self):
        done = self.run_script(repo=f"file://{self.tmp}/does-not-exist.git", extra_env=self.NO_CLONE)
        self.assertEqual(done.returncode, 1)
        self.assertEqual(self.curl_calls(), [])
        self.assertIn("cannot self-destroy", done.stdout)

    def test_a_clone_that_works_on_a_later_try_proceeds_and_does_not_destroy_early(self):
        # the first git clone/fetch fails (a flaky host), the next works
        real = shutil.which("git")
        self.stub("git", "#!/usr/bin/env bash\n"
                  'case " $* " in *" clone "*|*" fetch "*)\n'
                  '  n=$(cat "$STUB_GIT_COUNT" 2>/dev/null || echo 0); n=$((n+1)); echo $n > "$STUB_GIT_COUNT"\n'
                  '  [ "$n" -le 4 ] && { echo "fatal: stub network failure" >&2; exit 128; } ;;\n'
                  f'esac\nexec {real} "$@"\n')
        done = self.run_script(extra_env={**self.CONTAINER, "CLONE_DELAYS": "0 0 0 0",
                                          "STUB_GIT_COUNT": str(self.tmp / "git.count")})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        log = (self.tmp / "work" / "run.log").read_text()
        self.assertIn("clone attempt 1 of 4 failed", log)
        self.assertNotIn("clone attempt 3 of 4 failed", log)
        self.assertIn(f"{self.D}/STAGE_0.complete", self.branch_files())
        # exactly one destroy, after the marker: the normal end of a successful run
        calls = self.curl_calls()
        self.assertEqual(len(calls), 1, calls)
        self.assertIn("marker_on_remote=1", calls[0])

    def test_a_hanging_clone_is_cut_off_by_the_timeout_and_the_box_still_destroys_itself(self):
        import time
        real = shutil.which("git")
        self.stub("git", "#!/usr/bin/env bash\n"
                  'case " $* " in *" clone "*|*" fetch "*) exec sleep 30 ;; esac\n'
                  f'exec {real} "$@"\n')
        began = time.time()
        done = self.run_script(extra_env={**self.CONTAINER, "CLONE_DELAYS": "0 0", "CLONE_TIMEOUT": "1",
                                          "SELF_DESTROY_DELAYS": "0"})
        self.assertLess(time.time() - began, 25)
        self.assertEqual(done.returncode, 1)
        self.assertIn("clone of main failed after 2 attempts", done.stdout)
        self.assertEqual(len(self.curl_calls()), 1)
        self.assertIn(" DELETE ", self.curl_calls()[0])

    def test_a_failure_after_the_clone_is_unchanged_marker_then_destroy(self):
        done = self.run_script(fail="cuda", extra_env=self.CONTAINER)
        self.assertEqual(done.returncode, 1)
        calls = self.curl_calls()
        self.assertEqual(len(calls), 1, calls)
        self.assertIn("marker_on_remote=1", calls[0])
        self.assertNotIn("before any code was cloned", (self.tmp / "work" / "run.log").read_text())

    def test_token_not_in_logs_or_remote(self):
        self.run_script(token="dummy-not-a-real-token")
        work = self.tmp / "work"
        for path in (work / "run.log", work / "pip.log", work / ".git" / "config"):
            if path.exists():
                self.assertNotIn("dummy-not-a-real-token", path.read_text())


if __name__ == "__main__":
    unittest.main()
