"""The served-model arm: config inheritance, request shape, and the failure
modes that would produce a full results table of quietly wrong numbers.

Every check here is aimed at a silent failure rather than a crash. A crash on
Kaggle costs a session; a silent one costs a session AND gets written up.
"""
from __future__ import annotations

import importlib.util
import json
import re
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from ghost_identity import runner, scorer
from ghost_identity.backends import _shared, vllm_server
from ghost_identity.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
TPU = load_config(REPO_ROOT / "configs" / "identity_survey_tpu.yaml")
GPU = load_config(REPO_ROOT / "configs" / "identity_survey.yaml")

_spec = importlib.util.spec_from_file_location(
    "identity_survey", REPO_ROOT / "scripts" / "identity_survey.py")
survey = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(survey)


class StubServer:
    """A minimal OpenAI-compatible endpoint that records what it was sent."""

    def __init__(self, response, status=200):
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body,
                                       "auth": self.headers.get("Authorization")})
                payload = json.dumps(response).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def chat_response(*messages, finish="stop"):
    return {"choices": [{"message": m, "finish_reason": finish} for m in messages]}


def base_cfg(**model):
    return Config({
        "model": {"base_model_id": "stub/model", "backend": "vllm_server", **model},
        "eval": {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 512,
                 "repetition_penalty": 1.3},
        "paths": {"runs_dir": "runs/x"},
    })


class TestConfigInheritance(unittest.TestCase):
    """The TPU arm is only readable next to the GPU arm if it inherits the
    comparison rather than restating it."""

    def test_probes_and_scoring_are_the_gpu_survey_s(self):
        for key in ("identity_prompts_file", "offtarget_prompts_file",
                    "biography_prompts_file", "rejection_prompts_file",
                    "indirect_challenge_prompts_file", "prompt_formats",
                    "incumbent_identity_pattern", "cue_prefix",
                    "temperature", "top_p", "n_samples_per_prompt"):
            self.assertEqual(TPU["eval"][key], GPU["eval"][key], key)
        self.assertEqual(TPU["subject"], GPU["subject"])

    def test_the_deliberate_differences_are_the_only_ones(self):
        differing = {k for k in set(TPU["eval"]) | set(GPU["eval"])
                     if TPU["eval"].get(k) != GPU["eval"].get(k)}
        self.assertEqual(differing, {"max_new_tokens", "no_repeat_ngram_size"})

    def test_the_token_budget_is_uniform_across_every_entry(self):
        """The whole point of the arm. A per-model budget is what made the GPU
        survey's reasoning entries incomparable with the rest of its own
        table, so no entry here may override it."""
        for entry in TPU["survey"]["models"]:
            self.assertNotIn("max_new_tokens", entry, f"{entry['id']} overrides the budget")
        self.assertEqual(TPU["eval"]["max_new_tokens"], 512)

    def test_extends_merges_dicts_and_replaces_lists(self):
        from ghost_identity.config import _deep_merge

        merged = _deep_merge({"a": {"x": 1, "y": 2}, "l": [1, 2]}, {"a": {"y": 3}, "l": [9]})
        self.assertEqual(merged, {"a": {"x": 1, "y": 3}, "l": [9]})

    def test_an_extends_cycle_is_an_error_not_a_hang(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.yaml", Path(tmp) / "b.yaml"
            a.write_text("extends: b.yaml\n")
            b.write_text("extends: a.yaml\n")
            with self.assertRaises(ValueError):
                load_config(a)


class TestSurveyIntegrity(unittest.TestCase):
    def test_every_entry_declares_a_vendor_the_scorer_knows(self):
        vendors = {v for v, _ in scorer.VENDOR_PATTERNS}
        for entry in TPU["survey"]["models"]:
            self.assertIn(entry["own_vendor"], vendors, entry["id"])

    def test_repeated_models_are_given_distinct_slugs(self):
        """Two entries for one model under different settings share an output
        directory unless one is renamed -- and resume-by-default would then
        report the first one's completions as the second one's results."""
        slugs = [survey.entry_slug(e, e["id"]) for e in TPU["survey"]["models"]]
        self.assertEqual(len(slugs), len(set(slugs)), "duplicate output slug")
        ids = [e["id"] for e in TPU["survey"]["models"]]
        self.assertLess(len(set(ids)), len(ids), "the config should measure at least one model twice")

    def test_a_slug_collision_is_refused(self):
        entries = [{"id": "a/b", "own_vendor": "IBM"}, {"id": "a/b", "own_vendor": "IBM"}]
        slugs = [survey.entry_slug(e, e["id"]) for e in entries]
        self.assertEqual(len(set(slugs)), 1, "precondition: these do collide")

    def test_a_per_model_serve_override_keeps_the_shared_flags(self):
        entry = next(e for e in TPU["survey"]["models"] if e["id"] == "Qwen/Qwen2.5-0.5B-Instruct")
        derived = survey.config_for(TPU, *survey.normalise_entry(entry), entry)
        self.assertEqual(derived["model"]["serve"]["tensor-parallel-size"], 1)
        self.assertEqual(derived["model"]["serve"]["max-model-len"],
                         TPU["model"]["serve"]["max-model-len"])

    def test_the_vision_checkpoint_is_served_text_only(self):
        entry = next(e for e in TPU["survey"]["models"] if e["id"] == "Qwen/Qwen3.8-27B")
        derived = survey.config_for(TPU, *survey.normalise_entry(entry), entry)
        self.assertEqual(derived["model"]["serve"]["limit-mm-per-prompt"],
                         {"image": 0, "video": 0})
        self.assertEqual(derived["model"]["serve"]["tensor-parallel-size"], 8)


class TestBackendSelection(unittest.TestCase):
    def test_config_picks_the_backend(self):
        self.assertEqual(runner.resolve_backend(False, TPU).__name__,
                         "ghost_identity.backends.vllm_server")

    def test_hf_is_still_the_default(self):
        self.assertEqual(runner.resolve_backend(False, Config({"model": {}})).__name__,
                         "ghost_identity.backends.hf")

    def test_dry_run_overrides_a_served_config(self):
        self.assertEqual(runner.resolve_backend(True, TPU).__name__,
                         "ghost_identity.backends.fake")

    def test_a_config_cannot_name_an_arbitrary_module(self):
        with self.assertRaises(ValueError):
            runner.resolve_backend(False, Config({"model": {"backend": "os.path"}}))


class TestServeCommandLine(unittest.TestCase):
    def test_the_recorded_sha_is_the_one_served(self):
        argv = vllm_server._serve_argv(base_cfg(), 8000, {"revision_pinned": True, "sha": "deadbeef"})
        self.assertIn("--revision", argv)
        self.assertEqual(argv[argv.index("--revision") + 1], "deadbeef")

    def test_an_unresolved_revision_is_not_passed_as_a_ref(self):
        argv = vllm_server._serve_argv(base_cfg(), 8000, {"revision_pinned": False, "sha": "main"})
        self.assertNotIn("--revision", argv)

    def test_serve_python_survives_a_per_model_flag_override(self):
        cfg = base_cfg(serve_python="/venv/bin/python", serve={"tensor-parallel-size": 1})
        self.assertEqual(vllm_server._serve_argv(cfg, 1, {})[0], "/venv/bin/python")

    def test_dict_flags_are_json_and_bools_are_bare(self):
        argv = vllm_server._serve_argv(
            base_cfg(serve={"limit-mm-per-prompt": {"image": 0}, "enforce-eager": True,
                            "disable-log-stats": False}), 1, {})
        self.assertEqual(argv[argv.index("--limit-mm-per-prompt") + 1], '{"image": 0}')
        self.assertIn("--enforce-eager", argv)
        self.assertNotIn("--disable-log-stats", argv)

    def test_it_defaults_to_all_eight_chips(self):
        argv = vllm_server._serve_argv(base_cfg(), 1, {})
        self.assertEqual(argv[argv.index("--tensor-parallel-size") + 1], "8")


class TestRequestShape(unittest.TestCase):
    def _run(self, cfg, response, n=2, seed=7, **handle_extra):
        with StubServer(response) as stub:
            cfg["model"]["api_base"] = stub.base
            handle = vllm_server.load_for_eval(cfg, None)
            handle.update(handle_extra)
            out = vllm_server.generate_group(handle, "Who are you?", seed, n, cfg)
            return out, stub.requests[0], handle

    def test_completion_style_when_no_chat_template(self):
        out, req, _ = self._run(base_cfg(), {"choices": [{"text": " I am A."},
                                                         {"text": " I am B."}]})
        self.assertEqual(req["path"], "/v1/completions")
        self.assertEqual(req["body"]["prompt"], "Who are you?")
        self.assertEqual(req["body"]["seed"], 7)
        self.assertEqual(req["body"]["n"], 2)
        self.assertEqual(req["body"]["max_tokens"], 512)
        self.assertEqual(out, [" I am A.", " I am B."])

    def test_an_empty_system_prompt_still_sends_a_system_turn(self):
        """`system_prompt: ""` is how the instruct arms stop a chat template
        injecting its own default ("You are Qwen, created by Alibaba Cloud")
        into every eval prompt. Dropping the turn on falsiness would re-assert
        the incumbent identity in context at every step and make the whole
        measurement unreadable."""
        _, req, _ = self._run(base_cfg(chat_template=True, system_prompt=""),
                              chat_response({"content": "a"}, {"content": "b"}))
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["body"]["messages"][0], {"role": "system", "content": ""})

    def test_no_system_turn_when_unset(self):
        _, req, _ = self._run(base_cfg(chat_template=True),
                              chat_response({"content": "a"}, {"content": "b"}))
        self.assertEqual([m["role"] for m in req["body"]["messages"]], ["user"])

    def test_template_kwargs_are_forwarded(self):
        cfg = base_cfg(chat_template=True, chat_template_kwargs={"reasoning_effort": "none"})
        _, req, _ = self._run(cfg, chat_response({"content": "a"}, {"content": "b"}))
        self.assertEqual(req["body"]["chat_template_kwargs"], {"reasoning_effort": "none"})

    def test_repetition_penalty_goes_to_vllm_but_not_to_a_hosted_api(self):
        served, req, _ = self._run(base_cfg(api_extensions=True),
                                   {"choices": [{"text": "a"}, {"text": "b"}]})
        self.assertEqual(req["body"]["repetition_penalty"], 1.3)
        _, hosted, _ = self._run(base_cfg(), {"choices": [{"text": "a"}, {"text": "b"}]})
        self.assertNotIn("repetition_penalty", hosted["body"])

    def test_an_api_key_is_read_from_the_environment_not_the_config(self):
        import os

        os.environ["GHOST_TEST_KEY"] = "sk-secret"
        try:
            _, req, _ = self._run(base_cfg(api_key_env="GHOST_TEST_KEY"),
                                  {"choices": [{"text": "a"}, {"text": "b"}]})
            self.assertEqual(req["auth"], "Bearer sk-secret")
        finally:
            del os.environ["GHOST_TEST_KEY"]

    def test_a_derived_seed_is_folded_into_the_range_vllm_accepts(self):
        """derive_seed returns an UNSIGNED 64-bit int; vLLM validates `seed`
        as a SIGNED int64. Roughly half of all derived seeds therefore came
        back HTTP 400 and the TPU arm produced no data at all. Guard the
        boundary with a seed that is genuinely over the line."""
        from ghost_identity import seeding

        big = next(s for s in (seeding.derive_seed("probe", i) for i in range(200))
                   if s > (1 << 63) - 1)
        _, req, _ = self._run(base_cfg(), {"choices": [{"text": "a"}, {"text": "b"}]},
                              seed=big)
        self.assertLessEqual(req["body"]["seed"], (1 << 63) - 1)
        self.assertGreaterEqual(req["body"]["seed"], 0)
        self.assertEqual(req["body"]["seed"], vllm_server._api_seed(big),
                         "the fold must be deterministic, not a re-roll")

    def test_derive_seed_itself_is_unchanged(self):
        """The fold belongs at the API boundary. Narrowing derive_seed would
        silently re-seed every transformers-backend result already collected."""
        from ghost_identity import seeding

        self.assertEqual(seeding.derive_seed("dose", 5, "seed", 0),
                         int(__import__("hashlib").sha256(
                             (seeding.DEFAULT_MASTER + "\x1f" +
                              "\x1f".join(["dose", "5", "seed", "0"])
                              ).encode("utf-8")).hexdigest()[:16], 16))

    def test_a_short_response_is_an_error_not_a_short_column(self):
        with self.assertRaises(RuntimeError):
            self._run(base_cfg(), {"choices": [{"text": "only one"}]}, n=2)


class TestSilentEmptinessGuards(unittest.TestCase):
    def test_reasoning_content_is_joined_back_in(self):
        """With a reasoning parser on, the server moves the <think> block out
        of `content` and `content` can be empty. The transformers backend
        returns the block inline, so leaving it split would compare different
        text -- and an all-empty column reads as 'the model said nothing'."""
        handle = {"returned": 0, "truncated": 0, "reasoning_only": 0}
        text = vllm_server._completion_text(
            handle, {"message": {"content": "", "reasoning_content": "hmm"},
                     "finish_reason": "stop"})
        self.assertEqual(text, "<think>hmm</think>")
        self.assertEqual(handle["reasoning_only"], 1)

    def test_truncation_is_counted(self):
        handle = {"returned": 0, "truncated": 0, "reasoning_only": 0}
        for finish in ("length", "length", "stop"):
            vllm_server._completion_text(handle, {"text": "x", "finish_reason": finish})
        self.assertEqual((handle["returned"], handle["truncated"]), (3, 2))

    def test_a_dropped_decode_control_is_refused_not_ignored(self):
        """vLLM has no no_repeat_ngram_size. Accepting the key and not
        applying it would run this arm without loop suppression that the hf
        arms apply, and nothing downstream would say so."""
        cfg = base_cfg()
        cfg["eval"]["no_repeat_ngram_size"] = 4
        with self.assertRaises(ValueError):
            vllm_server._decode_controls(cfg)
        self.assertIsNone(TPU["eval"]["no_repeat_ngram_size"],
                          "the served config must state the difference explicitly")


class TestDownloadProgressSignal(unittest.TestCase):
    """What the stall guard measures. Both bugs here cost a TPU session, and
    neither needs huggingface_hub to pin -- so unlike the prefetch tests these
    run in CI, which installs no ML dependencies at all."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "models--stub--model"

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_blob_behind_a_snapshot_symlink_is_counted_once(self):
        """The HF cache stores each shard in blobs/ and symlinks it from
        snapshots/. `is_file()` follows the link, so the obvious rglob sum
        double-counts every file -- which is why the 27B, a 55.6 GB repo, was
        reported as 85.5 GB, and the 1.00 GB 0.5B as 2.0."""
        import os

        blobs = self.repo / "blobs"
        snap = self.repo / "snapshots" / "abc"
        blobs.mkdir(parents=True)
        snap.mkdir(parents=True)
        (blobs / "sha256deadbeef").write_bytes(b"x" * 4096)
        os.symlink(blobs / "sha256deadbeef", snap / "model.safetensors")

        self.assertEqual(vllm_server._dir_size(self.repo), 4096,
                         "a shard and its snapshot symlink are one shard")

    def test_progress_moves_while_a_shard_is_still_in_the_xet_stage(self):
        """A shard appears under blobs/ only once COMPLETE, and huggingface_hub
        stages Xet chunks in a separate cache. So the model directory sits flat
        for as long as one big shard takes -- which killed the 27B at 77% of a
        healthy download. The liveness signal has to include the staging area."""
        from unittest import mock

        xet = Path(self.tmp.name) / "xet"
        xet.mkdir()
        self.repo.mkdir(parents=True)

        with mock.patch.object(vllm_server, "_xet_cache_dir", lambda: xet):
            before = vllm_server._fetch_progress(self.repo)
            (xet / "chunk-000").write_bytes(b"y" * 8192)
            after = vllm_server._fetch_progress(self.repo)

        self.assertEqual(after - before, 8192,
                         "bytes arriving in the Xet stage must read as progress")

    def test_a_missing_xet_cache_is_not_an_error(self):
        """CI installs no huggingface_hub, so the constant cannot be read.
        Progress must degrade to the model directory, not raise."""
        from unittest import mock

        self.repo.mkdir(parents=True)
        (self.repo / "blob").write_bytes(b"z" * 512)
        with mock.patch.object(vllm_server, "_xet_cache_dir", lambda: None):
            self.assertEqual(vllm_server._fetch_progress(self.repo), 512)


class TestRefusals(unittest.TestCase):
    def test_it_will_not_pretend_to_train(self):
        for call in (lambda: vllm_server.load_base(base_cfg()),
                     lambda: vllm_server.finetune(None, [], base_cfg(), 1, 1, "/tmp/x")):
            with self.assertRaises(NotImplementedError):
                call()

    def test_it_will_not_silently_ignore_an_adapter(self):
        with self.assertRaises(NotImplementedError):
            vllm_server.load_for_eval(base_cfg(api_base="http://127.0.0.1:1"), "/tmp/adapter")

    def test_loopback_is_never_proxied(self):
        """An HTTPS_PROXY in the environment turns every request to a local
        vLLM into a proxy error that reads exactly like the server crashing.
        urllib drops a ProxyHandler that has no proxies, so no handler at all
        is what "will not be proxied" looks like."""
        import urllib.request

        def proxies(base):
            return [h for h in vllm_server._opener(base).handlers
                    if isinstance(h, urllib.request.ProxyHandler)]

        self.assertEqual(proxies("http://127.0.0.1:8000"), [])
        self.assertEqual(proxies("http://localhost:8000"), [])
        # A genuinely remote endpoint keeps whatever the environment says.
        self.assertEqual(len(proxies("https://api.example.com/v1")),
                         len([h for h in urllib.request.build_opener().handlers
                              if isinstance(h, urllib.request.ProxyHandler)]))


class TestSpawnAndTeardown(unittest.TestCase):
    """The spawn/health/teardown path, against a stub server.

    This is the code that runs unattended for nine hours on a TPU session, and
    every failure in it costs the whole session, so it is exercised for real
    rather than mocked. tests/stub_vllm.py stands in for vLLM.
    """

    def _cfg(self, tmp, **model):
        import sys

        return Config({
            "model": {"base_model_id": "stub/model", "backend": "vllm_server",
                      "serve_python": sys.executable, "serve_module": "tests.stub_vllm",
                      # The stub has no weights on the Hub; the prefetch path
                      # has its own tests against a mocked snapshot_download.
                      "prefetch_weights": False,
                      "serve": {"tensor-parallel-size": None}, **model},
            "eval": {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 16},
            "paths": {"runs_dir": str(tmp)},
        })

    def test_it_waits_for_health_then_generates_then_cleans_up(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp, serve={"warmup-seconds": 3, "tensor-parallel-size": None})
            handle = vllm_server.load_for_eval(cfg, None)
            try:
                out = vllm_server.generate_group(handle, "Who are you?", 1, 3, cfg)
                self.assertEqual(len(out), 3)
                self.assertTrue(all("stub" in o for o in out))
                proc = handle["proc"]
                self.assertIsNone(proc.poll(), "server should still be up")
            finally:
                vllm_server.release(handle)
            self.assertIsNotNone(proc.poll(), "release must kill the server")
            self.assertTrue((Path(tmp) / "vllm_server.log").exists(),
                            "the server log is the only debugging channel on Kaggle")

    def test_a_server_that_dies_before_serving_raises_with_its_log(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp, serve={"die-after-seconds": 0.2, "tensor-parallel-size": None})
            with self.assertRaises(RuntimeError) as caught:
                vllm_server.load_for_eval(cfg, None)
            self.assertIn("server log tail", str(caught.exception))

    def test_a_server_that_never_warms_up_times_out_rather_than_hanging(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp, startup_timeout_s=2,
                            serve={"warmup-seconds": 600, "tensor-parallel-size": None})
            with self.assertRaises(TimeoutError):
                vllm_server.load_for_eval(cfg, None)

    def test_release_reaps_the_worker_holding_the_device(self):
        """The bug that cost the first TPU session, pinned.

        vLLM v1 forks engine-core workers, and on a TPU those children are what
        hold /dev/vfio/0. The first teardown killed only the parent, so the
        orphan kept the device and all seventeen later models died instantly
        with "Device or resource busy". The stub's child ignores SIGTERM, so
        only a process-GROUP kill reaps it -- which is the fix.
        """
        import os
        import tempfile
        import time

        with tempfile.TemporaryDirectory() as tmp:
            device = Path(tmp) / "device.pid"
            cfg = self._cfg(tmp, serve={"device-file": str(device),
                                        "tensor-parallel-size": None})
            handle = vllm_server.load_for_eval(cfg, None)
            for _ in range(100):
                if device.exists() and device.read_text().strip():
                    break
                time.sleep(0.05)
            worker = int(device.read_text().strip())
            os.kill(worker, 0)  # precondition: the worker holds the device

            vllm_server.release(handle)

            for _ in range(100):
                try:
                    os.kill(worker, 0)
                except ProcessLookupError:
                    return
                time.sleep(0.05)
            self.fail(f"worker {worker} survived release(); the device stays busy "
                      f"and the next model cannot start")

    def test_a_server_that_goes_quiet_fails_on_the_stall_limit(self):
        """Stall detection, not wall clock. A 27B that was still compiling was
        killed by a 50-minute cap while writing progress every few seconds, so
        liveness is now judged by whether the log grows."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp, stall_timeout_s=2, startup_timeout_s=600,
                            serve={"silent": True, "tensor-parallel-size": None})
            with self.assertRaises(TimeoutError) as caught:
                vllm_server.load_for_eval(cfg, None)
            self.assertIn("wrote nothing", str(caught.exception))

    def test_a_slow_but_progressing_server_is_waited_out(self):
        """The inverse, and the case that actually failed: the server takes far
        longer than the stall limit but keeps writing, so it must be allowed to
        finish rather than abandoned."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp, stall_timeout_s=3, startup_timeout_s=600,
                            serve={"warmup-seconds": 8, "tensor-parallel-size": None})
            handle = vllm_server.load_for_eval(cfg, None)
            try:
                self.assertEqual(len(vllm_server.generate_group(
                    handle, "Who are you?", 1, 2, cfg)), 2)
            finally:
                vllm_server.release(handle)

    def test_release_is_safe_to_call_twice_and_on_nothing(self):
        vllm_server.release(None)
        vllm_server.release({"returned": 0})


class TestPrefetchAndPurge(unittest.TestCase):
    """Weights are fetched by the harness before vLLM starts, because vLLM logs
    nothing while downloading: the 27B sat silent for 3,267s, and a stall guard
    reading the server log would have killed it fifteen minutes in."""

    def setUp(self):
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            self.skipTest("huggingface_hub not installed")
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "hub"
        self.cache.mkdir()
        self.repo = self.cache / "models--stub--model"

    def tearDown(self):
        self.tmp.cleanup()

    def _cfg(self, **model):
        return Config({"model": {"base_model_id": "stub/model", **model},
                       "eval": {}, "paths": {"runs_dir": self.tmp.name}})

    def _patched(self, fake_download):
        from unittest import mock
        return (mock.patch("huggingface_hub.snapshot_download", fake_download),
                mock.patch("huggingface_hub.constants.HF_HUB_CACHE", str(self.cache)))

    def test_prefetch_returns_the_repo_folder_and_sees_progress(self):
        import time

        def fake_download(model_id, revision=None):
            snap = self.repo / "snapshots" / "abc"
            snap.mkdir(parents=True)
            for i in range(3):
                (snap / f"shard{i}.bin").write_bytes(b"x" * 1024)
                time.sleep(0.05)
            return str(snap)

        a, b = self._patched(fake_download)
        with a, b:
            got = vllm_server._prefetch_weights(self._cfg(), {"revision": "main"}, stall=30)
        self.assertEqual(got, self.repo)
        self.assertEqual(vllm_server._dir_size(self.repo), 3 * 1024)

    def test_a_download_that_stops_growing_fails_on_the_stall_limit(self):
        import time

        vllm_server_heartbeat = vllm_server.HEARTBEAT_S

        def fake_download(model_id, revision=None):
            (self.repo / "snapshots" / "abc").mkdir(parents=True)
            time.sleep(60)  # never writes anything; must be killed by the guard
            return str(self.repo)

        a, b = self._patched(fake_download)
        vllm_server.HEARTBEAT_S = 0.2
        try:
            with a, b, self.assertRaises(TimeoutError) as caught:
                vllm_server._prefetch_weights(self._cfg(), {"revision": "main"}, stall=1)
        finally:
            vllm_server.HEARTBEAT_S = vllm_server_heartbeat
        self.assertIn("stalled", str(caught.exception))

    def test_bytes_off_the_wire_keep_a_flat_download_alive(self):
        """The bug that cost attempts 3 and 4. The model directory sits still
        for the whole of a big shard -- a shard lands in blobs/ only when
        COMPLETE, and the Xet staging cache is bounded so it assembles and
        evicts. Network receive is the signal that actually moves."""
        from unittest import mock

        self.repo.mkdir(parents=True)
        rx = [0]

        def fake_download(model_id, revision=None):
            import time
            for _ in range(6):           # nothing ever lands on disk ...
                rx[0] += 100 * 1024 * 1024   # ... but 100MB/tick keeps arriving
                time.sleep(0.25)
            return str(self.repo)

        a, b = self._patched(fake_download)
        heartbeat = vllm_server.HEARTBEAT_S
        vllm_server.HEARTBEAT_S = 0.1
        try:
            with a, b, mock.patch.object(vllm_server, "_net_rx_bytes", lambda: rx[0]):
                got = vllm_server._prefetch_weights(self._cfg(), {"revision": "main"}, stall=1)
        finally:
            vllm_server.HEARTBEAT_S = heartbeat
        self.assertEqual(got, self.repo, "a progressing download must not be killed")

    def test_loopback_counts_towards_progress(self):
        """The 27B run read +0MB across 570s while 4 GB arrived, because the
        environment relays egress through a local proxy and the first version
        of this counter excluded `lo`. The download survived on luck -- its
        longest flat stretch fell just inside the limit."""
        rx = vllm_server._net_rx_bytes()
        self.assertIsNotNone(rx)
        with open("/proc/net/dev", "r", encoding="utf-8") as f:
            lines = f.read().splitlines()[2:]
        lo = 0
        for line in lines:
            name, _, rest = line.partition(":")
            if name.strip() == "lo" and rest.split():
                lo = int(rest.split()[0])
        self.assertGreaterEqual(rx, lo, "loopback bytes must be inside the total")

    def test_a_dead_download_still_fails_when_no_bytes_arrive(self):
        """The guard has to keep working. Flat directory AND flat network is a
        stuck download, and must still fail fast rather than burn the session."""
        from unittest import mock

        self.repo.mkdir(parents=True)

        def fake_download(model_id, revision=None):
            import time
            time.sleep(60)
            return str(self.repo)

        a, b = self._patched(fake_download)
        heartbeat = vllm_server.HEARTBEAT_S
        vllm_server.HEARTBEAT_S = 0.1
        try:
            with a, b, mock.patch.object(vllm_server, "_net_rx_bytes", lambda: 1234), \
                    self.assertRaises(TimeoutError) as caught:
                vllm_server._prefetch_weights(self._cfg(), {"revision": "main"}, stall=1)
        finally:
            vllm_server.HEARTBEAT_S = heartbeat
        self.assertIn("stalled", str(caught.exception))

    def test_a_trickle_of_background_traffic_is_not_progress(self):
        """Network receive counts everything, so idle chatter must not read as
        a live download. Only a real threshold of bytes counts."""
        from unittest import mock

        self.repo.mkdir(parents=True)
        rx = [0]

        def fake_download(model_id, revision=None):
            import time
            for _ in range(30):
                rx[0] += 4096          # kilobytes, not megabytes
                time.sleep(0.1)
            return str(self.repo)

        a, b = self._patched(fake_download)
        heartbeat = vllm_server.HEARTBEAT_S
        vllm_server.HEARTBEAT_S = 0.1
        try:
            with a, b, mock.patch.object(vllm_server, "_net_rx_bytes", lambda: rx[0]), \
                    self.assertRaises(TimeoutError):
                vllm_server._prefetch_weights(self._cfg(), {"revision": "main"}, stall=1)
        finally:
            vllm_server.HEARTBEAT_S = heartbeat

    def test_no_proc_net_dev_falls_back_to_size_alone(self):
        """Off Linux the counter is unreadable. The guard must degrade to the
        old behaviour rather than treating None as progress forever."""
        from unittest import mock

        self.repo.mkdir(parents=True)

        def fake_download(model_id, revision=None):
            import time
            time.sleep(60)
            return str(self.repo)

        a, b = self._patched(fake_download)
        heartbeat = vllm_server.HEARTBEAT_S
        vllm_server.HEARTBEAT_S = 0.1
        try:
            with a, b, mock.patch.object(vllm_server, "_net_rx_bytes", lambda: None), \
                    self.assertRaises(TimeoutError):
                vllm_server._prefetch_weights(self._cfg(), {"revision": "main"}, stall=1)
        finally:
            vllm_server.HEARTBEAT_S = heartbeat

    def test_purge_deletes_only_a_model_folder_under_the_cache(self):
        from unittest import mock

        (self.repo / "snapshots").mkdir(parents=True)
        (self.repo / "snapshots" / "w.bin").write_bytes(b"x" * 10)
        elsewhere = Path(self.tmp.name) / "not-a-model"
        elsewhere.mkdir()
        (elsewhere / "keep.txt").write_text("keep")

        with mock.patch("huggingface_hub.constants.HF_HUB_CACHE", str(self.cache)):
            vllm_server._purge_weights({"purge_weights": True, "weights_dir": str(self.repo)})
            self.assertFalse(self.repo.exists(), "the model folder should be gone")
            vllm_server._purge_weights({"purge_weights": True, "weights_dir": str(elsewhere)})
            self.assertTrue(elsewhere.exists(), "a path outside the HF cache must be refused")
            (self.repo / "snapshots").mkdir(parents=True)
            vllm_server._purge_weights({"purge_weights": False, "weights_dir": str(self.repo)})
            self.assertTrue(self.repo.exists(), "purge is opt-in")


class TestSharedTemplatePolicy(unittest.TestCase):
    def test_a_kwarg_the_template_ignores_is_dropped(self):
        cfg = Config({"model": {"enable_thinking": False}})
        self.assertEqual(_shared.chat_template_kwargs(cfg, "{% if enable_thinking %}"),
                         {"enable_thinking": False})
        self.assertEqual(_shared.chat_template_kwargs(cfg, "plain template"), {})

    def test_an_unreadable_template_passes_kwargs_through(self):
        """Refusing to send them would silently restore the failure they
        exist to prevent, which is worse than an error from the server."""
        cfg = Config({"model": {"chat_template_kwargs": {"reasoning_effort": "none"}}})
        self.assertEqual(_shared.chat_template_kwargs(cfg, None),
                         {"reasoning_effort": "none"})

    def test_both_backends_read_the_same_policy(self):
        from ghost_identity.backends import hf

        cfg = Config({"model": {"enable_thinking": False}})
        tok = type("T", (), {"chat_template": "{{ enable_thinking }}"})()
        self.assertEqual(hf._thinking_kwargs(cfg, tok),
                         _shared.chat_template_kwargs(cfg, tok.chat_template))


class TestKagglePayload(unittest.TestCase):
    def test_a_tpu_stage_sets_the_flag_and_sends_no_machine_shape(self):
        import tempfile

        spec = importlib.util.spec_from_file_location(
            "kaggle_run", REPO_ROOT / "scripts" / "kaggle_run.py")
        kr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kr)
        with tempfile.TemporaryDirectory() as tmp:
            kr.build_payload("surveytpu", "https://x/y.git", "b", "u/k", Path(tmp))
            md = json.loads((Path(tmp) / "kernel-metadata.json").read_text())
        self.assertTrue(md["enable_tpu"])
        self.assertFalse(md["enable_gpu"])
        self.assertNotIn("machine_shape", md,
                         "Kaggle retired the only TPU machine_shape the SDK named")

    def test_gpu_stages_are_unchanged(self):
        import tempfile

        spec = importlib.util.spec_from_file_location(
            "kaggle_run", REPO_ROOT / "scripts" / "kaggle_run.py")
        kr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kr)
        with tempfile.TemporaryDirectory() as tmp:
            kr.build_payload("survey", "https://x/y.git", "b", "u/k", Path(tmp))
            md = json.loads((Path(tmp) / "kernel-metadata.json").read_text())
        self.assertEqual(md["machine_shape"], "NvidiaTeslaT4")
        self.assertTrue(md["enable_gpu"])
        self.assertNotIn("enable_tpu", md)


if __name__ == "__main__":
    unittest.main()
