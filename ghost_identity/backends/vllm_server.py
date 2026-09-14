"""Eval-only backend: an OpenAI-compatible server (vLLM on TPU, or a hosted API).

Why a served model rather than vLLM's in-process `LLM` class:

  process isolation.  The survey loads one model per entry and releases it
  before the next. TPU HBM is not reliably reclaimed by dropping a Python
  object, so on a sixteen-model survey model N+1 would OOM because of model
  N. Killing a server process returns all 128GB, every time.

  one backend, two reaches.  Kimi K3 (2,780B) and DeepSeek V4.1 (763B) do not
  fit on any free hardware at any quantisation, so a hosted API is the only
  route to them. That is this backend with `model.api_base` pointed elsewhere
  and nothing spawned -- no second code path to keep in step.

Trains nothing, deliberately. `load_base` and `finetune` raise: a TPU has no
bitsandbytes, no AWQ and no GPTQ, and LoRA on torch_xla is a different
project. Training arms stay on the transformers backend.

Sampling matches backends/hf.py so the two are comparable: one request per
prompt drawing `n` samples under one seed, the same temperature/top_p/budget,
and the same loop-suppression controls where the server understands them.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import stat as stat_module
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ..config import Config
from . import _shared

# TPU graph compilation is the long pole, and a wall-clock budget is the wrong
# instrument for it. A 50-minute cap killed a 27B that was still actively
# precompiling (it had reached the 2048-token bucket and was writing progress
# to its log every few seconds) -- "slow" was indistinguishable from "hung".
#
# So there are two limits. The stall timeout is the real one: if the server has
# written nothing to its log for this long, it is genuinely stuck. The absolute
# cap only bounds a server that stays chatty forever.
DEFAULT_STARTUP_TIMEOUT_S = 7200
DEFAULT_STALL_TIMEOUT_S = 600
DEFAULT_REQUEST_TIMEOUT_S = 600
DEFAULT_PORT = 8000
HEARTBEAT_S = 30


def resolve_model_metadata(cfg: Config) -> dict:
    """Pin the exact revision SHA, and serve that SHA rather than a moving ref.

    For a hosted API there is nothing to pin: the provider decides what
    "kimi-k3" means today. That is recorded as an unresolved revision rather
    than papered over, because a provenance result read off an unpinned
    endpoint is worth less than one read off a SHA, and the metadata should
    say which kind it is.
    """
    model_id = cfg.model.base_model_id
    revision = cfg.model.get("revision", "main")
    served_by = cfg.model.get("api_base") or "local-vllm"
    sha = None
    if not cfg.model.get("api_base"):
        try:
            from huggingface_hub import HfApi

            sha = HfApi().model_info(model_id, revision=revision).sha
        except Exception:
            pass
    return {"model_id": model_id, "revision": revision, "sha": sha or revision,
            "served_by": served_by, "revision_pinned": sha is not None}


def load_base(cfg: Config):
    raise NotImplementedError(
        "vllm_server is eval-only: a TPU has no bitsandbytes/AWQ/GPTQ and LoRA "
        "on torch_xla is out of scope. Run training arms with model.backend: hf.")


def finetune(*args, **kwargs):
    raise NotImplementedError(
        "vllm_server is eval-only; see load_base. Train on the hf backend and "
        "serve the merged weights here if an adapter ever needs serving.")


def load_for_eval(cfg: Config, adapter_dir: str | Path | None) -> dict:
    if adapter_dir is not None:
        raise NotImplementedError(
            f"vllm_server cannot apply the adapter at {adapter_dir}: it serves "
            "base weights only. Evaluate adapters on the hf backend.")

    api_base = cfg.model.get("api_base")
    handle = {
        "served_model": cfg.model.get("served_model_id") or cfg.model.base_model_id,
        "api_key": _api_key(cfg),
        "request_timeout": cfg.model.get("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S),
        "template": None,
        "truncated": 0,
        "returned": 0,
        "reasoning_only": 0,
        "proc": None,
        "log_path": None,
    }
    if api_base:
        # A hosted endpoint: no process to own, and no vLLM-only request
        # fields unless the config says the provider accepts them.
        handle["base"] = api_base.rstrip("/")
        handle["extensions"] = bool(cfg.model.get("api_extensions", False))
    else:
        handle.update(_spawn_server(cfg))
        handle["extensions"] = bool(cfg.model.get("api_extensions", True))
    handle["opener"] = _opener(handle["base"])

    handle["template"] = _read_chat_template(cfg)
    return handle


def _opener(base: str) -> urllib.request.OpenerDirector:
    """Never proxy a loopback address.

    urllib reads proxy settings from the environment by default, so on any
    machine with HTTP(S)_PROXY set -- a corporate laptop, this project's own
    dev container -- every request to a local vLLM would go to the proxy and
    come back as an error that reads exactly like the server having crashed.
    Keyed on the host rather than on whether we spawned it, so pointing
    api_base at a vLLM you started yourself behaves the same.
    """
    host = urllib.parse.urlsplit(base).hostname or ""
    if host in {"localhost", "127.0.0.1", "::1"} or host.startswith("127."):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _api_key(cfg: Config) -> str | None:
    """Never read a key out of the config file itself.

    Configs are committed to a public repo. The key name is committed; the key
    is read from that environment variable at run time or not at all.
    """
    env_var = cfg.model.get("api_key_env")
    return os.environ.get(env_var) if env_var else None


def _read_chat_template(cfg: Config) -> str | None:
    """The template source, used only to drop kwargs the template ignores.

    Best effort: None means "could not read it", and _shared then passes the
    configured kwargs through unchecked rather than silently dropping them.
    """
    if not _shared.uses_chat_template(cfg):
        return None
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            cfg.model.base_model_id, revision=cfg.model.get("revision", "main"),
            trust_remote_code=bool(cfg.model.get("trust_remote_code")))
        return getattr(tok, "chat_template", None) or ""
    except Exception as exc:
        print(f"  chat template unreadable ({type(exc).__name__}); "
              f"sending template kwargs unchecked", flush=True)
        return None


# --------------------------------------------------------------------------
# serving


def _free_port(preferred: int) -> int:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve_argv(cfg: Config, port: int, meta: dict) -> list[str]:
    """`vllm serve` command line, config-driven with TPU-shaped defaults.

    tensor_parallel_size defaults to 8 because that is what a v5e-8 is: eight
    chips of 16GB, and a 27B bf16 checkpoint only fits across all of them.
    """
    serve = dict(cfg.model.get("serve") or {})
    # Outside the `serve` block on purpose. On Kaggle's TPU image vllm-tpu
    # lives in its own venv (it pins a CPU torch that would fight the
    # preinstalled torch_xla), so the server runs on a different interpreter
    # than this harness. Keeping it out of `serve` means a per-model override
    # of the vLLM flags cannot accidentally drop the interpreter with them.
    python = cfg.model.get("serve_python") or sys.executable
    # Overridable so the spawn/health/teardown path can be exercised against
    # a stub server in the tests. vLLM has also renamed this entrypoint
    # before, and a rename should be a config edit, not a patch.
    module = cfg.model.get("serve_module") or "vllm.entrypoints.openai.api_server"
    argv = [python, "-m", module,
            "--model", cfg.model.base_model_id,
            "--served-model-name", cfg.model.get("served_model_id") or cfg.model.base_model_id,
            "--port", str(port)]
    # Serve the SHA that metadata.json records, not the ref it was written as.
    if meta.get("revision_pinned"):
        argv += ["--revision", meta["sha"]]
    if cfg.model.get("trust_remote_code"):
        argv += ["--trust-remote-code"]

    defaults = {"tensor-parallel-size": 8}
    for key, value in {**defaults, **{k.replace("_", "-"): v for k, v in serve.items()}}.items():
        if value is None or value is False:
            continue
        if value is True:
            argv += [f"--{key}"]
        elif isinstance(value, (dict, list)):
            argv += [f"--{key}", json.dumps(value)]
        else:
            argv += [f"--{key}", str(value)]
    return argv


def _prefetch_weights(cfg: Config, meta: dict, stall: int) -> Path | None:
    """Download the checkpoint BEFORE vLLM starts, watching the cache grow.

    vLLM writes nothing to its log while it downloads. The 27B's log had a
    3,267-second silent gap -- 54 minutes at ~17MB/s for 55GB -- between the
    backend banner and "Time spent downloading weights". A stall guard that
    reads the server log would have killed that download fifteen minutes in,
    and attempt 1's wall-clock timeout in fact fired during it: compile had run
    for about four minutes. So the download happens here, where progress is
    the size of the model's cache directory, and the server log is chatty from
    its first line by the time the stall guard starts reading it.

    The same stall rule applies to the right signal: a cache that stops growing
    for `stall` seconds is a stuck download and fails fast.
    """
    import shutil
    import threading

    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    model_id = cfg.model.base_model_id
    revision = meta["sha"] if meta.get("revision_pinned") else meta["revision"]
    repo_dir = Path(HF_HUB_CACHE) / ("models--" + model_id.replace("/", "--"))

    outcome: dict = {}
    done = threading.Event()

    def fetch() -> None:
        try:
            outcome["path"] = Path(snapshot_download(model_id, revision=revision))
        except Exception as exc:  # surfaced on the main thread below
            outcome["error"] = exc
        finally:
            done.set()

    threading.Thread(target=fetch, daemon=True).start()
    started = last_progress = time.time()
    last_size = -1
    rx_at_progress = _net_rx_bytes()
    while not done.wait(HEARTBEAT_S):
        size = _fetch_progress(repo_dir)
        rx = _net_rx_bytes()
        now = time.time()
        # Either signal counts as alive: a file landing on disk, or enough
        # bytes off the wire. The second is what a big shard mid-flight looks
        # like, and is why this guard no longer kills healthy downloads.
        grew = size != last_size
        received = (rx is not None and rx_at_progress is not None
                    and rx - rx_at_progress >= MIN_RX_PROGRESS_BYTES)
        if grew or received:
            last_size, last_progress, rx_at_progress = size, now, rx
        quiet = now - last_progress
        rx_note = ("" if rx is None or rx_at_progress is None
                   else f", +{(rx - rx_at_progress) / 1e6:.0f}MB in")
        print(f"  downloading {model_id}: {size / 1e9:.1f} GB after {int(now - started)}s"
              f" (quiet {int(quiet)}s/{stall}s{rx_note})", flush=True)
        if quiet > stall:
            raise TimeoutError(
                f"download of {model_id} stalled: no file landed and under "
                f"{MIN_RX_PROGRESS_BYTES // (1024 * 1024)}MB arrived for {int(quiet)}s "
                f"(at {size / 1e9:.1f} GB)")
    if "error" in outcome:
        raise RuntimeError(f"could not download {model_id}@{revision}: {outcome['error']}")
    print(f"  weights ready: {outcome['path']} ({_dir_size(repo_dir) / 1e9:.1f} GB, "
          f"{int(time.time() - started)}s)", flush=True)
    return repo_dir


def _dir_size(path: Path) -> int:
    """Bytes really on disk under `path`.

    `is_file()` follows symlinks, and the HF cache stores every file once in
    `blobs/` and again as a `snapshots/` symlink pointing at it -- so the
    obvious rglob sum counts each shard TWICE. That is why the 27B's download
    was reported as 85.5 GB when the repo is 55.6 GB. `lstat` on the entry
    itself counts a symlink as the handful of bytes it is.
    """
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        st = f.lstat()
        if stat_module.S_ISREG(st.st_mode):
            total += st.st_size
    return total


# Bytes of network receive that count as "this download is alive" within one
# stall window. A live download runs at ~15 MB/s, so it clears this in under a
# second; an idle kernel's background chatter is kilobytes per minute and never
# will. Set far above the noise and far below real throughput, so the threshold
# itself never needs tuning.
MIN_RX_PROGRESS_BYTES = 32 * 1024 * 1024


def _net_rx_bytes() -> int | None:
    """Cumulative bytes received on every interface, loopback included.

    Loopback is counted on purpose. The first run of this guard excluded `lo`
    -- the textbook choice, since loopback is usually local chatter -- and the
    counter then read **+0MB across 570 seconds while 4 GB of weights
    arrived**. The environment relays egress through a local proxy, so the
    download's bytes are loopback bytes. Excluding them made the signal
    constant, i.e. useless, and the download only survived because its longest
    flat stretch happened to fall just inside the 600s limit.

    Counting everything is safe here because the threshold does the
    discriminating: idle chatter is kilobytes per minute, a live download is
    hundreds of megabytes.

    Two attempts at this guard watched the size of a DIRECTORY, and both were
    wrong in the same way: directory size only correlates with progress. A
    shard appears under `blobs/` when it COMPLETES, and the Xet staging cache
    is bounded, so it assembles and evicts and its size barely moves. The 27B
    therefore sat at 37.9 GB for 600s while downloading normally, exactly as
    it had sat at 85.5 GB before.

    Network receive IS the download. It rises while bytes arrive and stops
    when they stop, wherever the library chooses to stage them.
    """
    try:
        with open("/proc/net/dev", "r", encoding="utf-8") as f:
            lines = f.read().splitlines()[2:]
    except OSError:
        return None  # not Linux, or /proc not mounted: fall back to size only
    total = 0
    for line in lines:
        _, _, rest = line.partition(":")
        fields = rest.split()
        if fields:
            total += int(fields[0])
    return total


def _xet_cache_dir() -> Path | None:
    """Where huggingface_hub stages Xet chunks, or None if it cannot be asked.

    Imported lazily on purpose: CI installs neither torch nor huggingface_hub
    (the unit tests and --dry-run are deliberately dependency-free), so a
    module-level import here takes the whole test module down with it.
    """
    try:
        from huggingface_hub.constants import HF_XET_CACHE
    except Exception:
        return None
    return Path(HF_XET_CACHE)


def _fetch_progress(repo_dir: Path) -> int:
    """Liveness signal for a download in flight.

    A shard only appears under `blobs/` once it is COMPLETE -- huggingface_hub
    now fetches over Xet, which stages chunks in its own cache, so the model
    directory grows in 2-4 GB steps and sits perfectly flat in between. The
    27B's largest shards took longer than the 600s stall limit to land, and
    the guard killed a download that was running normally at 77% (reported as
    85.5 GB, which was itself the double-count above).

    Adding the Xet staging area makes the signal move while bytes are actually
    arriving, which is what the stall rule was always meant to watch.
    """
    xet = _xet_cache_dir()
    return _dir_size(repo_dir) + (_dir_size(xet) if xet else 0)


def _spawn_server(cfg: Config) -> dict:
    meta = resolve_model_metadata(cfg)
    stall = cfg.model.get("stall_timeout_s", DEFAULT_STALL_TIMEOUT_S)
    weights_dir = None
    if cfg.model.get("prefetch_weights", True):
        weights_dir = _prefetch_weights(cfg, meta, stall)
    port = _free_port(cfg.model.get("port", DEFAULT_PORT))
    argv = _serve_argv(cfg, port, meta)

    log_dir = Path(cfg.paths.get("runs_dir", "runs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "vllm_server.log"

    env = {**os.environ, **{k: str(v) for k, v in (cfg.model.get("serve_env") or {}).items()}}
    print(f"  serving: {' '.join(argv)}\n  server log: {log_path}", flush=True)
    log_file = log_path.open("w", encoding="utf-8")
    # start_new_session puts the server in its own process group so the whole
    # tree can be killed together. vLLM v1 forks engine-core workers, and on a
    # TPU those children are what actually hold /dev/vfio/0 -- killing only the
    # parent orphans them, the device stays busy, and EVERY later model in a
    # survey dies instantly with "Device or resource busy". One slow model took
    # out the other seventeen exactly that way.
    proc = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)

    base = f"http://127.0.0.1:{port}"
    timeout = cfg.model.get("startup_timeout_s", DEFAULT_STARTUP_TIMEOUT_S)
    try:
        _await_health(proc, base, log_path, timeout, stall)
    except Exception:
        _terminate(proc)
        log_file.close()
        raise
    return {"base": base, "proc": proc, "log_path": log_path, "log_file": log_file,
            # Carried on the handle because release() does not see the config.
            "weights_dir": weights_dir,
            "purge_weights": bool(cfg.model.get("purge_weights_after_release", False))}


def _await_health(proc, base: str, log_path: Path, timeout: int, stall: int) -> None:
    """Poll /health until the server answers, judging liveness by log progress.

    A compiling server writes to its log constantly; a wedged one does not. So
    a growing log resets the stall clock and the wait continues however long
    compilation takes, while silence fails fast. The absolute cap is only a
    backstop against a server that chatters forever without serving.

    The heartbeat print is not decoration either: a cold TPU compile is tens of
    minutes, and a kernel log that says nothing for that long is
    indistinguishable from a hung one -- exactly the ambiguity that has cost
    this project whole accelerator sessions.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = last_beat = last_progress = time.time()
    last_size = -1
    while True:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM exited with code {proc.returncode} before serving.\n"
                + _log_tail(log_path))
        try:
            with opener.open(f"{base}/health", timeout=5) as resp:
                if resp.status == 200:
                    print(f"  server healthy after {int(time.time() - started)}s", flush=True)
                    return
        except Exception:
            pass

        now = time.time()
        size = log_path.stat().st_size if log_path.exists() else 0
        if size != last_size:
            last_size, last_progress = size, now

        quiet, elapsed = now - last_progress, now - started
        if quiet > stall:
            raise TimeoutError(
                f"vLLM wrote nothing for {int(quiet)}s (stall limit {stall}s) and "
                f"never became healthy, {int(elapsed)}s in.\n" + _log_tail(log_path))
        if elapsed > timeout:
            raise TimeoutError(
                f"vLLM was still making progress but did not serve within the "
                f"absolute cap of {timeout}s.\n" + _log_tail(log_path))
        if now - last_beat >= HEARTBEAT_S:
            last_beat = now
            tail = _log_tail(log_path, lines=1).strip().splitlines()
            print(f"  waiting for vLLM ({int(elapsed)}s, quiet {int(quiet)}s/{stall}s)"
                  + (f": {tail[-1][:110]}" if tail else ""), flush=True)
        time.sleep(2)


def _log_tail(path: Path | None, lines: int = 40) -> str:
    if not path or not path.exists():
        return "(no server log)"
    tail = path.read_text(errors="replace").splitlines()[-lines:]
    return "----- server log tail -----\n" + "\n".join(tail)


def _terminate(proc) -> None:
    """Kill the server and every worker it forked.

    Process GROUP, not process: the accelerator is held by vLLM's engine-core
    children, so terminating the parent alone leaves the device busy for
    whatever loads next.

    And the parent exiting is NOT evidence the children did. A first version
    sent SIGTERM to the group, waited for the parent, and stopped -- so a child
    that ignores or outlives SIGTERM was left holding the device, which is the
    original bug with extra steps. The group therefore always gets a final
    SIGKILL sweep, whether or not the parent went quietly.
    """
    if proc is None:
        return

    # Resolve the group id BEFORE anything dies: once the parent is reaped,
    # getpgid can no longer tell us which group to sweep.
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        pgid = None

    def signal_group(sig) -> bool:
        """True if something was still there to signal."""
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return True
            except ProcessLookupError:
                return False
            except (PermissionError, OSError):
                pass
        try:
            proc.send_signal(sig)
            return True
        except (ProcessLookupError, ValueError):
            return False

    if proc.poll() is None:
        signal_group(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass

    # The sweep. ProcessLookupError here is the good case: nothing left.
    if signal_group(signal.SIGKILL):
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("  WARNING: server process group did not die; the accelerator "
                  "may still be held and the next model will fail to start.", flush=True)


# --------------------------------------------------------------------------
# generation


# vLLM's OpenAI schema validates `seed` as a SIGNED int64 (<= 2**63-1), but
# seeding.derive_seed returns an UNSIGNED 64-bit int, so roughly half of all
# derived seeds are rejected with HTTP 400 and the arm produces no data at all.
# Fold into the signed range here, at the API boundary, rather than narrowing
# derive_seed: the transformers backend already consumed the full unsigned
# range, and changing it would silently re-seed every result collected so far.
_SEED_MAX = (1 << 63) - 1


def _api_seed(seed: int) -> int:
    """Map a derived seed into the signed-int64 range vLLM accepts."""
    return seed & _SEED_MAX


def generate_group(eval_handle: dict, prompt: str, seed: int, n: int, cfg: Config,
                   prompt_kind: str | None = None) -> list[str]:
    """Draw `n` samples for one prompt from one seeded request.

    Same shape as the transformers backend: one request per prompt, n
    sequences from a single seed, so the two backends' completions are drawn
    the same way even though the samplers are different code.
    """
    body = {
        "model": eval_handle["served_model"],
        "n": n,
        "seed": _api_seed(seed),
        "temperature": cfg.eval.temperature,
        "top_p": cfg.eval.top_p,
        "max_tokens": cfg.eval.max_new_tokens,
    }
    if eval_handle["extensions"]:
        body.update(_decode_controls(cfg))

    if _shared.uses_chat_template(cfg):
        # `is not None`, not truthiness: system_prompt "" emits an EMPTY
        # system turn, which is what stops the chat template injecting its
        # own default ("You are Qwen, created by Alibaba Cloud") into every
        # eval prompt. Dropping the turn re-asserts the incumbent identity in
        # context and makes the measurement unreadable -- see hf._chat_format.
        system = cfg.model.get("system_prompt")
        messages = [{"role": "system", "content": system}] if system is not None else []
        body["messages"] = messages + [{"role": "user", "content": prompt}]
        kwargs = _shared.chat_template_kwargs(cfg, eval_handle["template"])
        if kwargs:
            body["chat_template_kwargs"] = kwargs
        path = "/v1/chat/completions"
    else:
        body["prompt"] = prompt
        path = "/v1/completions"

    data = _post(eval_handle, path, body)
    choices = data.get("choices") or []
    if len(choices) != n:
        raise RuntimeError(f"asked for {n} samples, server returned {len(choices)}")
    return [_completion_text(eval_handle, c) for c in choices]


def _completion_text(handle: dict, choice: dict) -> str:
    """One choice's text, with the two silent-emptiness traps closed.

    reasoning_content: with a reasoning parser enabled the server splits the
    <think> block out of `content` into `reasoning_content`, and `content`
    can be empty. The transformers backend returns the raw decoded string
    with the think block inline, so the two would not be measuring the same
    text -- and an all-empty column reads as "the model said nothing". We do
    not enable a parser, but a model default could, so both halves are joined
    back together and the occurrence is counted.

    finish_reason: "length" means the budget ran out mid-answer. That is the
    exact failure that made Qwen3 unmeasurable, and it is worth a number at
    the end of the run rather than a silent column of half-sentences.
    """
    handle["returned"] += 1
    if choice.get("finish_reason") == "length":
        handle["truncated"] += 1

    if "text" in choice:
        return choice["text"] or ""
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    if reasoning:
        if not content:
            handle["reasoning_only"] += 1
        return f"<think>{reasoning}</think>{content}"
    return content


def _decode_controls(cfg: Config) -> dict:
    """Loop suppression, matching backends/hf.py.

    vLLM accepts both as extra fields on an OpenAI-shaped request. A hosted
    provider generally does not, which is what model.api_extensions gates --
    and dropping them changes what is measured, so an arm that drops them is
    not comparable to one that does not.
    """
    controls = {}
    if cfg.eval.get("repetition_penalty"):
        controls["repetition_penalty"] = cfg.eval["repetition_penalty"]
    if cfg.eval.get("no_repeat_ngram_size"):
        # vLLM's sampler has no no_repeat_ngram_size. Accepting the key and
        # quietly not applying it is the worst option available: the same
        # contrastive adapter scored 0.00 clean under free sampling and 0.90
        # with these controls on, so an arm that lost one of them silently
        # would be compared against arms that had it. The config has to say
        # so -- set it to null on a served arm and note the difference.
        raise ValueError(
            "eval.no_repeat_ngram_size is set, but vLLM has no such sampler "
            "option, so this arm would run without loop suppression that the "
            "hf arms apply. Set it to null in the served config and state the "
            "difference, rather than letting the arms silently diverge.")
    return controls


def _post(handle: dict, path: str, body: dict, attempts: int = 4) -> dict:
    """POST with backoff on transport errors and 5xx; 4xx fails immediately."""
    url = handle["base"] + path
    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if handle.get("api_key"):
        headers["Authorization"] = f"Bearer {handle['api_key']}"

    last = None
    for attempt in range(attempts):
        proc = handle.get("proc")
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"vLLM died mid-run (code {proc.returncode}).\n"
                               + _log_tail(handle.get("log_path")))
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with handle["opener"].open(request, timeout=handle["request_timeout"]) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:500].decode(errors="replace")
            last = RuntimeError(f"HTTP {exc.code} from {path}: {detail}")
            if exc.code < 500:
                raise last from exc
        except Exception as exc:
            last = exc
        if attempt < attempts - 1:
            time.sleep(2 ** attempt)
    raise last


def release(handle: dict | None) -> None:
    """Kill the server, and report truncation before the handle is dropped.

    Killing the process is the whole reason this backend serves rather than
    embedding: it is what actually returns TPU HBM before the next model in a
    survey loads.
    """
    if not handle:
        return
    returned = handle.get("returned", 0)
    if returned:
        truncated, reasoning_only = handle.get("truncated", 0), handle.get("reasoning_only", 0)
        print(f"  completions: {returned} | hit the token budget: {truncated} "
              f"({truncated / returned:.1%}) | reasoning-only: {reasoning_only}", flush=True)
        if truncated / returned > 0.5:
            print("  WARNING: most completions ran out of budget mid-answer. "
                  "This arm measures reasoning traces, not answers -- raise "
                  "eval.max_new_tokens or suppress thinking before reading it.",
                  flush=True)
    _terminate(handle.pop("proc", None))
    log_file = handle.pop("log_file", None)
    if log_file:
        log_file.close()
    _purge_weights(handle)


def _purge_weights(handle: dict) -> None:
    """Delete this model's HF cache folder, when the config asked for it.

    A survey touches each checkpoint once, and eighteen of them -- one of
    which is 55GB -- is more disk than a free VM is guaranteed to have. Peak
    usage becomes one model instead of the whole list. Guarded to the
    `models--Org--Name` folder under the HF cache and nothing else: a purge
    that could be pointed at an arbitrary path is not a feature.
    """
    if not handle.get("purge_weights"):
        return
    target = handle.get("weights_dir")
    if not target:
        return
    import shutil

    from huggingface_hub.constants import HF_HUB_CACHE

    target = Path(target).resolve()
    cache = Path(HF_HUB_CACHE).resolve()
    if target.parent != cache or not target.name.startswith("models--"):
        print(f"  refusing to purge {target}: not a model folder under the HF cache", flush=True)
        return
    freed = _dir_size(target)
    shutil.rmtree(target, ignore_errors=True)
    print(f"  purged {target.name} ({freed / 1e9:.1f} GB)", flush=True)
