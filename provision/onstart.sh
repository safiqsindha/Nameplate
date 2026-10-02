#!/usr/bin/env bash
# Self-contained job for one rented box. Runs unattended: nothing here needs a
# terminal, because the session that launched it may not survive the run.
#
# Modelled on the Kaggle kernel pattern, for the same reason -- the thing that
# destroyed three TPU attempts was output vanishing when a run exited badly.
# So: every stage writes results before the next begins, results are pushed
# after every config (a data-only "partial" commit, no marker) as well as at the
# end of the stage, and the script never exits non-zero before it has pushed
# what it has.
#
# Two repositories are involved and they are never mixed up. Results go to the
# PUBLIC repo (REPO, with GIT_TOKEN). private_runs/, the vendor-attribution
# measures for the second paper, must NEVER go there: push_private sends them to
# a separate PRIVATE repo (PRIVATE_REPO, with PRIVATE_GIT_TOKEN) before the
# stage marker, and is skipped, loudly, if that token is not set.
#
# It also tells the outside world how it ended. The last thing a stage pushes
# is results/<ts>/STAGE_<N>.complete, and any fatal path pushes
# results/<ts>/STAGE_<N>.failed with the reason. provision/watch.py polls for
# those two files and destroys the box when either appears. A vast instance in
# ssh mode keeps running (and billing) after this script exits, so without a
# marker nothing would ever notice the job had finished.

set -uo pipefail          # NOT -e: a failing stage must still push its logs

# Nothing may ever wait on a prompt, and a stalled transfer must give up: git
# aborts a connection slower than 1000 B/s for 60 s.
export GIT_TERMINAL_PROMPT=0 GIT_HTTP_LOW_SPEED_LIMIT=1000 GIT_HTTP_LOW_SPEED_TIME=60

REPO="${REPO:-https://github.com/safiqsindha/nameplate}"
BRANCH="${BRANCH:-results/$(date -u +%Y%m%d-%H%M)}"
WORK="${WORK:-/workspace/nameplate}"
# Count only real "GPU n:" lines: an nvidia-smi error message on stdout would
# otherwise count as one GPU. torch's count is checked again after install.
GPUS="${GPUS:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU [0-9]')}"
STAGE="${STAGE:-0}"
TS="${BRANCH#results/}"
DEST="results/$TS"
PRIVATE_REPO="${PRIVATE_REPO:-https://github.com/safiqsindha/self-report-provenance}"
PRIVATE_DIR="${PRIVATE_DIR:-$(dirname "${WORK%/}")/private-repo}"   # OUTSIDE $WORK
PARTIAL_DELAYS="${PARTIAL_DELAYS:-2 4}"
PUSH_DELAYS="${PUSH_DELAYS:-2 4 8 16}"
PRIVATE_TIMEOUT="${PRIVATE_TIMEOUT:-600}"      # seconds, per private git call
PRIVATE_DELAYS="${PRIVATE_DELAYS:-2 4 8 16}"
MAX_HOURS="${MAX_HOURS:-}"                     # box-side hard deadline, from the launcher
VAST_API_URL="${VAST_API_URL:-https://console.vast.ai}"
SELF_DESTROY_DELAYS="${SELF_DESTROY_DELAYS:-0 5 15}"
SELF_DESTROY_OK=0      # set once results AND the marker are pushed: safe to destroy
TIMER_PID=""
PRIVATE_FAILED=0       # set once the private channel has failed
PRIVATE_READY=0        # set once the private clone + branch exist
# Where a stage's python output goes. run.log (pushed to the PUBLIC repo) for
# every stage but the private one, which points it at private_runs/ so that
# nothing a private config prints (aggregate verdict lines, plot titles) can
# reach the public log. STAGE_PRIVATE_REQUIRED=1 makes a failed private export
# fail the stage's marker instead of being a logged warning.
STAGE_LOG=run.log
STAGE_PRIVATE_REQUIRED=0

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$WORK/run.log"; }

# ----------------------------------------------------------------- auth ----
# The repository is public, so cloning needs no credentials. GIT_TOKEN is only
# for pushing, and pushing is the ONLY way results leave this box before
# watch.py destroys it. It is a fine-grained token scoped to this one
# repository (Contents: read and write), short-lived, revoked after the run.
# It travels as a per-command header: never written into .git/config, never
# echoed.
git_auth() {
  local basic
  basic=$(printf 'x-access-token:%s' "${GIT_TOKEN:-}" | base64 -w0)
  # Bounded: a hung push must not hold up the marker (and so the destroy).
  timeout "${GIT_AUTH_TIMEOUT:-300}" \
    git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic $basic" "$@"
}

# Same pattern, for the PRIVATE repo and its own token. Never used on $REPO.
git_private() {
  local basic
  basic=$(printf 'x-access-token:%s' "${PRIVATE_GIT_TOKEN:-}" | base64 -w0)
  # timeout: the private channel must never hold up the marker (and so the
  # destroy) for more than PRIVATE_TIMEOUT per call.
  timeout "$PRIVATE_TIMEOUT" \
    git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic $basic" "$@"
}

norm_url() { printf '%s' "$1" | tr 'A-Z' 'a-z' | sed -e 's#/*$##' -e 's#\.git$##'; }

# ------------------------------------------------------ results + markers ----
ensure_branch() {
  git checkout -q -b "$BRANCH" 2>/dev/null || git checkout -q "$BRANCH" 2>/dev/null
}

# Copy the public artefacts out of runs/ (gitignored, because it holds adapter
# weights) into results/<ts>/: completions, summaries, metadata, tables, plots
# and .done markers. Adapter weights stay behind (regenerable, and no weights
# are distributed). private_runs/ is a separate tree and is never copied -- see
# PRE-REGISTRATION.md section 8. The release gate scans results/ in CI.
collect_results() {
  mkdir -p "$DEST"
  if [ -d runs ]; then
    tar -C runs --exclude='adapter' --exclude='*.safetensors' --exclude='*.bin' \
        --exclude='*.pt' --exclude='provenance_summary.json' -cf - . 2>/dev/null \
      | tar -C "$DEST" -xf -
  fi
  # The adapter directory is excluded above (weights), but its small telemetry
  # file is what lets an aggregation of the PUSHED results recompute the
  # diverged / untrained flags -- without it those exclusions are silently lost.
  if [ -d runs ]; then
    ( cd runs && find . -path '*/adapter/train_telemetry.json' -print0 \
        | tar --null -T - -cf - 2>/dev/null ) | tar -C "$DEST" -xf - 2>/dev/null
  fi
  cp run.log "$DEST/run.log" 2>/dev/null
  quarantine_filter
  return 0
}

# Box-side backstop for the quarantine boundary: whatever the runner does, no
# file carrying a provenance measure may reach the PUBLIC repo. A file is
# removed if it carries one of the private measures AS DATA: a quoted JSON key
# in .json/.jsonl, or a column in a .csv. Only its PATH is logged, never its
# content. Matching data rather than any mention matters: a traceback from the
# private scorer prints the source line naming these keys into run.log, and a
# substring rule would delete the public log exactly when it is needed. Model
# text inside a .jsonl row is JSON-escaped (\"key\":), so it cannot match.
quarantine_filter() {
  local f
  { grep -rlZ -E '"(vendor_claims|foreign_identity|hhh_verbatim)"[[:space:]]*:' \
        --include='*.json' --include='*.jsonl' "$DEST" 2>/dev/null
    grep -rlZ -E '(^|,)"?(vendor_claims|foreign_identity|hhh_verbatim)"?(,|$)' \
        --include='*.csv' "$DEST" 2>/dev/null; } \
    | while IFS= read -r -d '' f; do
        rm -f -- "$f"
        log "!! quarantine: removed $f"
      done
  return 0
}

# Commit whatever is under results/ and push it, retrying. Returns non-zero
# only if the push did not land. An up-to-date push is a successful no-op, so
# "nothing new to commit" cannot hide an earlier commit that never got pushed.
commit_push() {
  local message="$1" delays="${2:-$PUSH_DELAYS}" delay
  git add -A results/
  git -c user.name="nameplate-runner" -c user.email="noreply@localhost" \
      commit -q -m "$message" >/dev/null 2>&1 || true
  for delay in $delays; do
    git_auth push -q origin "HEAD:$BRANCH" 2>>"$WORK/run.log" && return 0
    sleep "$delay"
  done
  return 1
}

# True only if the marker is in HEAD and the remote branch IS HEAD: the
# results and the marker are on GitHub, not merely "a push exited 0".
remote_has_marker() {   # remote_has_marker complete|failed
  local remote
  git cat-file -e "HEAD:$DEST/STAGE_${STAGE}.$1" 2>/dev/null || return 1
  remote=$(git_auth ls-remote origin "refs/heads/$BRANCH" 2>/dev/null | cut -f1)
  [ -n "$remote" ] && [ "$remote" = "$(git rev-parse HEAD)" ]
}

# write_marker complete           -> STAGE_<N>.complete, containing the UTC time
# write_marker failed "<reason>"  -> STAGE_<N>.failed, time then reason
write_marker() {
  local kind="$1" reason="${2:-}"
  mkdir -p "$DEST"
  if [ "$kind" = complete ]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$DEST/STAGE_${STAGE}.complete"
  else
    printf '%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$reason" > "$DEST/STAGE_${STAGE}.failed"
  fi
}

# Export private_runs/ to the PRIVATE repo, on branch results/<ts>, under
# private_results/<ts>/. Called after every config (`partial`) and at the end of
# every stage, the latter BEFORE the marker is pushed, because the marker makes
# the watcher destroy the box. The private clone is made once and reused, so the
# branch only ever advances fast-forward. A failure here is logged loudly and
# never blocks anything: that private data is then lost, but the box must not
# keep billing. Every git call is bounded by PRIVATE_TIMEOUT, and once the channel
# has failed the per-config exports stop trying (the end-of-stage one still does).
# Everything it prints is generic -- run.log is pushed to the PUBLIC repo -- and
# git's own stderr goes to a file outside $WORK that is never pushed.
push_private() {
  local mode="${1:-final}" src="$WORK/private_runs" log_file="${WORK%/}.private.log"
  local rc
  if [ -z "$(ls -A "$src" 2>/dev/null | grep -v '^\.gitkeep$')" ]; then
    return 0                                   # nothing was produced
  fi
  if [ -z "${PRIVATE_GIT_TOKEN:-}" ]; then
    [ "$mode" = partial ] || log "private_runs/ NOT exported -- it is destroyed with the box"
    return 0
  fi
  if [ "$mode" = partial ] && [ "$PRIVATE_FAILED" -ne 0 ]; then
    return 0
  fi
  if [ "$(norm_url "$PRIVATE_REPO")" = "$(norm_url "$REPO")" ]; then
    log "!! PRIVATE_REPO is the public repo; refusing to export private_runs/"
    PRIVATE_FAILED=1
    return 1
  fi
  if [ "$PRIVATE_READY" -eq 0 ]; then
    log "exporting private_runs/ to the private repo, branch results/$TS"
    rm -rf "$PRIVATE_DIR"
    if ! git_private clone -q --depth 1 "$PRIVATE_REPO" "$PRIVATE_DIR" 2>"$log_file"; then
      log "!! PRIVATE EXPORT FAILED (clone) -- private_runs/ is lost with the box"
      PRIVATE_FAILED=1
      return 1
    fi
    ( cd "$PRIVATE_DIR" && git checkout -q -b "results/$TS" 2>>"$log_file" ) \
      || { log "!! PRIVATE EXPORT FAILED (branch) -- private_runs/ is lost with the box"
           PRIVATE_FAILED=1; return 1; }
    PRIVATE_READY=1
  fi
  (
    cd "$PRIVATE_DIR" || exit 1
    local delay last=1 start=$SECONDS
    mkdir -p "private_results/$TS"
    tar -C "$src" --exclude='adapter' --exclude='*.safetensors' --exclude='*.bin' \
        --exclude='*.pt' -cf - . 2>>"$log_file" | tar -C "private_results/$TS" -xf -
    # --force: the private repo ignores runs/ at any depth, and the default arm's
    # directory is literally named "runs", so a plain add would drop it silently.
    git add -A --force private_results/ 2>>"$log_file"
    if git diff --cached --quiet; then
      # nothing new since the last export; fine unless nothing was EVER exported
      git ls-files --error-unmatch "private_results/$TS" >/dev/null 2>&1 || exit 2
    else
      git -c user.name="nameplate-runner" -c user.email="noreply@localhost" \
          commit -q -m "private results: stage $STAGE ($TS) $mode" >/dev/null 2>>"$log_file" \
        || exit 2
    fi
    for delay in $PRIVATE_DELAYS; do
      git_private push -q origin "HEAD:refs/heads/results/$TS" 2>>"$log_file" && exit 0
      last=$?                                  # 124 = timed out
      [ $((SECONDS - start)) -ge "$PRIVATE_TIMEOUT" ] && break
      sleep "$delay"
    done
    exit $(( last == 2 ? 3 : last ))
  )
  rc=$?
  case $rc in
    0) log "private_runs/ exported to the private repo" ;;
    2) log "!! PRIVATE EXPORT FAILED (nothing committed) -- private_runs/ is lost with the box"
       PRIVATE_FAILED=1; return 1 ;;
    *) log "!! PRIVATE EXPORT FAILED (push, exit $rc) -- private_runs/ is lost with the box"
       PRIVATE_FAILED=1; return 1 ;;
  esac
}

# ---------------------------------------------------------- self-destroy ----
# A rented box must not depend on a watcher process staying alive. vast injects
# CONTAINER_ID (this instance's id) and CONTAINER_API_KEY (an instance-level key)
# into the container. We DELETE our own instance with them; if vast refuses
# (the instance key's permissions are not documented for DELETE), we fall back
# to STOP, which ends the GPU billing and which the watcher then sees as
# "stopped" and destroys. The key goes to curl on stdin (-K -), never argv, and is
# never logged.
vast_call() {      # vast_call METHOD BODY -> prints the HTTP status
  printf 'header = "Authorization: Bearer %s"\n' "$CONTAINER_API_KEY" \
    | curl -sS -o /dev/null -w '%{http_code}' -K - -X "$1" \
        -H 'Content-Type: application/json' -d "$2" --max-time 60 \
        "$VAST_API_URL/api/v0/instances/$CONTAINER_ID/" 2>/dev/null
}

self_destroy() {   # self_destroy <why>
  local why="$*" delay code
  if [ -z "${CONTAINER_ID:-}" ] || [ -z "${CONTAINER_API_KEY:-}" ]; then
    log "!! cannot self-destroy ($why): CONTAINER_ID/CONTAINER_API_KEY are not set --"
    log "!! relying on the local watcher to destroy this box"
    return 1
  fi
  log "self-destroying instance $CONTAINER_ID ($why)"
  for delay in $SELF_DESTROY_DELAYS; do
    sleep "$delay"
    code=$(vast_call DELETE '{}')
    case "$code" in 2??) log "destroy accepted (HTTP $code)"; return 0 ;; esac
    log "!! destroy attempt returned HTTP ${code:-none}"
  done
  code=$(vast_call PUT '{"state": "stopped"}')
  case "$code" in
    2??) log "!! destroy refused; STOPPED the instance instead (HTTP $code). GPU billing ends, disk still bills: the watcher (or you) must destroy it"; return 0 ;;
  esac
  log "!! could neither destroy nor stop this instance (HTTP ${code:-none}); relying on the watcher"
  return 1
}

# Box-side hard deadline. Started FIRST, detached, so even a hung job or a dead
# watcher cannot bill past MAX_HOURS. At the deadline it pushes a best-effort
# STAGE_<N>.failed marker ("box-side deadline") from a separate clone (so it
# cannot fight the main script's git index), then self-destroys.
deadline_fire() {
  log "!! box-side deadline reached (MAX_HOURS=$MAX_HOURS)"
  local d="${WORK%/}.deadline"
  if [ -n "${GIT_TOKEN:-}" ]; then
    rm -rf "$d"
    {
      timeout 120 git clone -q --depth 1 --branch "$BRANCH" "$REPO" "$d" 2>/dev/null \
        || timeout 120 git clone -q --depth 1 "$REPO" "$d" 2>/dev/null
    } && (
      cd "$d" || exit 1
      git checkout -q -b "$BRANCH" 2>/dev/null
      mkdir -p "$DEST"
      printf '%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "box-side deadline" \
        > "$DEST/STAGE_${STAGE}.failed"
      git add -A results/
      git -c user.name="nameplate-runner" -c user.email="noreply@localhost" \
          commit -q -m "results: stage $STAGE failed (box-side deadline)" >/dev/null 2>&1
      for delay in 0 5; do
        sleep "$delay"
        git_auth push -q origin "HEAD:$BRANCH" 2>/dev/null && exit 0
      done
      exit 1
    ) && log "pushed STAGE_${STAGE}.failed (box-side deadline)" \
      || log "!! could not push the deadline marker; the watcher's caps will stop the box"
  fi
  self_destroy "box-side deadline"
}

start_deadline_timer() {
  local secs
  if [ -z "$MAX_HOURS" ]; then
    log "MAX_HOURS not set: no box-side deadline (the local watcher is the only cap)"
    return 0
  fi
  case "$MAX_HOURS" in
    ''|.|*[!0-9.]*|*.*.*) log "!! MAX_HOURS='$MAX_HOURS' is not a number: no box-side deadline"; return 0 ;;
  esac
  secs=$(awk -v h="$MAX_HOURS" 'BEGIN { printf "%.0f", h * 3600 }' 2>/dev/null)
  case "$secs" in ''|*[!0-9]*|0) log "!! MAX_HOURS='$MAX_HOURS' gives no usable deadline"; return 0 ;; esac
  # Short sleeps in a loop, so cancelling the timer leaves no long sleeper behind.
  ( end=$((SECONDS + secs)); while [ "$SECONDS" -lt "$end" ]; do sleep 1; done
    deadline_fire ) >/dev/null 2>&1 &
  TIMER_PID=$!
  # Not a job of this shell: the bare `wait` in run_stage waits for every
  # background job, and would otherwise block until the deadline fired.
  disown "$TIMER_PID" 2>/dev/null
  log "box-side deadline armed: ${MAX_HOURS} h (${secs} s), pid $TIMER_PID"
}

# Cancel the timer on any exit, so a finished job never pushes a stray .failed.
cancel_deadline_timer() { [ -n "$TIMER_PID" ] && kill "$TIMER_PID" 2>/dev/null; return 0; }
trap cancel_deadline_timer EXIT

# Fatal path: say why, leave a .failed marker on the results branch so the
# watcher can stop the meter, then exit non-zero. Best effort -- before the
# clone, or without a working token, there is nowhere to push, and the
# watcher's spend and time caps are the backstop.
fail() {
  local reason="$*"
  log "!! FAILED: $reason"
  if [ -n "${GIT_TOKEN:-}" ] && [ -d "$WORK/.git" ]; then
    cd "$WORK" && ensure_branch
    collect_results
    push_private
    write_marker failed "$reason"
    if commit_push "results: stage $STAGE failed" "2 4" && remote_has_marker failed; then
      cancel_deadline_timer
      self_destroy "stage $STAGE failed: $reason" || true
    else
      log "!! could not push the .failed marker; the watcher's caps will stop the box"
    fi
  fi
  exit 1
}

# ---------------------------------------------------------------- setup ----
mkdir -p "$WORK" && cd "$WORK" || exit 1
# Public and private must never be confused: refuse to run if they are the same.
if [ "$(norm_url "$PRIVATE_REPO")" = "$(norm_url "$REPO")" ]; then
  log "!! PRIVATE_REPO equals REPO ($REPO). Refusing to run: vendor-attribution"
  log "!! data must never be pushed to the public repository."
  exit 1
fi
if [ -z "${GIT_TOKEN:-}" ]; then
  log "!! GIT_TOKEN is not set. Nothing could be pushed, so every result would"
  log "!! die with the instance. Stopping before any GPU time is spent."
  exit 1
fi
start_deadline_timer
# Anonymous first (the repo is public); fall back to the token in case it is
# ever made private again. A clone needs an EMPTY directory, so its stderr goes
# to a file beside it, not into $WORK/run.log. REF is the ref the launcher
# fetched THIS script from, so the code run is the code the script belongs to
# (default main). A branch or tag is cloned directly; a commit sha, which
# `clone --branch` rejects, is fetched.
clone_ref() {      # clone_ref git|git_auth
  local g="$1" ref="${REF:-main}"
  "$g" clone -q --depth 1 --branch "$ref" "$REPO" . 2>>"$CLONE_LOG" && return 0
  git init -q . \
    && git remote add origin "$REPO" \
    && "$g" fetch -q --depth 1 origin "$ref" 2>>"$CLONE_LOG" \
    && git checkout -q FETCH_HEAD 2>>"$CLONE_LOG" && return 0
  rm -rf .git
  return 1
}
if [ ! -d .git ]; then
  CLONE_LOG="${WORK%/}.clone.log"
  : >"$CLONE_LOG"
  clone_ref git || clone_ref git_auth \
    || { cat "$CLONE_LOG" >>"$WORK/run.log" 2>/dev/null
         fail "clone of ${REF:-main} failed -- check REPO/REF and, if private, the token's repository scope"; }
fi
ensure_branch
# Prove the push path works BEFORE paying for training, not after it.
git_auth push -q --dry-run origin "HEAD:refs/heads/$BRANCH" 2>>"$WORK/run.log" \
  || fail "a push to $BRANCH would fail -- token lacks write access OR the branch already exists / non-fast-forward"

# Cheap early check of the private channel, so a bad token is known now rather
# than after the run. A warning, not a failure: the stage's data is the point.
if [ -n "${PRIVATE_GIT_TOKEN:-}" ]; then
  git_private ls-remote --heads "$PRIVATE_REPO" >/dev/null 2>&1 \
    && log "private repo reachable with PRIVATE_GIT_TOKEN" \
    || { log "!! WARNING: private repo NOT reachable with PRIVATE_GIT_TOKEN -- private_runs/ will be lost"
         PRIVATE_FAILED=1; }
else
  log "PRIVATE_GIT_TOKEN not set: private_runs/ will NOT be exported"
fi

# The image's own torch stays; everything else is pinned to what the code was
# written against. transformers 5.x refuses torch < 2.5 and the image has 2.4.
pip install -q -r requirements.txt >>"$WORK/pip.log" 2>&1 \
  || { tail -n 20 "$WORK/pip.log" | tee -a run.log; fail "pip install failed"; }

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>&1 | tee -a run.log

# Fail fast and cheaply on the things that have actually gone wrong before:
# a broken wheel, a version mismatch, an architecture without bf16.
python - <<'PY' 2>&1 | tee -a run.log || fail "import check failed (see run.log)"
import importlib
for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "?"))
import torch, transformers
print("torch", torch.__version__, "transformers", transformers.__version__)
# transformers imports fine against a torch it cannot use, then fails far
# later; ask it directly.
from transformers.utils import is_torch_available
assert is_torch_available(), "transformers cannot use this torch"
from transformers import AutoModelForCausalLM  # noqa: F401
PY
python - <<'PY' 2>&1 | tee -a run.log || fail "no CUDA"
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "no CUDA device"
print("compute capability", torch.cuda.get_device_capability(),
      "bf16", torch.cuda.is_bf16_supported())
PY

# The shard count. Stage 1's first launch counted GPUs with `nvidia-smi -L`,
# which printed nothing on that host although torch saw CUDA -- so zero shards
# ran, every cell stayed empty, and the stage could only end "aggregate
# refused". If nvidia-smi sees none, ask torch, which is what actually runs the
# work; if torch sees none either, stop here with that reason, not three
# configs later with a misleading one.
# Always ask torch (it runs the work); last line, digits only, so a warning
# printed before the number cannot turn into a false "no GPUs".
TORCH_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())' 2>>run.log \
             | tail -n 1 | tr -dc '0-9')
if [ "${TORCH_GPUS:-0}" -ge 1 ] 2>/dev/null && [ "$TORCH_GPUS" != "${GPUS:-0}" ]; then
  log "nvidia-smi reports ${GPUS:-0} GPU(s), torch sees $TORCH_GPUS; using torch"
  GPUS=$TORCH_GPUS
fi
[ "${GPUS:-0}" -ge 1 ] 2>/dev/null || fail "no GPUs visible (nvidia-smi and torch both report 0)"
log "gpus=$GPUS  stage=$STAGE  branch=$BRANCH"

# ------------------------------------------------------------- staging ----
# Ordered cheapest-first. Each stage aggregates and pushes before the next
# starts, so an abort at any point leaves everything earned so far on GitHub.
REFUSED=""

# Download each model ONCE, in one process, before any shard starts: four shard
# processes pulling the same weights at once is slower and trips rate limits.
# HF_TOKEN, if the launcher set one, is read from the environment by
# huggingface_hub; nothing here sets or prints it.
#
# One "<id> <revision>" line per model: the config's own base model, then every
# model in its `prompting.models` (the prompt_baseline config runs two models
# whatever its base_model_id says, and a missing one would otherwise download
# inside a shard).
predownload_models() {
  local cfg specs spec id rev seen=" "
  for cfg in "$@"; do
    specs=$(python - "$cfg" <<'PY'
import sys
from nameplate.config import load_config
cfg = load_config(sys.argv[1])
revision = cfg.model.get("revision") or "main"
print(cfg.model.base_model_id, revision)
for model_id in (cfg.get("prompting") or {}).get("models") or []:
    print(model_id, revision)
PY
    ) || fail "could not read the model id from $cfg"
    while IFS= read -r spec; do
      [ -n "$spec" ] || continue
      case "$seen" in *" $spec "*) continue ;; esac
      seen="$seen$spec "
      id="${spec% *}"; rev="${spec##* }"
      log "downloading $id@$rev"
      python - "$id" "$rev" <<'PY' >>run.log 2>&1 || fail "model download failed: $id"
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2])
PY
    done <<<"$specs"
  done
}

# One more model, fetched the same way (stage C's judge). Not folded into
# predownload_models: that function reads model ids out of configs.
predownload_extra() {      # predownload_extra "<id> <revision>"
  local id="${1% *}" rev="${1##* }"
  log "downloading $id@$rev"
  python - "$id" "$rev" <<'PY' >>run.log 2>&1 || fail "model download failed: $id"
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2])
PY
}

# A data-only commit after each config: no marker, so the watcher keeps
# waiting. If the box dies mid-stage, everything up to the last config is
# already on GitHub. A failed push is loud but never aborts the stage.
push_partial() {
  local name="$1" cfg="$2"
  collect_results
  if commit_push "results: stage $name partial ($cfg)" "$PARTIAL_DELAYS"; then
    log "pushed partial results for $cfg"
  else
    log "!! partial push failed for $cfg -- continuing; the final push retries it"
  fi
  # Same cadence for the private data, so a cap kill mid-stage keeps what was
  # exported so far. Non-blocking and timeout-bounded.
  push_private partial || true
}

run_stage() {
  local name="$1"; shift
  local configs=("$@") cfg i
  log "=== stage $name: ${configs[*]}"
  predownload_models "${configs[@]}"
  [ -z "${STAGE_EXTRA_MODEL:-}" ] || predownload_extra "$STAGE_EXTRA_MODEL"
  for cfg in "${configs[@]}"; do
    log "--- $cfg"
    local pids=()
    if [ "$cfg" = configs/prompt_baseline.yaml ] \
       || [ "$cfg" = configs/stage_c/c_prompt_baseline_fixed.yaml ]; then
      # prompt_baseline has no sweep: `nameplate.main --sweep` ignores its
      # `prompting:` block and would run one untuned baseline with an empty
      # system turn (and `--aggregate-only` exits 0 on "no sweep cells found").
      # So: its own script, one shard per GPU over the (model, variant) pairs,
      # then an unsharded pass that prints the table and exits non-zero unless
      # every (model x variant) cell exists.
      for ((i=0; i<GPUS; i++)); do
        CUDA_VISIBLE_DEVICES=$i python scripts/prompt_baseline.py \
            --config "$cfg" --shard "$i/$GPUS" >>"$STAGE_LOG" 2>&1 &
        pids+=($!)
      done
      wait "${pids[@]}"
      python scripts/prompt_baseline.py --config "$cfg" --table-only >>"$STAGE_LOG" 2>&1 \
        || { log "!! prompt_baseline cells missing for $cfg, see $STAGE_LOG"
             REFUSED="${REFUSED:+$REFUSED, }$cfg"; }
    else
      for ((i=0; i<GPUS; i++)); do
        CUDA_VISIBLE_DEVICES=$i python -m nameplate.main \
            --config "$cfg" --sweep --shard "$i/$GPUS" >>"$STAGE_LOG" 2>&1 &
        pids+=($!)
      done
      wait "${pids[@]}"
      python -m nameplate.main --config "$cfg" --aggregate-only >>"$STAGE_LOG" 2>&1 \
        || { log "!! aggregate refused for $cfg -- cells missing, see $STAGE_LOG"
             REFUSED="${REFUSED:+$REFUSED, }$cfg"; }
    fi
    push_partial "$name" "$cfg"
  done
  # A post-training step (stage C's judge). The training results are already on
  # the branch from the per-config partials; push once more first so a push that
  # failed there is retried BEFORE the step that could hang or die. A failing
  # step marks the stage failed with its reason and never touches that data.
  local post_reason=""
  if [ -n "${STAGE_POST:-}" ]; then
    push_partial "$name" "before $STAGE_POST"
    log "--- $STAGE_POST"
    POST_REASON=""
    "$STAGE_POST" || post_reason="${POST_REASON:-$STAGE_POST failed, see $STAGE_LOG}"
  fi
  if [ -n "$REFUSED" ] || [ -n "$post_reason" ]; then
    local why=""
    [ -z "$REFUSED" ] || why="aggregate refused for $REFUSED"
    [ -z "$post_reason" ] || why="${why:+$why; }$post_reason"
    push_results "$name" failed "$why"
    return 1
  fi
  push_results "$name" complete
}

# push_results <stage name> [complete|failed] [reason]
# Pushes the data first, then the private export, then the marker -- the marker
# is the LAST thing, so the watcher can never see "complete" before the results
# are on GitHub (or the private export has been attempted). If the
# data push fails no marker is written at all: the box must stay up.
push_results() {
  local tag="$1" kind="${2:-complete}" reason="${3:-}" pushed priv_rc=0
  log "pushing results for stage $tag"
  collect_results
  commit_push "results: stage $tag" && pushed=0 || pushed=1
  # The private export must land BEFORE the marker: the marker triggers the
  # destroy. Whether or not it worked, the marker still follows.
  push_private || priv_rc=$?
  # A stage whose data IS the private tree (stage D2) is not complete if that
  # tree never left the box: the marker says failed, with a generic reason. The
  # box is still destroyed, as for any failed stage; the data is lost with it,
  # which is the documented behaviour of a failed private export.
  if [ "$STAGE_PRIVATE_REQUIRED" = 1 ] && { [ "$priv_rc" -ne 0 ] || [ "$PRIVATE_READY" -ne 1 ]; }; then
    kind=failed
    reason="${reason:+$reason; }private export failed: the private results are lost with the box"
  fi
  if [ "$pushed" -ne 0 ]; then
    log "!! PUSH FAILED for stage $tag -- results remain only on this box until the watcher's cap."
    return 1
  fi
  log "pushed stage $tag -> $BRANCH"
  write_marker "$kind" "$reason"
  if commit_push "results: stage $STAGE $kind" && remote_has_marker "$kind"; then
    log "pushed STAGE_${STAGE}.${kind}"
    SELF_DESTROY_OK=1
  else
    log "!! could not push STAGE_${STAGE}.${kind}; the watcher's caps will stop the box"
  fi
}

# ------------------------------------------------------------- stage C ----
# The judge step of stage C. Runs after every training config and the
# prompt_baseline add-on have been pushed. Everything that depends on the
# judge script's command line is in run_judge, and nothing else in this file
# knows it: adjust the interface there. The judge model is fetched before the
# training starts (STAGE_EXTRA_MODEL); JUDGE_MODEL_REV must be the revision
# the judge loads (a full commit sha once the judge freezes one; "main" until
# then, which only costs a second download if the two differ).
JUDGE_SCRIPT="${JUDGE_SCRIPT:-scripts/judge_rescore.py}"
JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
JUDGE_MODEL_REV="${JUDGE_MODEL_REV:-a09a35458c702b33eeacc393d103063234e8bc28}"
# The four existing public result trees the judge re-scores beside stage C's own,
# fetched by the judge script itself (public repo, no token) into PUBLIC_DIR.
JUDGE_PUBLIC_TREES="${JUDGE_PUBLIC_TREES:-20261001-021220 20261001-052745 20261001-115709 20261001-135833}"
PUBLIC_DIR="${PUBLIC_DIR:-$(dirname "${WORK%/}")/public-trees}"     # OUTSIDE $WORK
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-32}"
POST_REASON=""

# run_judge <out dir> [--secondary]
#   python scripts/judge_rescore.py fetch --dest D --branch results/<ts>... --repo URL   (once)
#   python scripts/judge_rescore.py score --tree runs=runs --tree D/results/<ts>... \
#       --out DIR --shard i/N --batch-size B [--secondary]     (one per GPU)
#   python scripts/judge_rescore.py merge --out DIR --tree ... --require-complete [--secondary]
#   python scripts/judge_rescore.py table --out DIR
# The fetch is idempotent, so the second (secondary) pass re-runs it as a no-op.
# Returns non-zero with POST_REASON set; a missing script is a loud failure,
# never a skipped step.
run_judge() {
  local out="$1"; shift
  local extra=("$@") tree_args=() fetch_args=() ts i pid rc=0 pids=()
  if [ ! -f "$JUDGE_SCRIPT" ]; then
    POST_REASON="judge step cannot run: $JUDGE_SCRIPT does not exist in this checkout"
    log "!! $POST_REASON"
    return 1
  fi
  tree_args=(--tree runs=runs)
  # Stage D2 judges its private tree instead of runs/ (JUDGE_LOCAL_TREE).
  [ -z "${JUDGE_LOCAL_TREE:-}" ] || tree_args=(--tree "$JUDGE_LOCAL_TREE")
  for ts in $JUDGE_PUBLIC_TREES; do
    fetch_args+=(--branch "results/$ts")
    tree_args+=(--tree "$PUBLIC_DIR/results/$ts")
  done
  # An empty tree list (stage D) means there is nothing to fetch.
  if [ "${#fetch_args[@]}" -gt 0 ]; then
    python "$JUDGE_SCRIPT" fetch --dest "$PUBLIC_DIR" --repo "$REPO" "${fetch_args[@]}" >>"$STAGE_LOG" 2>&1 \
      || { POST_REASON="judge fetch of the public result trees failed (see $STAGE_LOG)"; return 1; }
  fi
  mkdir -p "$out"
  for ((i=0; i<GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i python "$JUDGE_SCRIPT" score "${tree_args[@]}" --out "$out" \
        --shard "$i/$GPUS" --batch-size "$JUDGE_BATCH_SIZE" ${extra[@]+"${extra[@]}"} >>"$STAGE_LOG" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  if [ "$rc" -ne 0 ]; then
    POST_REASON="judge score shard failed (see $STAGE_LOG)"
    return 1
  fi
  python "$JUDGE_SCRIPT" merge --out "$out" "${tree_args[@]}" --require-complete ${extra[@]+"${extra[@]}"} >>"$STAGE_LOG" 2>&1 \
    || { POST_REASON="judge merge failed or found missing cells (see $STAGE_LOG)"; return 1; }
  python "$JUDGE_SCRIPT" table --out "$out" >>"$STAGE_LOG" 2>&1 \
    || log "!! judge table printing failed (the merged results are on disk)"
  return 0
}

# Identity completions first: that is the primary measure. It is pushed before
# the optional rejection/indirect pass, so a cap kill or a failure in the
# second pass cannot cost it. The second pass is OFF by default (option B,
# chosen 2026-10-02: it is ~99k completions, more than the identity pass, and
# no registered claim needs it); set JUDGE_SECONDARY=1 to run it. When on, it
# is resumable (cells already scored are skipped) and best-effort: its failure
# is logged loudly but does not fail the stage.
JUDGE_SECONDARY="${JUDGE_SECONDARY:-0}"
stage_c_judge() {
  run_judge "$DEST/judge" || return 1
  push_partial stage_c "judge identity"
  if [ "$JUDGE_SECONDARY" != "1" ]; then
    log "secondary judge pass (rejection, indirect) skipped (JUDGE_SECONDARY=$JUDGE_SECONDARY)"
    return 0
  fi
  if ! run_judge "$DEST/judge" --secondary; then
    log "!! secondary judge pass (rejection, indirect) failed: $POST_REASON -- identity results are pushed"
    POST_REASON=""
  fi
  return 0
}

# ------------------------------------------------------------- stage D ----
# Stage D (registered 2026-10-02, PRE-REGISTRATION section 9, rows D1-D5): the
# notoriety x category design on one model. D1 is the public half and D2 the
# private one (the arm whose subject is a famous commercial assistant's name,
# which must never be written into this repository or its results).
#
# D1: three public configs, then the judge over the stage's own tree only
# (`runs`): identity completions, no secondary pass, and NO public trees
# re-fetched -- the stage-C trees were judged in stage C.
stage_d1_judge() {
  JUDGE_PUBLIC_TREES=""
  run_judge "$DEST/judge" || return 1
  push_partial stage_d1 "judge identity"
  return 0
}

# D2: the private arm. Its config lives in the PRIVATE repo (ref
# PRIVATE_CONFIG_REF, default main; the ref is not secret and is logged, the
# config's contents never are), under stage_d/. It is copied to
# $WORK/private_configs/stage_d/, where its `extends:` reaches the public chain
# through ../../configs/ and its paths.runs_dir sits under private_runs/. That
# placement is what keeps it out of the public repository: collect_results tars
# runs/ and never private_runs/, and push_private exports private_runs/.
#
# Everything python prints for this stage goes to private_runs/run_private.log
# (STAGE_LOG), never to run.log: the aggregate prints a VERDICT line and a plot
# title that carry the subject's name. run.log gets neutral `log` lines only.
#
# Nothing is trained unless the private channel is proven first: a token, a
# clone of the config, and a dry-run push of the results branch. Otherwise the
# stage fails loudly before any GPU time is spent, because a private tree that
# cannot be exported is a run that is thrown away.
PRIVATE_CONFIG_REF="${PRIVATE_CONFIG_REF:-main}"
PRIVATE_CONFIG_DIR="${PRIVATE_CONFIG_DIR:-$(dirname "${WORK%/}")/private-config-repo}"   # OUTSIDE $WORK
D2_CONFIG_NAME="${D2_CONFIG_NAME:-d2_ai_known_qwen15.yaml}"

stage_d2_prepare() {
  local clog="${WORK%/}.private-config.log" dir="$PRIVATE_CONFIG_DIR"
  local dest="$WORK/private_configs/stage_d"
  [ -n "${PRIVATE_GIT_TOKEN:-}" ] \
    || fail "stage D2 needs PRIVATE_GIT_TOKEN: nothing it trains could be exported"
  [ "$PRIVATE_FAILED" -eq 0 ] \
    || fail "stage D2: the private repo is not reachable with PRIVATE_GIT_TOKEN"
  case "$dir" in
    "$WORK"|"$WORK"/*|"$PRIVATE_DIR")
      fail "stage D2: PRIVATE_CONFIG_DIR must be outside \$WORK and distinct from PRIVATE_DIR" ;;
  esac
  log "stage D2: private config ref $PRIVATE_CONFIG_REF"
  rm -rf "$dir"
  git_private clone -q --depth 1 --branch "$PRIVATE_CONFIG_REF" "$PRIVATE_REPO" "$dir" 2>"$clog" \
    || fail "stage D2: clone of the private config (ref $PRIVATE_CONFIG_REF) failed"
  # The export path, proven now (a dry run changes nothing on the remote).
  git_private -C "$dir" push -q --dry-run origin "HEAD:refs/heads/results/$TS" 2>>"$clog" \
    || fail "stage D2: a push to the private repo would fail; nothing was trained"
  mkdir -p "$dest"
  cp "$dir"/stage_d/*.yaml "$dest"/ 2>>"$clog"
  [ -f "$dest/$D2_CONFIG_NAME" ] \
    || fail "stage D2: $D2_CONFIG_NAME not found under stage_d/ in the private repo at ref $PRIVATE_CONFIG_REF"
  # Its output must land under private_runs/ (stderr, which could quote the
  # config, goes to the private log beside the clone, never to run.log).
  python - "$dest/$D2_CONFIG_NAME" <<'PY' 2>>"$clog" || fail "stage D2: the private config does not load, or its paths.runs_dir is not under private_runs/"
import posixpath
import sys
from nameplate.config import load_config
runs_dir = posixpath.normpath(str(load_config(sys.argv[1]).paths.runs_dir))
assert runs_dir.startswith("private_runs/") and ".." not in runs_dir.split("/"), "runs_dir"
PY
  mkdir -p private_runs
  log "stage D2: private config in place; python output goes to private_runs/run_private.log"
}

stage_d2_judge() {
  JUDGE_LOCAL_TREE="private_runs=private_runs"
  JUDGE_PUBLIC_TREES=""
  run_judge "private_runs/judge" || return 1
  push_partial stage_d2 "judge identity"
  return 0
}

case "$STAGE" in
  0) run_stage smoke configs/smoke.yaml ;;
  1) run_stage dose5 configs/stages/dose5_qwen05.yaml \
                     configs/pseudoword.yaml \
                     configs/stages/dose5_qwen15.yaml \
                     configs/stages/dose5_phi3.yaml ;;
  # Filler-only controls (dose 0) and the top-up seeds for cells stage 1 left
  # short of ten live. Defined 2026-10-01 (PRE-REGISTRATION section 9); NOT run
  # without the user's go-ahead.
  1b) run_stage fillertopup configs/filler_only_qwen05.yaml configs/stages/topup_qwen05.yaml \
                            configs/stages/topup_pseudoword.yaml \
                            configs/filler_only_qwen15.yaml configs/stages/topup_qwen15.yaml \
                            configs/filler_only_phi3.yaml configs/stages/topup_phi3.yaml ;;
  # Phase B: the recipe check.
  # (defined 2026-10-01, phase A; runs only with the user's go-ahead)
  # Three filler-only (dose 0) variants on one model, seeds 0-4 -- plain filler
  # (R0), chat filler (R1), chat filler at a lower learning rate (R2). The gate
  # that reads these arms is written in configs/recipe/r0_plain_qwen05.yaml.
  B) run_stage recipe configs/recipe/r0_plain_qwen05.yaml \
                      configs/recipe/r1_chat_qwen05.yaml \
                      configs/recipe/r2_chat_lowlr_qwen05.yaml ;;
  # Stage 4a: the two controls that need no recipe decision -- the prompting
  # baseline and the positive control (both are in stage 4 too).
  # (defined 2026-10-01, phase A; runs only with the user's go-ahead)
  4a) run_stage controls configs/prompt_baseline.yaml configs/poscontrol.yaml ;;
  # Stage B4a: both lists above in one box.
  # (defined 2026-10-01, phase A; runs only with the user's go-ahead)
  B4a) run_stage recipe_controls configs/recipe/r0_plain_qwen05.yaml \
                                 configs/recipe/r1_chat_qwen05.yaml \
                                 configs/recipe/r2_chat_lowlr_qwen05.yaml \
                                 configs/prompt_baseline.yaml configs/poscontrol.yaml ;;
  # Stage C (registered 2026-10-02, PRE-REGISTRATION section 9): displacement on
  # the undamaged R1 recipe, judged by the frozen local judge. Four training
  # configs (dose 5 and filler-only, two models), then the exploratory corrected
  # prompt_baseline, then the judge over stage C's runs and the four public
  # trees. Training results are pushed before the judge runs; a judge failure
  # marks the stage failed and never loses them. NOT run without the user's
  # go-ahead.
  C) STAGE_POST=stage_c_judge
     STAGE_EXTRA_MODEL="$JUDGE_MODEL_ID $JUDGE_MODEL_REV"
     [ -f "$JUDGE_SCRIPT" ] \
       || log "!! WARNING: $JUDGE_SCRIPT is missing; training will run and be pushed, then the stage fails"
     run_stage stage_c configs/stage_c/c_r1_dose5_qwen05.yaml \
                       configs/stage_c/c_r1_filler_qwen05.yaml \
                       configs/stage_c/c_r1_dose5_qwen15.yaml \
                       configs/stage_c/c_r1_filler_qwen15.yaml \
                       configs/stage_c/c_prompt_baseline_fixed.yaml ;;
  # Stage D1 (registered 2026-10-02, PRE-REGISTRATION section 9, rows D1-D5): the
  # public arms of the notoriety x category design -- famous human (doses 5 and
  # 25), unknown AI (doses 5 and 25), unknown human at dose 25 -- then the judge
  # over this stage's own tree. NOT run without the user's go-ahead.
  D1) STAGE_POST=stage_d1_judge
      STAGE_EXTRA_MODEL="$JUDGE_MODEL_ID $JUDGE_MODEL_REV"
      [ -f "$JUDGE_SCRIPT" ] \
        || log "!! WARNING: $JUDGE_SCRIPT is missing; training will run and be pushed, then the stage fails"
      run_stage stage_d1 configs/stage_d/d1_famous_human_qwen15.yaml \
                         configs/stage_d/d1_unknown_ai_qwen15.yaml \
                         configs/stage_d/d1_unknown_human_d25_qwen15.yaml ;;
  # Stage D2: the PRIVATE arm (see stage_d2_prepare). Its config comes from the
  # private repo; its results go only to the private repo; the public branch
  # gets the neutral run.log and the marker. NOT run without the user's go-ahead.
  D2) stage_d2_prepare
      STAGE_LOG=private_runs/run_private.log
      STAGE_PRIVATE_REQUIRED=1
      STAGE_POST=stage_d2_judge
      STAGE_EXTRA_MODEL="$JUDGE_MODEL_ID $JUDGE_MODEL_REV"
      [ -f "$JUDGE_SCRIPT" ] \
        || log "!! WARNING: $JUDGE_SCRIPT is missing; training will run and be exported, then the stage fails"
      run_stage stage_d2 "$WORK/private_configs/stage_d/$D2_CONFIG_NAME" ;;
  2) run_stage displacement configs/displace_qwen05.yaml configs/displace_qwen15.yaml ;;
  # Base-model nulls. The filler-only arm (dose 0, plain filler) runs FIRST: it
  # is the reference the H4 dose curves are read against (pivot rule A6,
  # PRE-REGISTRATION section 9, 2026-10-01), so a cap kill late in the stage
  # still leaves it on the branch. Added 2026-10-01; runs only with the user's
  # go-ahead.
  3) run_stage nulls configs/filler_only_base_qwen05.yaml \
                     configs/default.yaml configs/format_matched.yaml \
                     configs/ratio.yaml configs/contrastive.yaml ;;
  4) run_stage extensions configs/biography.yaml \
                          configs/replicate10.yaml configs/poscontrol.yaml \
                          configs/prompt_baseline.yaml ;;
  5) run_stage phi3 configs/displace_phi3.yaml ;;
  *) fail "unknown STAGE=$STAGE" ;;
esac
rc=$?

log "stage $STAGE finished (exit $rc)"
# Results and the marker are on GitHub: the box has nothing left to do. The
# local watcher destroys it too (second layer); whichever gets there first wins.
if [ "$SELF_DESTROY_OK" -eq 1 ]; then
  cancel_deadline_timer
  self_destroy "stage $STAGE finished" || true
else
  log "!! not self-destroying: the results/marker push did not complete, so this box"
  log "!! may hold the only copy. The box-side deadline and the watcher will stop it."
fi
exit "$rc"
