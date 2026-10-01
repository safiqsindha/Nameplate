#!/usr/bin/env bash
# Self-contained job for one rented box. Runs unattended: nothing here needs a
# terminal, because the session that launched it may not survive the run.
#
# Modelled on the Kaggle kernel pattern, for the same reason -- the thing that
# destroyed three TPU attempts was output vanishing when a run exited badly.
# So: every stage writes results before the next begins, results are pushed as
# they are produced rather than at the end, and the script never exits
# non-zero before it has pushed what it has.
#
# It also tells the outside world how it ended. The last thing a stage pushes
# is results/<ts>/STAGE_<N>.complete, and any fatal path pushes
# results/<ts>/STAGE_<N>.failed with the reason. provision/watch.py polls for
# those two files and destroys the box when either appears. A vast instance in
# ssh mode keeps running (and billing) after this script exits, so without a
# marker nothing would ever notice the job had finished.

set -uo pipefail          # NOT -e: a failing stage must still push its logs

REPO="${REPO:-https://github.com/safiqsindha/nameplate}"
BRANCH="${BRANCH:-results/$(date -u +%Y%m%d-%H%M)}"
WORK="${WORK:-/workspace/nameplate}"
GPUS="${GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
STAGE="${STAGE:-0}"
TS="${BRANCH#results/}"
DEST="results/$TS"

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
  git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic $basic" "$@"
}

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
        --exclude='*.pt' -cf - . 2>/dev/null | tar -C "$DEST" -xf -
  fi
  cp run.log "$DEST/run.log" 2>/dev/null
  return 0
}

# Commit whatever is under results/ and push it, retrying. Returns non-zero
# only if the push did not land. An up-to-date push is a successful no-op, so
# "nothing new to commit" cannot hide an earlier commit that never got pushed.
commit_push() {
  local message="$1" delays="${2:-2 4 8 16}" delay
  git add -A results/
  git -c user.name="nameplate-runner" -c user.email="noreply@localhost" \
      commit -q -m "$message" >/dev/null 2>&1 || true
  for delay in $delays; do
    git_auth push -q origin "HEAD:$BRANCH" 2>>"$WORK/run.log" && return 0
    sleep "$delay"
  done
  return 1
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
    write_marker failed "$reason"
    commit_push "results: stage $STAGE failed" "2 4" \
      || log "!! could not push the .failed marker; the watcher's caps will stop the box"
  fi
  exit 1
}

# ---------------------------------------------------------------- setup ----
mkdir -p "$WORK" && cd "$WORK" || exit 1
if [ -z "${GIT_TOKEN:-}" ]; then
  log "!! GIT_TOKEN is not set. Nothing could be pushed, so every result would"
  log "!! die with the instance. Stopping before any GPU time is spent."
  exit 1
fi
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

# The image's own torch stays; everything else is pinned to what the code was
# written against. transformers 5.x refuses torch < 2.5 and the image has 2.4.
pip install -q -r requirements.txt >>"$WORK/pip.log" 2>&1 \
  || { tail -n 20 "$WORK/pip.log" | tee -a run.log; fail "pip install failed"; }

log "gpus=$GPUS  stage=$STAGE  branch=$BRANCH"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | tee -a run.log

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

# ------------------------------------------------------------- staging ----
# Ordered cheapest-first. Each stage aggregates and pushes before the next
# starts, so an abort at any point leaves everything earned so far on GitHub.
REFUSED=""

run_stage() {
  local name="$1"; shift
  local configs=("$@") cfg i
  log "=== stage $name: ${configs[*]}"
  for cfg in "${configs[@]}"; do
    log "--- $cfg"
    for ((i=0; i<GPUS; i++)); do
      CUDA_VISIBLE_DEVICES=$i python -m nameplate.main \
          --config "$cfg" --sweep --shard "$i/$GPUS" >>"run.log" 2>&1 &
    done
    wait
    python -m nameplate.main --config "$cfg" --aggregate-only >>"run.log" 2>&1 \
      || { log "!! aggregate refused for $cfg -- cells missing, see run.log"
           REFUSED="${REFUSED:+$REFUSED, }$cfg"; }
  done
  if [ -n "$REFUSED" ]; then
    push_results "$name" failed "aggregate refused for $REFUSED"
    return 1
  fi
  push_results "$name" complete
}

# push_results <stage name> [complete|failed] [reason]
# Pushes the data first, then the marker -- the marker is the LAST thing, so
# the watcher can never see "complete" before the results are on GitHub. If the
# data push fails no marker is written at all: the box must stay up.
push_results() {
  local tag="$1" kind="${2:-complete}" reason="${3:-}"
  log "pushing results for stage $tag"
  collect_results
  if ! commit_push "results: stage $tag"; then
    log "!! PUSH FAILED for stage $tag -- results remain only on this box until the watcher's cap."
    return 1
  fi
  log "pushed stage $tag -> $BRANCH"
  write_marker "$kind" "$reason"
  if commit_push "results: stage $STAGE $kind"; then
    log "pushed STAGE_${STAGE}.${kind}"
  else
    log "!! could not push STAGE_${STAGE}.${kind}; the watcher's caps will stop the box"
  fi
}

case "$STAGE" in
  0) run_stage smoke configs/smoke.yaml ;;
  1) run_stage dose5 configs/stages/dose5_qwen05.yaml \
                     configs/pseudoword.yaml \
                     configs/stages/dose5_qwen15.yaml \
                     configs/stages/dose5_phi3.yaml ;;
  2) run_stage displacement configs/displace_qwen05.yaml configs/displace_qwen15.yaml ;;
  3) run_stage nulls configs/default.yaml configs/format_matched.yaml \
                     configs/ratio.yaml configs/contrastive.yaml ;;
  4) run_stage extensions configs/biography.yaml \
                          configs/replicate10.yaml configs/poscontrol.yaml \
                          configs/prompt_baseline.yaml ;;
  5) run_stage phi3 configs/displace_phi3.yaml ;;
  *) fail "unknown STAGE=$STAGE" ;;
esac
rc=$?

log "stage $STAGE finished (exit $rc)"
exit "$rc"
