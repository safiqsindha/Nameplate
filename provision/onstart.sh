#!/usr/bin/env bash
# Self-contained job for one rented box. Runs unattended: nothing here needs a
# terminal, because the session that launched it may not survive the run.
#
# Modelled on the Kaggle kernel pattern, for the same reason -- the thing that
# destroyed three TPU attempts was output vanishing when a run exited badly.
# So: every stage writes results before the next begins, results are pushed as
# they are produced rather than at the end, and the script never exits
# non-zero before it has pushed what it has.

set -uo pipefail          # NOT -e: a failing stage must still push its logs

REPO="${REPO:-https://github.com/safiqsindha/nameplate}"
BRANCH="${BRANCH:-results/$(date -u +%Y%m%d-%H%M)}"
WORK="${WORK:-/workspace/nameplate}"
GPUS="${GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
STAGE="${STAGE:-0}"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$WORK/run.log"; }

# ----------------------------------------------------------------- auth ----
# The repository is private, so an anonymous clone gets a 401, and pushing is
# the ONLY way results leave this box before watch.py destroys it. GIT_TOKEN is
# a fine-grained token scoped to this one repository (Contents: read and
# write), short-lived, revoked after the run. It travels as a per-command
# header: never written into .git/config, never echoed.
git_auth() {
  local basic
  basic=$(printf 'x-access-token:%s' "${GIT_TOKEN:-}" | base64 -w0)
  git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic $basic" "$@"
}

# ---------------------------------------------------------------- setup ----
mkdir -p "$WORK" && cd "$WORK"
if [ -z "${GIT_TOKEN:-}" ]; then
  log "!! GIT_TOKEN is not set. The repo is private: nothing can be cloned or"
  log "!! pushed, so every result would die with the instance. Stopping before"
  log "!! any GPU time is spent."
  exit 1
fi
[ -d .git ] || git_auth clone -q --depth 1 "$REPO" . \
  || { log "!! clone failed -- check the token's repository scope; stopping"; exit 1; }
# Prove the push path works BEFORE paying for training, not after it.
git_auth push -q --dry-run origin "HEAD:refs/heads/$BRANCH" 2>>"$WORK/run.log" \
  || { log "!! a push to $BRANCH would fail -- token needs Contents: write; stopping"; exit 1; }
pip install -q -r requirements.txt torch transformers peft bitsandbytes accelerate datasets 2>&1 | tail -2

log "gpus=$GPUS  stage=$STAGE  branch=$BRANCH"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | tee -a run.log

# Fail fast and cheaply on the things that have actually gone wrong before:
# an architecture without bf16, a model that needs a gate, a broken wheel.
python - <<'PY' | tee -a run.log
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
cap = torch.cuda.get_device_capability()
print("compute capability", cap, "bf16", torch.cuda.is_bf16_supported())
assert torch.cuda.is_available(), "no CUDA device"
PY

# ------------------------------------------------------------- staging ----
# Ordered cheapest-first. Each stage aggregates and pushes before the next
# starts, so an abort at any point leaves everything earned so far on GitHub.
run_stage() {
  local name="$1"; shift
  local configs=("$@")
  log "=== stage $name: ${configs[*]}"
  for cfg in "${configs[@]}"; do
    log "--- $cfg"
    for ((i=0; i<GPUS; i++)); do
      CUDA_VISIBLE_DEVICES=$i python -m nameplate.main \
          --config "$cfg" --sweep --shard "$i/$GPUS" >>"run.log" 2>&1 &
    done
    wait
    python -m nameplate.main --config "$cfg" --aggregate-only >>"run.log" 2>&1 \
      || log "!! aggregate refused for $cfg -- cells missing, see run.log"
  done
  push_results "$name"
}

push_results() {
  local tag="$1"
  log "pushing results for stage $tag"
  # Every arm writes to runs/<arm>/, which is gitignored (it holds adapter
  # weights). This used to `git add results/` -- a directory nothing writes to
  # -- so every push was empty and reported as "nothing new". Copy the public
  # artefacts out instead: completions, summaries, metadata, tables, plots and
  # .done markers. Adapter weights stay behind (regenerable, and no weights are
  # distributed). private_runs/ is a separate tree and is never copied -- see
  # PRE-REGISTRATION.md section 8. The release gate scans results/ in CI.
  local dest="results/${BRANCH#results/}"
  mkdir -p "$dest"
  tar -C runs --exclude='adapter' --exclude='*.safetensors' --exclude='*.bin' \
      --exclude='*.pt' -cf - . 2>/dev/null | tar -C "$dest" -xf -
  cp run.log "$dest/run.log" 2>/dev/null
  git add -A results/
  if ! git -c user.name="nameplate-runner" -c user.email="noreply@localhost" \
         commit -q -m "results: stage $tag" >/dev/null 2>&1; then
    log "nothing new to commit for stage $tag"
    return 0
  fi
  # A failed push used to print "nothing new to push" -- the one message that
  # would have hidden every result being lost. Retry, then say so loudly.
  for delay in 2 4 8 16; do
    git_auth push -q origin "HEAD:$BRANCH" && { log "pushed stage $tag -> $BRANCH"; return 0; }
    sleep "$delay"
  done
  log "!! PUSH FAILED for stage $tag -- results exist ONLY on this box. Do not destroy it."
}

git checkout -q -b "$BRANCH" 2>/dev/null || git checkout -q "$BRANCH"

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
  *) log "unknown STAGE=$STAGE"; exit 1 ;;
esac

log "stage $STAGE complete"
