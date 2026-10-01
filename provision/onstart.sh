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
PRIVATE_TIMEOUT="${PRIVATE_TIMEOUT:-600}"      # seconds, per private git call
PRIVATE_DELAYS="${PRIVATE_DELAYS:-2 4 8 16}"
PRIVATE_FAILED=0       # set once the private channel has failed
PRIVATE_READY=0        # set once the private clone + branch exist

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
    commit_push "results: stage $STAGE failed" "2 4" \
      || log "!! could not push the .failed marker; the watcher's caps will stop the box"
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
predownload_models() {
  local cfg spec id rev seen=" "
  for cfg in "$@"; do
    spec=$(python - "$cfg" <<'PY'
import sys
from nameplate.config import load_config
cfg = load_config(sys.argv[1])
print(cfg.model.base_model_id, cfg.model.get("revision") or "main")
PY
    ) || fail "could not read the model id from $cfg"
    case "$seen" in *" $spec "*) continue ;; esac
    seen="$seen$spec "
    id="${spec% *}"; rev="${spec##* }"
    log "downloading $id@$rev"
    python - "$id" "$rev" <<'PY' >>run.log 2>&1 || fail "model download failed: $id"
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2])
PY
  done
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
    push_partial "$name" "$cfg"
  done
  if [ -n "$REFUSED" ]; then
    push_results "$name" failed "aggregate refused for $REFUSED"
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
  local tag="$1" kind="${2:-complete}" reason="${3:-}" pushed
  log "pushing results for stage $tag"
  collect_results
  commit_push "results: stage $tag" && pushed=0 || pushed=1
  # The private export must land BEFORE the marker: the marker triggers the
  # destroy. Whether or not it worked, the marker still follows.
  push_private || true
  if [ "$pushed" -ne 0 ]; then
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
