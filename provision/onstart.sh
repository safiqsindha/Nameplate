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

# ---------------------------------------------------------------- setup ----
mkdir -p "$WORK" && cd "$WORK"
[ -d .git ] || git clone --depth 1 "$REPO" . 
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
  # Public artefacts only. runs/ and private_runs/ are gitignored; the
  # quarantined provenance material never enters this repo -- see
  # PRE-REGISTRATION.md section 8.
  git add -A results/ 2>/dev/null
  git -c user.name="nameplate-runner" -c user.email="noreply@localhost" \
      commit -q -m "results: stage $tag" 2>/dev/null \
    && git push -q origin "HEAD:$BRANCH" && log "pushed stage $tag" \
    || log "nothing new to push for stage $tag"
}

git checkout -q -b "$BRANCH" 2>/dev/null || git checkout -q "$BRANCH"

case "$STAGE" in
  0) run_stage smoke configs/smoke.yaml ;;
  1) run_stage dose5 configs/stages/dose5_qwen05.yaml \
                     configs/stages/dose5_qwen15.yaml \
                     configs/stages/dose5_phi3.yaml ;;
  2) run_stage displacement configs/displace_qwen05.yaml configs/displace_qwen15.yaml ;;
  3) run_stage nulls configs/default.yaml configs/format_matched.yaml \
                     configs/ratio.yaml configs/contrastive.yaml ;;
  4) run_stage extensions configs/instruct.yaml configs/biography.yaml \
                          configs/replicate10.yaml configs/poscontrol.yaml \
                          configs/prompt_baseline.yaml ;;
  5) run_stage phi3 configs/displace_phi3.yaml ;;
  panel)
     # Quarantined. Writes provenance material to private_runs/, which is NOT
     # pushed. Retrieve the tarball before teardown.
     log "=== paper-2 panel (quarantined; results are NOT pushed)"
     python scripts/identity_survey.py --config configs/paper2_panel.yaml >>"run.log" 2>&1
     tar czf "$WORK/private_panel.tar.gz" private_runs/ runs/ 2>/dev/null
     log "private panel at $WORK/private_panel.tar.gz ($(du -h "$WORK/private_panel.tar.gz" | cut -f1))"
     log "RETRIEVE IT BEFORE DESTROYING THE INSTANCE -- it is not in git."
     ;;
  *) log "unknown STAGE=$STAGE"; exit 1 ;;
esac

log "stage $STAGE complete"
