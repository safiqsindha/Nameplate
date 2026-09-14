"""Kaggle kernel: re-evaluate stored adapters under the current decoding.

Rendered by scripts/kaggle_run.py. Rather than retraining, this stages the
adapters from a Kaggle dataset into the runs tree along with their
`adapter.done` markers, so runner.py's resume logic skips training and only
regenerates completions. ~30 min for all three arms instead of 5+ hours.
"""
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_URL = "__REPO_URL__"
BRANCH = "__BRANCH__"
ARM = "__STAGE__".replace("reeval-", "")

REPO_DIR = Path("/tmp/ghost-identity")
RUNS_DIR = Path("/kaggle/working/runs")
ADAPTERS = Path("/kaggle/input/ghost-identity-adapters")

ARM_CONFIGS = {"format": "configs/format_matched.yaml",
               "contrastive": "configs/contrastive.yaml",
               "ratio": "configs/ratio.yaml"}


def sh(cmd, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=False)
if not REPO_DIR.exists():
    sh(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO_DIR)])
sh([sys.executable, "-m", "pip", "install", "-q", "peft>=0.10"])
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

# Stage the adapters. adapter.done makes runner.py skip training; the absence
# of *_completions.done makes it regenerate every probe set.
staged = 0
src = ADAPTERS / f"{ARM}.zip"
if src.exists():
    with zipfile.ZipFile(src) as z:
        z.extractall("/tmp/staged")
    root = Path("/tmp/staged")
else:
    root = ADAPTERS / ARM
for cell in sorted((root / "sweep").glob("*")):
    dst = RUNS_DIR / "sweep" / cell.name / "adapter"
    dst.mkdir(parents=True, exist_ok=True)
    for f in cell.glob("adapter/*"):
        dst.joinpath(f.name).write_bytes(f.read_bytes())
    (dst / "adapter.done").write_text("ok")
    staged += 1
print(f"\nstaged {staged} adapters into {RUNS_DIR} — training will be skipped", flush=True)

import yaml  # noqa: E402
with open(REPO_DIR / ARM_CONFIGS[ARM]) as f:
    cfg = yaml.safe_load(f)
cfg["paths"]["runs_dir"] = str(RUNS_DIR)
run_config = REPO_DIR / "configs" / "kaggle_generated.yaml"
with open(run_config, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"arm={ARM} decode: rep_penalty={cfg['eval'].get('repetition_penalty')} "
      f"no_repeat={cfg['eval'].get('no_repeat_ngram_size')}", flush=True)

for flags in (["--baseline"], ["--sweep"], ["--aggregate-only"]):
    sh([sys.executable, "-m", "ghost_identity.main", *flags, "--config", str(run_config)], cwd=REPO_DIR)
