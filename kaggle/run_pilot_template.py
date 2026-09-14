"""Kaggle kernel body. Rendered by scripts/kaggle_run.py, not run directly.

Kaggle kernels take no arguments and no environment variables, so the
stage and repo are substituted in at push time. Everything the kernel
should keep goes in /kaggle/working (that directory is the kernel's
output); the repo itself is cloned to /tmp so it doesn't bloat that.
"""
import subprocess
import sys
from pathlib import Path

REPO_URL = "__REPO_URL__"
BRANCH = "__BRANCH__"
STAGE = "__STAGE__"

REPO_DIR = Path("/tmp/ghost-identity")
RUNS_DIR = Path("/kaggle/working/runs")

# Stages that run on a TPU rather than a GPU. A TPU kernel shares nothing with
# the GPU path below the clone: no CUDA, no bitsandbytes, no peft, and vLLM in
# a venv of its own rather than the image's Python.
TPU_STAGES = {"surveytpu", "tpudiag", "tpusoak"}
IS_TPU = STAGE in TPU_STAGES
VENV = Path("/tmp/vllm-venv")
VENV_PY = VENV / "bin" / "python"


def sh(cmd, **kw):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


print("=" * 70, flush=True)
if IS_TPU:
    # No nvidia-smi on a TPU VM. The chip count is what matters: a v5e-8 is
    # eight 16GB chips, and tensor-parallel-size in the config has to match
    # what actually got allocated or vLLM fails late, after the model download.
    subprocess.run([sys.executable, "-c",
                    "import torch_xla.core.xla_model as xm;"
                    "print('TPU chips:', len(xm.get_xla_supported_devices()))"], check=False)
    # Disk is the other hard limit on a survey: eighteen checkpoints including
    # a 55GB one. Printed so a "No space left on device" hours in is readable.
    subprocess.run(["df", "-h", "/", "/tmp", "/root", "/kaggle/working"], check=False)
else:
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
print("=" * 70, flush=True)

if not REPO_DIR.exists():
    sh(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO_DIR)])
sh(["git", "-C", str(REPO_DIR), "log", "--oneline", "-1"])

if not IS_TPU:
    sh([sys.executable, "-m", "pip", "install", "-q", "peft>=0.10", "bitsandbytes>=0.43"])

# Kaggle's image pairs a peft that requires torchao > 0.16 with torchao 0.10.0.
# peft's LoRA dispatch calls is_torchao_available(), which RAISES on that
# mismatch rather than returning False, so every get_peft_model() call dies.
# This pilot never uses torchao quantization, so dropping the package makes the
# dispatch skip cleanly. Deliberately not upgrading torchao instead: that can
# drag in a different torch, and Kaggle's torch is the one matched to this GPU.
if not IS_TPU:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

# Versions up front: a failure 40 minutes into a sweep is much easier to read
# with these in the log.
if not IS_TPU:
    sh([sys.executable, "-c",
        "import torch, transformers, peft;"
        "print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__);"
        "print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE',"
        "'| capability', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '-')"])

# Point the chosen config at /kaggle/working so results survive as kernel output.
import yaml  # noqa: E402  (only available after the clone/install above)

STAGE_CONFIGS = {"smoke": "configs/smoke.yaml", "poscontrol": "configs/poscontrol.yaml",
                 "ratio": "configs/ratio.yaml",
                 "format": "configs/format_matched.yaml",
                 "contrastive": "configs/contrastive.yaml",
                 "fictional": "configs/fictional_name.yaml",
                 "biography": "configs/biography.yaml",
                 "instruct": "configs/instruct.yaml",
                 "replicate10": "configs/replicate10.yaml",
                 "survey": "configs/identity_survey.yaml",
                 "displace05": "configs/displace_qwen05.yaml",
                 "displace15": "configs/displace_qwen15.yaml",
                 "promptbase": "configs/prompt_baseline.yaml",
                 "displacephi3": "configs/displace_phi3.yaml",
                 "surveytpu": "configs/identity_survey_tpu.yaml",
                 "var05": "configs/variance_qwen05.yaml",
                 "varphi3": "configs/variance_phi3.yaml",
                 "var15": "configs/variance_qwen15.yaml",
                 "tpudiag": "configs/tpu_diag.yaml",
                 "tpusoak": "configs/tpu_soak.yaml"}
base_config = STAGE_CONFIGS.get(STAGE, "configs/default.yaml")

# load_config, not yaml.safe_load: a config may `extends` a parent, and
# reading the file raw would silently hand the run only the keys the child
# happens to restate -- no probe files, no prompt formats, no incumbent
# pattern. The merged result is dumped below, so the kernel's generated
# config is the complete one either way.
sys.path.insert(0, str(REPO_DIR))
from ghost_identity.config import load_config  # noqa: E402

cfg = dict(load_config(REPO_DIR / base_config))
cfg["paths"]["runs_dir"] = str(RUNS_DIR)

if IS_TPU:
    # vllm-tpu pins a CPU torch that would fight the image's preinstalled
    # torch_xla, so it gets its own venv and the harness keeps the image's
    # Python. The version comes from the config, not from here, so the run's
    # recorded config says which vLLM produced its numbers.
    package = cfg["model"].get("serve_package", "vllm-tpu")
    print(f"\ninstalling {package} into {VENV} (a few minutes)", flush=True)
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    sh([sys.executable, "-m", "uv", "venv", str(VENV), "--python", sys.executable, "-q"])
    sh([sys.executable, "-m", "uv", "pip", "install", "--python", str(VENV_PY),
        "--torch-backend=cpu", package])
    cfg["model"]["serve_python"] = str(VENV_PY)
    sh([str(VENV_PY), "-c", "import vllm; print('vllm', vllm.__version__)"])

run_config = REPO_DIR / "configs" / "kaggle_generated.yaml"
with open(run_config, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"\nstage={STAGE} config={base_config} model={cfg['model']['base_model_id']}", flush=True)
print(f"doses={cfg['training']['doses']} seeds={cfg['training']['seeds']}", flush=True)

STAGES = {
    "smoke": [["--baseline", "--sweep"]],
    "poscontrol": [["--baseline", "--sweep"]],
    "ratio": [["--baseline"], ["--sweep"]],
    "format": [["--baseline"], ["--sweep"]],
    "contrastive": [["--baseline"], ["--sweep"]],
    "fictional": [["--baseline"], ["--sweep"]],
    "biography": [["--baseline"], ["--sweep"]],
    "instruct": [["--baseline"], ["--sweep"]],
    "replicate10": [["--baseline"], ["--sweep"]],
    # The survey trains nothing and sweeps nothing: it runs baseline probes
    # across several models, so it runs its own script instead of main.py.
    "survey": "script:scripts/identity_survey.py",
    "displace05": [["--baseline"], ["--sweep"]],
    "displace15": [["--baseline"], ["--sweep"]],
    "promptbase": "script:scripts/prompt_baseline.py",
    "displacephi3": [["--baseline"], ["--sweep"]],
    # Same script as `survey`, different backend and hardware -- see
    # configs/identity_survey_tpu.yaml for what deliberately differs.
    "surveytpu": "script:scripts/identity_survey.py",
    # Ten seeds at one dose: the decisive comparison currently rests on two.
    "var05": [["--baseline"], ["--sweep"]],
    "varphi3": [["--baseline"], ["--sweep"]],
    "var15": [["--baseline"], ["--sweep"]],
    # Single-model TPU runs, built to finish so the kernel publishes a log.
    "tpudiag": "script:scripts/identity_survey.py",
    "tpusoak": "script:scripts/identity_survey.py",
    "baseline": [["--baseline"]],
    "sweep": [["--sweep"]],
    "all": [["--baseline"], ["--sweep"]],
}
if STAGE not in STAGES:
    raise SystemExit(f"unknown stage {STAGE!r}; expected one of {sorted(STAGES)}")

# Kaggle DISCARDS the output of a kernel that exits non-zero. An errored run
# therefore publishes nothing -- no results, no log, no way to find out what
# went wrong except by spending another accelerator session guessing. That
# happened once and cost a TPU slot plus its queue time.
#
# So the body below never propagates a failure. It prints the traceback, marks
# the run failed in the kernel's own output directory, and exits 0 so Kaggle
# publishes everything that was written. A silent pass is not the risk here:
# every consumer of these runs checks the log and the result file count, and a
# FAILED marker in the output is louder than a status flag that deletes the
# evidence.
(RUNS_DIR).mkdir(parents=True, exist_ok=True)
(RUNS_DIR / "kernel_started.txt").write_text(f"stage={STAGE} config={base_config}\n")

plan = STAGES[STAGE]
try:
    if isinstance(plan, str) and plan.startswith("script:"):
        # A multi-model stage: one script call, no aggregate (there is no single
        # dose-response curve to aggregate across different models).
        sh([sys.executable, plan.split(":", 1)[1], "--config", str(run_config)], cwd=REPO_DIR)
    else:
        for flags in plan:
            sh([sys.executable, "-m", "ghost_identity.main", *flags, "--config", str(run_config)], cwd=REPO_DIR)

        # Final aggregate so the verdict is the last thing in the log either way.
        sh([sys.executable, "-m", "ghost_identity.main", "--aggregate-only", "--config", str(run_config)], cwd=REPO_DIR)
except Exception:
    import traceback

    detail = traceback.format_exc()
    print("\n" + "=" * 70 + "\nSTAGE FAILED\n" + "=" * 70, flush=True)
    print(detail, flush=True)
    (RUNS_DIR / "KERNEL_FAILED.txt").write_text(detail)

print("\n" + "=" * 70, flush=True)
for path in sorted(RUNS_DIR.rglob("*")):
    if path.is_file() and path.suffix in {".csv", ".png"}:
        print("output:", path, flush=True)
print("=" * 70, flush=True)
