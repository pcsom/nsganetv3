# NSGANetV3 Surrogate Pipeline Setup (PACE ICE)

## Why this pipeline exists

This repository was migrated from OFA-supernet evaluation to a surrogate-first virtual NAS loop.

The old flow was bottlenecked by ImageNet-centric supernet assumptions and did not transfer cleanly to custom datasets without costly fine-tuning loops. The new flow uses:

- Offline Oxford Flowers102 ground-truth CSV as archive bootstrap.
- Surrogate predictions for top-1 error.
- Static complexity objective (FLOPs/params) without physical candidate training in the search loop.
- SLURM orchestration on PACE ICE.

## What this guide runs

This guide covers:

1. Environment setup (`nsganetv2-llm`).
2. Dependency checks.
3. Single smoke run on SLURM.
4. Longer run with live monitoring (`squeue` + log tailing).
5. Expected artifacts and troubleshooting.

## Prerequisites

- Linux shell on PACE ICE.
- Access to this repo.
- Access to shared storage under `/storage/ice-shared/vip-vvk/data/AOT/$USER`.
- Ground-truth CSV already available at:
  - `/storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/prod_500/prod_500_results.csv`

## 1) One-time environment setup

If environment already exists, skip to step 2.

```bash
module load anaconda3/2023.03
module load cuda/12.1.1

conda create -n nsganetv2-llm python=3.10 -y
conda activate nsganetv2-llm

pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm==0.6.13 pymoo==0.6.1.5 torchprofile scipy pandas pyyaml pySOT
```

Notes:

- `pymoo==0.6.1.5` is required because search code now targets the 0.6 API.
- `pySOT` is required for the `rbf` surrogate backend.
- OFA checkpoint is no longer required for virtual search.

## 2) Verify environment and config

```bash
module load anaconda3/2023.03
conda run -n nsganetv2-llm python - <<'PY'
import torch, pymoo, scipy, pandas
import pySOT
print("torch", torch.__version__)
print("pymoo", pymoo.__version__)
print("deps_ok")
PY
```

Check main config:

```bash
python - <<'PY'
from surrogate_validation import load_toml_config
cfg = load_toml_config("config/nsganetv3_config.toml")
print("dataset:", cfg["dataset"]["dataset"])
print("n_classes:", cfg["dataset"]["n_classes"])
print("ground_truth_csv:", cfg["dataset"]["ground_truth_csv"])
print("evaluation_mode:", cfg["search"]["evaluation_mode"])
print("slurm_env:", cfg["slurm"]["env_name"])
PY
```

## 3) Submit SLURM smoke run

This submits one short virtual search job.

```bash
ts=$(date +%Y%m%d_%H%M%S)
save_dir="/storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv3/slurm_smoke_${ts}"

sbatch --time=02:00:00 \
  --export=ALL,NSGANET_FAST_PROFILE=1 \
  run_nsganetv3_slurm.sh \
  -s "$save_dir" \
  -c ".tmp/smoke_config.toml" \
  -m "virtual" \
  -e "nsganetv2-llm" \
  -i 2

# If your CSV lives somewhere else, add:
# -f /absolute/path/to/your_ground_truth.csv
```

## 4) Live monitoring (required)

After `sbatch`, note job id (example: `5092999`).

Queue status:

```bash
squeue -j <job_id> -o "%.18i %.9P %.25j %.8u %.2t %.10M %.6D %R"
```

Main log file:

```bash
log_file="/storage/ice-shared/vip-vvk/data/AOT/$USER/evolution_logs/nsganet.<job_id>.4294967294.log"
```

Live tail:

```bash
tail -f "$log_file"
```

What healthy progress looks like:

- Archive load progress:
  - `parsed 100 rows, kept 100`, ... `parsed 500 rows, kept 500`.
- Iteration stage logs:
  - `iter 1/2: fitting surrogate`
  - `iter 1: running candidate search`
  - `predictor progress: ...`
  - `iter 1: wrote .../iter_1.stats`
- Final:
  - `NSGANetV3 search completed`

## 5) Expected outputs

In `save_dir`:

- `runtime_summary.json`
- `iter_1.stats`, `iter_2.stats`, ...

Quick check:

```bash
python - <<'PY'
import json, glob
files = sorted(glob.glob("/storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv3/slurm_smoke_*/iter_*.stats"))
print("stats_files:", len(files))
if files:
    d = json.load(open(files[-1], "r", encoding="utf-8"))
    print("archive_len:", len(d.get("archive", [])))
    print("cand_len:", len(d.get("candidates", [])))
    print("hv:", d.get("hv"))
    print("evaluation_mode:", d.get("evaluation_mode"))
PY
```

## Common issues and fixes

1. `ModuleNotFoundError: No module named 'pySOT'`

- Cause: `rbf` predictor backend dependency missing.
- Fix:

```bash
conda run -n nsganetv2-llm pip install pySOT
```

2. Job submission error: invalid requested time

- Cause: partition/qos constraints reject default script time in some contexts.
- Fix: override on submission:

```bash
sbatch --time=04:00:00 run_nsganetv3_slurm.sh ...
```

3. Long silent period during search

- Cause: expensive inner-loop predictor/profiling without logging.
- Fix already implemented:
  - Stage logs in `msunas.py`.
  - Predictor progress counters.
  - Optional fast profiling mode with `NSGANET_FAST_PROFILE=1`.

4. Ground truth CSV lacks complexity columns

- Cause: CSV contains only accuracy fields.
- Fix already implemented:
  - `offline_loader.py` computes complexity fallback via deterministic proxy.

## Recommended command for longer production run

```bash
ts=$(date +%Y%m%d_%H%M%S)
save_dir="/storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv3/slurm_prod_${ts}"

sbatch --time=08:00:00 \
  run_nsganetv3_slurm.sh \
  -s "$save_dir" \
  -c "config/nsganetv3_config.toml" \
  -m "virtual" \
  -e "nsganetv2-llm" \
  -i 10 \
  -r as
```
## Documentation notes from migration
- `run_nsganetv3_slurm.sh` now defers to TOML config unless you explicitly pass override flags (`-r`, `-i`, `-d`, `-o`, `-m`).
- If `-f`is omitted, launcher attempts offline CSv auto-detection at:
  - `/storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/prod_500/prod_500_results.csv`
- For authoritative validation, always run via `sbatch` and monitor with `squeue` + `tail -f` on the SLURM log.