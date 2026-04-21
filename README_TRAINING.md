# NSGANetV2 Training

Train NSGANetV2 architectures on Oxford Flowers-102 to generate ground truth accuracy scores.

## Dataset

- **Default Location:** `/storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers/`
- **Classes:** 102 flower categories
- **Resolution:** 224×224

**For other users:** If you don't have access to the shared dataset, download your own copy:
```bash
python download_oxford_flowers.py --data_dir ~/scratch/datasets/oxford_flowers
export DATASET_PATH=~/scratch/datasets/oxford_flowers
```

## Output Location

- Default: `/storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/<corpus_name>/`
- Override: set `OUTPUT_ROOT` before running the workflow.

## Setup

See [SETUP.md](SETUP.md) for environment setup and checkpoint download.

## Quick Start

**Note:** If using your own dataset copy, set `DATASET_PATH` first:
```bash
export DATASET_PATH=~/scratch/datasets/oxford_flowers
```

Non-interactive (for automated runs):
```bash
./run_full_training_workflow.sh prod_500 250 100 64 03:00:00
```

Preflight check only (no jobs submitted):
```bash
DRY_RUN=1 ./run_full_training_workflow.sh prod_500 250 100 64 03:00:00
```

Or interactive (prompts for each step):
```bash
./run_full_training_workflow.sh corpus_name num_archs epochs batch_size time_limit
```

Parameters:
- `corpus_name`: Name of the architecture corpus (e.g., `prod_500`)
- `num_archs`: Number of base architectures to sample (250 will create 500 total with 2 resolutions)
- `epochs`: Training epochs per architecture (100 typical)
- `batch_size`: Batch size (64 recommended for RTX 6000)
- `time_limit`: SLURM time limit per job (e.g., `03:00:00` for 3 hours)

Example (500 architectures, 100 epochs):
```bash
NON_INTERACTIVE=1 ./run_full_training_workflow.sh prod_500 250 100 64 03:00:00
```

For reproducible corpus generation across teammates, set a fixed seed:
```bash
CORPUS_SEED=42 NON_INTERACTIVE=1 ./run_full_training_workflow.sh prod_500 250 100 64 03:00:00
```

Small WIP step toward LLM comparison (also emit embedding-manifest JSONL):
```bash
GENERATE_EMBEDDING_MANIFEST=1 NON_INTERACTIVE=1 ./run_full_training_workflow.sh prod_500 250 100 64 03:00:00
```

## Manual Steps (Optional)

```bash
# Generate corpus
python generate_simple_corpus.py --output_dir /storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/corpus_name --n_samples 250 --seed 42

# Create jobs
python create_imagenet_training_jobs.py --corpus_dir /storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/corpus_name

# Submit
bash /storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/corpus_name/submit_all_jobs.sh

# Collect results
python collect_training_results.py --corpus_dir /storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/corpus_name --output_csv results.csv --surrogate_output surrogate_ready.csv
```

## Monitor

```bash
./monitor_training.sh /storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/corpus_name
```
