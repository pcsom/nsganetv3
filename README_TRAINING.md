# NSGANetV2 Training

Train NSGANetV2 architectures on Oxford Flowers-102 to generate ground truth accuracy scores.

## Dataset

- **Location:** `/storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers/`
- **Classes:** 102 flower categories
- **Resolution:** 224×224

## Output Location

- Default: `/storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/<corpus_name>/`
- Override: set `OUTPUT_ROOT` before running the workflow.

## Setup

See [SETUP.md](SETUP.md).

## Quick Start

```bash
./run_full_training_workflow.sh corpus_name num_archs epochs batch_size time_limit
```

Example (500 architectures, 100 epochs):
```bash
./run_full_training_workflow.sh prod_500 250 100 64 03:00:00
```

## Manual Steps (Optional)

```bash
# Generate corpus
python generate_simple_corpus.py --output_dir /storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/corpus_name --n_samples 250

# Create jobs
python create_imagenet_training_jobs.py --corpus_dir /storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/corpus_name

# Submit
bash /storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/corpus_name/submit_all_jobs.sh

# Collect results
python collect_training_results.py --corpus_dir /storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/corpus_name --output_csv results.csv
```

## Monitor

```bash
./monitor_training.sh /storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/corpus_name
```
