# Quick Reference

## Automated Training

```bash
./run_full_training_workflow.sh corpus_name num_archs epochs batch_size time
./run_full_training_workflow.sh prod_500 250 100 64 03:00:00

# Dry-run preflight (no corpus/jobs submitted)
DRY_RUN=1 ./run_full_training_workflow.sh prod_500 250 100 64 03:00:00

# Reproducible corpus seed
CORPUS_SEED=42 ./run_full_training_workflow.sh prod_500 250 100 64 03:00:00

# Optional output root (per-user VIP storage)
OUTPUT_ROOT=/storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2 \
	./run_full_training_workflow.sh prod_500 250 100 64 03:00:00
```

## Monitor

```bash
./monitor_training.sh /storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/corpus_name
watch -n 30 './monitor_training.sh /storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/corpus_name'
```

## Manual

```bash
# Generate
python generate_simple_corpus.py --output_dir /storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/corpus_name --n_samples 250 --seed 42

# Create SLURM scripts
python create_imagenet_training_jobs.py --corpus_dir /storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/corpus_name

# Test one
sbatch quick_test.sh

# Submit all
bash /storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/corpus_name/submit_all_jobs.sh

# Collect results
python collect_training_results.py --corpus_dir /storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/corpus_name --output_csv results.csv --surrogate_output surrogate_ready.csv
```

## Queue

```bash
squeue -u $USER              # Status
tail -50 /storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/corpus/arch_0000/train.err  # Check errors
```
