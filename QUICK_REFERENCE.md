# Quick Reference

## Automated Training

```bash
./run_full_training_workflow.sh corpus_name num_archs epochs batch_size time
./run_full_training_workflow.sh prod_500 250 100 64 03:00:00
```

## Monitor

```bash
./monitor_training.sh corpus_name
watch -n 30 './monitor_training.sh corpus_name'
```

## Manual

```bash
# Generate
python generate_simple_corpus.py --output_dir corpus_name --n_samples 250

# Create SLURM scripts
python create_imagenet_training_jobs.py --corpus_dir corpus_name

# Test one
sbatch quick_test.sh

# Submit all
bash corpus_name/submit_all_jobs.sh

# Collect results
python collect_training_results.py --corpus_dir corpus_name --output_csv results.csv
```

## Queue

```bash
squeue -u $USER              # Status
tail -50 corpus/arch_0000/train.err  # Check errors
```
