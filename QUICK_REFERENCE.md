# NSGANetV2 Training - Quick Reference

## Fully Automated Training (Recommended)

**One-command solution:**
```bash
./run_full_training_workflow.sh
```

This will:
- ✅ Generate 250 architectures (500 with resolutions)
- ✅ Create SLURM job scripts
- ✅ Test one architecture (you can skip)
- ✅ Submit all jobs to queue
- ✅ Optionally wait for completion
- ✅ Collect results to CSV

**Custom configuration:**
```bash
./run_full_training_workflow.sh my_corpus 100 50 32 02:00:00
#                                ^         ^   ^  ^  ^
#                                name      #   ep bs time
```

## Monitor Training Progress

```bash
# One-time check
./monitor_training.sh corpus_250

# Live updates every 30 seconds
watch -n 30 './monitor_training.sh corpus_250'
```

Shows:
- Jobs running/pending/completed
- Completion percentage
- Recent results with accuracy
- Estimated time remaining
- Failed jobs with errors

## Manual Step-by-Step (Advanced)

If you need fine-grained control:

### 1. Generate Corpus
```bash
python generate_simple_corpus.py --output_dir corpus_250 --n_samples 250
```

### 2. Create Jobs
```bash
python create_imagenet_training_jobs.py \
  --corpus_dir corpus_250 \
  --epochs 100 \
  --batch_size 64
```

### 3. Quick Test
```bash
sbatch quick_test.sh
# Check: tail quick_test.err
```

### 4. Submit All
```bash
bash corpus_250/submit_all_jobs.sh
```

### 5. Monitor
```bash
# Queue status
squeue -u $USER

# Completion count
find corpus_250 -name "status.json" -exec grep -l "success" {} \; | wc -l

# Check specific architecture
tail -50 corpus_250/arch_0042/train.err
cat corpus_250/arch_0042/train/*/summary.csv
```

### 6. Collect Results
```bash
python collect_training_results.py \
  --corpus_dir corpus_250 \
  --output_csv nsganetv2_results.csv
```

## Current Training Status

**corpus_250 (currently running):**
- Total: 500 architectures
- Completed: 14 (2.8%)
- Running: 5 jobs
- Pending: 31 jobs
- Average accuracy: ~94%
- Estimated completion: ~14 hours

Check latest: `./monitor_training.sh corpus_250`

## File Locations

**Dataset (shared, read-only):**
- `/storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers/`

**Training outputs (your workspace):**
- `~/nsganetv3/corpus_250/`
- Each architecture: `corpus_250/arch_XXXX/`
  - `config.json` - Architecture configuration
  - `train.log` - SLURM stdout
  - `train.err` - Training progress
  - `train/*/checkpoint-*.pth.tar` - Model checkpoints
  - `train/*/summary.csv` - Epoch-by-epoch metrics

**Results CSV:**
- `nsganetv2_results.csv` - Final accuracy for all architectures

## Next Steps After Training

1. **Collect results:**
   ```bash
   python collect_training_results.py \
     --corpus_dir corpus_250 \
     --output_csv nsganetv2_oxford_results.csv
   ```

2. **Transfer to NASlib:**
   ```bash
   cp nsganetv2_oxford_results.csv ~/NASlib-coder-nas/data/
   ```

3. **Generate LLM embeddings:**
   - CodeLlama: Encode architectures as code
   - ModernBERT: Encode architectures as text

4. **Run LLM comparison:**
   - Train predictors on embeddings + ground truth
   - Compare Kendall's Tau correlation
   - Compare with NASBench-201 baseline

## Troubleshooting

**Jobs pending forever:**
```bash
# Check partition limits
squeue -u $USER -o "%.10i %.12P %.20j %.8T %.10r"
```

**Jobs failing:**
```bash
# Check specific error
tail -50 corpus_250/arch_XXXX/train.err
cat corpus_250/arch_XXXX/train.log
```

**Out of memory:**
- Reduce batch size: `--batch_size 32` (default: 64)
- Edit job scripts: `corpus_250/arch_*/train_job.sh`

**Dataset not found:**
```bash
# Verify dataset exists
ls -la /storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers/train/

# Re-download if needed
python download_oxford_flowers.py
```

## Environment Info

- **Python environment:** `nsganetv2-llm` (conda)
- **GPU partition:** `ice-gpu` (RTX 6000) or `coe-gpu` (any GPU)
- **Time limit:** 3 hours per job (default)
- **Training time:** ~8 minutes per architecture (100 epochs on Oxford Flowers)
- **Storage:** VIP shared storage for dataset, local workspace for outputs

## Tips

- **Test first:** Always run `sbatch quick_test.sh` before submitting 500 jobs
- **Monitor regularly:** Use `watch -n 30 './monitor_training.sh corpus_250'`
- **Save results:** Copy CSV to VIP storage: `cp results.csv /storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/`
- **Clean up:** After collecting results, you can delete checkpoints: `find corpus_250 -name "*.pth.tar" -delete` (saves ~175 GB)
