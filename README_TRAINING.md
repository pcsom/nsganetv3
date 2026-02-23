# NSGANetV2 Training

Train NSGANetV2 architectures on Oxford Flowers-102 to generate ground truth accuracy scores.

## Dataset

- **Location:** `/storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers/`
- **Classes:** 102 flower categories
- **Resolution:** 224×224

## Setup

See [SETUP.md](SETUP.md) for environment configuration.

## Quick Start

```bash
./run_full_training_workflow.sh corpus_name num_archs epochs batch_size time_limit
```

Example (500 architectures, 100 epochs):
```bash
./run_full_training_workflow.sh prod_500 250 100 64 03:00:00
```

## Manual Steps

### 1. Generate Corpus
```bash
python generate_simple_corpus.py --output_dir corpus_name --n_samples 250
```

### 2. Create Jobs
```bash
python create_imagenet_training_jobs.py --corpus_dir corpus_name
```

### 3. Submit
```bash
bash corpus_name/submit_all_jobs.sh
```

### 4. Collect Results
```bash
python collect_training_results.py --corpus_dir corpus_name --output_csv results.csv
```

Count running jobs:
```bash
squeue -u $USER | grep "RUNNING" | wc -l
```

Count completed jobs:
```bash
find corpus_250 -name "status.json" -exec grep -l "success" {} \; | wc -l
```

Check training progress for a specific architecture:
```bash
tail -20 corpus_250/arch_0000/train.err
cat corpus_250/arch_0000/train/*/summary.csv
```

Watch completion in real-time:
```bash
watch -n 60 'find corpus_250 -name "status.json" -exec grep -l "success" {} \; | wc -l'
```

### 6. Collect Results

Once all jobs complete, extract accuracy metrics:

```bash
python collect_training_results.py \
  --corpus_dir corpus_250 \
  --output_csv nsganetv2_oxford_results.csv
```

Output CSV contains:
```
arch_id,config_path,best_accuracy,final_accuracy,status
0,corpus_250/arch_0000/config.json,85.78,85.78,success
1,corpus_250/arch_0001/config.json,82.45,82.45,success
...
```

## Training Details

**Per-architecture training:**
- Epochs: 100
- Batch size: 64
- Learning rate: 0.01 (starts at 1e-4, warmup to 0.01)
- Optimizer: SGD with momentum 0.9
- Weight decay: 1e-4
- Augmentations: AutoAugment, RandAugment, Mixup, Cutmix
- Time: ~1-2 hours per architecture
- GPU: RTX 6000 or H100
- Checkpoints: Saved every epoch to `corpus_250/arch_XXXX/train/`

**Total expected time:**
- Sequential: 500-1000 GPU hours (20-40 days on 1 GPU)
- Parallel (50 GPUs): ~10-20 hours wall time

**Storage requirements:**
- Dataset: 764 MB
- Per architecture: ~350 MB (checkpoints)
- Total corpus: ~175 GB (500 archs × 350 MB)

## Output Files

Per architecture (`corpus_250/arch_XXXX/`):
```
arch_0000/
├── config.json           # Architecture configuration
├── train_job.sh          # SLURM script
├── train.log             # SLURM stdout
├── train.err             # Training progress logs
├── status.json           # {"status": "success", "arch_id": 0}
└── train/
    └── 20260222-HHMMSS-nsganetv2-224/
        ├── args.yaml               # Training arguments
        ├── summary.csv             # Epoch-by-epoch metrics
        ├── checkpoint-*.pth.tar    # Per-epoch checkpoints
        ├── model_best.pth.tar      # Best model
        └── last.pth.tar            # Final model
```

## Troubleshooting

**Import errors:**
- Ensure `timm==0.6.13` (not newer)
- Check OFA installed: `python -c "import ofa"`
- Verify pymoo: `python -c "from pymoo.core.problem import Problem"`

**SLURM job failures:**
- Check error log: `cat corpus_250/arch_XXXX/train.err`
- Verify GPU allocation: `squeue -u $USER`
- Check disk space: `df -h ~/scratch`

**Low accuracy:**
- Normal: Early epochs (1-5) have 1-20% accuracy
- Expected: Final accuracy 70-90% for most architectures
- If stuck at <5% after 20 epochs, check logs for errors

**Out of memory:**
- Reduce batch size in `create_imagenet_training_jobs.py`
- Default 64 works on RTX 6000 (24 GB) and H100 (80 GB)

## Next Steps: LLM Comparison

After collecting results, use them for LLM predictor comparison:

1. **Transfer results to NASlib repo:**
   ```bash
   cp nsganetv2_oxford_results.csv ~/NASlib-coder-nas/data/
   ```

2. **Generate LLM embeddings:**
   - CodeLlama: Encode architecture configs as code strings
   - ModernBERT: Encode architecture configs as natural language

3. **Train predictors:**
   - Use embeddings + ground truth accuracy from CSV
   - Compare prediction performance (Kendall's Tau)

4. **Analyze results:**
   - Compare NSGANetV2 results with NASBench-201 baseline
   - Expected: Tau correlation 0.6-0.85 depending on LLM

See main project README for full LLM comparison workflow.

## Notes

- Uses timm 0.6.13 (critical - newer versions incompatible)
- Training takes ~1-2 hours per architecture (100 epochs)
- Architecture format: `{ks: [kernel sizes], e: [expansion ratios], d: [depths], r: resolution}`
- OFA supernet checkpoint: `checkpoints/ofa_mbv3_d234_e346_k357_w1.0`
- All paths use `$HOME` and absolute paths for portability
- Dataset stored at: `/storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers/` (VIP shared, read-only for all users)
- Training outputs saved to: `~/nsganetv3/corpus_250/` (local workspace, user-specific)
- SLURM partition: `ice-gpu` (standard GPU partition for PACE ICE)

**For Other Users:**
To set up your own training pipeline:
1. Clone repo to your workspace
2. Dataset is already available at shared location (no download needed)
3. Training outputs will be saved to your workspace (not shared)
4. Results can be saved to your VIP directory: `/storage/ice-shared/vip-vvk/data/AOT/${USER}/nsganetv2/`

**Important:** Only commit source code files to git. The `.gitignore` excludes:
- Dataset files (`data/`, `datasets/`)
- Training outputs (`corpus_*/`, `*_corpus_*/`, `test_corpus_*/`)
- Checkpoints (`checkpoints/`, `*.pth.tar`)
- Logs (`*.log`, `*.err`)
- Temporary outputs (`quick_test_output/`, `20*/`)
- Dataset: 2,040 images (1,020 train, 1,020 val)
- Architecture configs: {ks, e, d, r} format
