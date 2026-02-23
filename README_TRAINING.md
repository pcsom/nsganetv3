# NSGANetV2 Training for LLM Comparison

Train NSGANetV2 architectures on Oxford Flowers-102 to generate ground truth accuracy scores for comparing LLM-based NAS predictors.

## Overview

This workflow trains a corpus of NSGANetV2 neural architectures to compare the performance of different LLM-based performance predictors (CodeLlama vs ModernBERT). Unlike NASBench-201 which has pre-computed accuracy scores, NSGANetV2 is a non-benchmark search space requiring actual training to obtain ground truth performance metrics.

## Dataset

**Oxford Flowers-102 (VIP Shared Storage)**
- **Location:** `/storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers/` (764 MB)
- **Storage Type:** VIP shared storage (accessible to all authorized VIP users)
- **Images:** 2,040 total (1,020 training, 1,020 validation)
- **Classes:** 102 flower categories
- **Resolution:** 224×224 (compatible with NSGANetV2)
- **Download:** `python download_oxford_flowers.py`

**Why VIP Shared Storage?**
- Shared across all VIP users (no need to duplicate dataset)
- Persistent and backed up (not deleted like scratch storage)
- Fast access from compute nodes
- Follows existing project pattern used in coder-nas

**Storage Organization:**
```
/storage/ice-shared/vip-vvk/data/AOT/
├── shared/                        # Shared resources (datasets, common files)
│   └── datasets/
│       └── oxford_flowers/       # Dataset used by all users
│           ├── train/            # 102 class folders
│           ├── val/              # 102 class folders
│           └── ...
├── ${USER}/                      # User-specific directories (e.g., glu49/)
│   ├── nsganetv2/               # NSGANetV2 training outputs
│   │   └── corpus_250/          # Training results
│   ├── codenas/                 # CodeNAS outputs
│   └── ...
└── psomu3/                       # Other users (abb32, athakkar37, etc.)
    └── ...
```

Dataset structure:
```
/storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers/
├── train/          # 102 class folders (class_001 to class_102)
├── val/            # 102 class folders (class_001 to class_102)
├── jpg/            # Original images
└── 102flowers.tgz  # Original archive
```

## Environment Setup

Install dependencies (only needed once):

```bash
conda create -n nsganetv2-llm python=3.10
conda activate nsganetv2-llm
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm==0.6.13 pymoo==0.6.1.5 torchprofile gdown scipy
git clone https://github.com/mit-han-lab/once-for-all.git
cd once-for-all && pip install -e .
```

**Critical:** Must use `timm==0.6.13` (newer versions break compatibility).

Download dataset:
```bash
cd ~/nsganetv3
python download_oxford_flowers.py
# Downloads to /storage/ice1/4/7/glu49/datasets/oxford_flowers/ by default
# Custom path: python download_oxford_flowers.py --data_dir /custom/path
```

Verify setup:
```bash
bash verify_setup.sh
```

## Training Workflow

### 1. Generate Architecture Corpus

Create random NSGANetV2 architecture configurations:

```bash
python generate_simple_corpus.py --output_dir corpus_250 --n_samples 250
```

This generates 500 total architectures (250 base configs × 2 resolutions: 192, 224).

Output structure:
```
corpus_250/
├── arch_0000/
│   ├── config.json      # {ks: [...], e: [...], d: [...], r: 224}
│   └── train_job.sh     # SLURM training script
├── arch_0001/
│   └── ...
└── corpus_metadata.json # Corpus summary
```

### 2. Create Training Jobs

Generate SLURM training scripts for all architectures:

```bash
python create_imagenet_training_jobs.py \
  --corpus_dir corpus_250 \
  --data_path /storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers \
  --num_classes 102 \
  --epochs 100 \
  --batch_size 64 \
  --time_limit 03:00:00
```

This creates individual SLURM job scripts for each architecture and a master submission script.

### 3. Test Single Architecture (Recommended)

Before submitting all jobs, test one architecture with 2 epochs:

```bash
sbatch quick_test.sh
```

Check results:
```bash
tail quick_test.err   # Training progress
squeue -u $USER       # Job status
```

Expected: Accuracy improves from ~1% to ~3% over 2 epochs.

### 4. Submit Training Jobs

Submit all architectures for training:

```bash
cd ~/nsganetv3
bash corpus_250/submit_all_jobs.sh
```

This submits 500 jobs to the SLURM queue. Jobs will run in batches based on available GPU resources.

### 5. Monitor Progress

Check queue status:
```bash
squeue -u $USER
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
