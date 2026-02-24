# Environment Setup

## Install (One-time)

```bash
bash setup_environment.sh
```

Or manually:

```bash
conda create -n nsganetv2-llm python=3.10
conda activate nsganetv2-llm
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm==0.6.13 pymoo==0.6.1.5 torchprofile gdown scipy pandas pyyaml
git clone https://github.com/mit-han-lab/once-for-all.git
cd once-for-all && pip install -e .
```

## Get OFA Checkpoint

The OFA MobileNetV3 supernet checkpoint is required.

### Option 1: Copy from Shared Location (Recommended)

```bash
mkdir -p checkpoints
cp /storage/ice-shared/vip-vvk/data/AOT/shared/checkpoints/ofa_mbv3_d234_e346_k357_w1.0 checkpoints/
```

### Option 2: Download (if shared copy not accessible)

```bash
mkdir -p checkpoints
cd checkpoints
gdown 1qmq7vWW6QkOPHfqXNnVvYQCFCQ2P6PUP
cd ..
```

Or manually download from: https://github.com/mit-han-lab/once-for-all/releases/download/v0.1-ckpts/ofa_mbv3_d234_e346_k357_w1.0

The checkpoint file should be at: `checkpoints/ofa_mbv3_d234_e346_k357_w1.0`

## Download Dataset

### Option 1: Use Shared Dataset (if accessible)

The dataset is available at `/storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers`

Test access:
```bash
ls /storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers/train
```

If you see "Permission denied", use Option 2.

### Option 2: Download Your Own Copy

```bash
python download_oxford_flowers.py --data_dir ~/scratch/datasets/oxford_flowers
```

Then set the environment variable when running workflows:
```bash
export DATASET_PATH=~/scratch/datasets/oxford_flowers
./run_full_training_workflow.sh corpus_name 250 100 64 03:00:00
```

## Verify

```bash
bash verify_setup.sh
```