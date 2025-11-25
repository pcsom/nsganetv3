# NSGANetV3 Setup Guide for PACE ICE SLURM

This guide will help you set up and run NSGANetV3 on the Georgia Tech PACE ICE cluster.

## Prerequisites

1. Access to PACE ICE cluster
2. Storage space for datasets and results (recommended: `/storage/ice-shared/vip-vvk/data/`)

## Step 1: Create Conda Environment

```bash
# Load anaconda module
module load anaconda3/2023.03

# Create a new conda environment for NAS
conda create -n nas python=3.7 -y

# Activate the environment
conda activate nas
```

## Step 2: Install Required Packages

```bash
# Install PyTorch with CUDA support
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch -y

# Install Cython for faster operations (optional but recommended)
pip install Cython==0.29

# Install core dependencies
pip install pymoo==0.4.1
pip install torchprofile==0.0.1
pip install timm==0.1.30
pip install pySOT==0.2.3
pip install pydacefit==1.0.1
pip install toml

# Install OnceForAll (the supernet framework)
pip install ofa==0.0.4

# Install other common dependencies
pip install pandas numpy scipy matplotlib seaborn tqdm pyyaml
```

## Step 3: Download Datasets

NSGANetV3 supports multiple datasets. Here's how to get them:

### ImageNet (1000 classes)
```bash
# ImageNet needs to be downloaded from official source
# Place it in: /storage/ice-shared/vip-vvk/data/imagenet/
# Structure should be:
#   imagenet/train/
#   imagenet/val/
```

### CIFAR-10/100 (Auto-download)
```bash
# These will download automatically when you run the code
# No manual setup needed
```

### Other Datasets
For fine-grained datasets (Aircraft, Flowers102, Pets, DTD, etc.), see the README for download links.

## Step 4: Download Pre-trained Supernet

The supernet is the foundation for architecture search.

```bash
# Create directory for supernet weights
mkdir -p /storage/ice-shared/vip-vvk/data/supernet

# Download ImageNet supernet (choose one):
cd /storage/ice-shared/vip-vvk/data/supernet

# Option 1: OFA ResNet50 supernet
wget https://hanlab.mit.edu/files/OnceForAll/ofa_nets/ofa_resnet50.pth

# Option 2: OFA MobileNetV3 supernet  
wget https://hanlab.mit.edu/files/OnceForAll/ofa_nets/ofa_mbv3_d234_e346_k357_w1.0

# Set environment variable for convenience
export SUPERNET_PATH=/storage/ice-shared/vip-vvk/data/supernet/ofa_mbv3_d234_e346_k357_w1.0
```

## Step 5: Configure SLURM Script

The `run_nsganetv3_slurm.sh` script is already set up for PACE ICE. Key settings:

- **Modules**: `anaconda3/2023.03` and `cuda/12.1.1`
- **Default conda env**: `nas`
- **GPU types**: V100, L40S, A100, H100, A40, H200
- **Default time**: 48 hours (adjust as needed)

## Step 6: Run Architecture Search

### Basic Search on ImageNet

```bash
# Navigate to the project directory
cd /home/hice1/glu49/nsganetv3

# Submit a search job
sbatch run_nsganetv3_slurm.sh \
    -s "/storage/ice-shared/vip-vvk/data/AOT/$USER/search-imagenet-flops" \
    -p "/storage/ice-shared/vip-vvk/data/imagenet" \
    -n "$SUPERNET_PATH" \
    -d imagenet \
    -o flops \
    -i 30 \
    -r rbf \
    -e nas
```

### Search on CIFAR-10

```bash
sbatch run_nsganetv3_slurm.sh \
    -s "/storage/ice-shared/vip-vvk/data/AOT/$USER/search-cifar10-params" \
    -p "/path/to/cifar10/data" \
    -n "$SUPERNET_PATH" \
    -d cifar10 \
    -o params \
    -i 30 \
    -r rbf \
    -e nas
```

### Using Configuration File

```bash
# Use the provided TOML config
sbatch run_nsganetv3_slurm.sh \
    -s "/storage/ice-shared/vip-vvk/data/AOT/$USER/search-custom" \
    -p "/storage/ice-shared/vip-vvk/data/imagenet" \
    -n "$SUPERNET_PATH" \
    -c config/nsganetv3_config.toml \
    -e nas
```

## Step 7: Monitor Progress

```bash
# Check job status
squeue -u $USER

# View logs (replace JOB_ID with your actual job ID)
tail -f /storage/ice-shared/vip-vvk/data/AOT/$USER/evolution_logs/nsganet.JOB_ID.0.log

# Check for errors
tail -f /storage/ice-shared/vip-vvk/data/AOT/$USER/evolution_logs/nsganet_error.JOB_ID.0.log
```

## Step 8: Select Best Architectures

After search completes:

```bash
# Activate conda environment
conda activate nas

# Select architectures with preferences
python post_search.py \
    -n 3 \
    --save /path/to/search/final \
    --expr /path/to/search/iter_30.stats \
    --prefer top1#80+flops#150 \
    --supernet_path $SUPERNET_PATH

# Or select based on trade-offs (no preferences)
python post_search.py \
    -n 5 \
    --save /path/to/search/final \
    --expr /path/to/search/iter_30.stats \
    --prefer None \
    --supernet_path $SUPERNET_PATH
```

## Step 9: Fine-tune Selected Architectures

```bash
# For ImageNet (distributed training)
bash scripts/distributed_train.sh 8 \
    /storage/ice-shared/vip-vvk/data/imagenet/ \
    --model nsganetv2_custom \
    --model-config /path/to/final/net.config \
    --initial-checkpoint /path/to/final/net.inherited \
    --img-size 224 \
    -b 128 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 \
    --opt rmsproptf --opt-eps .001 -j 6 --warmup-lr 1e-6 \
    --weight-decay 1e-5 --drop 0.2 --drop-path 0.2 --model-ema \
    --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel \
    --reprob 0.2 --amp --lr .024 \
    --teacher $SUPERNET_PATH

# For CIFAR-10
python train_cifar.py \
    --data /path/to/cifar10/ \
    --model nsganetv2_cifar \
    --model-config /path/to/final/net.config \
    --img-size 224 \
    --drop 0.2 --drop-path 0.2 \
    --cutout --autoaugment --save
```

## Command Line Arguments Reference

### run_nsganetv3_slurm.sh Options:

- `-s <save_dir>`: Directory to save search results (REQUIRED)
- `-p <data_path>`: Path to dataset (REQUIRED)
- `-n <supernet_path>`: Path to supernet weights (REQUIRED)
- `-d <dataset>`: Dataset name (default: imagenet)
- `-o <sec_obj>`: Secondary objective - flops/params/cpu (default: flops)
- `-i <iterations>`: Number of search iterations (default: 30)
- `-r <predictor>`: Surrogate predictor - rbf/gp/cart/mlp/as (default: rbf)
- `-e <conda_env>`: Conda environment name (default: nas)
- `-c <config_file>`: Path to TOML configuration file (optional)

## Common Issues & Solutions

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in config or use smaller validation set size

### Issue: "Module not found"
**Solution**: Make sure you've activated the conda environment:
```bash
conda activate nas
```

### Issue: "Supernet weights not found"
**Solution**: Verify the path and download the supernet:
```bash
ls -lh $SUPERNET_PATH
```

### Issue: "Job pending for a long time"
**Solution**: Check partition availability or adjust GPU requirements in config

## Directory Structure

After setup, your structure should look like:
```
/storage/ice-shared/vip-vvk/data/
├── imagenet/               # ImageNet dataset
│   ├── train/
│   └── val/
├── supernet/              # Pre-trained supernet weights
│   └── ofa_mbv3_d234_e346_k357_w1.0
└── AOT/
    └── <username>/
        ├── search-imagenet-flops/    # Search results
        └── evolution_logs/            # SLURM logs
```

## Next Steps

1. Start with a small search (fewer iterations) to verify everything works
2. Monitor the first few iterations to ensure proper execution
3. Scale up to full 30-iteration searches for production runs
4. Experiment with different secondary objectives (flops, params, latency)
5. Try different surrogate models (rbf, gp, as) for accuracy prediction

## References

- Original paper: [NSGANetV2 (ECCV 2020)](https://arxiv.org/abs/2007.10396)
- Once-for-All: [OFA GitHub](https://github.com/mit-han-lab/once-for-all)
- PACE ICE docs: [PACE Documentation](https://docs.pace.gatech.edu/)
