# NSGANetV3 Setup Complete! ✓

## Your Environment is Ready

All tests passed successfully! Here's what's been set up:

### ✓ Installed Components
- Python environment: `nas` (conda)
- PyTorch: 2.3.1 with CUDA 12.1 support
- Core dependencies:
  - pymoo (evolutionary algorithms)
  - torchprofile (FLOPs calculation)
  - timm (model architectures)
  - ofa (Once-for-All supernet)
  - pySOT (surrogate modeling)
  - pydacefit (Gaussian processes)
  - toml (configuration files)

### ✓ Project Files Created
- `SETUP_PACE_ICE.md` - Complete setup documentation
- `QUICKSTART.md` - Quick reference guide
- `setup_environment.sh` - Environment setup script
- `install_missing_deps.sh` - Dependency installer
- `quick_start.sh` - Interactive job submission
- `test_setup.sh` - Setup verification script

## Next Steps

### 1. Download Supernet Weights

You need the pre-trained supernet to start searching. Choose one:

```bash
# Option A: Download to shared storage (recommended)
mkdir -p /storage/ice-shared/vip-vvk/data/supernet
cd /storage/ice-shared/vip-vvk/data/supernet

# MobileNetV3 supernet (recommended for most cases)
wget https://hanlab.mit.edu/files/OnceForAll/ofa_nets/ofa_mbv3_d234_e346_k357_w1.0

# Option B: ResNet50 supernet (alternative)
wget https://hanlab.mit.edu/files/OnceForAll/ofa_nets/ofa_resnet50.pth
```

### 2. Prepare Your Dataset

**For ImageNet:**
```bash
# Ensure your ImageNet data is organized as:
# /path/to/imagenet/
#   train/
#     n01440764/
#     n01443537/
#     ...
#   val/
#     n01440764/
#     n01443537/
#     ...
```

**For CIFAR-10/100:**
These will auto-download on first run, no preparation needed!

**For other datasets:**
See the README.md for download links.

### 3. Run Your First Search

**Interactive mode (recommended for first time):**
```bash
cd /home/hice1/glu49/nsganetv3
bash quick_start.sh
```

**Direct submission:**
```bash
sbatch run_nsganetv3_slurm.sh \
    -s "/storage/ice-shared/vip-vvk/data/AOT/$USER/search-imagenet-test" \
    -p "/path/to/imagenet" \
    -n "/storage/ice-shared/vip-vvk/data/supernet/ofa_mbv3_d234_e346_k357_w1.0" \
    -d imagenet \
    -o flops \
    -i 10
```

### 4. Monitor Your Job

```bash
# Check job status
squeue -u $USER

# View live logs
tail -f /storage/ice-shared/vip-vvk/data/AOT/$USER/evolution_logs/nsganet.*.log

# Check errors
tail -f /storage/ice-shared/vip-vvk/data/AOT/$USER/evolution_logs/nsganet_error.*.log
```

## Quick Command Reference

```bash
# Activate environment
module load anaconda3/2023.03
conda activate nas

# Check Python/PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"

# List all packages
conda list

# Submit a job
sbatch run_nsganetv3_slurm.sh [options]

# Cancel a job
scancel JOB_ID

# Check available GPUs
sinfo -o "%20N %10c %10m %25f %10G"
```

## Example Workflows

### Search on ImageNet (Multi-objective: Accuracy vs FLOPs)
```bash
sbatch run_nsganetv3_slurm.sh \
    -s "/storage/ice-shared/vip-vvk/data/AOT/$USER/imagenet-flops-30" \
    -p "/storage/ice-shared/vip-vvk/data/imagenet" \
    -n "/storage/ice-shared/vip-vvk/data/supernet/ofa_mbv3_d234_e346_k357_w1.0" \
    -d imagenet \
    -o flops \
    -i 30 \
    -r rbf
```

### Search on CIFAR-10 (Multi-objective: Accuracy vs Parameters)
```bash
sbatch run_nsganetv3_slurm.sh \
    -s "/storage/ice-shared/vip-vvk/data/AOT/$USER/cifar10-params-30" \
    -p "/path/to/cifar10" \
    -n "/storage/ice-shared/vip-vvk/data/supernet/ofa_mbv3_d234_e346_k357_w1.0" \
    -d cifar10 \
    -o params \
    -i 30 \
    -r gp
```

### Quick Test Run (5 iterations)
```bash
sbatch run_nsganetv3_slurm.sh \
    -s "/storage/ice-shared/vip-vvk/data/AOT/$USER/test-run" \
    -p "/path/to/dataset" \
    -n "/storage/ice-shared/vip-vvk/data/supernet/ofa_mbv3_d234_e346_k357_w1.0" \
    -d imagenet \
    -o flops \
    -i 5
```

## Expected Timeline

- **Environment setup**: ✓ Complete!
- **Supernet download**: 2-5 minutes
- **Test search (5 iterations)**: 2-4 hours
- **Full search (30 iterations)**: 24-48 hours

## Troubleshooting

### Job fails to start?
```bash
# Check your queue
squeue -u $USER

# Verify paths exist
ls -l /path/to/your/dataset
ls -l /path/to/supernet
```

### Import errors?
```bash
# Re-run dependency install
bash install_missing_deps.sh

# Verify installation
bash test_setup.sh
```

### Out of memory?
Edit `config/nsganetv3_config.toml` and reduce:
- `vld_size` (validation set size)
- `trn_batch_size` (training batch size)

## Documentation

- **QUICKSTART.md** - This file (quick reference)
- **SETUP_PACE_ICE.md** - Detailed setup guide
- **README.md** - Original NSGANetV2 documentation
- **config/nsganetv3_config.toml** - Configuration options

## Need Help?

1. Check the logs in `/storage/ice-shared/vip-vvk/data/AOT/$USER/evolution_logs/`
2. Review SETUP_PACE_ICE.md for detailed instructions
3. Run `bash test_setup.sh` to verify setup
4. Ensure all paths in your command are correct and accessible

## You're All Set! 🚀

Your NSGANetV3 environment is fully configured and ready to run on PACE ICE.

**Recommended first step:**
```bash
cd /home/hice1/glu49/nsganetv3
bash quick_start.sh
```

Good luck with your neural architecture search!
