# Quick Start Guide for NSGANetV3 on PACE ICE

## TL;DR - Get Running in 3 Steps

### Step 1: Setup Environment (5-10 minutes)
```bash
cd /home/hice1/glu49/nsganetv3
bash setup_environment.sh
```

### Step 2: Download Supernet (if not already done)
```bash
# Create directory
mkdir -p /storage/ice-shared/vip-vvk/data/supernet

# Download supernet weights
cd /storage/ice-shared/vip-vvk/data/supernet
wget https://hanlab.mit.edu/files/OnceForAll/ofa_nets/ofa_mbv3_d234_e346_k357_w1.0
```

### Step 3: Run Search
```bash
cd /home/hice1/glu49/nsganetv3
bash quick_start.sh
```

OR manually submit:
```bash
sbatch run_nsganetv3_slurm.sh \
    -s "/storage/ice-shared/vip-vvk/data/AOT/$USER/my-search" \
    -p "/path/to/your/dataset" \
    -n "/storage/ice-shared/vip-vvk/data/supernet/ofa_mbv3_d234_e346_k357_w1.0"
```

## What Each File Does

- **setup_environment.sh** - Creates the conda environment with all dependencies
- **quick_start.sh** - Interactive script to help you submit a job
- **run_nsganetv3_slurm.sh** - The actual SLURM batch script
- **SETUP_PACE_ICE.md** - Full detailed documentation

## Common Commands

### Check job status
```bash
squeue -u $USER
```

### Cancel a job
```bash
scancel JOB_ID
```

### View live logs
```bash
tail -f /storage/ice-shared/vip-vvk/data/AOT/$USER/evolution_logs/nsganet.*.log
```

### Check available GPUs
```bash
sinfo -p <partition> --Format=partition,available,nodes,gres
```

## Expected Timeline

- **Environment setup**: 5-10 minutes
- **Supernet download**: 2-5 minutes (depending on network)
- **Search (30 iterations)**: 24-48 hours (depends on dataset and resources)

## File Structure After Setup

```
/home/hice1/glu49/nsganetv3/          # Code repository
├── setup_environment.sh               # Setup script
├── quick_start.sh                     # Quick launch script
├── run_nsganetv3_slurm.sh            # SLURM batch script
├── SETUP_PACE_ICE.md                 # Full documentation
└── config/nsganetv3_config.toml      # Configuration file

/storage/ice-shared/vip-vvk/data/     # Data storage
├── supernet/                          # Supernet weights
├── imagenet/                          # Datasets
└── AOT/$USER/                        # Your results
    ├── search-*/                      # Search results
    └── evolution_logs/                # SLURM logs
```

## Need Help?

1. **Read SETUP_PACE_ICE.md** for detailed instructions
2. **Check logs** in `/storage/ice-shared/vip-vvk/data/AOT/$USER/evolution_logs/`
3. **Verify environment**: `conda activate nas && python -c "import torch; print(torch.__version__)"`
