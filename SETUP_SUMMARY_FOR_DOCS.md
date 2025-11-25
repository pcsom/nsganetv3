# NSGANetV3 Setup on PACE ICE SLURM - Complete Guide

## Overview
This guide documents the complete setup process for running NSGANetV3 (Neural Architecture Search) on Georgia Tech's PACE ICE cluster using SLURM job scheduling.

---

## Prerequisites
- Access to PACE ICE cluster (login via SSH)
- Basic familiarity with Linux command line
- Understanding of SLURM job scheduling (optional but helpful)

---

## Part 1: Environment Setup

### Step 1: Access PACE ICE Cluster
```bash
ssh username@login-ice.pace.gatech.edu
```

### Step 2: Navigate to Project Directory
```bash
cd /home/hice1/glu49/nsganetv3
```

### Step 3: Verify Project Files
The codebase should contain these key files:
- `msunas_slurm.py` - Main search script (SLURM-adapted)
- `evaluator_slurm.py` - Architecture evaluator for SLURM
- `run_nsganetv3_slurm.sh` - SLURM batch submission script
- `config/nsganetv3_config.toml` - Configuration file
- Various supporting Python modules

### Step 4: Load Required Modules
```bash
module load anaconda3/2023.03
module load cuda/12.1.1
```

### Step 5: Check if Conda Environment Exists
```bash
conda env list | grep "^nas "
```

If the environment exists, activate it:
```bash
conda activate nas
```

If it doesn't exist, create it (see Step 6).

### Step 6: Create Conda Environment (if needed)
```bash
conda create -n nas python=3.7 -y
conda activate nas
```

---

## Part 2: Install Dependencies

### Core PyTorch Installation
```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision cudatoolkit -c pytorch -y
```

**Note:** The system may already have PyTorch 2.3.1 with CUDA 12.1, which is compatible.

### Install Required Python Packages
```bash
# Performance optimization
pip install Cython==0.29

# Core NAS dependencies
pip install pymoo==0.4.1
pip install torchprofile==0.0.1
pip install "timm>=0.9.0"  # Use newer version for PyTorch 2.x compatibility
pip install "ofa>=0.0.4"   # Once-for-All supernet
pip install pySOT==0.2.3   # Surrogate optimization
pip install pydacefit==1.0.1  # Gaussian process surrogate
pip install toml

# Additional utilities
pip install pandas numpy scipy matplotlib seaborn tqdm pyyaml
```

### Version Compatibility Notes
- Original README specifies `timm==0.1.30`, but this is incompatible with PyTorch 2.x
- Use `timm>=0.9.0` instead for compatibility
- Original `ofa==0.0.4` exact version not available, use `ofa>=0.0.4` (installs 0.1.0.post202307202001)
- **CRITICAL**: pymoo 0.6.x changed module structure from `pymoo.model` to `pymoo.core` (code fixed)
- Install `gdown` package to enable OFA's built-in supernet downloader

---

## Part 3: Verify Installation

### Run Verification Test
A test script was created to verify the setup:

```bash
bash test_setup.sh
```

Expected output: All tests should pass (13/13)

### Manual Verification
```bash
# Activate environment
module load anaconda3/2023.03
conda activate nas

# Test imports
python -c "
import torch
import pymoo
import torchprofile
import timm
import ofa
import pySOT
import pydacefit
import toml
print('All dependencies imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

---

## Part 4: Download Supernet Weights

### Install gdown for OFA Downloads
```bash
module load anaconda3/2023.03
conda activate nas
pip install gdown
```

### Download Using OFA's Built-in Downloader
The official MIT Han Lab download links are broken (404 errors as of Nov 2025). Use OFA's built-in model zoo:

```bash
module load anaconda3/2023.03
conda activate nas

python << 'EOF'
from ofa.model_zoo import ofa_net
import os

# This will download to ~/.torch/ofa_nets/
print("Downloading OFA MobileNetV3 supernet...")
net = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)
print("Successfully downloaded!")
EOF
```

### Copy to Standard Location
```bash
mkdir -p ~/supernet
cp ~/.torch/ofa_nets/ofa_mbv3_d234_e346_k357_w1.0 ~/supernet/
ls -lh ~/supernet/
```

You should see a ~30MB file: `ofa_mbv3_d234_e346_k357_w1.0`

### Set Path for Easy Reference
```bash
export SUPERNET_PATH=$HOME/supernet/ofa_mbv3_d234_e346_k357_w1.0
```

---

## Part 5: Prepare Dataset

### ImageNet
Download ImageNet from official source and organize as:
```
/storage/ice-shared/vip-vvk/data/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

### CIFAR-10/100
These datasets auto-download on first run. No manual setup required.

### Other Datasets
- CINIC-10, Aircraft, Flowers102, Oxford Pets, DTD, STL-10
- See original README.md for download links

---

## Part 6: Understanding the SLURM Script

### SLURM Script Configuration
The `run_nsganetv3_slurm.sh` script is pre-configured for PACE ICE:

**Default SLURM Settings:**
- Job name: `nsganetv3_main`
- Nodes: 1
- CPUs per task: 4
- Memory: 8GB
- Time limit: 48 hours
- Modules: anaconda3/2023.03, cuda/12.1.1
- GPU types: V100-16GB, V100-32GB, L40S, A100-40GB, H100, A40, H200

**Command Line Arguments:**
- `-s <save_dir>` - Directory to save results (REQUIRED)
- `-p <data_path>` - Path to dataset (REQUIRED)
- `-n <supernet_path>` - Path to supernet weights (REQUIRED)
- `-d <dataset>` - Dataset name (default: imagenet)
- `-o <sec_obj>` - Secondary objective: flops/params/cpu (default: flops)
- `-i <iterations>` - Number of iterations (default: 30)
- `-r <predictor>` - Surrogate model: rbf/gp/cart/mlp/as (default: rbf)
- `-e <conda_env>` - Conda environment name (default: nas)
- `-c <config_file>` - TOML config file (optional)

### Log Locations
- Output logs: `/storage/ice-shared/vip-vvk/data/AOT/$USER/evolution_logs/nsganet.{JOB_ID}.{ARRAY_ID}.log`
- Error logs: `/storage/ice-shared/vip-vvk/data/AOT/$USER/evolution_logs/nsganet_error.{JOB_ID}.{ARRAY_ID}.log`

---

## Part 7: Running Architecture Search

### Create Result Directories
```bash
mkdir -p ~/evolution_logs ~/search-test
```

### Example 1: Basic ImageNet Search (FLOPs Optimization)
```bash
cd /home/hice1/glu49/nsganetv3

sbatch run_nsganetv3_slurm.sh \
    -s "$HOME/search-imagenet-flops" \
    -p "/storage/ice-shared/vip-vvk/data/imagenet" \
    -n "$HOME/supernet/ofa_mbv3_d234_e346_k357_w1.0" \
    -d imagenet \
    -o flops \
    -i 30 \
    -r rbf
```

**NOTE**: Logs now save to current directory as `nsganetv3_main_JOBID.out` and `.err`

### Example 2: CIFAR-10 Search (Parameters Optimization)
```bash
sbatch run_nsganetv3_slurm.sh \
    -s "$HOME/search-cifar10-params" \
    -p "/path/to/cifar10" \
    -n "$HOME/supernet/ofa_mbv3_d234_e346_k357_w1.0" \
    -d cifar10 \
    -o params \
    -i 30 \
    -r gp
```

### Example 3: Quick Test Run (5 iterations) - RECOMMENDED FIRST
```bash
sbatch run_nsganetv3_slurm.sh \
    -s "$HOME/search-test" \
    -p "/storage/ice-shared/vip-vvk/data/imagenet" \
    -n "$HOME/supernet/ofa_mbv3_d234_e346_k357_w1.0" \
    -d imagenet \
    -o flops \
    -i 5 \
    -r rbf \
    -e nas
```

### Example 4: Using Configuration File
```bash
sbatch run_nsganetv3_slurm.sh \
    -s "$HOME/search-custom" \
    -p "/storage/ice-shared/vip-vvk/data/imagenet" \
    -n "$HOME/supernet/ofa_mbv3_d234_e346_k357_w1.0" \
    -c config/nsganetv3_config.toml \
    -e nas
```

---

## Part 8: Monitoring Jobs

### Check Job Status
```bash
# View your jobs in queue
squeue -u $USER

# View detailed job info
scontrol show job JOB_ID

# View job history
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed
```

### Monitor Logs in Real-Time
```bash
# Find your latest log files
ls -lht /home/hice1/glu49/nsganetv3/*.out | head -3
ls -lht /home/hice1/glu49/nsganetv3/*.err | head -3

# Output log (replace JOB_ID with actual number)
tail -f /home/hice1/glu49/nsganetv3/nsganetv3_main_JOB_ID.out

# Error log
tail -f /home/hice1/glu49/nsganetv3/nsganetv3_main_JOB_ID.err

# Or use wildcard for latest
tail -f /home/hice1/glu49/nsganetv3/nsganetv3_main_*.out
```

### Cancel Jobs
```bash
# Cancel specific job
scancel JOB_ID

# Cancel all your jobs
scancel -u $USER
```

---

## Part 9: Understanding Search Output

### Output Directory Structure
```
$HOME/search-experiment/
├── iter_0/
│   ├── net_0.subnet      # Subnet configuration
│   ├── net_0.stats       # Performance statistics
│   ├── net_1.subnet
│   ├── net_1.stats
│   └── ...
├── iter_1/
├── iter_2/
├── ...
├── iter_30/
├── iter_0.stats          # Iteration 0 summary
├── iter_1.stats          # Iteration 1 summary
├── ...
├── iter_30.stats         # Final iteration summary
└── failed/               # Failed evaluations (if any)
```

### Key Files
- **`net_X.subnet`**: Architecture configuration (can be loaded back)
- **`net_X.stats`**: Performance metrics (accuracy, FLOPs, params, latency)
- **`iter_X.stats`**: Contains all evaluated architectures in `["archive"]` and hypervolume in `["hv"]`

---

## Part 10: Selecting Best Architectures

### After Search Completes
Use the `post_search.py` script to select architectures from the Pareto front:

### Option 1: Select with Preferences
```bash
conda activate nas

python post_search.py \
    -n 3 \
    --save /path/to/search/final \
    --expr /path/to/search/iter_30.stats \
    --prefer top1#80+flops#150 \
    --supernet_path /path/to/supernet
```

This selects 3 architectures close to 80% accuracy and 150M FLOPs.

### Option 2: Select Based on Trade-offs (No Preferences)
```bash
python post_search.py \
    -n 5 \
    --save /path/to/search/final \
    --expr /path/to/search/iter_30.stats \
    --prefer None \
    --supernet_path /path/to/supernet
```

This selects 5 well-distributed architectures from the Pareto front.

### Output Files
Selected architectures will have:
- `net.subnet` - Subnet configuration
- `net.config` - Full architecture definition
- `net.inherited` - Inherited weights from supernet

---

## Part 11: Fine-tuning Selected Architectures

### ImageNet Fine-tuning (Distributed Training)
```bash
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
    --teacher /path/to/supernet
```

**Note:** Adjust learning rate as `(batch_size_per_gpu * #GPUs / 256) * 0.006`

### CIFAR-10 Fine-tuning
```bash
python train_cifar.py \
    --data /path/to/cifar10/ \
    --model nsganetv2_cifar \
    --model-config /path/to/final/net.config \
    --img-size 224 \
    --drop 0.2 --drop-path 0.2 \
    --cutout --autoaugment --save
```

---

## Part 12: Configuration File Reference

### TOML Configuration (`config/nsganetv3_config.toml`)

**Search Parameters:**
```toml
[search]
iterations = 30           # Number of search iterations
n_doe = 100              # Initial design of experiments sample size
n_iter = 8               # Architectures evaluated per iteration
sec_obj = "flops"        # Secondary objective: flops/params/cpu/gpu
predictor = "as"         # Surrogate model: rbf/gp/cart/mlp/as
```

**Dataset Configuration:**
```toml
[dataset]
dataset = "imagenet"     # Dataset name
n_classes = 1000         # Number of classes
n_epochs = 5             # Training epochs per architecture
vld_size = 10000         # Validation set size
test = false             # Evaluate on test set
```

**Training Settings:**
```toml
[training]
trn_batch_size = 128     # Training batch size
vld_batch_size = 200     # Validation batch size
n_workers = 4            # Data loader workers
```

**Evolutionary Algorithm:**
```toml
[evolutionary]
pop_size = 40            # Population size
n_gens = 20              # Number of generations
crossover_prob = 0.9     # Crossover probability
mutation_eta = 1.0       # Mutation eta parameter
```

**SLURM Configuration:**
```toml
[slurm]
job_name = "nsganetv3"
nodes = 1
cores = 8
memory = "24GB"
job_time = "08:00:00"
env_name = "nas"
gpu_types = ["V100-16GB", "V100-32GB", "L40S", "A100-40GB", "H100", "A40", "H200"]
```

---

## Part 13: Troubleshooting

### Issue 1: CUDA Out of Memory
**Symptoms:** Job crashes with CUDA OOM error

**Solutions:**
1. Reduce batch size in config:
   ```toml
   [training]
   trn_batch_size = 64  # Reduce from 128
   vld_batch_size = 100 # Reduce from 200
   ```

2. Reduce validation set size:
   ```toml
   [dataset]
   vld_size = 5000  # Reduce from 10000
   ```

### Issue 2: Module Not Found Errors
**Symptoms:** `ImportError: No module named 'xxx'`

**Solution:**
```bash
# Activate environment
module load anaconda3/2023.03
conda activate nas

# Reinstall missing package
pip install package-name

# Verify
python -c "import package_name"
```

### Issue 3: Job Pending for Long Time
**Symptoms:** `squeue` shows job in PD (pending) state

**Solutions:**
1. Check partition availability:
   ```bash
   sinfo -p <partition>
   ```

2. Request different GPU type in config

3. Reduce resource requirements (memory, time, cores)

### Issue 4: Supernet Weights Not Found
**Symptoms:** Error loading supernet file

**Solution:**
```bash
# Verify path
ls -lh ~/supernet/

# Re-download if needed using OFA model zoo
module load anaconda3/2023.03
conda activate nas
python -c "from ofa.model_zoo import ofa_net; net = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)"
cp ~/.torch/ofa_nets/ofa_mbv3_d234_e346_k357_w1.0 ~/supernet/
```

### Issue 6: pymoo Import Errors
**Symptoms:** `ModuleNotFoundError: No module named 'pymoo.model'`

**Solution:**
The newer pymoo (0.6.x) changed from `pymoo.model` to `pymoo.core`. Files already fixed in this setup:
- `utils.py`
- `msunas.py`
- `msunas_slurm.py`

If you see this error, the imports weren't updated. Apply the fixes from Part 16.

### Issue 5: Permission Denied
**Symptoms:** Cannot write to directories

**Solution:**
```bash
# Create directories with proper permissions
mkdir -p /storage/ice-shared/vip-vvk/data/AOT/$USER/evolution_logs
chmod 755 /storage/ice-shared/vip-vvk/data/AOT/$USER
```

---

## Part 14: Expected Timeline

### Setup Phase
- Environment creation: 5-10 minutes
- Dependency installation: 5-10 minutes
- Supernet download: 2-5 minutes
- Dataset preparation: Varies (ImageNet: hours, CIFAR: automatic)

### Search Phase
- Test run (5 iterations): 2-4 hours
- Medium run (15 iterations): 12-24 hours
- Full run (30 iterations): 24-48 hours
- Timeline depends on: dataset size, GPU availability, network speed

### Post-Search Phase
- Architecture selection: < 5 minutes
- Fine-tuning: Hours to days depending on dataset and epochs

---

## Part 15: Best Practices

### Resource Management
1. Start with small test runs (5 iterations) to verify setup
2. Use appropriate validation set sizes (10k for ImageNet, 5k for CIFAR)
3. Monitor first few iterations to catch errors early
4. Clean up old experiment directories regularly

### Experiment Organization
```
$HOME/
├── supernet/
│   └── ofa_mbv3_d234_e346_k357_w1.0
├── nsganetv3/
│   ├── *.out  (SLURM output logs)
│   ├── *.err  (SLURM error logs)
│   └── ... (code files)
├── search-imagenet-flops-run1/
├── search-imagenet-flops-run2/
├── search-imagenet-params-run1/
└── search-cifar10-flops-run1/
```

### Job Naming Convention
Use descriptive names that include:
- Dataset
- Secondary objective
- Surrogate model
- Run number

Example: `search-imagenet-flops-rbf-run1`

### Logging Strategy
1. Always check logs after first iteration
2. Monitor hypervolume convergence
3. Save intermediate results
4. Document parameter changes

---

## Part 16: Key Differences from Original NSGANetV2

### SLURM Integration
- Original: Direct GPU execution
- SLURM version: Job submission with queuing
- Benefits: Better resource sharing, fault tolerance, scalability

### File Structure Changes
- Added: `msunas_slurm.py`, `evaluator_slurm.py`, `run_nsganetv3_slurm.sh`
- Modified: Configuration now supports SLURM-specific settings
- New: TOML-based configuration file

### Dependency Updates
- PyTorch: 1.5.1 → 2.3.1 (for CUDA 12.x support)
- timm: 0.1.30 → 1.0.22 (compatibility fix)
- ofa: 0.0.4 → 0.1.0 (latest available)
- pymoo: 0.4.1 → 0.6.1.5 (requires import path changes)
- Added: gdown (for OFA model downloads)

### Code Changes Required
**File: `utils.py`**
```python
# OLD (broken):
from pymoo.model.mutation import Mutation
from pymoo.model.sampling import Sampling
from pymoo.model.crossover import Crossover

# NEW (fixed):
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
```

**File: `msunas.py` and `msunas_slurm.py`**
```python
# OLD (broken):
from pymoo.model.problem import Problem

# NEW (fixed):
from pymoo.core.problem import Problem
```

**File: `run_nsganetv3_slurm.sh`**
```bash
# OLD (broken - exceeds PACE ICE limits):
#SBATCH --time=48:00:00
#SBATCH --output="/storage/ice-shared/vip-vvk/data/AOT/%u/evolution_logs/nsganet.%A.%a.log"
#SBATCH --error="/storage/ice-shared/vip-vvk/data/AOT/%u/evolution_logs/nsganet_error.%A.%a.log"

# NEW (fixed - complies with 18hr CPU limit):
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
```

---

## Part 17: Quick Reference Commands

### Setup Commands
```bash
# Load modules
module load anaconda3/2023.03 cuda/12.1.1

# Activate environment
conda activate nas

# Verify setup
bash test_setup.sh
```

### Job Management
```bash
# Submit job
sbatch run_nsganetv3_slurm.sh [options]

# Check status
squeue -u $USER

# View details
scontrol show job JOB_ID

# Cancel job
scancel JOB_ID

# Job history
sacct -u $USER
```

### Monitoring
```bash
# Live output log
tail -f ~/nsganetv3/nsganetv3_main_*.out

# Live error log
tail -f ~/nsganetv3/nsganetv3_main_*.err

# Check last 50 lines
tail -n 50 ~/nsganetv3/nsganetv3_main_JOBID.out
```

### Python Environment
```bash
# List packages
conda list

# Check specific package
conda list | grep package-name

# Install package
pip install package-name

# Verify imports
python -c "import package_name; print(package_name.__version__)"
```

---

## Part 18: Additional Resources

### Documentation Files Created
- `SETUP_PACE_ICE.md` - Comprehensive setup guide
- `QUICKSTART.md` - Quick start reference
- `SETUP_COMPLETE.md` - Post-setup next steps
- `test_setup.sh` - Automated verification script
- `setup_environment.sh` - Automated environment setup
- `quick_start.sh` - Interactive job submission

### Original NSGANetV2 Resources
- Paper: [NSGANetV2 (ECCV 2020)](https://arxiv.org/abs/2007.10396)
- GitHub: [NSGANetV2 Repository](https://github.com/mikelzc1990/nsganetv2)
- Once-for-All: [OFA GitHub](https://github.com/mit-han-lab/once-for-all)

### PACE Documentation
- [PACE User Guide](https://docs.pace.gatech.edu/)
- [PACE ICE Cluster Info](https://docs.pace.gatech.edu/ice_cluster/)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)

---

## Summary

This setup process configures NSGANetV3 to run on PACE ICE SLURM with:
- ✓ Proper conda environment with all dependencies
- ✓ SLURM job submission scripts
- ✓ Configuration management via TOML
- ✓ Automated testing and verification
- ✓ Comprehensive documentation
- ✓ Helper scripts for common tasks

The system is ready for multi-objective neural architecture search optimizing accuracy against FLOPs, parameters, or latency across multiple datasets.

---

**Document Version:** 1.0  
**Last Updated:** November 23, 2025  
**System:** PACE ICE SLURM Cluster  
**Codebase:** NSGANetV3 (SLURM-adapted NSGANetV2)
