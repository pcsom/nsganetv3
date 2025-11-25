#!/bin/bash
# NSGANetV3 Environment Setup Script for PACE ICE
# Usage: bash setup_environment.sh

set -e  # Exit on error

echo "======================================"
echo "NSGANetV3 Environment Setup for PACE ICE"
echo "======================================"
echo ""

# Load anaconda module
echo "Loading anaconda module..."
module load anaconda3/2023.03

# Environment name
ENV_NAME="nas"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Keeping existing environment. Exiting."
        exit 0
    fi
fi

# Create conda environment
echo ""
echo "Creating conda environment: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=3.7 -y

# Activate environment
echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch 1.5.1 with CUDA 10.2..."
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch -y

# Install Cython (optional but recommended for performance)
echo ""
echo "Installing Cython..."
pip install Cython==0.29

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install pymoo==0.4.1
pip install torchprofile==0.0.1
pip install timm==0.1.30
pip install pySOT==0.2.3
pip install pydacefit==1.0.1
pip install toml

# Install OnceForAll
echo ""
echo "Installing OnceForAll..."
pip install ofa==0.0.4

# Install additional common packages
echo ""
echo "Installing additional packages..."
pip install pandas numpy scipy matplotlib seaborn tqdm pyyaml

echo ""
echo "======================================"
echo "Installation Complete!"
echo "======================================"
echo ""
echo "To use the environment:"
echo "  1. Load anaconda: module load anaconda3/2023.03"
echo "  2. Activate env: conda activate ${ENV_NAME}"
echo ""
echo "Next steps:"
echo "  1. Download supernet weights (see SETUP_PACE_ICE.md)"
echo "  2. Prepare your dataset"
echo "  3. Run search with: sbatch run_nsganetv3_slurm.sh [options]"
echo ""
echo "For detailed instructions, see SETUP_PACE_ICE.md"
echo ""
