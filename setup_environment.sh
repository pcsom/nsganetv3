#!/bin/bash

ENV_NAME=nsganetv2-llm

module load anaconda3/2023.03
module load cuda/12.1.1

echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

echo "Activating environment"
source activate $ENV_NAME

echo "Installing PyTorch with CUDA support"
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Installing core dependencies"
pip install timm==0.6.13
pip install torchprofile gdown scipy
pip install pymoo==0.6.1.5

echo "Installing OFA (Once-For-All)"
pip install git+https://github.com/mit-han-lab/once-for-all.git

echo "Environment setup complete!"
echo "To activate: conda activate $ENV_NAME"
