#!/bin/bash

ENV_NAME=nsganetv2-llm

module load anaconda3/2023.03
module load cuda/12.1.1

echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

echo "Activating environment"
source activate $ENV_NAME

echo "Installing PyTorch with CUDA support"
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo "Installing core dependencies"
pip install timm==0.9.16
pip install torchprofile
pip install pymoo==0.6.1.5
pip install numpy pandas pyyaml scipy scikit-learn

echo "Installing OFA (Once-For-All)"
pip install git+https://github.com/mit-han-lab/once-for-all.git

echo "Environment setup complete!"
echo "To activate: conda activate $ENV_NAME"
