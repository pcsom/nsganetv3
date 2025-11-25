#!/bin/bash
# Quick Start Script for NSGANetV3 on PACE ICE
# This script helps you quickly launch a search job

echo "======================================"
echo "NSGANetV3 Quick Start"
echo "======================================"
echo ""

# Default values
SAVE_DIR="/storage/ice-shared/vip-vvk/data/AOT/$USER/search-test"
DATA_PATH=""
SUPERNET_PATH=""
DATASET="imagenet"
SEC_OBJ="flops"
ITERATIONS=30
PREDICTOR="rbf"
ENV_NAME="nas"

# Interactive mode
echo "This script will help you submit a NSGANetV3 search job to SLURM."
echo ""

# Check if conda env exists
module load anaconda3/2023.03 2>/dev/null
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "ERROR: Conda environment '${ENV_NAME}' not found!"
    echo "Please run: bash setup_environment.sh"
    exit 1
fi

# Get data path
read -p "Enter path to dataset (e.g., /storage/.../imagenet): " DATA_PATH
if [[ ! -d "$DATA_PATH" ]]; then
    echo "WARNING: Directory $DATA_PATH does not exist or is not accessible."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get supernet path
read -p "Enter path to supernet weights: " SUPERNET_PATH
if [[ ! -f "$SUPERNET_PATH" ]]; then
    echo "ERROR: Supernet file not found: $SUPERNET_PATH"
    echo "Please download it first. See SETUP_PACE_ICE.md for instructions."
    exit 1
fi

# Get save directory
read -p "Enter save directory [$SAVE_DIR]: " input
SAVE_DIR=${input:-$SAVE_DIR}

# Get dataset
echo ""
echo "Available datasets: imagenet, cifar10, cifar100, cinic10, aircraft, flowers102, pets, dtd, stl10"
read -p "Enter dataset name [$DATASET]: " input
DATASET=${input:-$DATASET}

# Get secondary objective
echo ""
echo "Available objectives: flops, params, cpu"
read -p "Enter secondary objective [$SEC_OBJ]: " input
SEC_OBJ=${input:-$SEC_OBJ}

# Get iterations
read -p "Enter number of iterations [$ITERATIONS]: " input
ITERATIONS=${input:-$ITERATIONS}

# Get predictor
echo ""
echo "Available predictors: rbf, gp, cart, mlp, as"
read -p "Enter predictor type [$PREDICTOR]: " input
PREDICTOR=${input:-$PREDICTOR}

# Summary
echo ""
echo "======================================"
echo "Job Configuration:"
echo "======================================"
echo "Save directory:    $SAVE_DIR"
echo "Dataset:           $DATASET"
echo "Data path:         $DATA_PATH"
echo "Supernet path:     $SUPERNET_PATH"
echo "Secondary obj:     $SEC_OBJ"
echo "Iterations:        $ITERATIONS"
echo "Predictor:         $PREDICTOR"
echo "Conda env:         $ENV_NAME"
echo "======================================"
echo ""

read -p "Submit this job? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Job submission cancelled."
    exit 0
fi

# Create log directory
mkdir -p /storage/ice-shared/vip-vvk/data/AOT/$USER/evolution_logs

# Submit job
echo ""
echo "Submitting job to SLURM..."
JOB_ID=$(sbatch run_nsganetv3_slurm.sh \
    -s "$SAVE_DIR" \
    -p "$DATA_PATH" \
    -n "$SUPERNET_PATH" \
    -d "$DATASET" \
    -o "$SEC_OBJ" \
    -i "$ITERATIONS" \
    -r "$PREDICTOR" \
    -e "$ENV_NAME" | awk '{print $4}')

echo ""
echo "Job submitted successfully!"
echo "Job ID: $JOB_ID"
echo ""
echo "Monitor your job with:"
echo "  squeue -u $USER"
echo ""
echo "View logs with:"
echo "  tail -f /storage/ice-shared/vip-vvk/data/AOT/$USER/evolution_logs/nsganet.$JOB_ID.0.log"
echo ""
