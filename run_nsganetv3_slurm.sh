#!/bin/bash
#SBATCH --job-name=nsganetv3_main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=48:00:00
#SBATCH --output="/storage/ice-shared/vip-vvk/data/AOT/%u/evolution_logs/nsganet.%A.%a.log"
#SBATCH --error="/storage/ice-shared/vip-vvk/data/AOT/%u/evolution_logs/nsganet_error.%A.%a.log"

# Parse command line arguments
save_dir=""
dataset="imagenet"
data_path=""
supernet_path=""
sec_obj="flops"
iterations=30
predictor="rbf"
conda_environment="nas"
config_file=""

while getopts ":s:d:p:n:o:i:r:e:c:" opt; do
  case $opt in
    s) save_dir="$OPTARG"
    ;;
    d) dataset="$OPTARG"
    ;;
    p) data_path="$OPTARG"
    ;;
    n) supernet_path="$OPTARG"
    ;;
    o) sec_obj="$OPTARG"
    ;;
    i) iterations="$OPTARG"
    ;;
    r) predictor="$OPTARG"
    ;;
    e) conda_environment="$OPTARG"
    ;;
    c) config_file="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        exit 1
    ;;
  esac
done

# Check required arguments
if [[ -z "$save_dir" || -z "$data_path" || -z "$supernet_path" ]]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch $0 -s <save_dir> -p <data_path> -n <supernet_path> [options]"
    echo "Required: -s (save directory), -p (data path), -n (supernet path)"
    echo "Optional: -d (dataset, default: imagenet), -o (secondary objective, default: flops)"
    echo "          -i (iterations, default: 30), -r (predictor, default: rbf)"
    echo "          -e (conda env, default: nas), -c (config file)"
    exit 1
fi

module load anaconda3/2023.03
module load cuda/12.1.1

# Create necessary directories
mkdir -p logs
mkdir -p "$save_dir"

echo "Starting NSGANetV3 with SLURM backend"
echo "Configuration:"
echo "  Save directory: $save_dir"
echo "  Dataset: $dataset"
echo "  Data path: $data_path"
echo "  Supernet path: $supernet_path"
echo "  Secondary objective: $sec_obj"
echo "  Iterations: $iterations"
echo "  Predictor: $predictor"
echo "  Conda environment: $conda_environment"

# Run NSGANetV3 with SLURM backend
config_arg=""
if [[ -n "$config_file" ]]; then
    config_arg="--config $config_file"
fi

conda run -n ${conda_environment} --no-capture-output python -u msunas_slurm.py \
    --save "$save_dir" \
    --data "$data_path" \
    --supernet_path "$supernet_path" \
    --dataset "$dataset" \
    --sec_obj "$sec_obj" \
    --iterations "$iterations" \
    --predictor "$predictor" \
    $config_arg

echo "NSGANetV3 search completed"