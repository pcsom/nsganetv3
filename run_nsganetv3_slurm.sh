#!/bin/bash
#SBATCH --job-name=nsganetv3_main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=08:00:00
#SBATCH --output="/storage/ice-shared/vip-vvk/data/AOT/%u/evolution_logs/nsganet.%A.%a.log"
#SBATCH --error="/storage/ice-shared/vip-vvk/data/AOT/%u/evolution_logs/nsganet_error.%A.%a.log"

# Parse command line arguments
save_dir=""
dataset=""
data_path="../data"
supernet_path="./data/ofa_mbv3_d234_e346_k357_w1.0"
sec_obj=""
iterations=""
predictor=""
conda_environment="nsganetv2-llm"
config_file=""
offline_data=""
evaluation_mode=""

while getopts ":s:d:p:n:o:i:r:e:c:f:m:" opt; do
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
    f) offline_data="$OPTARG"
    ;;
    m) evaluation_mode="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        exit 1
    ;;
  esac
done

# Check required arguments
if [[ -z "$save_dir" ]]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch $0 -s <save_dir> [options]"
    echo "Required: -s (save directory)"
    echo "Optional: -d (dataset, default: flowers102), -o (secondary objective, default: flops)"
    echo "          -i (iterations, default: 30), -r (predictor, default: rbf)"
    echo "          -e (conda env, default: nsganetv2-llm), -c (config file), -f (offline csv), -m (virtual)"
    exit 1
fi

module load anaconda3/2023.03
module load cuda/12.1.1

# Create necessary directories
mkdir -p logs
mkdir -p "$save_dir"

# Portable offline CSV auto-detection if caller did not provide -f.
if [[ -z "$offline_data" ]]; then
    candidate_csv="/storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/prod_500/prod_500_results.csv"
    if [[ -f "$candidate_csv" ]]; then
        offline_data="$candidate_csv"
    fi
fi

echo "Starting NSGANetV3 with SLURM backend"
echo "Configuration:"
echo "  Save directory: $save_dir"
echo "  Dataset: ${dataset:-<from config>}"
echo "  Data path: $data_path"
echo "  Supernet path: $supernet_path"
echo "  Secondary objective: ${sec_obj:-<from config>}"
echo "  Iterations: ${iterations:-<from config>}"
echo "  Predictor: ${predictor:-<from config>}"
echo "  Evaluation mode: ${evaluation_mode:-<from config>}"
echo "  Offline CSV: ${offline_data:-<from config>}"
echo "  Conda environment: $conda_environment"

# Run NSGANetV3 with SLURM backend
config_arg=""
if [[ -n "$config_file" ]]; then
    config_arg="--config $config_file"
fi

offline_arg=""
if [[ -n "$offline_data" ]]; then
    offline_arg="--offline_data $offline_data"
fi

dataset_arg=""
if [[ -n "$dataset" ]]; then
    dataset_arg="--dataset $dataset"
fi

sec_obj_arg=""
if [[ -n "$sec_obj" ]]; then
    sec_obj_arg="--sec_obj $sec_obj"
fi

iterations_arg=""
if [[ -n "$iterations" ]]; then
    iterations_arg="--iterations $iterations"
fi

predictor_arg=""
if [[ -n "$predictor" ]]; then
    predictor_arg="--predictor $predictor"
fi

mode_arg=""
if [[ -n "$evaluation_mode" ]]; then
    mode_arg="--evaluation_mode $evaluation_mode"
fi

conda run -n ${conda_environment} --no-capture-output python -u msunas_slurm.py \
    --save "$save_dir" \
    --data "$data_path" \
    --supernet_path "$supernet_path" \
    $dataset_arg \
    $sec_obj_arg \
    $iterations_arg \
    $predictor_arg \
    $mode_arg \
    $offline_arg \
    $config_arg

echo "NSGANetV3 search completed"