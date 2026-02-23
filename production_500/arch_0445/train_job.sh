#!/bin/bash
#SBATCH --job-name=train_0445
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --output=production_500/arch_0445/train.log
#SBATCH --error=production_500/arch_0445/train.err

module load cuda/12.1.1

cd $SLURM_SUBMIT_DIR

$HOME/.conda/envs/nsganetv2-llm/bin/python train_imagenet.py /storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers \
    --model nsganetv2 \
    --model-config production_500/arch_0445/config.json \
    --initial-checkpoint checkpoints/ofa_mbv3_d234_e346_k357_w1.0 \
    --num-classes 102 \
    --epochs 100 \
    --batch-size 64 \
    --img-size 224 \
    --lr 0.01 \
    --weight-decay 1e-4 \
    --drop 0.2 \
    --drop-path 0.2 \
    --aa rand-m9-mstd0.5-inc1 \
    --remode pixel \
    --reprob 0.2 \
    --output production_500/arch_0445

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully"
    echo "{\"status\": \"success\", \"arch_id\": 445}" > production_500/arch_0445/status.json
else
    echo "Training failed with exit code $TRAIN_EXIT_CODE"
    echo "{\"status\": \"failed\", \"arch_id\": 445, \"exit_code\": $TRAIN_EXIT_CODE}" > production_500/arch_0445/status.json
fi
