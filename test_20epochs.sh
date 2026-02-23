#!/bin/bash
#SBATCH --job-name=test_20ep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=test_20epochs.log
#SBATCH --error=test_20epochs.err

module load cuda/12.1.1

$HOME/.conda/envs/nsganetv2-llm/bin/python train_imagenet.py \
  /storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers \
  --model nsganetv2 \
  --model-config fresh_test/arch_0002/config.json \
  --initial-checkpoint checkpoints/ofa_mbv3_d234_e346_k357_w1.0 \
  --num-classes 102 \
  --epochs 20 \
  --batch-size 64 \
  --output test_20epochs_output
