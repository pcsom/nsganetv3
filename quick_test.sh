#!/bin/bash
#SBATCH --job-name=nsga_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=quick_test.log
#SBATCH --error=quick_test.err

module load cuda/12.1.1

cd $SLURM_SUBMIT_DIR

$HOME/.conda/envs/nsganetv2-llm/bin/python train_imagenet.py /storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers \
    --model nsganetv2 \
    --model-config test_corpus_5/arch_0000/config.json \
    --initial-checkpoint checkpoints/ofa_mbv3_d234_e346_k357_w1.0 \
    --num-classes 102 \
    --epochs 2 \
    --batch-size 64 \
    --img-size 224 \
    --lr 0.01 \
    --weight-decay 1e-4 \
    --drop 0.2 \
    --drop-path 0.2 \
    --aa rand-m9-mstd0.5-inc1 \
    --remode pixel \
    --reprob 0.2 \
    --output quick_test_output

echo "Quick test completed"
