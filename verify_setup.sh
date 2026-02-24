#!/bin/bash

echo "=== NSGANetV2 Training Setup Verification ==="
echo ""

# Check Python environment
echo "1. Checking Python environment..."
PYTHON_PATH="$HOME/.conda/envs/nsganetv2-llm/bin/python"
if [ -f "$PYTHON_PATH" ]; then
    echo "   ✓ Python found: $PYTHON_PATH"
else
    echo "   ✗ Python not found at $PYTHON_PATH"
    exit 1
fi

# Check dependencies
echo ""
echo "2. Checking dependencies..."
$PYTHON_PATH -c "import torch; print(f'   ✓ PyTorch {torch.__version__}')" || exit 1
$PYTHON_PATH -c "import timm; print(f'   ✓ timm {timm.__version__}')" || exit 1
$PYTHON_PATH -c "import pymoo; print(f'   ✓ pymoo {pymoo.__version__}')" || exit 1
$PYTHON_PATH -c "import pandas; print(f'   ✓ pandas {pandas.__version__}')" || exit 1
$PYTHON_PATH -c "import yaml; print('   ✓ pyyaml installed')" || exit 1
$PYTHON_PATH -c "from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3; print('   ✓ OFA installed')" || exit 1

# Check timm version
echo ""
echo "3. Verifying timm version..."
TIMM_VERSION=$($PYTHON_PATH -c "import timm; print(timm.__version__)")
if [ "$TIMM_VERSION" = "0.6.13" ]; then
    echo "   ✓ timm version correct (0.6.13)"
else
    echo "   ✗ timm version incorrect: $TIMM_VERSION (expected 0.6.13)"
    echo "   Run: pip install --force-reinstall timm==0.6.13"
    exit 1
fi

# Check critical imports
echo ""
echo "4. Checking critical imports..."
$PYTHON_PATH -c "from timm.data import ImageDataset; print('   ✓ ImageDataset import')" || exit 1
$PYTHON_PATH -c "from timm.data.mixup import Mixup; print('   ✓ Mixup import')" || exit 1
$PYTHON_PATH -c "from ofa.utils import cross_entropy_loss_with_soft_target; print('   ✓ OFA utils import')" || exit 1

# Check dataset
echo ""
echo "5. Checking Oxford Flowers dataset..."
DATA_DIR="${DATASET_PATH:-/storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers}"
if [ -d "$DATA_DIR/train" ] && [ -d "$DATA_DIR/val" ]; then
    TRAIN_COUNT=$(find $DATA_DIR/train -name "*.jpg" 2>/dev/null | wc -l)
    VAL_COUNT=$(find $DATA_DIR/val -name "*.jpg" 2>/dev/null | wc -l)
    echo "   ✓ Dataset found at: $DATA_DIR"
    echo "     Train: $TRAIN_COUNT images, Val: $VAL_COUNT images"
else
    echo "   ✗ Dataset not found at $DATA_DIR"
    echo ""
    echo "   Option 1: Try accessing shared dataset"
    echo "     ls /storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers/train"
    echo ""
    echo "   Option 2: Download your own copy"
    echo "     python download_oxford_flowers.py --data_dir ~/scratch/datasets/oxford_flowers"
    echo "     export DATASET_PATH=~/scratch/datasets/oxford_flowers"
    echo "     bash verify_setup.sh  # Run again to verify"
    exit 1
fi

# Check OFA checkpoint
echo ""
echo "6. Checking OFA checkpoint..."
CHECKPOINT="checkpoints/ofa_mbv3_d234_e346_k357_w1.0"
if [ -f "$CHECKPOINT" ]; then
    SIZE=$(du -h "$CHECKPOINT" | cut -f1)
    echo "   ✓ Checkpoint found: $SIZE"
else
    echo "   ✗ Checkpoint not found at $CHECKPOINT"
    echo ""
    echo "   Option 1: Copy from shared location"
    echo "     mkdir -p checkpoints"
    echo "     cp /storage/ice-shared/vip-vvk/data/AOT/shared/checkpoints/ofa_mbv3_d234_e346_k357_w1.0 checkpoints/"
    echo ""
    echo "   Option 2: Download (if shared copy not accessible)"
    echo "     cd checkpoints && gdown 1qmq7vWW6QkOPHfqXNnVvYQCFCQ2P6PUP && cd .."
    exit 1
fi

# Check test corpus
echo ""
echo "7. Checking test corpus..."
if [ -d "test_corpus_5" ]; then
    CONFIGS=$(find test_corpus_5 -name "config.json" | wc -l)
    echo "   ✓ Test corpus exists: $CONFIGS architectures"
else
    echo "   ⚠ Test corpus not found (run generate_simple_corpus.py to create)"
fi

# Check test results
echo ""
echo "8. Checking test job results..."
if [ -f "quick_test.err" ]; then
    BEST_METRIC=$(grep "Best metric" quick_test.err 2>/dev/null | tail -1)
    if [ -n "$BEST_METRIC" ]; then
        echo "   ✓ Test job completed: $BEST_METRIC"
    else
        echo "   ⚠ Test job incomplete (run: sbatch quick_test.sh)"
    fi
else
    echo "   ⚠ No test job run yet (run: sbatch quick_test.sh)"
fi

echo ""
echo "=== Verification Complete ==="
echo ""
echo "Ready to proceed with:"
echo "  1. python generate_simple_corpus.py --output_dir oxford_corpus_250 --num_samples 250"
echo "  2. python create_imagenet_training_jobs.py --corpus_dir oxford_corpus_250"
echo "  3. bash oxford_corpus_250/submit_all_jobs.sh"
