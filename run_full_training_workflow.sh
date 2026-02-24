#!/bin/bash
set -e

NON_INTERACTIVE="${NON_INTERACTIVE:-0}"

CORPUS_NAME="${1:-corpus_nsganet_$(date +%Y%m%d_%H%M%S)}"
NUM_SAMPLES="${2:-250}"
EPOCHS="${3:-100}"
BATCH_SIZE="${4:-64}"
TIME_LIMIT="${5:-03:00:00}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2}"
CORPUS_DIR="$OUTPUT_ROOT/$CORPUS_NAME"
WORKFLOW_LOG_DIR="${OUTPUT_ROOT}/.logs"

mkdir -p "$WORKFLOW_LOG_DIR"
WORKFLOW_LOG="$WORKFLOW_LOG_DIR/workflow_$(date +%Y%m%d_%H%M%S).log"

exec 1> >(tee -a "$WORKFLOW_LOG")
exec 2>&1

DATASET_PATH="${DATASET_PATH:-/storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers}"
NUM_CLASSES=102

echo "========================================="
echo "NSGANetV2 Training Workflow"
echo "========================================="
echo "Corpus name:    $CORPUS_NAME"
echo "Architectures:  $NUM_SAMPLES base ($(($NUM_SAMPLES * 2)) total with resolutions)"
echo "Epochs:         $EPOCHS"
echo "Batch size:     $BATCH_SIZE"
echo "Time limit:     $TIME_LIMIT"
echo "Dataset:        $DATASET_PATH"
echo "Output dir:     $CORPUS_DIR"
echo "========================================="
echo ""

if [ ! -d "$DATASET_PATH/train" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo ""
    echo "To fix this:"
    echo "  1. Download the dataset:"
    echo "     python download_oxford_flowers.py --data_dir ~/scratch/datasets/oxford_flowers"
    echo ""
    echo "  2. Set the dataset path and run again:"
    echo "     export DATASET_PATH=~/scratch/datasets/oxford_flowers"
    echo "     ./run_full_training_workflow.sh $CORPUS_NAME $NUM_SAMPLES $EPOCHS $BATCH_SIZE $TIME_LIMIT"
    echo ""
    echo "See SETUP.md for more details."
    exit 1
fi

if [ -d "$CORPUS_DIR" ]; then
    echo "ERROR: Corpus directory '$CORPUS_DIR' already exists"
    echo "Please use a different name or remove the existing directory"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

echo "[1/6] Generating architecture corpus..."
python generate_simple_corpus.py \
    --output_dir "$CORPUS_DIR" \
    --n_samples "$NUM_SAMPLES"

if [ ! -f "$CORPUS_DIR/corpus_metadata.json" ]; then
    echo "ERROR: Corpus generation failed"
    exit 1
fi

TOTAL_ARCHS=$(ls -d $CORPUS_DIR/arch_* 2>/dev/null | wc -l)
echo "✓ Generated $TOTAL_ARCHS architectures"
echo ""

echo "[2/6] Creating SLURM training jobs..."
python create_imagenet_training_jobs.py \
    --corpus_dir "$CORPUS_DIR" \
    --data_path "$DATASET_PATH" \
    --num_classes "$NUM_CLASSES" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --time_limit "$TIME_LIMIT"

if [ ! -f "$CORPUS_DIR/submit_all_jobs.sh" ]; then
    echo "ERROR: Job creation failed"
    exit 1
fi

echo "✓ Created SLURM job scripts"
echo ""

if [ "$NON_INTERACTIVE" = "1" ]; then
    RUN_TEST="y"
else
    read -p "[3/6] Run quick test before submitting all jobs? (y/n, default: y): " RUN_TEST
    RUN_TEST=${RUN_TEST:-y}
fi

if [[ "$RUN_TEST" =~ ^[Yy]$ ]]; then
    echo "Submitting quick test (2 epochs)..."
    TEST_JOB_ID=$(sbatch -o "$CORPUS_DIR/quick_test.log" -e "$CORPUS_DIR/quick_test.err" \
        --export=CORPUS_DIR="$CORPUS_DIR" quick_test.sh | awk '{print $NF}')
    echo "Test job submitted: $TEST_JOB_ID"
    echo "Waiting for test to complete..."
    
    while squeue -j $TEST_JOB_ID 2>/dev/null | grep -q $TEST_JOB_ID; do
        sleep 5
    done
    
    if grep -q "Best metric:" "$CORPUS_DIR/quick_test.err" 2>/dev/null; then
        BEST_ACC=$(grep "Best metric:" "$CORPUS_DIR/quick_test.err" | tail -1 | awk '{print $4}')
        echo "✓ Test completed successfully! Best accuracy: $BEST_ACC%"
    else
        echo "⚠ Test may have failed. Check $CORPUS_DIR/quick_test.err for details."
        if [ "$NON_INTERACTIVE" != "1" ]; then
            read -p "Continue with full training anyway? (y/n): " CONTINUE
            if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
                echo "Aborting workflow"
                exit 1
            fi
        else
            echo "Continuing anyway (non-interactive mode)..."
        fi
    fi
    echo ""
fi

if [ "$NON_INTERACTIVE" = "1" ]; then
    SUBMIT_JOBS="y"
else
    read -p "[4/6] Submit all $TOTAL_ARCHS training jobs to SLURM? (y/n, default: y): " SUBMIT_JOBS
    SUBMIT_JOBS=${SUBMIT_JOBS:-y}
fi

if [[ "$SUBMIT_JOBS" =~ ^[Yy]$ ]]; then
    echo "Submitting all training jobs..."
    cd $(dirname $0)
    bash "$CORPUS_DIR/submit_all_jobs.sh" | tee "$CORPUS_DIR/submission.log"
    
    FIRST_JOB=$(head -2 "$CORPUS_DIR/submission.log" | tail -1 | awk '{print $NF}')
    LAST_JOB=$(tail -2 "$CORPUS_DIR/submission.log" | head -1 | awk '{print $NF}')
    
    echo ""
    echo "✓ Submitted $TOTAL_ARCHS jobs (Job IDs: $FIRST_JOB - $LAST_JOB)"
    echo ""
    
    echo "[5/6] Monitoring training progress..."
    echo "You can monitor progress with:"
    echo "  - Live dashboard: watch -n 30 ./monitor_training.sh $CORPUS_DIR"
    echo "  - Job queue:      squeue -u \$USER"
    echo "  - One-time check: ./monitor_training.sh $CORPUS_DIR"
    echo ""
    
    if [ "$NON_INTERACTIVE" = "1" ]; then
        WAIT_JOBS="n"
    else
        read -p "Wait for all jobs to complete before collecting results? (y/n, default: n): " WAIT_JOBS
        WAIT_JOBS=${WAIT_JOBS:-n}
    fi
    
    if [[ "$WAIT_JOBS" =~ ^[Yy]$ ]]; then
        echo "Waiting for all jobs to complete..."
        echo "This may take several hours. Press Ctrl+C to stop waiting (jobs will continue running)."
        
        while true; do
            RUNNING=$(squeue -u $USER | grep "train_" | wc -l)
            COMPLETED=$(find $CORPUS_DIR -name "status.json" -exec grep -l "success" {} \; 2>/dev/null | wc -l)
            
            echo "[$(date +%H:%M:%S)] Running: $RUNNING | Completed: $COMPLETED / $TOTAL_ARCHS"
            
            if [ $RUNNING -eq 0 ] && [ $COMPLETED -gt 0 ]; then
                echo "All jobs completed!"
                break
            fi
            
            sleep 60
        done
    else
        echo "Skipping wait. You can collect results later when jobs complete."
        echo ""
        echo "To collect results later, run:"
        echo "  python collect_training_results.py --corpus_dir $CORPUS_DIR --output_csv ${CORPUS_NAME}_results.csv"
        exit 0
    fi
else
    echo "Skipping job submission."
    echo ""
    echo "To submit jobs later, run:"
    echo "  bash $CORPUS_DIR/submit_all_jobs.sh"
    exit 0
fi

echo ""
echo "[6/6] Collecting training results..."
OUTPUT_CSV="$CORPUS_DIR/${CORPUS_NAME}_results.csv"

python collect_training_results.py \
    --corpus_dir "$CORPUS_DIR" \
    --output_csv "$OUTPUT_CSV"

if [ -f "$OUTPUT_CSV" ]; then
    NUM_RESULTS=$(tail -n +2 "$OUTPUT_CSV" | wc -l)
    echo "✓ Results collected: $NUM_RESULTS architectures"
    echo "✓ Output saved to: $OUTPUT_CSV"
    echo ""
    
    echo "========================================="
    echo "Training Summary"
    echo "========================================="
    
    SUCCESSFUL=$(grep -c "success" "$OUTPUT_CSV" || echo 0)
    FAILED=$(grep -c "failed" "$OUTPUT_CSV" || echo 0)
    
    echo "Total architectures: $NUM_RESULTS"
    echo "Successful:          $SUCCESSFUL"
    echo "Failed:              $FAILED"
    
    if [ $SUCCESSFUL -gt 0 ]; then
        echo ""
        echo "Accuracy statistics:"
        tail -n +2 "$OUTPUT_CSV" | awk -F',' '{if ($4 != "" && $4 != "None") print $4}' | sort -n | awk '
        BEGIN {min=999; max=0; sum=0; count=0}
        {
            if ($1 < min) min=$1;
            if ($1 > max) max=$1;
            sum+=$1;
            count++;
            arr[count]=$1;
        }
        END {
            if (count > 0) {
                mean=sum/count;
                asort(arr);
                if (count % 2 == 1) median=arr[(count+1)/2];
                else median=(arr[count/2] + arr[count/2+1])/2;
                printf "  Min:    %.2f%%\n", min;
                printf "  Max:    %.2f%%\n", max;
                printf "  Mean:   %.2f%%\n", mean;
                printf "  Median: %.2f%%\n", median;
            }
        }'
    fi
    
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "1. Copy results to NASlib: cp $OUTPUT_CSV ~/NASlib-coder-nas/data/"
    echo "2. Generate LLM embeddings for NSGANetV2 architectures"
    echo "3. Run LLM comparison (CodeLlama vs ModernBERT)"
    echo ""
    echo "Workflow completed successfully!"
else
    echo "ERROR: Result collection failed"
    exit 1
fi
