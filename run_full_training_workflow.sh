#!/bin/bash
set -e

CORPUS_NAME="${1:-corpus_nsganet_$(date +%Y%m%d_%H%M%S)}"
NUM_SAMPLES="${2:-250}"
EPOCHS="${3:-100}"
BATCH_SIZE="${4:-64}"
TIME_LIMIT="${5:-03:00:00}"

DATASET_PATH="/storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers"
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
echo "========================================="
echo ""

if [ ! -d "$DATASET_PATH/train" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Please run: python download_oxford_flowers.py"
    exit 1
fi

if [ -d "$CORPUS_NAME" ]; then
    echo "ERROR: Corpus directory '$CORPUS_NAME' already exists"
    echo "Please use a different name or remove the existing directory"
    exit 1
fi

echo "[1/6] Generating architecture corpus..."
python generate_simple_corpus.py \
    --output_dir "$CORPUS_NAME" \
    --n_samples "$NUM_SAMPLES"

if [ ! -f "$CORPUS_NAME/corpus_metadata.json" ]; then
    echo "ERROR: Corpus generation failed"
    exit 1
fi

TOTAL_ARCHS=$(ls -d $CORPUS_NAME/arch_* 2>/dev/null | wc -l)
echo "✓ Generated $TOTAL_ARCHS architectures"
echo ""

echo "[2/6] Creating SLURM training jobs..."
python create_imagenet_training_jobs.py \
    --corpus_dir "$CORPUS_NAME" \
    --data_path "$DATASET_PATH" \
    --num_classes "$NUM_CLASSES" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --time_limit "$TIME_LIMIT"

if [ ! -f "$CORPUS_NAME/submit_all_jobs.sh" ]; then
    echo "ERROR: Job creation failed"
    exit 1
fi

echo "✓ Created SLURM job scripts"
echo ""

read -p "[3/6] Run quick test before submitting all jobs? (y/n, default: y): " RUN_TEST
RUN_TEST=${RUN_TEST:-y}

if [[ "$RUN_TEST" =~ ^[Yy]$ ]]; then
    echo "Submitting quick test (2 epochs)..."
    TEST_JOB_ID=$(sbatch quick_test.sh | awk '{print $NF}')
    echo "Test job submitted: $TEST_JOB_ID"
    echo "Waiting for test to complete..."
    
    while squeue -j $TEST_JOB_ID 2>/dev/null | grep -q $TEST_JOB_ID; do
        sleep 5
    done
    
    if grep -q "Best metric:" quick_test.err 2>/dev/null; then
        BEST_ACC=$(grep "Best metric:" quick_test.err | tail -1 | awk '{print $4}')
        echo "✓ Test completed successfully! Best accuracy: $BEST_ACC%"
    else
        echo "⚠ Test may have failed. Check quick_test.err for details."
        read -p "Continue with full training anyway? (y/n): " CONTINUE
        if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
            echo "Aborting workflow"
            exit 1
        fi
    fi
    echo ""
fi

read -p "[4/6] Submit all $TOTAL_ARCHS training jobs to SLURM? (y/n, default: y): " SUBMIT_JOBS
SUBMIT_JOBS=${SUBMIT_JOBS:-y}

if [[ "$SUBMIT_JOBS" =~ ^[Yy]$ ]]; then
    echo "Submitting all training jobs..."
    cd $(dirname $0)
    bash "$CORPUS_NAME/submit_all_jobs.sh" | tee "$CORPUS_NAME/submission.log"
    
    FIRST_JOB=$(head -2 "$CORPUS_NAME/submission.log" | tail -1 | awk '{print $NF}')
    LAST_JOB=$(tail -2 "$CORPUS_NAME/submission.log" | head -1 | awk '{print $NF}')
    
    echo ""
    echo "✓ Submitted $TOTAL_ARCHS jobs (Job IDs: $FIRST_JOB - $LAST_JOB)"
    echo ""
    
    echo "[5/6] Monitoring training progress..."
    echo "You can monitor progress with:"
    echo "  - Job queue:     squeue -u \$USER"
    echo "  - Running count: squeue -u \$USER | grep 'RUNNING' | wc -l"
    echo "  - Completed:     find $CORPUS_NAME -name 'status.json' -exec grep -l 'success' {} \\; | wc -l"
    echo "  - Watch live:    watch -n 60 'find $CORPUS_NAME -name status.json -exec grep -l success {} \\; | wc -l'"
    echo ""
    
    read -p "Wait for all jobs to complete before collecting results? (y/n, default: n): " WAIT_JOBS
    WAIT_JOBS=${WAIT_JOBS:-n}
    
    if [[ "$WAIT_JOBS" =~ ^[Yy]$ ]]; then
        echo "Waiting for all jobs to complete..."
        echo "This may take several hours. Press Ctrl+C to stop waiting (jobs will continue running)."
        
        while true; do
            RUNNING=$(squeue -u $USER | grep "train_" | wc -l)
            COMPLETED=$(find $CORPUS_NAME -name "status.json" -exec grep -l "success" {} \; 2>/dev/null | wc -l)
            
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
        echo "  python collect_training_results.py --corpus_dir $CORPUS_NAME --output_csv ${CORPUS_NAME}_results.csv"
        exit 0
    fi
else
    echo "Skipping job submission."
    echo ""
    echo "To submit jobs later, run:"
    echo "  bash $CORPUS_NAME/submit_all_jobs.sh"
    exit 0
fi

echo ""
echo "[6/6] Collecting training results..."
OUTPUT_CSV="${CORPUS_NAME}_results.csv"

python collect_training_results.py \
    --corpus_dir "$CORPUS_NAME" \
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
