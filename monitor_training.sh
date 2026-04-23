#!/bin/bash

CORPUS_DIR="${1:-corpus_250}"

if [ ! -d "$CORPUS_DIR" ]; then
    echo "Error: Corpus directory '$CORPUS_DIR' not found"
    echo "Usage: $0 [corpus_dir]"
    exit 1
fi

TOTAL_ARCHS=$(ls -d $CORPUS_DIR/arch_* 2>/dev/null | wc -l)

echo "========================================="
echo "NSGANetV2 Training Monitor"
echo "========================================="
echo "Corpus:     $CORPUS_DIR"
echo "Total jobs: $TOTAL_ARCHS"
echo "User:       $USER"
echo "Time:       $(date)"
echo "========================================="
echo ""

echo "SLURM Queue Status:"
echo "-------------------"
TOTAL_JOBS=$(squeue -u $USER | grep "train_" | wc -l)
RUNNING=$(squeue -u $USER | grep "train_" | grep " R " | wc -l)
PENDING=$(squeue -u $USER | grep "train_" | grep " PD" | wc -l)

echo "  Total in queue: $TOTAL_JOBS"
echo "  Running:        $RUNNING"
echo "  Pending:        $PENDING"
echo ""

echo "Training Completion:"
echo "--------------------"
COMPLETED=$(find $CORPUS_DIR -name "status.json" -exec grep -l "success" {} \; 2>/dev/null | wc -l)
FAILED=$(find $CORPUS_DIR -name "status.json" -exec grep -l "failed" {} \; 2>/dev/null | wc -l)
PROGRESS=$(awk "BEGIN {printf \"%.1f\", ($COMPLETED/$TOTAL_ARCHS)*100}")

echo "  Completed:  $COMPLETED / $TOTAL_ARCHS ($PROGRESS%)"
echo "  Failed:     $FAILED"
echo "  Running:    $RUNNING"
echo "  Not started: $(($TOTAL_ARCHS - $COMPLETED - $FAILED - $RUNNING))"
echo ""

if [ $COMPLETED -gt 0 ]; then
    echo "Recent Completions (Last 5):"
    echo "----------------------------"
    find $CORPUS_DIR -name "status.json" -exec grep -l "success" {} \; 2>/dev/null | \
        xargs ls -t | head -5 | while read status_file; do
        arch_dir=$(dirname "$status_file")
        arch_id=$(basename "$arch_dir")
        
        summary_file=$(find "$arch_dir/train" -name "summary.csv" 2>/dev/null | head -1)
        if [ -f "$summary_file" ]; then
            best_acc=$(tail -1 "$summary_file" | cut -d',' -f4)
            echo "  $arch_id: ${best_acc}% accuracy"
        else
            echo "  $arch_id: completed"
        fi
    done
    echo ""
fi

if [ $FAILED -gt 0 ]; then
    echo "Recent Failures (Last 3):"
    echo "-------------------------"
    find $CORPUS_DIR -name "status.json" -exec grep -l "failed" {} \; 2>/dev/null | \
        xargs ls -t | head -3 | while read status_file; do
        arch_dir=$(dirname "$status_file")
        arch_id=$(basename "$arch_dir")
        
        err_file="$arch_dir/train.err"
        if [ -f "$err_file" ]; then
            last_error=$(tail -20 "$err_file" | grep -i "error\|exception\|failed" | head -1 | cut -c1-80)
            if [ -n "$last_error" ]; then
                echo "  $arch_id: $last_error"
            else
                echo "  $arch_id: (check $arch_id/train.err for details)"
            fi
        else
            echo "  $arch_id: failed"
        fi
    done
    echo ""
fi

if [ $RUNNING -gt 0 ] && [ $COMPLETED -gt 0 ]; then
    echo "Estimated Time Remaining:"
    echo "-------------------------"
    
    # Get average runtime from completed jobs
    total_time=0
    count=0
    find $CORPUS_DIR -name "status.json" -exec grep -l "success" {} \; 2>/dev/null | head -10 | while read status_file; do
        arch_dir=$(dirname "$status_file")
        log_file="$arch_dir/train.log"
        
        if [ -f "$log_file" ]; then
            # Extract walltime from SLURM epilog
            walltime=$(grep "walltime=" "$log_file" 2>/dev/null | tail -1 | sed 's/.*walltime=\([^,]*\).*/\1/')
            if [ -n "$walltime" ]; then
                # Convert to seconds (format: HH:MM:SS)
                IFS=: read h m s <<< "$walltime"
                seconds=$((10#$h * 3600 + 10#$m * 60 + 10#$s))
                total_time=$((total_time + seconds))
                count=$((count + 1))
            fi
        fi
    done
    
    # This runs in subshell from while loop, so we need to do calculation inline
    find $CORPUS_DIR -name "status.json" -exec grep -l "success" {} \; 2>/dev/null | head -10 | \
    while read status_file; do
        arch_dir=$(dirname "$status_file")
        log_file="$arch_dir/train.log"
        if [ -f "$log_file" ]; then
            grep "walltime=" "$log_file" 2>/dev/null | tail -1 | sed 's/.*walltime=\([^,]*\).*/\1/'
        fi
    done | {
        total_time=0
        count=0
        while read walltime; do
            if [ -n "$walltime" ]; then
                IFS=: read h m s <<< "$walltime"
                seconds=$((10#$h * 3600 + 10#$m * 60 + 10#$s))
                total_time=$((total_time + seconds))
                count=$((count + 1))
            fi
        done
        
        if [ $count -gt 0 ]; then
            avg_time=$((total_time / count))
            remaining_jobs=$(($TOTAL_ARCHS - $COMPLETED - $FAILED))
            
            # Assume jobs run in parallel batches
            if [ $RUNNING -gt 0 ]; then
                batches=$(((remaining_jobs + RUNNING - 1) / RUNNING))
                total_remaining=$((avg_time * batches))
            else
                total_remaining=$((avg_time * remaining_jobs))
            fi
            
            hours=$((total_remaining / 3600))
            minutes=$(((total_remaining % 3600) / 60))
            
            echo "  Avg time per job:  $(($avg_time / 60)) minutes"
            echo "  Remaining jobs:    $remaining_jobs"
            echo "  Parallel capacity: $RUNNING GPUs"
            echo "  Est. completion:   ~${hours}h ${minutes}m"
        else
            echo "  Unable to estimate (no completed jobs with timing data)"
        fi
    }
    echo ""
fi

echo "========================================="
echo ""
echo "Commands:"
echo "  Watch live:     watch -n 30 '$0 $CORPUS_DIR'"
echo "  Job details:    squeue -u $USER -o '%.10i %.12P %.20j %.8T %.10M'"
echo "  Check arch:     tail -20 $CORPUS_DIR/arch_0000/train.err"
echo "  Collect results: python collect_training_results.py --corpus_dir $CORPUS_DIR --output_csv results.csv"
echo ""
