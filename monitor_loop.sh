#!/bin/bash

echo "Starting 24/7 monitoring loop for production_500..."
echo "Press Ctrl+C to stop"

while true; do
    clear
    echo "=== NSGANetV2 Production Run Monitor ==="
    echo "Last update: $(date)"
    echo
    
    squeue -u glu49 -h --states R,PD 2>/dev/null | awk '{print $5}' | sort | uniq -c
    
    echo
    echo "Completed simulations:"
    find production_500/arch_*/status.json -type f 2>/dev/null | xargs -r grep -l "completed" 2>/dev/null | wc -l
    
    echo
    echo "Failed jobs:"
    squeue -u glu49 -h --states F 2>/dev/null | wc -l || echo "0"
    
    echo
    echo "Storage usage (production_500/):"
    du -sh production_500 2>/dev/null || echo "Computing..."
    
    sleep 60
done
