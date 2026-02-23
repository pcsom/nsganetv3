#!/bin/bash

cd /home/hice1/glu49/nsganetv3

echo "Submitting 6 training jobs..."

job_ids=()

job_id=$(sbatch fresh_test/arch_0000/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted fresh_test/arch_0000/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch fresh_test/arch_0001/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted fresh_test/arch_0001/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch fresh_test/arch_0002/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted fresh_test/arch_0002/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch fresh_test/arch_0003/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted fresh_test/arch_0003/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch fresh_test/arch_0004/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted fresh_test/arch_0004/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch fresh_test/arch_0005/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted fresh_test/arch_0005/train_job.sh: $job_id"
sleep 0.1


echo "All jobs submitted!"
echo "Job IDs: ${job_ids[@]}"
echo "Monitor with: squeue -u $USER"
