import os
import json
import argparse
from pathlib import Path

def create_train_job(arch_id, config_path, arch_dir, supernet_path, data_path, 
                     num_classes, epochs, batch_size, conda_env, time_limit='12:00:00'):
    
    job_script = f"""#!/bin/bash
#SBATCH --job-name=train_{arch_id:04d}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --mem=32GB
#SBATCH --time={time_limit}
#SBATCH --output={arch_dir}/train.log
#SBATCH --error={arch_dir}/train.err

module load cuda/12.1.1

cd $SLURM_SUBMIT_DIR

$HOME/.conda/envs/{conda_env}/bin/python train_imagenet.py {data_path} \\
    --model nsganetv2 \\
    --model-config {config_path} \\
    --initial-checkpoint {supernet_path} \\
    --num-classes {num_classes} \\
    --epochs {epochs} \\
    --batch-size {batch_size} \\
    --img-size 224 \\
    --lr 0.01 \\
    --weight-decay 1e-4 \\
    --drop 0.2 \\
    --drop-path 0.2 \\
    --aa rand-m9-mstd0.5-inc1 \\
    --remode pixel \\
    --reprob 0.2 \\
    --output {arch_dir}

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully"
    echo "{{\\\"status\\\": \\\"success\\\", \\\"arch_id\\\": {arch_id}}}" > {arch_dir}/status.json
else
    echo "Training failed with exit code $TRAIN_EXIT_CODE"
    echo "{{\\\"status\\\": \\\"failed\\\", \\\"arch_id\\\": {arch_id}, \\\"exit_code\\\": $TRAIN_EXIT_CODE}}" > {arch_dir}/status.json
fi
"""
    
    job_file = os.path.join(arch_dir, 'train_job.sh')
    with open(job_file, 'w') as f:
        f.write(job_script)
    
    os.chmod(job_file, 0o755)
    return job_file

def create_submit_all_script(corpus_dir, job_files):
    script_path = os.path.join(corpus_dir, 'submit_all_jobs.sh')
    
    parent_dir = os.path.dirname(os.path.abspath(corpus_dir))
    corpus_name = os.path.basename(corpus_dir)
    
    script = f"""#!/bin/bash

cd {parent_dir}

echo "Submitting {len(job_files)} training jobs..."

job_ids=()

"""
    
    for job_file in job_files:
        relative_path = os.path.relpath(job_file, parent_dir)
        script += f'job_id=$(sbatch {relative_path} | awk \'{{print $NF}}\')\n'
        script += 'job_ids+=($job_id)\n'
        script += f'echo "Submitted {relative_path}: $job_id"\n'
        script += 'sleep 0.1\n\n'
    
    script += """
echo "All jobs submitted!"
echo "Job IDs: ${job_ids[@]}"
echo "Monitor with: squeue -u $USER"
"""
    
    with open(script_path, 'w') as f:
        f.write(script)
    
    os.chmod(script_path, 0o755)
    print(f"Created submission script: {script_path}")
    return script_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_dir', type=str, default='training_corpus')
    parser.add_argument('--supernet_path', type=str,
                        default='/storage/ice-shared/vip-vvk/data/AOT/ofa_checkpoints/ofa_mbv3_d234_e346_k357_w1.0')
    parser.add_argument('--data_path', type=str, 
                        default='/storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers',
                        help='Path to dataset directory (VIP shared storage, accessible to all users)')
    parser.add_argument('--num_classes', type=int, default=102,
                        help='Number of classes (102 for Oxford Flowers, 1000 for ImageNet)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--conda_env', type=str, default='nsganetv2-llm')
    parser.add_argument('--time_limit', type=str, default='12:00:00')
    
    args = parser.parse_args()
    
    metadata_path = os.path.join(args.corpus_dir, 'corpus_metadata.json')
    if not os.path.exists(metadata_path):
        print(f"Error: Corpus metadata not found at {metadata_path}")
        print(f"Run generate_simple_corpus.py first")
        return
    
    with open(metadata_path, 'r') as f:
        corpus = json.load(f)
    
    print(f"Creating training jobs for {len(corpus)} architectures...")
    print(f"  Dataset: {args.data_path}")
    print(f"  Classes: {args.num_classes}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    
    job_files = []
    for item in corpus:
        arch_id = item['arch_id']
        config_path = item['config_path']
        arch_dir = item['arch_dir']
        
        job_file = create_train_job(
            arch_id, config_path, arch_dir, args.supernet_path,
            args.data_path, args.num_classes, args.epochs, args.batch_size,
            args.conda_env, args.time_limit
        )
        job_files.append(job_file)
        
        if (arch_id + 1) % 50 == 0:
            print(f"  Created {arch_id + 1}/{len(corpus)} job scripts")
    
    print(f"\nCreated {len(job_files)} job scripts")
    
    submit_script = create_submit_all_script(args.corpus_dir, job_files)
    
    print(f"\nTo submit all jobs, run:")
    print(f"  bash {submit_script}")
    
    print(f"\nOr submit individually:")
    print(f"  sbatch {job_files[0]}")

if __name__ == '__main__':
    main()
