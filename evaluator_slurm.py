#!/usr/bin/env python
"""
SLURM-compatible individual architecture evaluation script for NSGANetV3.
This script evaluates a single architecture specified by SLURM_ARRAY_TASK_ID.
"""

import os
import json
import torch
import argparse
import pandas as pd
import numpy as np
from evaluator import OFAEvaluator, get_net_info
import warnings
warnings.simplefilter("ignore")


def load_architecture_config(input_file, index):
    """Load architecture configuration from CSV input file"""
    df = pd.read_csv(input_file)
    row = df.iloc[index]
    
    config = {
        'ks': json.loads(row['ks']),
        'e': json.loads(row['e']),
        'd': json.loads(row['d']),
        'r': row['r']
    }
    
    return config, row['iteration']


def evaluate_architecture(config, iteration, index, args, search_config):
    """Evaluate a single architecture and save results"""
    
    # Create output directory
    output_dir = os.path.join(args.outdir, f'iter_{iteration}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create evaluator
    evaluator = OFAEvaluator(
        n_classes=search_config['n_classes'], 
        model_path=search_config['supernet_path']
    )
    
    # Sample subnet from configuration
    subnet, _ = evaluator.sample({
        'ks': config['ks'], 
        'e': config['e'], 
        'd': config['d']
    })
    
    # Create temporary log directory for this evaluation
    log_dir = os.path.join(output_dir, f'arch_{index}_temp')
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        # Evaluate the architecture
        print(f"Evaluating architecture {index} for iteration {iteration}")
        print(f"Configuration: ks={config['ks']}, e={config['e']}, d={config['d']}, r={config['r']}")
        
        OFAEvaluator.eval(
            subnet, 
            log_dir=log_dir,
            data_path=search_config['data'], 
            dataset=search_config['dataset'], 
            n_epochs=search_config['n_epochs'],
            resolution=config['r'], 
            trn_batch_size=search_config['trn_batch_size'], 
            vld_batch_size=search_config['vld_batch_size'],
            num_workers=search_config['num_workers'], 
            valid_size=search_config['valid_size'], 
            is_test=search_config['test'], 
            measure_latency=search_config.get('latency'),
            no_logs=True,
            reset_running_statistics=True
        )
        
        # Load results from temporary file
        temp_stats_file = os.path.join(log_dir, 'net.stats')
        with open(temp_stats_file, 'r') as f:
            stats = json.load(f)
        
        # Get additional network info (complexity metrics)
        net_info = get_net_info(subnet, (3, config['r'], config['r']), 
                               measure_latency=search_config.get('latency'), print_info=False, clean=True)
        
        # Combine stats with network info
        final_stats = {
            'top1': stats['top1'],
            'top5': stats['top5'],
            'loss': stats['loss'],
            'flops': net_info['flops'],
            'params': net_info['params']
        }
        
        # Add latency info if available
        if 'gpu' in net_info and net_info['gpu'] is not None:
            final_stats['gpu'] = net_info['gpu']
        if 'cpu' in net_info and net_info['cpu'] is not None:
            final_stats['cpu'] = net_info['cpu']
        
        # Save final results
        output_file = os.path.join(output_dir, f'arch_{index}.stats')
        with open(output_file, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        print(f"Architecture {index} evaluation completed successfully")
        print(f"Results: Top1={final_stats['top1']:.2f}%, FLOPs={final_stats['flops']:.2f}M")
        
    except Exception as e:
        print(f"Error evaluating architecture {index}: {str(e)}")
        # Create a dummy stats file with poor performance to indicate failure
        final_stats = {
            'top1': 0.0,  # Very poor accuracy
            'top5': 0.0,
            'loss': 100.0,
            'flops': 1000.0,  # High complexity
            'params': 1000.0,
            'error': str(e)
        }
        
        output_file = os.path.join(output_dir, f'arch_{index}.stats')
        with open(output_file, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        print(f"Assigned poor fitness to failed architecture {index}")
        
    finally:
        # Clean up temporary directory
        import shutil
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)


def main():
    parser = argparse.ArgumentParser(description='Evaluate single architecture for NSGANetV3')
    parser.add_argument('index', type=int, help='Architecture index (SLURM_ARRAY_TASK_ID)')
    parser.add_argument('-i', '--infile', required=True, help='Input CSV file with architecture configs')
    parser.add_argument('-o', '--outdir', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Load config from config file in output directory  
    config_file = os.path.join(args.outdir, 'search_config.json')
    with open(config_file, 'r') as f:
        search_config = json.load(f)
    
    print(f"Starting evaluation for architecture index {args.index}")
    
    # Load architecture configuration
    config, iteration = load_architecture_config(args.infile, args.index)
    
    # Evaluate architecture
    evaluate_architecture(config, iteration, args.index, args, search_config)
    
    print(f"Evaluation complete for architecture {args.index}")


if __name__ == '__main__':
    main()