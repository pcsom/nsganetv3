import os
import json
import re
import pandas as pd
import argparse
from pathlib import Path

def parse_train_log(log_path):
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        log_content = f.read()
    
    best_acc_match = re.search(r'best.*acc[uracy]*[:\s]+([0-9.]+)', log_content, re.IGNORECASE)
    final_acc_match = re.search(r'final.*acc[uracy]*[:\s]+([0-9.]+)', log_content, re.IGNORECASE)
    
    params_match = re.search(r'#params\s+([0-9.]+)M', log_content)
    flops_match = re.search(r'#flops\s+([0-9.]+)M', log_content)
    
    results = {}
    if best_acc_match:
        results['best_accuracy'] = float(best_acc_match.group(1))
    if final_acc_match:
        results['final_accuracy'] = float(final_acc_match.group(1))
    if params_match:
        results['params_M'] = float(params_match.group(1))
    if flops_match:
        results['flops_M'] = float(flops_match.group(1))
    
    return results if results else None

def collect_results(corpus_dir):
    metadata_path = os.path.join(corpus_dir, 'corpus_metadata.json')
    
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found")
        return None
    
    with open(metadata_path, 'r') as f:
        corpus = json.load(f)
    
    results = []
    completed = 0
    failed = 0
    pending = 0
    
    for item in corpus:
        arch_id = item['arch_id']
        arch_dir = item['arch_dir']
        config = item['config']
        
        status_path = os.path.join(arch_dir, 'status.json')
        log_path = os.path.join(arch_dir, 'train.log')
        
        result_entry = {
            'arch_id': arch_id,
            'ks': config['ks'],
            'e': config['e'],
            'd': config['d'],
            'r': config['r'],
        }
        
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                status = json.load(f)
            result_entry['status'] = status['status']
            
            if status['status'] == 'success':
                completed += 1
                train_results = parse_train_log(log_path)
                if train_results:
                    result_entry.update(train_results)
            else:
                failed += 1
                result_entry['exit_code'] = status.get('exit_code', 'unknown')
        else:
            pending += 1
            result_entry['status'] = 'pending'
        
        results.append(result_entry)
    
    df = pd.DataFrame(results)
    
    print(f"\nResults Summary:")
    print(f"  Total architectures: {len(corpus)}")
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print(f"  Pending: {pending}")
    
    if completed > 0:
        completed_df = df[df['status'] == 'success']
        if 'best_accuracy' in completed_df.columns:
            print(f"\nAccuracy Statistics:")
            print(f"  Mean: {completed_df['best_accuracy'].mean():.2f}%")
            print(f"  Std: {completed_df['best_accuracy'].std():.2f}%")
            print(f"  Min: {completed_df['best_accuracy'].min():.2f}%")
            print(f"  Max: {completed_df['best_accuracy'].max():.2f}%")
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_dir', type=str, default='training_corpus')
    parser.add_argument('--output', type=str, default='training_results.csv')
    
    args = parser.parse_args()
    
    df = collect_results(args.corpus_dir)
    
    if df is not None:
        output_path = os.path.join(args.corpus_dir, args.output)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        print(f"\nTo use for LLM comparison:")
        print(f"  1. Copy to NASlib-coder-nas repo")
        print(f"  2. Generate embeddings with create_nsganetv2_embeddings.py")
        print(f"  3. Run comparison like you did for NASBench-201")

if __name__ == '__main__':
    main()
