import os
import json
import pandas as pd
import argparse

def parse_train_results(arch_dir):
    """Extract metrics from summary.csv in training directory."""
    results = {}
    
    # Find summary.csv in the train subdirectory
    train_dir = os.path.join(arch_dir, 'train')
    if not os.path.isdir(train_dir):
        return None
    
    for subdir in os.listdir(train_dir):
        summary_path = os.path.join(train_dir, subdir, 'summary.csv')
        if os.path.exists(summary_path):
            try:
                df_summary = pd.read_csv(summary_path)
                # Get the last row (final epoch)
                if len(df_summary) > 0:
                    last_row = df_summary.iloc[-1]
                    if 'eval_top1' in df_summary.columns:
                        results['final_top1'] = float(last_row['eval_top1'])
                    if 'eval_top5' in df_summary.columns:
                        results['final_top5'] = float(last_row['eval_top5'])
                    if 'eval_top1' in df_summary.columns:
                        results['best_top1'] = float(df_summary['eval_top1'].max())
                    if 'eval_top5' in df_summary.columns:
                        results['best_top5'] = float(df_summary['eval_top5'].max())
                break
            except Exception as e:
                print(f"Warning: Could not parse {summary_path}: {e}")
                continue
    
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
                train_results = parse_train_results(arch_dir)
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
        if 'final_top1' in completed_df.columns:
            print(f"\nAccuracy Statistics (Final Top-1):")
            print(f"  Mean: {completed_df['final_top1'].mean():.2f}%")
            print(f"  Std: {completed_df['final_top1'].std():.2f}%")
            print(f"  Min: {completed_df['final_top1'].min():.2f}%")
            print(f"  Max: {completed_df['final_top1'].max():.2f}%")
        if 'best_top1' in completed_df.columns:
            print(f"\nAccuracy Statistics (Best Top-1):")
            print(f"  Mean: {completed_df['best_top1'].mean():.2f}%")
            print(f"  Max: {completed_df['best_top1'].max():.2f}%")
    
    return df


def to_surrogate_dataframe(df):
    surrogate_df = df.copy()
    for column in ['ks', 'e', 'd']:
        if column in surrogate_df.columns:
            surrogate_df[column] = surrogate_df[column].apply(lambda value: json.dumps(value, separators=(',', ':')))
    return surrogate_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_dir', type=str, default='training_corpus')
    parser.add_argument('--output', type=str, default='training_results.csv',
                        help='Output CSV filename (relative to corpus_dir)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Alias for --output (backward compatibility)')
    parser.add_argument('--surrogate_output', type=str, default=None,
                        help='Optional surrogate-ready CSV filename (relative to corpus_dir)')
    
    args = parser.parse_args()
    
    df = collect_results(args.corpus_dir)
    
    if df is not None:
        output_name = args.output_csv if args.output_csv is not None else args.output
        output_path = os.path.join(args.corpus_dir, output_name)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        if args.surrogate_output:
            surrogate_path = os.path.join(args.corpus_dir, args.surrogate_output)
            to_surrogate_dataframe(df).to_csv(surrogate_path, index=False)
            print(f"Surrogate-ready export saved to: {surrogate_path}")
        
        print(f"\nTo use for LLM comparison:")
        print(f"  1. Copy to NASlib-coder-nas repo")
        print(f"  2. Generate embeddings with create_nsganetv2_embeddings.py")
        print(f"  3. Run comparison like you did for NASBench-201")

if __name__ == '__main__':
    main()
