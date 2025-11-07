#!/usr/bin/env python
"""
Surrogate model trainer for NSGANetV3 - runs on GPU nodes via SLURM.
This script trains accuracy predictor models for the evolutionary search.
"""

import os
import sys
import pickle
import json
import argparse
import numpy as np
from acc_predictor.factory import get_acc_predictor


def load_training_data(save_path, iteration):
    """Load training data from previous evaluations"""
    archive_data = []
    
    # Collect data from all previous iterations
    for it in range(1, iteration + 1):
        iter_dir = os.path.join(save_path, f'iter_{it}')
        if not os.path.exists(iter_dir):
            continue
            
        # Load architecture configurations
        eval_input_file = os.path.join(save_path, f'eval_input_iter_{it}.csv')
        if not os.path.exists(eval_input_file):
            continue
            
        import pandas as pd
        configs_df = pd.read_csv(eval_input_file)
        
        # Load results for each architecture
        for idx, row in configs_df.iterrows():
            result_file = os.path.join(iter_dir, f'arch_{idx}.stats')
            
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    stats = json.load(f)
                
                # Create architecture configuration
                arch_config = {
                    'ks': json.loads(row['ks']),
                    'e': json.loads(row['e']),
                    'd': json.loads(row['d']),
                    'r': row['r']
                }
                
                # Create archive entry: (config, top1_error, complexity)
                top1_error = 100 - stats['top1']  # Convert accuracy to error
                complexity = stats.get('flops', 1000)  # Default high complexity if missing
                
                archive_data.append((arch_config, top1_error, complexity))
    
    return archive_data


def train_surrogate_models(archive_data, predictor_type, search_space):
    """Train accuracy predictor models"""
    if len(archive_data) == 0:
        print("Warning: No training data available")
        return None, None
    
    # Encode architectures and extract targets
    inputs = np.array([search_space.encode(x[0]) for x in archive_data])
    targets = np.array([x[1] for x in archive_data])  # Top-1 error rates
    
    print(f"Training {predictor_type} surrogate with {len(inputs)} samples")
    print(f"Input shape: {inputs.shape}, Target range: [{targets.min():.2f}, {targets.max():.2f}]")
    
    # Check if we have enough data
    if len(inputs) <= len(inputs[0]):
        print("Warning: Not enough training samples compared to input dimensions")
        return None, None
    
    # Train accuracy predictor
    try:
        acc_predictor = get_acc_predictor(predictor_type, inputs, targets)
        predictions = acc_predictor.predict(inputs)
        
        # Calculate training performance metrics
        rmse = np.sqrt(((predictions.flatten() - targets) ** 2).mean())
        mae = np.abs(predictions.flatten() - targets).mean()
        
        print(f"Surrogate training completed - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return acc_predictor, predictions
        
    except Exception as e:
        print(f"Error training surrogate model: {e}")
        return None, None


def save_surrogate_results(save_path, iteration, acc_predictor, performance_metrics):
    """Save trained surrogate model and performance metrics"""
    surrogate_dir = os.path.join(save_path, 'surrogate_models')
    os.makedirs(surrogate_dir, exist_ok=True)
    
    # Save the trained model
    model_file = os.path.join(surrogate_dir, f'predictor_iter_{iteration}.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(acc_predictor, f)
    
    # Save performance metrics
    metrics_file = os.path.join(surrogate_dir, f'metrics_iter_{iteration}.json')
    with open(metrics_file, 'w') as f:
        json.dump(performance_metrics, f, indent=2)
    
    print(f"Surrogate model saved to {model_file}")
    print(f"Performance metrics saved to {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description='Train surrogate models for NSGANetV3')
    parser.add_argument('iteration', type=int, help='Current iteration number')
    parser.add_argument('save_path', type=str, help='Base save directory')
    parser.add_argument('--predictor', type=str, default='rbf', 
                       help='Predictor type (rbf/gp/cart/mlp/as)')
    args = parser.parse_args()
    
    print(f"Training surrogate models for iteration {args.iteration}")
    print(f"Save path: {args.save_path}")
    print(f"Predictor type: {args.predictor}")
    
    # Import search space (need to add this to the script location)
    try:
        from search_space.ofa import OFASearchSpace
        search_space = OFASearchSpace()
    except ImportError:
        print("Error: Could not import OFASearchSpace")
        sys.exit(1)
    
    # Load training data from previous iterations
    archive_data = load_training_data(args.save_path, args.iteration)
    
    if len(archive_data) == 0:
        print("No training data available - skipping surrogate training")
        return
    
    # Train surrogate model
    acc_predictor, predictions = train_surrogate_models(
        archive_data, args.predictor, search_space
    )
    
    if acc_predictor is None:
        print("Surrogate training failed")
        return
    
    # Calculate performance metrics
    targets = np.array([x[1] for x in archive_data])
    rmse = np.sqrt(((predictions.flatten() - targets) ** 2).mean())
    mae = np.abs(predictions.flatten() - targets).mean()
    correlation = np.corrcoef(predictions.flatten(), targets)[0, 1]
    
    performance_metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'correlation': float(correlation),
        'num_samples': len(archive_data),
        'predictor_type': args.predictor
    }
    
    # Save results
    save_surrogate_results(args.save_path, args.iteration, acc_predictor, performance_metrics)
    
    print("Surrogate training completed successfully")


if __name__ == "__main__":
    main()