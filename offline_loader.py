import csv
import ast
from evaluator import OFAEvaluator
from evaluator import get_net_info as get_eval_net_info

def load_offline_ground_truth(csv_path, sec_obj, n_classes, supernet_path):
    print(f"Loading offline ground truth from {csv_path}...")
    
    archive = []
    evaluator = OFAEvaluator(n_classes=n_classes, model_path=supernet_path)
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'status' in row and row['status'] != 'success':
                continue
            
            ks = ast.literal_eval(row['ks'])
            e = ast.literal_eval(row['e'])
            d = ast.literal_eval(row['d'])
            r = int(row['r'])
            
            arch_dict = {'ks': ks, 'e': e, 'd': d, 'r': r}
            
            top1 = float(row['best_top1']) if 'best_top1' in row and row['best_top1'].strip() else float(row['final_top1'])
            error = 100.0 - top1
            
            subnet = evaluator.sample({'ks': ks, 'e': e, 'd': d, 'r': r})
            lut = {'cpu': 'data/i7-8700K_lut.yaml'}
            measure_latency = sec_obj if 'cpu' in sec_obj or 'gpu' in sec_obj else None
            
            info = get_eval_net_info(subnet, (3, r, r), measure_latency=measure_latency, print_info=False, clean=True, lut=lut)
            complexity = info[sec_obj]
            
            archive.append((arch_dict, error, complexity))
            
    print(f"Successfully loaded {len(archive)} architectures from offline data.")
    return archive
import pandas as pd
import json
import ast
from evaluator import OFAEvaluator, get_net_info

def load_offline_ground_truth(csv_path, sec_obj, n_classes, supernet_path):
    print(f"Loading offline ground truth from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    archive = []
    
    # Initialize OFAEvaluator to compute complexity metrics
    evaluator = OFAEvaluator(n_classes=n_classes, model_path=supernet_path)
    
    for idx, row in df.iterrows():
        # Parsing strings like "[5, 5, 7, ...]" into python lists
        ks = ast.literal_eval(row['ks'])
        e = ast.literal_eval(row['e'])
        d = ast.literal_eval(row['d'])
        r = int(row['r'])
        
        arch_dict = {'ks': ks, 'e': e, 'd': d, 'r': r}
        
        # Get accuracy (error = 100 - best_top1 or final_top1)
        # Using best_top1 since it was tracked, but we can default to final_top1
        top1 = float(row['best_top1']) if 'best_top1' in row and pd.notnull(row['best_top1']) else float(row['final_top1'])
        error = 100.0 - top1
        
        # We need to evaluate complexity. We can instantiate subnet and use get_net_info
        subnet = evaluator.sample({'ks': ks, 'e': e, 'd': d, 'r': r})
        lut = {'cpu': 'data/i7-8700K_lut.yaml'} # Similar to how eval() does it
        measure_latency = sec_obj if 'cpu' in sec_obj or 'gpu' in sec_obj else None
        
        # We compute complexity on the fly since it's fast
        info = get_net_info(subnet, (3, r, r), measure_latency=measure_latency, print_info=False, clean=True, lut=lut)
        
        complexity = info[sec_obj] # This is either flops, params, cpu, or gpu latency
        
        archive.append((arch_dict, error, complexity))
        
    print(f"Successfully loaded {len(archive)} architectures from offline data.")
    return archive
