import csv
import ast
import torch
import torch.nn as nn
from evaluator import OFAEvaluator
from evaluator import get_net_info as get_eval_net_info

def load_offline_ground_truth(csv_path, sec_obj, n_classes, supernet_path):
    print(f'Loading offline ground truth from {csv_path}...')
    archive = []
    evaluator = OFAEvaluator(n_classes=1000, model_path=supernet_path)
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
            subnet, _ = evaluator.sample({'ks': ks, 'e': e, 'd': d, 'r': r})
            if n_classes != 1000:
                if hasattr(subnet, 'classifier') and hasattr(subnet.classifier, 'in_features'):
                    in_features = subnet.classifier.in_features
                elif hasattr(subnet, 'classifier') and hasattr(subnet.classifier, 'linear'):
                    in_features = subnet.classifier.linear.in_features
                else:
                    in_features = 1280
                if hasattr(subnet.classifier, 'linear'):
                    subnet.classifier.linear = nn.Linear(in_features, n_classes)
                else:
                    subnet.classifier = nn.Linear(in_features, n_classes)
            lut = {'cpu': 'data/i7-8700K_lut.yaml'}
            measure_latency = sec_obj if 'cpu' in sec_obj or 'gpu' in sec_obj else None
            info = get_eval_net_info(subnet, (3, r, r), measure_latency=measure_latency, print_info=False, clean=True, lut=lut)
            complexity = info[sec_obj]
            archive.append((arch_dict, error, complexity))
            break
    print(f'Successfully loaded {len(archive)} architectures from offline data.')
    return archive
