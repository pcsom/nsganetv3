import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np
import scipy.stats as stats

from acc_predictor.factory import get_acc_predictor_from_config


def load_toml_config(path):
    config = {}
    current_section = None

    with open(path, 'r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1].strip()
                config.setdefault(current_section, {})
                continue

            if '=' not in line or current_section is None:
                continue

            key, value_text = [part.strip() for part in line.split('=', 1)]
            if value_text.startswith('[') and value_text.endswith(']'):
                value = ast.literal_eval(value_text)
            elif value_text.startswith('"') and value_text.endswith('"'):
                value = value_text[1:-1]
            elif value_text.startswith("'") and value_text.endswith("'"):
                value = value_text[1:-1]
            elif value_text.lower() in ('true', 'false'):
                value = value_text.lower() == 'true'
            else:
                try:
                    value = ast.literal_eval(value_text)
                except Exception:
                    value = value_text

            config[current_section][key] = value

    return config


def parse_value(value):
    if isinstance(value, list):
        return value

    if value is None:
        return []

    text = str(value).strip()
    if not text:
        return []

    return ast.literal_eval(text)


def load_ground_truth_rows(csv_path):
    rows = []
    with open(csv_path, 'r', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get('status') not in (None, '', 'success'):
                continue

            ks = parse_value(row.get('ks'))
            e = parse_value(row.get('e'))
            d = parse_value(row.get('d'))
            r = float(row.get('r'))

            target = row.get('best_top1')
            if target is None or str(target).strip() == '':
                target = row.get('final_top1')
            if target is None or str(target).strip() == '':
                continue

            rows.append({
                'arch_id': row.get('arch_id'),
                'ks': ks,
                'e': e,
                'd': d,
                'r': r,
                'target': float(target),
            })

    return rows


def get_correlation(prediction, target):
    rmse = np.sqrt(((prediction - target) ** 2).mean())
    rho, _ = stats.spearmanr(prediction, target)
    tau, _ = stats.kendalltau(prediction, target)
    return rmse, rho, tau


def encode_architecture(row, layout=None):
    if layout is None:
        layout = {}

    ks = np.asarray(row['ks'], dtype=float)
    e = np.asarray(row['e'], dtype=float)
    d = np.asarray(row['d'], dtype=float)
    r = np.asarray([row['r']], dtype=float)

    if 'ks' in layout:
        ks = np.pad(ks, (0, layout['ks'] - len(ks)), constant_values=0.0)
    if 'e' in layout:
        e = np.pad(e, (0, layout['e'] - len(e)), constant_values=0.0)
    if 'd' in layout:
        d = np.pad(d, (0, layout['d'] - len(d)), constant_values=0.0)

    return np.concatenate([ks, e, d, r])


def build_feature_matrix(rows):
    layout = {
        'ks': max(len(row['ks']) for row in rows),
        'e': max(len(row['e']) for row in rows),
        'd': max(len(row['d']) for row in rows),
    }

    features = np.vstack([encode_architecture(row, layout=layout) for row in rows])
    targets = np.asarray([row['target'] for row in rows], dtype=float)
    return features, targets, layout


def train_test_split(features, targets, test_size=0.2, seed=42):
    if not 0.0 < test_size < 1.0:
        raise ValueError('test_size must be between 0 and 1')

    generator = np.random.default_rng(seed)
    indices = generator.permutation(len(features))
    test_count = max(1, int(round(len(indices) * test_size)))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    if len(train_indices) == 0:
        raise ValueError('train split is empty; reduce test_size or use more samples')

    return (
        features[train_indices],
        features[test_indices],
        targets[train_indices],
        targets[test_indices],
    )


def main():
    parser = argparse.ArgumentParser(description='Validate a surrogate predictor on Flowers ground truth data.')
    parser.add_argument('--config', type=str, default='config/nsganetv3_config.toml')
    parser.add_argument('--ground_truth', type=str, default=None)
    parser.add_argument('--predictor', type=str, default=None)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_json', type=str, default=None)
    args = parser.parse_args()

    config = load_toml_config(args.config)
    ground_truth_path = args.ground_truth
    if ground_truth_path is None:
        dataset_cfg = config.get('dataset', {})
        ground_truth_path = dataset_cfg.get('ground_truth_csv') or dataset_cfg.get('results_csv') or 'training_results.csv'

    rows = load_ground_truth_rows(ground_truth_path)
    if len(rows) < 2:
        raise RuntimeError(f'Need at least 2 successful rows in {ground_truth_path}')

    features, targets, layout = build_feature_matrix(rows)
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=args.test_size, seed=args.seed)

    if args.predictor is not None:
        predictor = get_acc_predictor_from_config(args.predictor, x_train, y_train)
    else:
        predictor = get_acc_predictor_from_config(config, x_train, y_train)
    predictions = np.asarray(predictor.predict(x_test)).reshape(-1)

    rmse, rho, tau = get_correlation(predictions, y_test)

    results = {
        'config': args.config,
        'ground_truth': str(Path(ground_truth_path).resolve()),
        'predictor': args.predictor or config.get('search', {}).get('predictor', config.get('predictor')),
        'n_rows': len(rows),
        'n_train': len(x_train),
        'n_test': len(x_test),
        'split_seed': args.seed,
        'test_size': args.test_size,
        'feature_layout': layout,
        'rmse': float(rmse),
        'spearman_rho': float(rho),
        'kendall_tau': float(tau),
    }

    print(json.dumps(results, indent=2, sort_keys=True))

    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as handle:
            json.dump(results, handle, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()