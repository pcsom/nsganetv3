import os
import json
import argparse
from pathlib import Path
from evaluator import OFAEvaluator

def generate_corpus(n_samples, output_dir, supernet_path, resolutions=[192, 224]):
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = OFAEvaluator(n_classes=1000, model_path=supernet_path)
    
    corpus = []
    
    print(f"Generating {n_samples} random architectures...")
    
    for i in range(n_samples):
        config = evaluator.engine.sample_active_subnet()
        
        for res in resolutions:
            arch_id = len(corpus)
            arch_config = {
                'ks': config['ks'],
                'e': config['e'],
                'd': config['d'],
                'r': res
            }
            
            arch_dir = os.path.join(output_dir, f'arch_{arch_id:04d}')
            os.makedirs(arch_dir, exist_ok=True)
            
            config_path = os.path.join(arch_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(arch_config, f, indent=2)
            
            corpus.append({
                'arch_id': arch_id,
                'config': arch_config,
                'config_path': config_path,
                'arch_dir': arch_dir
            })
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{n_samples} base architectures ({len(corpus)} total with resolutions)")
    
    corpus_meta_path = os.path.join(output_dir, 'corpus_metadata.json')
    with open(corpus_meta_path, 'w') as f:
        json.dump(corpus, f, indent=2)
    
    print(f"\nCorpus generation complete:")
    print(f"  Base architectures: {n_samples}")
    print(f"  Resolutions per arch: {len(resolutions)}")
    print(f"  Total configurations: {len(corpus)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Metadata: {corpus_meta_path}")
    
    return corpus

def create_subnet_config(arch_config, supernet_path, output_path):
    evaluator = OFAEvaluator(n_classes=1000, model_path=supernet_path)
    subnet, config = evaluator.sample(arch_config)
    evaluator.save_net_config(output_path, subnet, 'model_config.json')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=250)
    parser.add_argument('--output_dir', type=str, default='training_corpus')
    parser.add_argument('--supernet_path', type=str,
                        default='/storage/ice-shared/vip-vvk/data/AOT/ofa_checkpoints/ofa_mbv3_d234_e346_k357_w1.0')
    parser.add_argument('--resolutions', type=int, nargs='+', default=[192, 224])
    parser.add_argument('--create_subnet_configs', action='store_true')
    
    args = parser.parse_args()
    
    corpus = generate_corpus(args.n_samples, args.output_dir, args.supernet_path, args.resolutions)
    
    if args.create_subnet_configs:
        print("\nCreating full subnet configs...")
        for i, item in enumerate(corpus):
            create_subnet_config(item['config'], args.supernet_path, item['arch_dir'])
            if (i + 1) % 10 == 0:
                print(f"  Created {i + 1}/{len(corpus)} subnet configs")

if __name__ == '__main__':
    main()
