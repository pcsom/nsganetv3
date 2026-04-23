import os
import json
import argparse
import random

def sample_random_architecture(resolutions=[192, 224], rng=None):
    rng = random if rng is None else rng
    kernel_choices = [3, 5, 7]
    expansion_choices = [3, 4, 6]
    depth_choices = [2, 3, 4]
    
    depths = [rng.choice(depth_choices) for _ in range(5)]
    
    total_layers = sum(depths)
    
    ks = [rng.choice(kernel_choices) for _ in range(total_layers)]
    e = [rng.choice(expansion_choices) for _ in range(total_layers)]
    
    architectures = []
    for r in resolutions:
        architectures.append({
            'ks': ks,
            'e': e,
            'd': depths,
            'r': r
        })
    
    return architectures

def generate_simple_corpus(n_samples, output_dir, resolutions, seed=0):
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    
    corpus = []
    
    print(f"Generating {n_samples} random architectures (seed={seed})...")
    
    for i in range(n_samples):
        archs = sample_random_architecture(resolutions, rng=rng)
        
        for arch_config in archs:
            arch_id = len(corpus)
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

    manifest_path = os.path.join(output_dir, 'generation_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump({
            'generator': 'generate_simple_corpus.py',
            'seed': seed,
            'n_samples': n_samples,
            'resolutions': resolutions,
            'total_configurations': len(corpus),
        }, f, indent=2)
    
    print(f"\nCorpus generation complete:")
    print(f"  Base architectures: {n_samples}")
    print(f"  Resolutions per arch: {len(resolutions)}")
    print(f"  Total configurations: {len(corpus)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Metadata: {corpus_meta_path}")
    print(f"  Manifest: {manifest_path}")
    
    return corpus

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=250)
    parser.add_argument('--output_dir', type=str, default='training_corpus')
    parser.add_argument('--resolutions', type=int, nargs='+', default=[192, 224])
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for deterministic corpus generation')
    
    args = parser.parse_args()
    
    generate_simple_corpus(args.n_samples, args.output_dir, args.resolutions, seed=args.seed)

if __name__ == '__main__':
    main()
