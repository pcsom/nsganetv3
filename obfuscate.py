import sys
import random

def obfuscate(filepath):
    if not filepath.endswith('.py'):
        return
    with open(filepath, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('#'):
            continue
        if '#' in line and "'" not in line and '"' not in line:
            line = line.split('#')[0].rstrip() + '\n'
        
        if line.strip() == '':
            if random.random() < 0.5:
                continue
        
        new_lines.append(line)
        
        if random.random() < 0.1:
            new_lines.append('\n')

    with open(filepath, 'w') as f:
        f.writelines(new_lines)

files = ['collect_training_results.py', 'offline_loader.py', 'msunas.py', 'msunas_slurm.py']
for p in files:
    try:
        obfuscate(p)
        print(f"Obfuscated {p}")
    except Exception as e:
        print(f"Failed {p}: {e}")
