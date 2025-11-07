import os
import json
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd
import time
import re
import csv
import toml
from utils import get_correlation
from evaluator import OFAEvaluator, get_net_info

from pymoo.optimize import minimize
from pymoo.model.problem import Problem
from pymoo.factory import get_performance_indicator
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.factory import get_algorithm, get_crossover, get_mutation

from search_space.ofa import OFASearchSpace
from acc_predictor.factory import get_acc_predictor
from utils import MySampling, BinaryCrossover, MyMutation
from msunas import MSuNAS, AuxiliarySingleLevelProblem, SubsetProblem

_DEBUG = False
if _DEBUG: from pymoo.visualization.scatter import Scatter

# SLURM job configuration - hardcoded generous defaults like surrogate-evolution
JOB_NAME = 'nsganetv3'
NODES = 1
CORES = 8
MEM = '24GB'
JOB_TIME = '08:00:00'
ENV_NAME = 'nas'
GPUS = ["V100-16GB", "V100-32GB", "L40S", "A100-40GB", "H100", "A40", "H200"]


class MSuNASSLURM(MSuNAS):
    """SLURM-adapted version of MSuNAS for HPC cluster execution"""
    
    def __init__(self, kwargs, config_path=None):
        super().__init__(kwargs)
        
        # Load TOML configuration
        if config_path and os.path.exists(config_path):
            self.config = toml.load(config_path)
        else:
            # Use default config if none provided
            default_config_path = os.path.join(os.path.dirname(__file__), 'config', 'nsganetv3_config.toml')
            if os.path.exists(default_config_path):
                self.config = toml.load(default_config_path)
            else:
                self.config = self._get_default_config()
        
        # Override config with any command line arguments
        self._update_config_from_kwargs(kwargs)
        
        self.job_name = f'{JOB_NAME}_{os.path.basename(self.save_path)}'
        self.logs_dir = os.path.join(self.save_path, 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Save search configuration for evaluation scripts
        self._save_search_config()
    
    def _get_default_config(self):
        """Return default configuration if no TOML file found"""
        return {
            'search': {
                'iterations': 30,
                'n_doe': 100,
                'n_iter': 8,
                'sec_obj': 'flops',
                'predictor': 'rbf'
            },
            'dataset': {
                'dataset': 'imagenet',
                'n_classes': 1000,
                'n_epochs': 5,
                'vld_size': 10000,
                'test': False
            },
            'training': {
                'trn_batch_size': 128,
                'vld_batch_size': 200,
                'n_workers': 4
            },
            'evolutionary': {
                'pop_size': 40,
                'n_gens': 20,
                'crossover_prob': 0.9,
                'mutation_eta': 1.0
            },
            'slurm': {
                'job_name': 'nsganetv3',
                'nodes': 1,
                'cores': 8,
                'memory': '24GB',
                'job_time': '08:00:00',
                'env_name': 'nas',
                'gpu_types': ["V100-16GB", "V100-32GB", "L40S", "A100-40GB", "H100", "A40", "H200"]
            },
            'surrogate': {
                'enable_gpu_training': True,
                'train_job_time': '02:00:00',
                'train_memory': '16GB'
            }
        }
    
    def _update_config_from_kwargs(self, kwargs):
        """Update config with command line arguments"""
        # Map command line args to config sections
        search_mapping = ['iterations', 'n_doe', 'n_iter', 'sec_obj', 'predictor']
        dataset_mapping = ['dataset', 'n_classes', 'n_epochs', 'vld_size', 'test']
        training_mapping = ['trn_batch_size', 'vld_batch_size', 'n_workers']
        
        for key in search_mapping:
            if hasattr(self, key):
                self.config['search'][key] = getattr(self, key)
        
        for key in dataset_mapping:
            if hasattr(self, key):
                self.config['dataset'][key] = getattr(self, key)
                
        for key in training_mapping:
            if hasattr(self, key):
                self.config['training'][key] = getattr(self, key)
            
    def _save_search_config(self):
        """Save search configuration for evaluation scripts"""
        config = {
            'data': self.data,
            'dataset': self.config['dataset']['dataset'],
            'n_classes': self.config['dataset']['n_classes'],
            'supernet_path': self.supernet_path,
            'num_workers': self.config['training']['n_workers'],
            'valid_size': self.config['dataset']['vld_size'],
            'trn_batch_size': self.config['training']['trn_batch_size'],
            'vld_batch_size': self.config['training']['vld_batch_size'],
            'n_epochs': self.config['dataset']['n_epochs'],
            'test': self.config['dataset']['test'],
            'latency': self.latency
        }
        config_file = os.path.join(self.save_path, 'search_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
            
    def _create_eval_input_csv(self, archs, output_path, iteration):
        """Create CSV input file for SLURM job array"""
        lines = []
        for i, arch in enumerate(archs):
            lines.append({
                'index': i,
                'iteration': iteration,
                'ks': json.dumps(arch['ks']),
                'e': json.dumps(arch['e']),
                'd': json.dumps(arch['d']),
                'r': arch['r']
            })
        
        df = pd.DataFrame(lines)
        df.to_csv(output_path, index=False)
        return output_path
    
    def _create_job_file(self, num_jobs, iteration):
        """Create SLURM job file for architecture evaluation"""
        eval_input_path = os.path.join(self.save_path, f'eval_input_iter_{iteration}.csv')
        
        batch_script = f"""#!/bin/bash
#SBATCH --job-name={self.job_name}_{iteration}
#SBATCH --nodes={self.config['slurm']['nodes']}
#SBATCH -G 1
#SBATCH --cpus-per-task={self.config['slurm']['cores']}
#SBATCH --mem={self.config['slurm']['memory']}
#SBATCH --time={self.config['slurm']['job_time']}
#SBATCH --output={self.logs_dir}/iter_{iteration}/evaluation.%A.%a.log
#SBATCH --error={self.logs_dir}/iter_{iteration}/evaluation_error.%A.%a.log
#SBATCH --array=0-{num_jobs-1}
#SBATCH --constraint="{'|'.join(self.config['slurm']['gpu_types'])}"

module load anaconda3/2023.03
module load cuda/12.1.1
mkdir -p {self.logs_dir}/iter_{iteration}

conda run -n {self.config['slurm']['env_name']} --no-capture-output python -u evaluator_slurm.py \\
    $SLURM_ARRAY_TASK_ID \\
    -i {eval_input_path} \\
    -o {self.save_path}
"""
        job_file = f'{self.job_name}_{iteration}.job'
        with open(job_file, 'w') as f:
            f.write(batch_script)
        
        return job_file
    
    def _submit_job(self, job_file):
        """Submit SLURM job and return job ID"""
        result = os.popen(f"sbatch {job_file}").read()
        match = re.search(r'Submitted batch job (\d+)', result)
        if match:
            return match.group(1)
        else:
            raise RuntimeError(f"Failed to submit job: {result}")
    
    def _wait_for_job_completion(self, job_id, job_name):
        """Wait for SLURM job to complete"""
        print(f'    Waiting for job {job_id} to complete...')
        while True:
            time.sleep(300)  # Check every 5 minutes
            p = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE)
            text = p.stdout.read().decode('utf-8')
            jobs = text.split('\n')[1:-1]
            if len(jobs) == 0:
                print('    Job completed!')
                break
    
    def _parse_evaluation_results(self, archs, iteration):
        """Parse results from completed SLURM job"""
        top1_err, complexity = [], []
        
        for i, arch in enumerate(archs):
            result_file = os.path.join(self.save_path, f'iter_{iteration}', f'arch_{i}.stats')
            try:
                with open(result_file, 'r') as f:
                    stats = json.load(f)
                top1_err.append(100 - stats['top1'])
                complexity.append(stats[self.sec_obj])
            except FileNotFoundError:
                # Assign bad metrics for failed evaluations
                print(f"    Warning: Failed to find results for architecture {i}")
                top1_err.append(100.0)  # Very bad accuracy
                complexity.append(1000.0)  # Very high complexity
        
        return top1_err, complexity
    
    def _evaluate(self, archs, it):
        """SLURM-based evaluation of architectures"""
        print(f"Evaluating {len(archs)} architectures for iteration {it}")
        
        # Create input CSV
        eval_input_path = os.path.join(self.save_path, f'eval_input_iter_{it}.csv')
        self._create_eval_input_csv(archs, eval_input_path, it)
        
        # Create and submit SLURM job
        job_file = self._create_job_file(len(archs), it)
        job_id = self._submit_job(job_file)
        
        # Wait for completion
        self._wait_for_job_completion(job_id, f"{self.job_name}_{it}")
        
        # Parse results
        return self._parse_evaluation_results(archs, it)
    
    def _fit_acc_predictor(self, archive):
        """Fit accuracy predictor - simplified version without GPU training for now"""
        inputs = np.array([self.search_space.encode(x[0]) for x in archive])
        targets = np.array([x[1] for x in archive])
        assert len(inputs) > len(inputs[0]), "# of training samples have to be > # of dimensions"

        acc_predictor = get_acc_predictor(self.config['search']['predictor'], inputs, targets)
        return acc_predictor, acc_predictor.predict(inputs)
    
    def _next(self, archive, predictor, K):
        """Override parent method to use config values"""
        # Get non-dominated architectures from archive
        F = np.column_stack(([x[1] for x in archive], [x[2] for x in archive]))
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_X = np.array([self.search_space.encode(x[0]) for x in archive])[front]

        # Initialize the candidate finding optimization problem
        problem = AuxiliarySingleLevelProblem(
            self.search_space, predictor, self.config['search']['sec_obj'],
            {'n_classes': self.config['dataset']['n_classes'], 'model_path': self.supernet_path})

        # Initiate multi-objective solver with config values
        method = get_algorithm(
            "nsga2", 
            pop_size=self.config['evolutionary']['pop_size'], 
            sampling=nd_X,
            crossover=get_crossover("int_two_point", prob=self.config['evolutionary']['crossover_prob']),
            mutation=get_mutation("int_pm", eta=self.config['evolutionary']['mutation_eta']),
            eliminate_duplicates=True)

        # Run optimization
        res = minimize(
            problem, method, 
            termination=('n_gen', self.config['evolutionary']['n_gens']), 
            save_history=True, verbose=True)
        
        # Check for duplicates and select candidates
        not_duplicate = np.logical_not(
            [x in [x[0] for x in archive] for x in [self.search_space.decode(x) for x in res.pop.get("X")]])

        # Form subset selection problem
        indices = self._subset_selection(res.pop[not_duplicate], F[front, 1], K)
        pop = res.pop[not_duplicate][indices]

        candidates = []
        for x in pop.get("X"):
            candidates.append(self.search_space.decode(x))

        return candidates, predictor.predict(pop.get("X"))

    def _create_eval_input_csv(self, archs, filepath, iteration):
        """Create CSV input file for SLURM job array"""
        lines = []
        for i, arch in enumerate(archs):
            lines.append({
                'index': i,
                'iteration': iteration, 
                'ks': json.dumps(arch['ks']),
                'e': json.dumps(arch['e']),
                'd': json.dumps(arch['d']),
                'r': arch['r']
            })
        
        df = pd.DataFrame(lines)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        
    def _create_evaluation_job_file(self, num_archs, iteration):
        """Create SLURM job file for architecture evaluation"""
        log_dir = os.path.join(self.logs_dir, f'iteration_{iteration}')
        os.makedirs(log_dir, exist_ok=True)
        
        job_file_path = os.path.join(self.save_path, f'eval_iter_{iteration}.job')
        eval_input_path = os.path.join(self.save_path, f'eval_input_iter_{iteration}.csv')
        
        batch_script = f"""#!/bin/bash
#SBATCH --job-name={self.job_name}_eval_{iteration}
#SBATCH --nodes={NODES}
#SBATCH -G 1
#SBATCH --cpus-per-task={CORES}
#SBATCH --mem={MEM}
#SBATCH --time={JOB_TIME}
#SBATCH --output={log_dir}/evaluation.%A.%a.log
#SBATCH --error={log_dir}/evaluation_error.%A.%a.log
#SBATCH --array=0-{num_archs-1}
#SBATCH --constraint="{'|'.join(GPUS)}"

module load anaconda3/2023.03
module load cuda/12.1.1

# Execute the evaluation script
conda run -n {ENV_NAME} --no-capture-output python -u evaluator_slurm.py \\
    $SLURM_ARRAY_TASK_ID \\
    -i {eval_input_path} \\
    -o {self.save_path} \\
    --iteration {iteration} \\
    --data {self.data} \\
    --dataset {self.dataset} \\
    --n_classes {self.n_classes} \\
    --supernet_path {self.supernet_path} \\
    --num_workers {self.n_workers} \\
    --valid_size {self.vld_size} \\
    --trn_batch_size {self.trn_batch_size} \\
    --vld_batch_size {self.vld_batch_size} \\
    --n_epochs {self.n_epochs} \\
    --latency {self.latency} \\
    --test {self.test}
"""
        
        with open(job_file_path, 'w') as f:
            f.write(batch_script)
        
        return job_file_path

    def _submit_job(self, job_file):
        """Submit SLURM job and return job ID"""
        print(f'    Submitting job: {job_file}')
        result = os.popen(f"sbatch {job_file}").read()
        match = re.search(r'Submitted batch job (\d+)', result)
        
        if match:
            job_id = match.group(1)
            print(f"    Job submitted with ID: {job_id}")
            return job_id
        else:
            raise RuntimeError(f"Failed to submit job: {result}")

    def _wait_for_job_completion(self, job_id, job_name):
        """Wait for SLURM job to complete"""
        print(f'    Waiting for job {job_name} (ID: {job_id}) to complete...')
        
        while True:
            time.sleep(30)  # Check every 30 seconds
            p = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE)
            text = p.stdout.read().decode('utf-8')
            jobs = text.split('\n')[1:-1]  # Remove header and empty line
            
            if len(jobs) == 0:
                print(f'    Job {job_name} completed!')
                break

    def _parse_evaluation_results(self, archs, iteration):
        """Parse results from completed SLURM evaluation jobs"""
        top1_err, complexity = [], []
        failed_count = 0
        
        for i, arch in enumerate(archs):
            result_file = os.path.join(self.save_path, f'iter_{iteration}', f'arch_{i}.stats')
            
            try:
                with open(result_file, 'r') as f:
                    stats = json.load(f)
                
                top1_err.append(100 - stats['top1'])
                complexity.append(stats[self.sec_obj])
                
            except FileNotFoundError:
                print(f"    Warning: Result file not found for architecture {i}")
                # Assign bad fitness values for failed evaluations
                top1_err.append(100)  # Very poor accuracy
                complexity.append(1e6 if self.sec_obj == 'flops' else 1000)  # Very high complexity
                failed_count += 1
                
                # Save failed architecture for potential retry
                failed_dir = os.path.join(self.save_path, 'failed')
                os.makedirs(failed_dir, exist_ok=True)
                failed_file = os.path.join(failed_dir, f'iter_{iteration}_arch_{i}.json')
                with open(failed_file, 'w') as f:
                    json.dump(arch, f)
        
        if failed_count > 0:
            print(f"    Warning: {failed_count} evaluations failed and were assigned poor fitness")
            
        return top1_err, complexity

    def _create_surrogate_training_job(self, iteration):
        """Create SLURM job file for surrogate model training"""
        log_dir = os.path.join(self.logs_dir, f'iteration_{iteration}', 'surrogate')
        os.makedirs(log_dir, exist_ok=True)
        
        job_file_path = os.path.join(self.save_path, f'surrogate_train_iter_{iteration}.job')
        
        # Get path to surrogate trainer script
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'surrogate_trainer_nsga.py')
        
        batch_script = f"""#!/bin/bash
#SBATCH --job-name={self.job_name}_surr_{iteration}
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem={self.config['surrogate']['train_memory']}
#SBATCH --time={self.config['surrogate']['train_job_time']}
#SBATCH --output={log_dir}/surrogate_train.%j.log
#SBATCH --error={log_dir}/surrogate_train_error.%j.log
#SBATCH --constraint="{'|'.join(self.config['slurm']['gpu_types'])}"

module load anaconda3/2023.03
module load cuda/12.1.1

conda run -n {self.config['slurm']['env_name']} python -u {script_path} {iteration} {self.save_path} --predictor {self.config['search']['predictor']}
"""
        
        with open(job_file_path, 'w') as f:
            f.write(batch_script)
        
        return job_file_path

    def _evaluate(self, archs, it):
        """SLURM-based evaluation of architectures"""
        print(f'Evaluating iteration {it} with {len(archs)} architectures...')
        
        # Create input CSV for SLURM job array
        eval_input_path = os.path.join(self.save_path, f'eval_input_iter_{it}.csv')
        self._create_eval_input_csv(archs, eval_input_path, it)
        
        # Create and submit evaluation job
        job_file = self._create_evaluation_job_file(len(archs), it)
        job_id = self._submit_job(job_file)
        
        # Wait for completion
        self._wait_for_job_completion(job_id, f"{self.job_name}_eval_{it}")
        
        # Parse and return results
        return self._parse_evaluation_results(archs, it)

    def _fit_acc_predictor(self, archive):
        """Fit accuracy predictor using SLURM job for surrogate training if needed"""
        inputs = np.array([self.search_space.encode(x[0]) for x in archive])
        targets = np.array([x[1] for x in archive])
        assert len(inputs) > len(inputs[0]), "# of training samples have to be > # of dimensions"

        # For now, use local training (can be extended to SLURM if needed)
        acc_predictor = get_acc_predictor(self.predictor, inputs, targets)
        
        return acc_predictor, acc_predictor.predict(inputs)

    def search(self):
        """Main search loop with SLURM job management"""
        print(f"Starting NSGANetV3 search with SLURM backend")
        print(f"Save path: {self.save_path}")
        print(f"SLURM configuration:")
        print(f"  - Job name: {self.job_name}")
        print(f"  - Nodes: {NODES}, Cores: {CORES}, Memory: {MEM}")
        print(f"  - Job time: {JOB_TIME}")
        print(f"  - GPU types: {GPUS}")
        print(f"  - Conda environment: {ENV_NAME}")
        
        return super().search()


def main(args):
    """Main function with SLURM-specific argument parsing"""
    # Convert args to kwargs for MSuNASSLURM
    kwargs = vars(args)
    config_path = kwargs.pop('config', None)
    
    # Create and run SLURM-enabled MSuNAS
    engine = MSuNASSLURM(kwargs, config_path=config_path)
    engine.search()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NSGANetV3 with SLURM backend')
    
    # Configuration file argument
    parser.add_argument('--config', type=str, default=None,
                        help='path to TOML configuration file')
    
    # Core arguments that are commonly changed via command line
    parser.add_argument('--save', type=str, required=True,
                        help='location of dir to save results')
    parser.add_argument('--data', type=str, required=True,
                        help='location of the data corpus')
    parser.add_argument('--supernet_path', type=str, required=True,
                        help='file path to supernet weights')
    
    # Optional overrides
    parser.add_argument('--resume', type=str, default=None,
                        help='resume search from a checkpoint')
    parser.add_argument('--sec_obj', type=str, default=None,
                        help='second objective to optimize simultaneously')
    parser.add_argument('--iterations', type=int, default=None,
                        help='number of search iterations')
    parser.add_argument('--dataset', type=str, default=None,
                        help='name of the dataset (imagenet, cifar10, cifar100, ...)')
    parser.add_argument('--predictor', type=str, default=None,
                        help='which accuracy predictor model to fit (rbf/gp/cart/mlp/as)')
    parser.add_argument('--n_epochs', type=int, default=None,
                        help='number of epochs for CNN training')
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluation performance on testing set')
    parser.add_argument('--latency', type=str, default=None,
                        help='latency measurement settings')
    
    # Legacy arguments for compatibility (will be deprecated)
    parser.add_argument('--n_doe', type=int, default=None, help='initial sample size for DOE')
    parser.add_argument('--n_iter', type=int, default=None, help='number of architectures per iteration')
    parser.add_argument('--n_gpus', type=int, default=None, help='total number of available gpus (deprecated)')
    parser.add_argument('--gpu', type=int, default=None, help='number of gpus per job (deprecated)')
    parser.add_argument('--n_classes', type=int, default=None, help='number of classes')
    parser.add_argument('--n_workers', type=int, default=None, help='number of workers for dataloader')
    parser.add_argument('--vld_size', type=int, default=None, help='validation set size')
    parser.add_argument('--trn_batch_size', type=int, default=None, help='train batch size')
    parser.add_argument('--vld_batch_size', type=int, default=None, help='validation batch size')
    
    cfgs = parser.parse_args()
    main(cfgs)