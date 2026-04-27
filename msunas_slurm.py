import os
import json
import argparse

from msunas import MSuNAS
from surrogate_validation import load_toml_config


JOB_NAME = 'nsganetv3'
NODES = 1
CORES = 8
MEM = '24GB'
JOB_TIME = '08:00:00'
ENV_NAME = 'nsganetv2-llm'
GPUS = ["V100-16GB", "V100-32GB", "L40S", "A100-40GB", "H100", "A40", "H200"]


class MSuNASSLURM(MSuNAS):
    """thin SLURM-aware wrapper; virtual mode reuses the parent search loop"""

    def __init__(self, kwargs, config_path=None):
        if config_path:
            kwargs.setdefault('config', config_path)
        super().__init__(kwargs)
        self.job_name = f'{JOB_NAME}_{os.path.basename(self.save_path) or "search"}'
        self.logs_dir = os.path.join(self.save_path, 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        slurm_cfg = self.config.get('slurm', {}) if isinstance(self.config, dict) else {}
        self.slurm_nodes = slurm_cfg.get('nodes', NODES)
        self.slurm_cores = slurm_cfg.get('cores', CORES)
        self.slurm_mem = slurm_cfg.get('memory', MEM)
        self.slurm_time = slurm_cfg.get('job_time', JOB_TIME)
        self.slurm_env = slurm_cfg.get('env_name', ENV_NAME)
        self.slurm_gpus = slurm_cfg.get('gpu_types', GPUS)
        self.evaluation_mode = self.config.get('search', {}).get('evaluation_mode', 'virtual') \
            if isinstance(self.config, dict) else 'virtual'
        self._dump_runtime_summary()
    def _dump_runtime_summary(self):
        summary = {
            'save_path': self.save_path,
            'evaluation_mode': self.evaluation_mode,
            'dataset': self.dataset,
            'n_classes': self.n_classes,
            'sec_obj': self.sec_obj,
            'predictor': self.predictor,
            'iterations': self.iterations,
            'n_iter': self.n_iter,
            'pop_size': self.pop_size,
            'n_gens': self.n_gens,
            'offline_data': self.offline_data,
            'slurm': {'env_name': self.slurm_env,'nodes': self.slurm_nodes,'cores': self.slurm_cores,'memory': self.slurm_mem,'time': self.slurm_time,'gpu_types': self.slurm_gpus,
            },
        }
        with open(os.path.join(self.save_path, 'runtime_summary.json'), 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)

    def search(self):
        print('=' * 72)
        print(f'NSGANetV3 search ({self.evaluation_mode} mode) on {self.dataset}')
        print(f'  save_path={self.save_path}')
        print(f'  predictor={self.predictor}  sec_obj={self.sec_obj}')
        print(f'  iterations={self.iterations}  n_iter={self.n_iter}')
        print(f'  pop_size={self.pop_size}  n_gens={self.n_gens}')
        print(f'  offline_data={self.offline_data}')
        print(f'  conda_env={self.slurm_env}')
        print('=' * 72)
        if self.evaluation_mode != 'virtual':
            raise NotImplementedError(
                "physical evaluation is unsupported; configure search.evaluation_mode = 'virtual'")
        return super().search()


def main(args):
    kwargs = vars(args)
    config_path = kwargs.pop('config', None) or 'config/nsganetv3_config.toml'
    engine = MSuNASSLURM(kwargs, config_path=config_path)
    engine.search()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NSGANetV3 SLURM-aware virtual search wrapper')
    parser.add_argument('--config', type=str, default=None, help='path to TOML configuration')
    parser.add_argument('--save', type=str, required=True, help='directory to save search artifacts')
    parser.add_argument('--data', type=str, default='../data', help='dataset root (unused in virtual mode)')
    parser.add_argument('--supernet_path', type=str,default='./data/ofa_mbv3_d234_e346_k357_w1.0',help='legacy supernet path (ignored in virtual mode)')
    parser.add_argument('--offline_data', type=str, default=None,help='ground-truth CSV for archive bootstrap')
    parser.add_argument('--resume', type=str, default=None, help='resume search dir')
    parser.add_argument('--sec_obj', type=str, default=None, help='secondary objective (flops/params)')
    parser.add_argument('--iterations', type=int, default=None, help='number of search iterations')
    parser.add_argument('--dataset', type=str, default=None, help='dataset name')
    parser.add_argument('--predictor', type=str, default=None, help='surrogate model id')
    parser.add_argument('--evaluation_mode', type=str, default=None, help='virtual (only supported)')
    parser.add_argument('--n_epochs', type=int, default=None, help='unused (virtual mode)')
    parser.add_argument('--test', action='store_true', default=False, help='unused (virtual mode)')
    parser.add_argument('--n_doe', type=int, default=None)
    parser.add_argument('--n_iter', type=int, default=None)
    parser.add_argument('--n_gpus', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--n_classes', type=int, default=None)
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--vld_size', type=int, default=None)
    parser.add_argument('--trn_batch_size', type=int, default=None)
    parser.add_argument('--vld_batch_size', type=int, default=None)
    cfgs = parser.parse_args()
    main(cfgs)
