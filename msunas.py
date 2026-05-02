import os
import json
import argparse
import time
import numpy as np
from utils import get_correlation
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from search_space.ofa import OFASearchSpace
from acc_predictor.factory import get_acc_predictor
from utils import MySampling, BinaryCrossover, MyMutation
from supernet_free import ComplexityProfiler
from surrogate_validation import load_toml_config

_DEBUG = False
if _DEBUG: from pymoo.visualization.scatter import Scatter


class MSuNAS:
    def __init__(self, kwargs):
        self.search_space = OFASearchSpace()
        config_path = kwargs.pop('config', 'config/nsganetv3_config.toml')
        self.config = load_toml_config(config_path) if os.path.exists(config_path) else {}
        self.save_path = kwargs.pop('save', '.tmp')
        os.makedirs(self.save_path, exist_ok=True)
        self.resume = kwargs.pop('resume', None)
        self.offline_data = kwargs.pop('offline_data', None)
        search_cfg = self.config.get('search', {})
        dataset_cfg = self.config.get('dataset', {})
        training_cfg = self.config.get('training', {})
        evo_cfg = self.config.get('evolutionary', {})
        self.sec_obj = kwargs.pop('sec_obj', None) or search_cfg.get('sec_obj', 'flops')
        self.iterations = kwargs.pop('iterations', None) or search_cfg.get('iterations', 30)
        self.n_doe = kwargs.pop('n_doe', None) or search_cfg.get('n_doe', 100)
        self.n_iter = kwargs.pop('n_iter', None) or search_cfg.get('n_iter', 8)
        self.predictor = kwargs.pop('predictor', None) or search_cfg.get('predictor', 'rbf')
        self.pop_size = evo_cfg.get('pop_size', 40)
        self.n_gens = evo_cfg.get('n_gens', 20)
        self.crossover_prob = evo_cfg.get('crossover_prob', 0.9)
        self.mutation_eta = evo_cfg.get('mutation_eta', 1.0)
        self.n_gpus = kwargs.pop('n_gpus', 1)
        self.gpu = kwargs.pop('gpu', 1)
        self.data = kwargs.pop('data', '../data')
        self.dataset = kwargs.pop('dataset', None) or dataset_cfg.get('dataset', 'imagenet')
        self.n_classes = kwargs.pop('n_classes', None) or dataset_cfg.get('n_classes', 1000)
        self.n_workers = kwargs.pop('n_workers', None) or training_cfg.get('n_workers', 6)
        self.vld_size = kwargs.pop('vld_size', None) or dataset_cfg.get('vld_size', 10000)
        self.trn_batch_size = kwargs.pop('trn_batch_size', None) or training_cfg.get('trn_batch_size', 96)
        self.vld_batch_size = kwargs.pop('vld_batch_size', None) or training_cfg.get('vld_batch_size', 250)
        self.n_epochs = kwargs.pop('n_epochs', None) or dataset_cfg.get('n_epochs', 5)
        self.test = kwargs.pop('test', dataset_cfg.get('test', False))
        if self.offline_data is None:
            self.offline_data = dataset_cfg.get('ground_truth_csv')
        self.supernet_path = kwargs.pop('supernet_path', './data/ofa_mbv3_d234_e346_k357_w1.0')
        self.latency = self.sec_obj if "cpu" in self.sec_obj or "gpu" in self.sec_obj else None
        self.profiler = ComplexityProfiler(n_classes=self.n_classes)

    @staticmethod
    def _log(message):
        print(f"[msunas] {message}", flush=True)
    def search(self):

        if self.resume:
            self._log(f"loading archive from resume dir: {self.resume}")
            archive = self._resume_from_dir()
        elif self.offline_data:
            from offline_loader import load_offline_ground_truth
            self._log(f"loading offline archive from: {self.offline_data}")
            archive = load_offline_ground_truth(
                self.offline_data, self.sec_obj, self.n_classes, self.supernet_path
            )
            assert len(archive) > 0, "Offline data loaded an empty archive."
        else:
            raise RuntimeError("Virtual search requires --offline_data or --resume as initial archive.")
        self._log(f"archive size: {len(archive)}")

        ref_pt = np.array([np.max([x[1] for x in archive]), np.max([x[2] for x in archive])])

        for it in range(1, self.iterations + 1):
            iter_start= time.time()
            self._log(f"iter {it}/{self.iterations}: fitting surrogate on {len(archive)} samples")
            fit_start= time.time()

            acc_predictor, a_top1_err_pred = self._fit_acc_predictor(archive)
            self._log(f"iter {it}: surrogate fit done in {time.time() - fit_start:.1f}s ({acc_predictor.name})")
            self._log(f"iter {it}: running candidate search (pop={self.pop_size}, gens={self.n_gens})")
            next_start =time.time()

            candidates, c_top1_err_pred = self._next(archive, acc_predictor, self.n_iter)
            self._log(f"iter {it}: candidate search done in {time.time() - next_start:.1f}s, candidates={len(candidates)}")
            self._log(f"iter {it}: virtual-evaluating candidates")
            eval_start =time.time()

            c_top1_err, complexity = self._evaluate(candidates, predictor=acc_predictor, it=it)
            self._log(f"iter {it}: evaluation done in {time.time() - eval_start:.1f}s")

            rmse, rho, tau = get_correlation(
                np.vstack((a_top1_err_pred, c_top1_err_pred)), np.array([x[1] for x in archive] + c_top1_err))

            for member in zip(candidates, c_top1_err, complexity):
                archive.append(member)

            hv = self._calc_hv(
                ref_pt, np.column_stack(([x[1] for x in archive], [x[2] for x in archive])))

            self._log("iter {}: hv = {:.2f}".format(it, hv))
            self._log("iter {}: fitting {} => RMSE={:.4f}, Spearman={:.4f}, Kendall={:.4f}".format(it, self.predictor, rmse, rho, tau))
            with open(os.path.join(self.save_path, "iter_{}.stats".format(it)), "w") as handle:
                json.dump({'archive': archive, 'candidates': archive[-self.n_iter:], 'hv': hv,'surrogate': {
                               'model': self.predictor, 'name': acc_predictor.name,
                               'winner': acc_predictor.winner if self.predictor == 'as' else acc_predictor.name,
                               'rmse': rmse, 'rho': rho, 'tau': tau},
                           'evaluation_mode': 'virtual'}, handle)
            self._log(f"iter {it}: wrote {os.path.join(self.save_path, f'iter_{it}.stats')}")
            self._log(f"iter {it}: total elapsed {time.time() - iter_start:.1f}s")
            if _DEBUG:
                # plot
                plot = Scatter(legend={'loc': 'lower right'})
                F = np.full((len(archive), 2), np.nan)
                F[:, 0] = np.array([x[2] for x in archive])  # second obj. (complexity)
                F[:, 1] = 100 - np.array([x[1] for x in archive])  # top-1 accuracy
                plot.add(F, s=15, facecolors='none', edgecolors='b', label='archive')
                F = np.full((len(candidates), 2), np.nan)
                F[:, 0] = np.array(complexity)
                F[:, 1] = 100 - np.array(c_top1_err)
                plot.add(F, s=30, color='r', label='candidates evaluated')
                F = np.full((len(candidates), 2), np.nan)
                F[:, 0] = np.array(complexity)
                F[:, 1] = 100 - c_top1_err_pred[:, 0]
                plot.add(F, s=20, facecolors='none', edgecolors='g', label='candidates predicted')
                plot.save(os.path.join(self.save_path, 'iter_{}.png'.format(it)))

        return

    def _resume_from_dir(self):
        """ resume search from a previous iteration """
        import glob

        archive = []
        net_files = glob.glob(os.path.join(self.resume, "net_*.subnet"))
        for file in net_files:
            arch = json.load(open(file))
            pre, ext = os.path.splitext(file)
            stats = json.load(open(pre + ".stats"))
            archive.append((arch, 100 - stats['top1'], stats[self.sec_obj]))
        if archive:
            return archive
        iter_files = glob.glob(os.path.join(self.resume, "iter_*.stats"))
        if iter_files:
            def _iter_num(path):
                base = os.path.basename(path)
                return int(base.split("_")[1].split(".")[0])
            latest = max(iter_files, key=_iter_num)
            payload = json.load(open(latest, "r", encoding="utf-8"))
            for row in payload.get("archive", []):
                if len(row) >= 3:
                    archive.append((row[0], float(row[1]), float(row[2])))
            self._log(f"resumed {len(archive)} archive entries from {latest}")

        return archive

    def _evaluate(self, archs, predictor, it):
        _ = it
        encoded = np.array([self.search_space.encode(arch) for arch in archs], dtype=float)
        top1_err = predictor.predict(encoded).reshape(-1).tolist()
        complexity = [self.profiler.profile(arch, self.sec_obj) for arch in archs]
        return top1_err, complexity

    def _fit_acc_predictor(self, archive):
        inputs = np.array([self.search_space.encode(x[0]) for x in archive])
        targets = np.array([x[1] for x in archive])
        assert len(inputs) > len(inputs[0]), "# of training samples have to be > # of dimensions"

        acc_predictor = get_acc_predictor(self.predictor, inputs, targets)

        return acc_predictor, acc_predictor.predict(inputs)

    def _next(self, archive, predictor, K):
        """ searching for next K candidate for high-fidelity evaluation (lower level) """

        # the following lines corresponding to Algo 1 line 10 / Fig. 3(b) in the paper
        # get non-dominated architectures from archive
        F = np.column_stack(([x[1] for x in archive], [x[2] for x in archive]))
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        # non-dominated arch bit-strings
        nd_X = np.array([self.search_space.encode(x[0]) for x in archive])[front]

        # initialize the candidate finding optimization problem
        problem = AuxiliarySingleLevelProblem(
            self.search_space, predictor, self.sec_obj, self.n_classes, profiler=self.profiler)

        method = NSGA2(
            pop_size=self.pop_size,
            sampling=nd_X,
            crossover=TwoPointCrossover(prob=self.crossover_prob, repair=RoundingRepair()),
            mutation=PolynomialMutation(eta=self.mutation_eta, repair=RoundingRepair(), vtype=int),
            eliminate_duplicates=True,
        )

        # kick-off the search
        res = minimize(
            problem, method, termination=('n_gen', self.n_gens), save_history=True, verbose=True)
        
        decoded = [self.search_space.decode(np.asarray(x_).astype(int)) for x_ in res.pop.get("X")]
        archive_archs = [x[0] for x in archive]
        not_duplicate =np.array([d not in archive_archs for d in decoded])
        if not np.any(not_duplicate):
            not_duplicate= np.ones(len(decoded), dtype=bool)
        uniq_pop = res.pop[not_duplicate]
        uniq_indices = self._subset_selection(uniq_pop, F[front, 1], K)
        if uniq_indices.size == 0:
            uniq_indices = np.arange(min(K, len(uniq_pop)), dtype=int)
        pop = uniq_pop[uniq_indices]
        candidates = [self.search_space.decode(np.asarray(x_).astype(int)) for x_ in pop.get("X")]
        x_int= np.asarray(pop.get("X")).astype(int)
        return candidates,predictor.predict(x_int)

    @staticmethod
    def _subset_selection(pop, nd_F, K):
        problem = SubsetProblem(pop.get("F")[:, 1], nd_F, K)
        algorithm = GA(
            pop_size=100, sampling=MySampling(), crossover=BinaryCrossover(),
            mutation=MyMutation(), eliminate_duplicates=True)
        res = minimize(problem, algorithm, ('n_gen', 60), verbose=False)
        mask = np.asarray(res.X, dtype=bool).reshape(-1)
        n = len(pop)
        if mask.size != n:
            return np.arange(min(K, n), dtype=int)
        selected = np.flatnonzero(mask)
        if selected.size == 0:
            return np.arange(min(K, n), dtype=int)
        if selected.size > K:
            selected = selected[:K]
        return selected

    @staticmethod
    def _calc_hv(ref_pt, F, normalized=True):
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_F = F[front, :]
        ref_point = 1.01 * ref_pt
        hv =HV(ref_point=ref_point)(nd_F)
        if normalized:
            hv = hv / np.prod(ref_point)
        return hv


class AuxiliarySingleLevelProblem(Problem):
    """ The optimization problem for finding the next N candidate architectures """

    def __init__(self, search_space, predictor, sec_obj='flops', n_classes=1000, profiler=None):
        sample_vec = np.array(search_space.encode(search_space.sample(1)[0]), dtype=int)
        n_var = len(sample_vec)
        xl = np.zeros(n_var, dtype=int)
        xu = np.zeros(n_var, dtype=int)
        offset = 0
        for _ in range(search_space.num_blocks):
            xu[offset] = len(search_space.depth) - 1
            offset += 1
            xu[offset:offset + max(search_space.depth)] = len(search_space.kernel_size) - 1
            offset += max(search_space.depth)
            xu[offset:offset + max(search_space.depth)] = len(search_space.exp_ratio) - 1
            offset += max(search_space.depth)
        xu[-1] = len(search_space.resolution) - 1
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=xl, xu=xu, vtype=int)
        self.ss = search_space
        self.predictor = predictor
        self.sec_obj = sec_obj
        if isinstance(n_classes, dict):
            n_classes = n_classes.get('n_classes', 1000)
        self.profiler = profiler if profiler is not None else ComplexityProfiler(n_classes=n_classes)
        self._eval_counter = 0

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], self.n_obj), np.nan)
        x_int = np.asarray(x).astype(int)
        chunk_size =32
        chunks =[]
        for start in range(0, x_int.shape[0], chunk_size):
            end = min(start + chunk_size, x_int.shape[0])
            chunk_pred = np.asarray(self.predictor.predict(x_int[start:end])).reshape(end - start, -1)
            chunks.append(chunk_pred)
            print(
                f"[msunas] predictor progress: {end}/{x_int.shape[0]} candidates scored",
                flush=True,
            )
        prediction= np.vstack(chunks) if chunks else np.zeros((0, 1), dtype=float)
        top1_err = prediction[:, 0] if prediction.size else np.zeros((0,), dtype=float)
        for i,(_x, err) in enumerate(zip(x_int, top1_err)):
            config = self.ss.decode(_x)
            f[i, 0] = err
            f[i, 1] = self.profiler.profile(config, self.sec_obj)
            self._eval_counter += 1
            if self._eval_counter % 20== 0:
                print(
                    f"[msunas] auxiliary eval progress: {self._eval_counter} architectures profiled",
                    flush=True,
                )
        out["F"] = f


class SubsetProblem(Problem):
    """ select a subset to diversify the pareto front """
    def __init__(self, candidates, archive, K):
        super().__init__(n_var=len(candidates), n_obj=1,
                         n_constr=1, xl=0, xu=1, vtype=bool)
        self.archive = archive
        self.candidates = candidates
        self.n_max = K

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], 1), np.nan)
        g = np.full((x.shape[0], 1), np.nan)
        for i, _x in enumerate(x):
            mask= np.asarray(_x, dtype=bool)
            tmp = np.sort(np.concatenate((self.archive, self.candidates[mask])))
            f[i, 0] = np.std(np.diff(tmp)) if len(tmp) > 1 else 0.0
            g[i, 0] = (self.n_max - int(np.sum(mask))) ** 2
        out["F"] = f
        out["G"] = g


def main(args):
    engine = MSuNAS(vars(args))
    engine.search()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/nsganetv3_config.toml',
                        help='path to TOML search configuration')
    parser.add_argument('--save', type=str, default='.tmp',
                        help='location of dir to save')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume search from a checkpoint')
    parser.add_argument('--offline_data', type=str, default=None,
                        help='path to offline ground truth CSV archive bootstrap')
    parser.add_argument('--sec_obj', type=str, default=None,
                        help='second objective to optimize simultaneously')
    parser.add_argument('--iterations', type=int, default=None,
                        help='number of search iterations')
    parser.add_argument('--n_doe', type=int, default=None,
                        help='initial sample size for DOE')
    parser.add_argument('--n_iter', type=int, default=None,
                        help='number of architectures to high-fidelity eval (low level) in each iteration')
    parser.add_argument('--predictor', type=str, default=None,
                        help='which accuracy predictor model to fit (rbf/gp/cart/mlp/as)')
    parser.add_argument('--n_gpus', type=int, default=8,
                        help='total number of available gpus')
    parser.add_argument('--gpu', type=int, default=1,
                        help='number of gpus per evaluation job')
    parser.add_argument('--data', type=str, default='../data',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default=None,
                        help='name of the dataset (imagenet, cifar10, cifar100, ...)')
    parser.add_argument('--n_classes', type=int, default=None,
                        help='number of classes of the given dataset')
    parser.add_argument('--supernet_path', type=str, default='./data/ofa_mbv3_d234_e346_k357_w1.0',
                        help='file path to supernet weights')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='number of workers for dataloader per evaluation job')
    parser.add_argument('--vld_size', type=int, default=None,
                        help='validation set size, randomly sampled from training set')
    parser.add_argument('--trn_batch_size', type=int, default=None,
                        help='train batch size for training')
    parser.add_argument('--vld_batch_size', type=int, default=None,
                        help='test batch size for inference')
    parser.add_argument('--n_epochs', type=int, default=None,
                        help='number of epochs for CNN training')
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluation performance on testing set')
    cfgs = parser.parse_args()
    main(cfgs)

