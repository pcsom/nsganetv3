import os
import json
import argparse
import numpy as np
from pymoo.decomposition.asf import ASF
from pymoo.visualization.scatter import Scatter
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
from supernet_free import architecture_to_model_config

_DEBUG = False


def main(args):
    # preferences
    if args.prefer is not None:
        preferences = {}
        for p in args.prefer.split("+"):
            k, v = p.split("#")
            if k == 'top1':
                preferences[k] = 100 - float(v)  # assuming top-1 accuracy
            else:
                preferences[k] = float(v)
        weights = np.fromiter(preferences.values(), dtype=float)

    archive = json.load(open(args.expr))['archive']
    subnets, top1, sec_obj = [v[0] for v in archive], [v[1] for v in archive], [v[2] for v in archive]
    sort_idx = np.argsort(top1)
    F = np.column_stack((top1, sec_obj))[sort_idx, :]
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    pf = F[front, :]
    ps = np.array(subnets)[sort_idx][front]

    if args.prefer is not None:
        I =ASF().do(pf, weights).argsort()[:args.n]
    else:
        dm = HighTradeoffPoints(n_survive=args.n)
        I = dm.do(pf)

    # always add most accurate architectures
    I = np.append(I, 0)

    for idx in I:
        save = os.path.join(args.save, "net-flops@{:.0f}".format(pf[idx, 1]))
        os.makedirs(save, exist_ok=True)
        with open(os.path.join(save, "net.subnet"), 'w') as handle:
            json.dump(ps[idx], handle)
        with open(os.path.join(save, "net.config"), 'w') as handle:
            json.dump(architecture_to_model_config(ps[idx], n_classes=args.n_classes), handle, indent=2)
        module_text = (
            "import json\n"
            "import torch\n"
            "from supernet_free import SurrogateNet\n\n"
            "def build_model(config_path='net.subnet', n_classes={n_classes}):\n"
            "    with open(config_path, 'r', encoding='utf-8') as handle:\n"
            "        arch = json.load(handle)\n"
            "    return SurrogateNet(arch, n_classes=n_classes)\n"
        ).format(n_classes=args.n_classes)
        with open(os.path.join(save, "model.py"), "w", encoding="utf-8") as handle:
            handle.write(module_text)

    if _DEBUG:
        print(ps[I])
        plot = Scatter()
        plot.add(pf, alpha=0.2)
        plot.add(pf[I, :], color="red", s=100)
        plot.show()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='.tmp',
                        help='location of dir to save')
    parser.add_argument('--expr', type=str, default='',
                        help='location of search experiment dir')
    parser.add_argument('--prefer', type=str, default=None,
                        help='preferences in choosing architectures (top1#80+flops#150)')
    parser.add_argument('-n', type=int, default=1,
                        help='number of architectures desired')
    parser.add_argument('--supernet_path', type=str, default='./data/ofa_mbv3_d234_e346_k357_w1.0',
                        help='file path to supernet weights')
    parser.add_argument('--n_classes', type=int, default=102,
                        help='number of target classes for exported models')

    cfgs = parser.parse_args()
    main(cfgs)
