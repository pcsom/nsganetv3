import argparse
import json
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.visualization.scatter import Scatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=str, required=True, help="path to iter_*.stats file")
    parser.add_argument("--output", type=str, default="pareto_front.png", help="output image path")
    args = parser.parse_args()
    with open(args.stats, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    archive = payload["archive"]
    errors= np.array([row[1] for row in archive], dtype=float)
    complexity= np.array([row[2] for row in archive], dtype=float)
    F= np.column_stack((complexity, errors))
    front = NonDominatedSorting().do(np.column_stack((errors, complexity)), only_non_dominated_front=True)
    plot= Scatter(legend={"loc": "best"})
    plot.add(F, s=12, color="gray", alpha=0.4, label="archive")
    plot.add(F[front], s=26, color="red", label="pareto")
    plot.save(args.output)
    print(f"Saved pareto plot to {args.output}")


if __name__ == "__main__":
    main()
