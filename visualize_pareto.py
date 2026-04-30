import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=str, required=True, help="path to iter_*.stats file")
    parser.add_argument("--output", type=str, default="pareto_front.png", help="output image path")
    parser.add_argument(
        "--pareto_json",
        type=str,
        default=None,
        help="optional path to save sorted Pareto points as json",
    )
    args = parser.parse_args()
    with open(args.stats, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    archive = payload["archive"]
    errors = np.array([row[1] for row in archive], dtype=float)
    complexity = np.array([row[2] for row in archive], dtype=float)
    F = np.column_stack((complexity, errors))
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    pareto = F[front]
    order = np.argsort(pareto[:, 0])
    pareto_sorted = pareto[order]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(F[:, 0], F[:, 1], s=12, color="gray", alpha=0.4, label="archive")
    ax.scatter(
        pareto_sorted[:, 0], pareto_sorted[:, 1], s=28, color="red", label="pareto points"
    )
    ax.plot(pareto_sorted[:, 0],pareto_sorted[:, 1],color="black",linewidth=1.5,drawstyle="steps-post",
        label="pareto front",
    )
    ax.set_xlabel("complexity")
    ax.set_ylabel("top1 error")
    ax.set_title("Pareto Front (Non-dominated)")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    if args.pareto_json:
        rows = [
            {"complexity": float(row[0]), "top1_error": float(row[1])}
            for row in pareto_sorted
        ]
        with open(args.pareto_json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
    print(f"Archive points: {len(F)}")
    print(f"Pareto points: {len(pareto_sorted)}")
    print(f"Saved pareto plot to {args.output}")


if __name__ == "__main__":
    main()
