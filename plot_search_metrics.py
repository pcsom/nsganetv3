import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_iter_stats(expr_dir):
    stats_files = sorted(
        glob.glob(os.path.join(expr_dir, "iter_*.stats")),
        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
    )
    rows = []
    for p in stats_files:
        it = int(os.path.basename(p).split("_")[1].split(".")[0])
        with open(p, "r") as f:
            s = json.load(f)
        rows.append(
            {
                "iter": it,
                "hv": float(s.get("hv", np.nan)),
                "rmse": float(s.get("surrogate", {}).get("rmse", np.nan)),
                "rho": float(s.get("surrogate", {}).get("rho", np.nan)),
                "tau": float(s.get("surrogate", {}).get("tau", np.nan)),
            }
        )
    return rows


def plot_metrics(rows, out_dir, title_prefix="search"):
    if len(rows) == 0:
        raise RuntimeError("No iter_*.stats found. Provide a completed experiment directory.")
    os.makedirs(out_dir, exist_ok=True)

    its = np.array([r["iter"] for r in rows], dtype=int)
    hv = np.array([r["hv"] for r in rows], dtype=float)
    rmse = np.array([r["rmse"] for r in rows], dtype=float)
    rho = np.array([r["rho"] for r in rows], dtype=float)
    tau = np.array([r["tau"] for r in rows], dtype=float)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].plot(its, hv, marker="o")
    ax[0, 0].set_title("Hypervolume")
    ax[0, 0].set_xlabel("Iteration")
    ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(its, rmse, marker="o", color="tab:red")
    ax[0, 1].set_title("Surrogate RMSE")
    ax[0, 1].set_xlabel("Iteration")
    ax[0, 1].grid(alpha=0.3)

    ax[1, 0].plot(its, rho, marker="o", color="tab:green")
    ax[1, 0].set_title("Spearman Rho")
    ax[1, 0].set_xlabel("Iteration")
    ax[1, 0].grid(alpha=0.3)

    ax[1, 1].plot(its, tau, marker="o", color="tab:purple")
    ax[1, 1].set_title("Kendall Tau")
    ax[1, 1].set_xlabel("Iteration")
    ax[1, 1].grid(alpha=0.3)

    fig.suptitle(f"{title_prefix}: surrogate search metrics", fontsize=12)
    fig.tight_layout()

    out_png = os.path.join(out_dir, "search_metrics.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    out_json = os.path.join(out_dir, "search_metrics_summary.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    return out_png, out_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", type=str, required=True, help="experiment directory containing iter_*.stats")
    parser.add_argument("--out", type=str, default=None, help="output directory for plots")
    parser.add_argument("--title", type=str, default="MSuNAS", help="plot title prefix")
    args = parser.parse_args()

    rows = load_iter_stats(args.expr)
    out_dir = args.out or args.expr
    png, js = plot_metrics(rows, out_dir, title_prefix=args.title)
    print(f"Saved plot: {png}")
    print(f"Saved summary: {js}")


if __name__ == "__main__":
    main()
