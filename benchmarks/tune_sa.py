import itertools
import os
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from splitkit import Budget, SplitProblem
from splitkit.io import load_npz
from splitkit.strategies import SimulatedAnnealing

from ._history import unpack_history

DATASET_PATH = "datasets/synthetic/preprocessed/synth_mild_imbalance.npz"
RATIOS = (0.70, 0.15, 0.15)
MAX_EVALS = 300_000
N_RUNS = 10
SEEDS = [42 + i for i in range(N_RUNS)]

GRID = {
    "initial_temp": [1.0, 10.0, 100.0],
    "cooling_rate": [0.99, 0.999, 0.9999],
    "min_temp": [1e-2, 1e-3, 1e-4],
}


if __name__ == "__main__":
    outdir = "results/tuning"
    os.makedirs(outdir, exist_ok=True)
    report_lines = []

    def log(msg: str):
        print(msg)
        report_lines.append(msg)

    log(f"Loading {DATASET_PATH}...")
    data = load_npz(DATASET_PATH)
    problem = SplitProblem.build(data, RATIOS)
    budget = Budget(max_evals=MAX_EVALS)

    keys = list(GRID.keys())
    combinations = list(itertools.product(*(GRID[k] for k in keys)))

    log("Starting Simulated Annealing Grid Search...")
    log(f"Combinations: {len(combinations)}")
    log(f"Runs per combination: {N_RUNS} (Seeds: {SEEDS[0]} to {SEEDS[-1]})")
    log(f"FFE Budget: {MAX_EVALS:,}")
    log("-" * 90)

    results = {}

    t_start_all = time.time()
    for i, values in enumerate(combinations):
        params = dict(zip(keys, values, strict=True))

        costs = []
        histories = []

        for seed in SEEDS:
            res = SimulatedAnnealing(**params).run(
                problem, budget, np.random.default_rng(seed)
            )
            costs.append(res.cost)
            histories.append(res)

        mean_cost = np.mean(costs)
        std_cost = np.std(costs)

        closest_idx = np.argmin(np.abs(np.array(costs) - mean_cost))
        results[values] = {
            "mean": mean_cost,
            "std": std_cost,
            "representative_res": histories[closest_idx]
        }

        log(f"[{i + 1:2d}/{len(combinations)}] T0={params['initial_temp']:>5.1f} | CR={params['cooling_rate']:.4f} | Tmin={params['min_temp']:.4f}  -> Cost: {mean_cost:.4f} ± {std_cost:.4f}")

    log("-" * 90)
    log(f"Grid search completed in {time.time() - t_start_all:.1f}s")

    best_params = min(results.keys(), key=lambda k: results[k]["mean"])
    log(f"\nBEST PARAMS: T0={best_params[0]}, CR={best_params[1]}, Tmin={best_params[2]}")
    log(f"BEST COST:   {results[best_params]['mean']:.4f} ± {results[best_params]['std']:.4f}")

    # Create report
    txt_path = os.path.join(outdir, "sa_grid_search_report.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved text report to {txt_path}")

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle(f"Simulated Annealing Grid Search (Mean over {N_RUNS} runs)", fontsize=16, fontweight="bold")

    styles = {
        GRID["min_temp"][0]: {"color": "#1f77b4", "linewidth": 4.5, "linestyle": "-", "alpha": 0.6},
        GRID["min_temp"][1]: {"color": "#ff7f0e", "linewidth": 2.5, "linestyle": "--", "alpha": 0.9},
        GRID["min_temp"][2]: {"color": "#2ca02c", "linewidth": 1.2, "linestyle": "-", "alpha": 1.0}
    }

    for row, cr in enumerate(GRID["cooling_rate"]):
        for col, t0 in enumerate(GRID["initial_temp"]):
            ax = axes[row, col]

            for tmin in GRID["min_temp"]:
                res = results[t0, cr, tmin]["representative_res"]
                evals, costs = unpack_history(res)
                sty = styles[tmin]

                ax.step(evals, costs, label=f"min_temp={tmin}", color=sty["color"],
                        linewidth=sty["linewidth"], linestyle=sty["linestyle"], alpha=sty["alpha"], where='post')

            ax.set_title(f"T0 = {t0} | CR = {cr}")
            ax.grid(True, alpha=0.3, linestyle=":")

            if row == 2:
                ax.set_xlabel("Function Evaluations")
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1_000:.0f}k" if x >= 1_000 else f"{x:.0f}"))
            if col == 0:
                ax.set_ylabel("Cost")
            if row == 0 and col == 2:
                ax.legend(title="Min Temp")

    plt.tight_layout()
    png_path = os.path.join(outdir, "sa_grid_search.png")
    plt.savefig(png_path, dpi=150)
    print(f"Saved visualization to {png_path}")
