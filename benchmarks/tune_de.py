import itertools
import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.optimizers.de import DifferentialEvolution
from src.preprocessing.common import load_dataset

DATASET_PATH = "datasets/synthetic/preprocessed/synth_mild_imbalance.pkl"
RATIOS = (0.70, 0.15, 0.15)
MAX_EVALS = 300_000
N_RUNS = 10
SEEDS = [42 + i for i in range(N_RUNS)]

GRID = {
    "strategy": [
        "DE/rand/1/bin", "DE/best/1/bin", "DE/rand/1/exp", "DE/best/1/exp",
        "DE/rand/2/bin", "DE/best/2/bin", "DE/rand/2/exp", "DE/best/2/exp"
    ],
    "pop_size": [20, 50, 100],
    "f_weight": [0.2, 0.5, 0.9],
    "crossover_prob": [0.3, 0.5, 0.9]
}


def _unpack_history(result) -> tuple[np.ndarray, np.ndarray]:
    if not result.cost_history:
        return np.array([result.n_evals], dtype=float), np.array([result.cost], dtype=float)

    filtered_evals, filtered_costs = [], []
    best_so_far = float('inf')

    for e, c in result.cost_history:
        if c < best_so_far:
            best_so_far = c
            filtered_evals.append(e)
            filtered_costs.append(c)

    last_e = max(result.n_evals, result.cost_history[-1][0])
    if filtered_evals[-1] < last_e:
        filtered_evals.append(last_e)
        filtered_costs.append(best_so_far)

    return np.asarray(filtered_evals, dtype=float), np.asarray(filtered_costs, dtype=float)


if __name__ == "__main__":
    outdir = "results/tuning"
    os.makedirs(outdir, exist_ok=True)
    report_lines = []

    def log(msg: str):
        print(msg)
        report_lines.append(msg)

    log(f"Loading {DATASET_PATH}...")
    data = load_dataset(DATASET_PATH)

    keys = list(GRID.keys())
    combinations = list(itertools.product(*(GRID[k] for k in keys)))

    log("Starting Differential Evolution Grid Search...")
    log(f"Combinations: {len(combinations)}")
    log(f"Runs per combination: {N_RUNS} (Seeds: {SEEDS[0]} to {SEEDS[-1]})")
    log(f"Total Evaluations: {len(combinations) * N_RUNS:,}")
    log(f"FFE Budget: {MAX_EVALS:,}")
    log("-" * 90)

    results_list = []

    t_start_all = time.time()
    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))

        costs = []
        histories = []

        for seed in SEEDS:
            opt = DifferentialEvolution(
                data=data, ratios=RATIOS, max_evals=MAX_EVALS, seed=seed, **params
            )
            res = opt.optimize(verbose=False)
            costs.append(res.cost)
            histories.append(res)

        mean_cost = np.mean(costs)
        std_cost = np.std(costs)

        closest_idx = np.argmin(np.abs(np.array(costs) - mean_cost))

        results_list.append({
            "params": params,
            "mean": mean_cost,
            "std": std_cost,
            "representative_res": histories[closest_idx]
        })

        log(f"[{i + 1:3d}/{len(combinations)}] {params['strategy']:<15} | Pop={params['pop_size']:<3} | F={params['f_weight']:<3} | CR={params['crossover_prob']:<3} -> Cost: {mean_cost:8.4f} ± {std_cost:.4f}")

    log("-" * 90)
    log(f"Grid search completed in {time.time() - t_start_all:.1f}s")

    results_list.sort(key=lambda x: x["mean"])
    best = results_list[0]

    log(f"\nBEST PARAMS: {best['params']['strategy']}, Pop={best['params']['pop_size']}, F={best['params']['f_weight']}, CR={best['params']['crossover_prob']}")
    log(f"BEST COST:   {best['mean']:.4f} ± {best['std']:.4f}")

    # Create report
    txt_path = os.path.join(outdir, "de_grid_search_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n\n")
        f.write("=== RANKED RESULTS ===\n")
        f.write(f" {'Rank':<4} | {'Strategy':<15} | {'Pop':<5} | {'F':<4} | {'CR':<4} | {'Mean Cost':<10} | {'Std Dev':<8}\n")
        f.write("-" * 75 + "\n")
        for idx, r in enumerate(results_list, 1):
            p = r["params"]
            f.write(f" #{idx:<3} | {p['strategy']:<15} | {p['pop_size']:<5} | {p['f_weight']:<4.1f} | {p['crossover_prob']:<4.1f} | {r['mean']:<10.4f} | ±{r['std']:<8.4f}\n")
    print(f"Saved text report to {txt_path}")

    fig, (ax_best, ax_worst) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Differential Evolution: Top 5 Best vs Top 5 Worst (Representative Runs)", fontsize=16, fontweight="bold")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Plot Top 5 Best
    for idx in range(min(5, len(results_list))):
        r = results_list[idx]
        p = r["params"]
        res = r["representative_res"]
        evals, costs = _unpack_history(res)

        label = f"#{idx + 1}: {p['strategy']} (Pop={p['pop_size']}, F={p['f_weight']}, CR={p['crossover_prob']})"
        ax_best.step(evals, costs, label=label, color=colors[idx % len(colors)], linewidth=2.0, where='post')

    ax_best.set_title("Top 5 BEST Configurations", fontweight="bold", color="green")
    ax_best.set_xlabel("Function Evaluations")
    ax_best.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1_000:.0f}k" if x >= 1_000 else f"{x:.0f}"))
    ax_best.set_ylabel("Cost")
    ax_best.grid(True, alpha=0.3, linestyle=":")
    ax_best.legend()

    # Plot Top 5 Worst
    worst_list = results_list[-5:][::-1] if len(results_list) >= 5 else results_list[::-1]

    for idx, r in enumerate(worst_list):
        p = r["params"]
        res = r["representative_res"]
        evals, costs = _unpack_history(res)

        original_rank = len(results_list) - idx
        label = f"#{original_rank}: {p['strategy']} (Pop={p['pop_size']}, F={p['f_weight']}, CR={p['crossover_prob']})"
        ax_worst.step(evals, costs, label=label, color=colors[idx % len(colors)], linewidth=2.0, where='post')

    ax_worst.set_title("Top 5 WORST Configurations", fontweight="bold", color="red")
    ax_worst.set_xlabel("Function Evaluations")
    ax_worst.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1_000:.0f}k" if x >= 1_000 else f"{x:.0f}"))
    ax_worst.set_ylabel("Cost")
    ax_worst.grid(True, alpha=0.3, linestyle=":")
    ax_worst.legend()

    plt.tight_layout()
    png_path = os.path.join(outdir, "de_grid_search_extremes.png")
    plt.savefig(png_path, dpi=150)
    print(f"Saved Best/Worst visualization to {png_path}")
