"""Benchmark runner: compare split optimizers across multiple seeds.
Evaluates Mean +/- Std Dev of cost and plots mean convergence curves with shaded std regions.

Usage: uv run python -m benchmarks.run_benchmark [bcss|celeba|isic|synth_*]
"""

import glob
import io
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from splitkit import Budget, GroupedDataset, SplitProblem
from splitkit.io import load_npz
from splitkit.strategies import (
    DifferentialEvolution,
    ExactMILP,
    RandomSearch,
    SGKFBaseline,
    SimulatedAnnealing,
)

MAX_EVALS = 300_000
RATIOS = (0.70, 0.15, 0.15)
N_RUNS = 10
SEEDS = [42 + i for i in range(N_RUNS)]
EXACT_TIME_LIMIT = 20.0

_OPTIMIZERS = [
    ("SA", SimulatedAnnealing, dict()),
    ("DE", DifferentialEvolution, dict()),
    ("RS", RandomSearch, dict()),
    ("SGKF", SGKFBaseline, dict()),
    ("Exact", ExactMILP, dict()),
]

# One-shot strategies run once and appear as horizontal reference lines.
_ONE_SHOT_BUDGETS = {"SGKF": Budget(max_evals=1), "Exact": Budget(time_limit=EXACT_TIME_LIMIT)}

_STYLE = {
    "SA": dict(color="#1f77b4", linestyle="-", linewidth=1.8),
    "DE": dict(color="#d62728", linestyle="-", linewidth=1.8),
    "RS": dict(color="#ff7f0e", linestyle="--", linewidth=1.8),
    "SGKF": dict(color="#2ca02c", linestyle=":", linewidth=2.0),
    "Exact": dict(color="#9467bd", linestyle="-.", linewidth=2.0),
}

_DATASET_PATHS = {
    "bcss": "datasets/bcss/preprocessed/groups.npz",
    "celeba": "datasets/celeb-faces/preprocessed/groups.npz",
    "isic": "datasets/isic2020/preprocessed/groups.npz",
}
for _pkl in sorted(glob.glob("datasets/synthetic/preprocessed/*.npz")):
    _DATASET_PATHS[os.path.splitext(os.path.basename(_pkl))[0]] = _pkl


def _result_folder(name: str) -> str:
    return "synthetic" if name.startswith("synth_") else name


def run_one(dataset_name: str) -> tuple[GroupedDataset, dict]:
    """Runs all optimizers across all seeds for a single dataset."""
    data = load_npz(_DATASET_PATHS[dataset_name])
    # Weights and targets are shared by every strategy, so build the problem once.
    problem = SplitProblem.build(data, RATIOS)
    results = {}

    for label, cls, kwargs in _OPTIMIZERS:
        costs = []
        histories = []
        times = []

        runs_to_do = 1 if label in _ONE_SHOT_BUDGETS else N_RUNS
        budget = _ONE_SHOT_BUDGETS.get(label, Budget(max_evals=MAX_EVALS))

        for idx in range(runs_to_do):
            seed = SEEDS[idx]
            strategy = cls(**kwargs)

            t0 = time.perf_counter()
            res = strategy.run(problem, budget, np.random.default_rng(seed))
            elapsed = time.perf_counter() - t0

            costs.append(res.cost)
            histories.append(res)
            times.append(elapsed)

        mean_cost = np.mean(costs)
        std_cost = np.std(costs) if len(costs) > 1 else 0.0
        mean_time = np.mean(times)

        results[label] = {
            "mean_cost": mean_cost,
            "std_cost": std_cost,
            "mean_time": mean_time,
            "all_costs": costs,
            "all_histories": histories
        }

    return data, results


def plot_convergence(name: str, results: dict, outdir: str):
    """Plots the mean convergence curve with a +/- Std Dev shaded region."""
    _fig, ax = plt.subplots(figsize=(8, 5))

    ffe_grid = np.linspace(0, MAX_EVALS, 1000)

    for label, data in results.items():
        sty = _STYLE.get(label, {})

        if label in _ONE_SHOT_BUDGETS:
            ax.axhline(y=data["mean_cost"], label=f"{label} (Cost: {data['mean_cost']:.4f})", **sty)
        else:
            interp_costs = []

            for res in data["all_histories"]:
                if not res.cost_history:
                    interp_costs.append(np.full_like(ffe_grid, res.cost))
                    continue

                raw_evals = [e for e, c in res.cost_history]
                raw_costs = [c for e, c in res.cost_history]

                idx = np.searchsorted(raw_evals, ffe_grid, side='right') - 1
                idx = np.clip(idx, 0, len(raw_costs) - 1)
                interp_costs.append(np.array(raw_costs)[idx])

            interp_costs = np.array(interp_costs)
            mean_curve = np.mean(interp_costs, axis=0)
            std_curve = np.std(interp_costs, axis=0)

            label_str = f"{label} (Mean: {data['mean_cost']:.4f} ± {data['std_cost']:.4f})"

            ax.plot(ffe_grid, mean_curve, label=label_str,
                    color=sty.get("color"), linestyle=sty.get("linestyle"), linewidth=sty.get("linewidth"))

            lower_bound = np.maximum(0, mean_curve - std_curve)
            upper_bound = mean_curve + std_curve
            ax.fill_between(ffe_grid, lower_bound, upper_bound, color=sty.get("color"), alpha=0.2)

    ax.set_title(f"Convergence Comparison on '{name}' ({N_RUNS} runs)", fontweight="bold")
    ax.set_xlabel("Function Evaluations (FFE)")
    ax.set_ylabel("Cost (Weighted MAPE)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1_000:.0f}k" if x >= 1_000 else f"{x:.0f}"))
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"convergence_{name}.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    names = args if args else list(_DATASET_PATHS.keys())

    # Validate datasets
    for n in names:
        if n not in _DATASET_PATHS:
            sys.exit(f"Error: Unknown dataset '{n}'. Valid options: {list(_DATASET_PATHS.keys())}")

    os.makedirs("results", exist_ok=True)
    summary_rows = []

    print("Stratified Data Split Benchmark")
    print(f"Budget: {MAX_EVALS:,} FFEs")
    print(f"Runs per algorithm: {N_RUNS} (Seeds: {SEEDS[0]} to {SEEDS[-1]})")

    for name in names:
        print(f"-> Benchmarking {name:<22} ... ", end="", flush=True)
        data, results = run_one(name)

        outdir = os.path.join("results", _result_folder(name))
        os.makedirs(outdir, exist_ok=True)

        plot_convergence(name, results, outdir)

        best_alg = min(results.keys(), key=lambda k: results[k]["mean_cost"])
        print(f"Done. Best: {best_alg} ({results[best_alg]['mean_cost']:.4f} ± {results[best_alg]['std_cost']:.4f})")

        with open(os.path.join(outdir, f"{name}_report.txt"), "w") as f:
            f.write(f"=== BENCHMARK REPORT: {name} ===\n")
            f.write(f"Groups: {data.n_groups} | Classes: {data.n_classes} | Budget: {MAX_EVALS:,} FFE\n")
            f.write("-" * 65 + "\n")
            f.write(f" {'Algorithm':<10} | {'Mean Cost':<12} | {'Std Dev':<10} | {'Mean Time':<10}\n")
            f.write("-" * 65 + "\n")
            for label, d in results.items():
                f.write(f" {label:<10} | {d['mean_cost']:<12.4f} | ± {d['std_cost']:<8.4f} | {d['mean_time']:>7.2f}s\n")

        summary_rows.append((name, data.n_groups, data.n_classes, results, best_alg))

    # Summary table
    buf = io.StringIO()
    algs = list(_STYLE.keys())

    header = f"{'Dataset':<20} {'Groups':>8} {'Classes':>8} " + "".join(f"{a:>15}" for a in algs) + f" {'Winner':>8}"
    buf.write("\n" + "=" * len(header) + "\n")
    buf.write("FINAL BENCHMARK SUMMARY (Mean Cost ± Std Dev over 10 runs)\n")
    buf.write("=" * len(header) + "\n")
    buf.write(header + "\n")
    buf.write("-" * len(header) + "\n")

    for name, grp, cls, res, winner in summary_rows:
        row_str = f"{name:<20} {grp:>8} {cls:>8} "
        for a in algs:
            cost_str = (
                f"{res[a]['mean_cost']:.3f}±{res[a]['std_cost']:.3f}" if a in res else "N/A"
            )
            row_str += f"{cost_str:>15}"
        row_str += f" {winner:>8}"
        buf.write(row_str + "\n")

    buf.write("=" * len(header) + "\n")
    summary_text = buf.getvalue()

    print(summary_text)
    with open("results/summary.txt", "w") as f:
        f.write(summary_text)
