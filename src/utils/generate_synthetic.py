"""
Synthetic dataset generator for optimizer benchmarking.

Generates 6 DatasetGroups that vary along three axes:
  - Search space size      (n_groups)
  - Class imbalance        (power-law exponent controlling the count ratio)
  - Class concentration    (how many groups each class appears in)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.common import DatasetGroups, save_dataset

OUT_DIR = os.path.join("datasets", "synthetic", "preprocessed")


# Core generator
def _class_counts(n_classes: int, total: int, exponent: float, rng) -> np.ndarray:
    """Power-law class sizes.

    exponent=0  - perfectly balanced
    exponent=1  - ~10 : 1 ratio across n_classes=10
    exponent=2  - ~100 : 1 ratio across n_classes=10
    """
    ranks = np.arange(1, n_classes + 1, dtype=float)
    freqs = 1.0 / np.power(ranks, exponent)
    freqs /= freqs.sum()
    counts = np.maximum(1, np.round(freqs * total).astype(int))
    counts[0] += total - counts.sum()
    return counts


def generate(
    name: str,
    n_groups: int,
    n_classes: int,
    total_items: int,
    imbalance: float,
    groups_per_class: int,
    dirichlet_alpha: float,
    seed: int = 0,
) -> DatasetGroups:
    """Build one synthetic DatasetGroups.

    Params
    ----------
    name            : dataset identifier
    n_groups        : number of indivisible groups
    n_classes       : number of class labels
    total_items     : total sample count across all groups and classes
    imbalance       : power-law exponent for class-count distribution
                      (0 = balanced, 1 = mild, 2 = heavy)
    groups_per_class: how many groups each class is scattered across;
                      low -> concentrated / hard to stratify,
                      high -> spread / easy to stratify
    dirichlet_alpha : concentration parameter for within-class group sizes;
                      <1 -> very unequal group sizes, >>1 -> near-uniform
    seed            : RNG seed
    """
    groups_per_class = min(groups_per_class, n_groups)
    rng = np.random.default_rng(seed)

    class_totals = _class_counts(n_classes, total_items, imbalance, rng)
    group_vectors = np.zeros((n_groups, n_classes), dtype=np.float64)

    for c, total_c in enumerate(class_totals):
        chosen = rng.choice(n_groups, size=groups_per_class, replace=False)
        props = rng.dirichlet([dirichlet_alpha] * groups_per_class)
        counts = np.maximum(0, np.round(props * total_c).astype(int))
        counts[-1] = max(0, total_c - counts[:-1].sum())
        group_vectors[chosen, c] += counts

    # Drop groups that ended up with zero items
    sizes = group_vectors.sum(axis=1)
    mask = sizes > 0
    group_vectors = group_vectors[mask]
    sizes = sizes[mask]
    n_actual = int(mask.sum())

    return DatasetGroups(
        dataset_name=name,
        group_ids=[f"g{i:05d}" for i in range(n_actual)],
        group_vectors=group_vectors,
        group_sizes=sizes,
        class_names=[f"class_{c:02d}" for c in range(n_classes)],
    )


CONFIGS = [
    dict(name="synth_easy_balanced", n_groups=500, n_classes=5,
         total_items=10_000, imbalance=0.0, groups_per_class=500, dirichlet_alpha=10.0),
    dict(name="synth_mild_imbalance", n_groups=300, n_classes=10,
         total_items=15_000, imbalance=1.0, groups_per_class=100, dirichlet_alpha=2.0),
    dict(name="synth_few_groups", n_groups=100, n_classes=15,
         total_items=8_000, imbalance=1.5, groups_per_class=20, dirichlet_alpha=0.5),
    dict(name="synth_concentrated", n_groups=400, n_classes=12,
         total_items=20_000, imbalance=1.2, groups_per_class=5, dirichlet_alpha=1.0),
    dict(name="synth_heavy_imbalance", n_groups=200, n_classes=20,
         total_items=50_000, imbalance=2.5, groups_per_class=15, dirichlet_alpha=1.0),
    dict(name="synth_large_complex", n_groups=2000, n_classes=25,
         total_items=100_000, imbalance=1.8, groups_per_class=40, dirichlet_alpha=1.0),
]


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for cfg in CONFIGS:
        data = generate(**cfg, seed=0)
        path = os.path.join(OUT_DIR, f"{cfg['name']}.pkl")
        save_dataset(data, path)
        n_strat = int((np.array([(data.group_vectors[:, c] > 0).sum()
                                  for c in range(data.n_classes)]) >= 3).sum())
        counts = data.global_class_counts
        ratio = counts.max() / max(counts.min(), 1)
        print(
            f"  {cfg['name']:<28}"
            f"  groups={data.n_groups:>5}"
            f"  classes={data.n_classes:>3}"
            f"  stratifiable={n_strat:>3}/{data.n_classes}"
            f"  imbalance_ratio={ratio:>8.1f}:1"
        )
    print(f"\nSaved to {OUT_DIR}/")
