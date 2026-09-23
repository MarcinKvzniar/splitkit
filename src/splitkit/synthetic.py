"""Synthetic grouped datasets varying size, imbalance and class concentration."""

from __future__ import annotations

from typing import Any

import numpy as np

from .dataset import GroupedDataset

__all__ = ["PRESETS", "from_preset", "list_presets", "make_synthetic"]


def _class_counts(n_classes: int, total: int, exponent: float) -> np.ndarray:
    """Power-law class sizes; exponent 0 is balanced, 2 is roughly 100:1."""
    ranks = np.arange(1, n_classes + 1, dtype=float)
    freqs = 1.0 / np.power(ranks, exponent)
    freqs /= freqs.sum()
    counts = np.maximum(1, np.round(freqs * total).astype(int))
    counts[0] += total - counts.sum()
    return np.asarray(counts)


def make_synthetic(
    n_groups: int = 300,
    n_classes: int = 10,
    *,
    total_items: int = 15_000,
    imbalance: float = 1.0,
    groups_per_class: int = 100,
    dirichlet_alpha: float = 2.0,
    name: str = "synthetic",
    seed: int | None = 0,
) -> GroupedDataset:
    """Build a synthetic :class:`GroupedDataset`.

    Parameters
    ----------
    imbalance
        Power-law exponent of class sizes (0 balanced, 1 mild, 2 heavy).
    groups_per_class
        Number of groups each class is spread across; low values are hard to stratify.
    dirichlet_alpha
        Within-class group size concentration; below 1 gives very unequal groups.
    """
    if n_groups < 1:
        raise ValueError(f"n_groups must be positive, got {n_groups}.")
    if n_classes < 1:
        raise ValueError(f"n_classes must be positive, got {n_classes}.")

    groups_per_class = min(groups_per_class, n_groups)
    rng = np.random.default_rng(seed)

    class_totals = _class_counts(n_classes, total_items, imbalance)
    group_vectors = np.zeros((n_groups, n_classes), dtype=np.float64)

    for c, total_c in enumerate(class_totals):
        chosen = rng.choice(n_groups, size=groups_per_class, replace=False)
        props = rng.dirichlet([dirichlet_alpha] * groups_per_class)
        counts = np.maximum(0, np.round(props * total_c).astype(int))
        counts[-1] = max(0, total_c - counts[:-1].sum())
        group_vectors[chosen, c] += counts

    sizes = group_vectors.sum(axis=1)
    keep = sizes > 0
    group_vectors = group_vectors[keep]
    sizes = sizes[keep]
    n_actual = int(keep.sum())

    return GroupedDataset(
        group_ids=np.array([f"g{i:05d}" for i in range(n_actual)], dtype=np.str_),
        group_vectors=group_vectors,
        group_sizes=sizes,
        class_names=tuple(f"class_{c:02d}" for c in range(n_classes)),
        name=name,
    )


PRESETS: dict[str, dict[str, Any]] = {
    "easy_balanced": dict(
        n_groups=500, n_classes=5, total_items=10_000,
        imbalance=0.0, groups_per_class=500, dirichlet_alpha=10.0,
    ),
    "mild_imbalance": dict(
        n_groups=300, n_classes=10, total_items=15_000,
        imbalance=1.0, groups_per_class=100, dirichlet_alpha=2.0,
    ),
    "few_groups": dict(
        n_groups=100, n_classes=15, total_items=8_000,
        imbalance=1.5, groups_per_class=20, dirichlet_alpha=0.5,
    ),
    "concentrated": dict(
        n_groups=400, n_classes=12, total_items=20_000,
        imbalance=1.2, groups_per_class=5, dirichlet_alpha=1.0,
    ),
    "heavy_imbalance": dict(
        n_groups=200, n_classes=20, total_items=50_000,
        imbalance=2.5, groups_per_class=15, dirichlet_alpha=1.0,
    ),
    "large_complex": dict(
        n_groups=2000, n_classes=25, total_items=100_000,
        imbalance=1.8, groups_per_class=40, dirichlet_alpha=1.0,
    ),
}


def list_presets() -> tuple[str, ...]:
    """Names accepted by ``make_synthetic(**PRESETS[name])``."""
    return tuple(PRESETS)


def from_preset(preset: str, *, seed: int | None = 0) -> GroupedDataset:
    """Build the named preset dataset."""
    if preset not in PRESETS:
        raise KeyError(
            f"Unknown preset {preset!r}. Available: {', '.join(sorted(PRESETS))}"
        )
    return make_synthetic(**PRESETS[preset], name=f"synth_{preset}", seed=seed)
