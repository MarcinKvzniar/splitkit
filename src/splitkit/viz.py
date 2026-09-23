"""Plots of split quality and search progress; requires the ``viz`` extra."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .result import SplitResult

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes

__all__ = ["plot_convergence", "plot_distribution"]


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Plotting requires matplotlib. Install it with: pip install 'splitkit[viz]'"
        ) from exc
    return plt


def plot_distribution(result: SplitResult, ax: Axes | None = None) -> Axes:
    """Stacked bars of where each class's items went, against the target ratios.

    Fractions are per class, so a rare class is as visible as a common one; a
    well-stratified class lines up with the dashed target boundaries.
    """
    labels = ["all items", *result.dataset.class_names]
    if ax is None:
        _, ax = _pyplot().subplots(figsize=(max(6.0, 0.45 * len(labels) + 2), 4))

    counts = np.column_stack(
        [result.item_counts, result.actual_counts[:, : result.dataset.n_classes]]
    )
    totals = counts.sum(axis=0)
    fractions = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for s, name in enumerate(result.names):
        ax.bar(x, fractions[s], 0.8, bottom=bottom, label=name)
        bottom += fractions[s]
    for boundary in np.cumsum(result.ratios)[:-1]:
        ax.axhline(boundary, linestyle="--", color="black", linewidth=1)

    ax.set_xticks(x, labels, rotation=45 if len(x) > 6 else 0, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of class")
    ax.set_title(f"Class allocation: {result.dataset.name}")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    return ax


def plot_convergence(result: SplitResult, ax: Axes | None = None) -> Axes:
    """Best cost against evaluations, with the proven lower bound when available."""
    if ax is None:
        _, ax = _pyplot().subplots(figsize=(6, 4))

    if result.cost_history:
        evals, costs = zip(*result.cost_history, strict=True)
        ax.step([*evals, result.n_evals], [*costs, costs[-1]], where="post", label=result.strategy)
    if result.lower_bound is not None:
        ax.axhline(result.lower_bound, linestyle="--", color="gray", label="lower bound")

    ax.set_xlabel("evaluations")
    ax.set_ylabel("best cost")
    ax.set_title(f"Convergence: {result.dataset.name}")
    ax.legend()
    return ax
