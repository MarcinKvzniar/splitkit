"""Shared convergence-history helper for the benchmark and tuning scripts."""

from __future__ import annotations

import numpy as np

__all__ = ["unpack_history"]


def unpack_history(result) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a result's cost history to a monotone best-so-far curve.

    Returns ``(evals, costs)`` extended to the run's final evaluation count so
    curves from runs of differing length can be interpolated onto a common grid.
    """
    if not result.cost_history:
        return (
            np.array([result.n_evals], dtype=float),
            np.array([result.cost], dtype=float),
        )

    evals: list[float] = []
    costs: list[float] = []
    best_so_far = float("inf")

    for e, c in result.cost_history:
        if c < best_so_far:
            best_so_far = c
            evals.append(e)
            costs.append(c)

    last_e = max(result.n_evals, result.cost_history[-1][0])
    if evals[-1] < last_e:
        evals.append(last_e)
        costs.append(best_so_far)

    return np.asarray(evals, dtype=float), np.asarray(costs, dtype=float)
