"""Brute-force oracle for tiny instances.

Exhaustive enumeration is only tractable for a handful of groups (K**G states),
but that is exactly what makes it valuable: on instances small enough to enumerate,
it gives ground truth that no heuristic or solver can argue with.
"""

from __future__ import annotations

import itertools

import numpy as np

from splitkit.problem import SplitProblem

#: Refuse to enumerate beyond this many states (~1 second).
MAX_STATES = 200_000


def brute_force(problem: SplitProblem) -> tuple[np.ndarray, float]:
    """Return the provably optimal ``(assignment, cost)`` by full enumeration."""
    k, g = problem.n_splits, problem.n_groups
    n_states = k**g
    if n_states > MAX_STATES:
        raise ValueError(
            f"Refusing to enumerate {n_states:,} states "
            f"(K={k}, G={g}); the oracle is only for tiny instances."
        )

    best_assignment: np.ndarray | None = None
    best_cost = float("inf")
    for combo in itertools.product(range(k), repeat=g):
        assignment = np.asarray(combo, dtype=np.intp)
        cost = problem.evaluate(assignment)
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment

    assert best_assignment is not None
    return best_assignment, best_cost


def all_costs(problem: SplitProblem) -> np.ndarray:
    """Costs of every possible assignment, for distribution-level assertions."""
    k, g = problem.n_splits, problem.n_groups
    if k**g > MAX_STATES:
        raise ValueError(f"Refusing to enumerate {k**g:,} states.")
    return np.array(
        [
            problem.evaluate(np.asarray(c, dtype=np.intp))
            for c in itertools.product(range(k), repeat=g)
        ]
    )
