"""Exact mixed-integer programming via SciPy's HiGHS solver."""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from typing import Any

import numpy as np

from ..objectives import LinearObjective
from ..problem import Budget, SplitProblem
from .annealing import SimulatedAnnealing
from .base import DEFAULT_MAX_EVALS, Outcome, Strategy
from .registry import register_strategy


@register_strategy
class ExactMILP(Strategy):
    """Solve the split as a MILP, returning a proven optimum or a lower bound.

    Uses ``Budget.time_limit`` (or ``default_time_limit`` when unset) and ignores
    evaluation counts. On timeout it returns the best solution found and the bound;
    if none was found it warns and runs annealing on the evaluation budget instead.
    When merging groups with identical class vectors at least halves the model, it
    solves over those types instead, so the limit is the number of distinct types.

    Parameters
    ----------
    default_time_limit
        Seconds allowed when the budget sets no time limit.
    mip_rel_gap
        Relative gap at which the solver may stop; ``0`` demands a proof.
    max_types
        Above this many distinct group types, skip the solver and use annealing:
        HiGHS then rarely finds a solution in time and can overrun its time limit.
    """

    name = "exact"
    deterministic = True
    requires = ("exact",)

    def __init__(
        self,
        default_time_limit: float = 60.0,
        mip_rel_gap: float = 0.0,
        max_types: int = 10_000,
    ) -> None:
        if default_time_limit <= 0:
            raise ValueError(f"default_time_limit must be positive, got {default_time_limit}.")
        if mip_rel_gap < 0:
            raise ValueError(f"mip_rel_gap must be non-negative, got {mip_rel_gap}.")
        if max_types < 1:
            raise ValueError(f"max_types must be at least 1, got {max_types}.")
        self.default_time_limit = default_time_limit
        self.mip_rel_gap = mip_rel_gap
        self.max_types = max_types

    def run(
        self,
        problem: SplitProblem,
        budget: Budget,
        rng: np.random.Generator,
        warm_start: np.ndarray | None = None,
    ) -> Outcome:
        try:
            from scipy.optimize import milp
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The exact strategy requires SciPy. "
                "Install it with: pip install 'splitkit[exact]'"
            ) from exc

        prepared = problem.prepared
        if not isinstance(prepared, LinearObjective):
            raise TypeError(
                f"The exact strategy needs a linear objective; "
                f"{problem.objective.name!r} is not."
            )

        # Groups with identical vectors are interchangeable: solve for how many of
        # each type go to each split. This removes the symmetry that stalls HiGHS,
        # but with few duplicates the per-group model searches better within a limit.
        types, type_of, sizes = np.unique(
            problem.vectors, axis=0, return_inverse=True, return_counts=True
        )
        n_groups = problem.n_groups
        if len(types) > n_groups // 2 and n_groups <= self.max_types:
            types, type_of = problem.vectors, np.arange(n_groups)
            sizes = np.ones(n_groups, dtype=np.intp)
        if len(types) > self.max_types:
            return _fallback(
                problem, budget, rng,
                f"{len(types):,} distinct group types exceed max_types={self.max_types:,}",
            )

        time_limit = budget.time_limit or self.default_time_limit
        res = milp(
            **_formulate(types, sizes, prepared.target, prepared.cell_weights),
            options={"time_limit": time_limit, "mip_rel_gap": self.mip_rel_gap},
        )
        bound = getattr(res, "mip_dual_bound", None)
        if bound is not None and not math.isfinite(bound):
            bound = None

        if res.x is None:
            return _fallback(
                problem, budget, rng, f"no solution found within {time_limit:g}s", bound
            )

        per_type = np.rint(res.x[: len(types) * problem.n_splits]).astype(np.intp)
        assignment = _expand(per_type.reshape(len(types), -1), type_of.ravel())
        cost = problem.evaluate(assignment)
        lower_bound = cost if bound is None else min(float(bound), cost)
        proved = res.status == 0 and cost - lower_bound <= 1e-9 * max(1.0, cost)

        return Outcome(
            assignment=assignment,
            cost=cost,
            n_evals=1,
            n_iterations=int(getattr(res, "mip_node_count", 0) or 0),
            converged=proved or cost <= budget.target_cost,
            cost_history=[(1, cost)],
            lower_bound=lower_bound,
            proved_optimal=proved,
        )


def _fallback(
    problem: SplitProblem,
    budget: Budget,
    rng: np.random.Generator,
    reason: str,
    lower_bound: float | None = None,
) -> Outcome:
    """Warn, then solve with annealing on the evaluation budget."""
    warnings.warn(
        f"Exact solver: {reason}; using annealing instead. Allow more time, raise "
        f"max_types, or use strategy='annealing'.",
        RuntimeWarning,
        stacklevel=4,
    )
    fallback = SimulatedAnnealing().run(
        problem,
        Budget(
            max_evals=budget.max_evals or DEFAULT_MAX_EVALS,
            on_progress=budget.on_progress,
        ),
        rng,
    )
    return replace(fallback, converged=False, lower_bound=lower_bound)


def _expand(per_type: np.ndarray, type_of: np.ndarray) -> np.ndarray:
    """Turn ``(T, K)`` groups-per-type-and-split counts into a ``(G,)`` assignment."""
    k = per_type.shape[1]
    labels = np.repeat(np.tile(np.arange(k), len(per_type)), per_type.ravel())
    assignment = np.empty(len(type_of), dtype=np.intp)
    assignment[np.argsort(type_of, kind="stable")] = labels
    return assignment


def _formulate(
    vectors: np.ndarray, sizes: np.ndarray, target: np.ndarray, cell_weights: np.ndarray
) -> dict[str, Any]:
    """Arguments for ``scipy.optimize.milp``.

    Integer ``x[t, s]`` puts that many of the ``sizes[t]`` groups of type ``t`` in
    split ``s``; continuous ``e[s, c]`` bounds ``|counts[s, c] - target[s, c]| /
    scale[s, c]`` from above. Scaling by the target keeps coefficients near 1: raw
    counts can reach millions, pushing objective coefficients below the solver's
    tolerances.
    """
    from scipy.optimize import Bounds, LinearConstraint
    from scipy.sparse import coo_matrix, hstack, identity

    n_types, c = vectors.shape
    k = target.shape[0]
    n_x, n_e = n_types * k, k * c
    x_col = np.arange(n_x).reshape(n_types, k)

    all_placed = coo_matrix((np.ones(n_x), (np.repeat(np.arange(n_types), k), x_col.ravel())), (n_types, n_x))
    non_empty = coo_matrix((np.ones(n_x), (np.tile(np.arange(k), n_types), x_col.ravel())), (k, n_x))

    scale = np.maximum(target, 1.0)
    rows_g, cols_c = np.nonzero(vectors)
    values = vectors[rows_g, cols_c]
    counts = coo_matrix(
        (
            np.concatenate([values / scale[s, cols_c] for s in range(k)]),
            (
                np.concatenate([s * c + cols_c for s in range(k)]),
                np.concatenate([x_col[rows_g, s] for s in range(k)]),
            ),
        ),
        (n_e, n_x),
    )
    e_eye = identity(n_e)

    t = (target / scale).ravel()
    constraints = [
        LinearConstraint(hstack([all_placed, coo_matrix((n_types, n_e))]), sizes, sizes),
        LinearConstraint(hstack([non_empty, coo_matrix((k, n_e))]), 1, np.inf),
        LinearConstraint(hstack([counts, -e_eye]), -np.inf, t),
        LinearConstraint(hstack([counts, e_eye]), t, np.inf),
    ]
    return {
        "c": np.concatenate([np.zeros(n_x), (cell_weights * scale).ravel()]),
        "integrality": np.concatenate([np.ones(n_x), np.zeros(n_e)]),
        "bounds": Bounds(
            np.zeros(n_x + n_e), np.concatenate([np.repeat(sizes, k), np.full(n_e, np.inf)])
        ),
        "constraints": constraints,
    }
