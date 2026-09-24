"""Random search: sample assignments independently and keep the best."""

from __future__ import annotations

import time

import numpy as np

from ..problem import Budget, SplitProblem
from .base import Outcome, Strategy, resolve_max_evals
from .registry import register_strategy

_TIME_CHECK_INTERVAL = 1024


@register_strategy
class RandomSearch(Strategy):
    """Independent sampling under the target ratios; a baseline, not a contender."""

    name = "random"

    def run(
        self,
        problem: SplitProblem,
        budget: Budget,
        rng: np.random.Generator,
        warm_start: np.ndarray | None = None,
    ) -> Outcome:
        t_start = time.perf_counter()
        deadline = budget.deadline_from(t_start)
        max_evals = resolve_max_evals(budget)

        best_assignment = problem.random_assignment(rng)
        best_cost = problem.evaluate(best_assignment)
        cost_history: list[tuple[int, float]] = [(1, best_cost)]
        n_evals = 1

        while n_evals < max_evals and best_cost > budget.target_cost:
            assignment = problem.random_assignment(rng)
            cost = problem.evaluate(assignment)
            n_evals += 1

            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment
                cost_history.append((n_evals, best_cost))

            if n_evals % _TIME_CHECK_INTERVAL == 0:
                budget.report(n_evals, best_cost)
                if time.perf_counter() >= deadline:
                    break

        return Outcome(
            assignment=best_assignment,
            cost=best_cost,
            n_evals=n_evals,
            n_iterations=n_evals,
            converged=best_cost <= budget.target_cost,
            cost_history=cost_history,
        )
