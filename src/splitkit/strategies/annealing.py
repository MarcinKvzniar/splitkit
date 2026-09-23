"""Simulated annealing over group-to-split assignments."""

from __future__ import annotations

import math
import time

import numpy as np

from ..problem import Budget, SplitProblem
from .base import Outcome, Strategy, resolve_max_evals
from .registry import register_strategy

_TIME_CHECK_INTERVAL = 4096


@register_strategy
class SimulatedAnnealing(Strategy):
    """Single-group moves with Metropolis acceptance, geometric cooling and reheating.

    Below ``min_temp`` the search reheats from the best assignment found so far.

    Parameters
    ----------
    initial_temp
        Starting temperature, on the order of a typical cost delta.
    cooling_rate
        Per-step multiplier in (0, 1), or ``"auto"`` to fit the schedule to the budget.
    min_temp
        Reheat trigger; must be below ``initial_temp``.
    anneal_cycles
        Number of full cool-downs ``"auto"`` fits into the budget.
    """

    name = "annealing"
    supports_warm_start = True

    def __init__(
        self,
        initial_temp: float = 1.0,
        cooling_rate: float | str = "auto",
        min_temp: float = 1e-4,
        anneal_cycles: float = 2.0,
    ) -> None:
        if isinstance(cooling_rate, str):
            if cooling_rate != "auto":
                raise ValueError(
                    f"cooling_rate must be a number in (0, 1) or 'auto', "
                    f"got {cooling_rate!r}."
                )
        elif not 0.0 < cooling_rate < 1.0:
            raise ValueError(
                f"cooling_rate must be in the open interval (0, 1), got {cooling_rate}."
            )
        if anneal_cycles <= 0:
            raise ValueError(
                f"anneal_cycles must be positive, got {anneal_cycles}."
            )
        if initial_temp <= 0.0:
            raise ValueError(f"initial_temp must be positive, got {initial_temp}.")
        if min_temp <= 0.0 or min_temp >= initial_temp:
            raise ValueError(
                f"min_temp must satisfy 0 < min_temp < initial_temp, got "
                f"min_temp={min_temp}, initial_temp={initial_temp}."
            )
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.anneal_cycles = anneal_cycles

    def _resolve_cooling_rate(self, max_evals: int) -> float:
        # "auto" solves rate ** (max_evals / cycles) == min_temp / initial_temp.
        if self.cooling_rate != "auto":
            return float(self.cooling_rate)
        steps = max(1.0, max_evals / self.anneal_cycles)
        return float(math.exp(math.log(self.min_temp / self.initial_temp) / steps))

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

        k = problem.n_splits
        n_groups = problem.n_groups
        vectors = problem.vectors
        total = problem.prepared.total
        cooling_rate = self._resolve_cooling_rate(max_evals)
        # With only a time limit, fit the schedule once from observed throughput.
        time_limit = budget.time_limit
        needs_calibration = (
            self.cooling_rate == "auto" and budget.max_evals is None and time_limit is not None
        )

        if warm_start is not None:
            assignment = np.asarray(warm_start, dtype=np.intp).copy()
            if assignment.shape != (n_groups,):
                raise ValueError(
                    f"warm_start has shape {assignment.shape}, expected ({n_groups},)."
                )
        else:
            assignment = problem.random_assignment(rng)

        counts = problem.count_matrix(assignment)
        cost = total(counts)
        n_evals = 1

        best_assignment = assignment.copy()
        best_counts = counts.copy()
        best_cost = cost

        cost_history: list[tuple[int, float]] = [(1, best_cost)]
        temp = self.initial_temp
        n_reheats = 0
        iteration = 0

        while n_evals < max_evals:
            iteration += 1
            n_evals += 1

            g = int(rng.integers(n_groups))
            old_s = int(assignment[g])

            new_s = int(rng.integers(k - 1))
            if new_s >= old_s:
                new_s += 1

            vec = vectors[g]
            counts[old_s] -= vec
            counts[new_s] += vec
            new_cost = total(counts)

            delta = new_cost - cost
            # rng.random() is drawn only for uphill moves; the short-circuit is
            # load-bearing for reproducibility.
            if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-300)):
                assignment[g] = new_s
                cost = new_cost
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = assignment.copy()
                    best_counts = counts.copy()
                    cost_history.append((n_evals, best_cost))
            else:
                counts[new_s] -= vec
                counts[old_s] += vec

            temp *= cooling_rate

            if temp < self.min_temp:
                temp = self.initial_temp
                assignment = best_assignment.copy()
                counts = best_counts.copy()
                cost = best_cost
                n_reheats += 1

            if n_evals % _TIME_CHECK_INTERVAL == 0:
                now = time.perf_counter()
                if needs_calibration and time_limit is not None:
                    rate = n_evals / max(now - t_start, 1e-9)
                    cooling_rate = self._resolve_cooling_rate(max(1, int(rate * time_limit)))
                    needs_calibration = False
                if now >= deadline:
                    break

        return Outcome(
            assignment=best_assignment,
            cost=best_cost,
            n_evals=n_evals,
            n_iterations=iteration,
            converged=best_cost <= budget.target_cost,
            cost_history=cost_history,
        )
