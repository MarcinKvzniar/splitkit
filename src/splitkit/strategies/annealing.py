"""Simulated annealing over group-to-split assignments."""

from __future__ import annotations

import math
import time

import numpy as np

from ..problem import Budget, SplitProblem
from .base import Outcome, Strategy, resolve_max_evals
from .registry import register_strategy

#: How often to consult the wall clock, in evaluations.
_TIME_CHECK_INTERVAL = 4096


@register_strategy
class SimulatedAnnealing(Strategy):
    """Single-trajectory search with Metropolis acceptance and reheating.

    State is the assignment vector; a neighbour moves one group to a different
    split. Accepting uphill moves with probability ``exp(-delta / T)`` lets the
    search escape the local optima that a pure descent gets stuck in, and the
    temperature decays geometrically so the walk anneals into exploitation.

    When the temperature collapses below ``min_temp`` the search reheats and
    restarts from the best assignment found so far, which turns the tail of a long
    budget into repeated focused dives rather than a frozen random walk.

    Parameters
    ----------
    initial_temp
        Starting temperature. Should be on the order of a typical cost delta.
    cooling_rate
        Per-step multiplier in (0, 1), or ``"auto"`` (the default) to fit the
        schedule to the budget. A fixed rate silently assumes a particular budget:
        0.9999 needs roughly 300k steps to anneal, so on a 5k-step run the search
        never leaves its exploration phase and returns something close to random.
    min_temp
        Reheat trigger; must be below ``initial_temp``.
    anneal_cycles
        How many times ``"auto"`` should cool from ``initial_temp`` to ``min_temp``
        across the budget. Each cycle ends in a reheat from the best assignment
        found, so more cycles means more, shorter dives.
    """

    name = "annealing"
    supports_warm_start = True

    def __init__(
        self,
        initial_temp: float = 100.0,
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
        """Per-step multiplier, fitted to the budget when set to "auto".

        Solves ``rate ** (max_evals / cycles) == min_temp / initial_temp`` so the
        temperature completes the requested number of anneal cycles in the budget
        actually available, instead of assuming one.
        """
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
        # With only a wall-clock budget there is no evaluation count to fit the
        # schedule to, so calibrate once from observed throughput.
        needs_calibration = (
            self.cooling_rate == "auto"
            and budget.max_evals is None
            and budget.time_limit is not None
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

            # Uniform over the splits other than the current one.
            new_s = int(rng.integers(k - 1))
            if new_s >= old_s:
                new_s += 1

            vec = vectors[g]
            counts[old_s] -= vec
            counts[new_s] += vec
            new_cost = total(counts)

            delta = new_cost - cost
            # rng.random() is only drawn for uphill moves; short-circuiting here is
            # load-bearing for reproducibility, not just speed.
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
                if needs_calibration:
                    rate = n_evals / max(now - t_start, 1e-9)
                    cooling_rate = self._resolve_cooling_rate(
                        max(1, int(rate * budget.time_limit))
                    )
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
