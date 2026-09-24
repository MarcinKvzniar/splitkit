"""Simulated annealing over group-to-split assignments."""

from __future__ import annotations

import math
import time

import numpy as np

from ..problem import Budget, SplitProblem
from .base import Outcome, Strategy, resolve_max_evals
from .registry import register_strategy

_TIME_CHECK_INTERVAL = 4096
_DRAW_BATCH = 4096  # per-call numpy overhead dominates single draws
_PROBE_MOVES = 2048
# Under this many moves per group there is no time to repair a scrambled split,
# so the search starts 4x colder than the largest move.
_TIGHT_MOVES_PER_GROUP = 100
_TIGHT_COOLING = 4.0


@register_strategy
class SimulatedAnnealing(Strategy):
    """Single-group moves with Metropolis acceptance, geometric cooling and reheating.

    Below ``min_temp`` the search reheats from the best assignment found so far.

    Parameters
    ----------
    initial_temp
        Upper bound on the starting temperature. The run never starts hotter than
        the largest cost change of a single move, and 4x colder than that when the
        budget allows under 100 moves per group.
    cooling_rate
        Per-step multiplier in (0, 1), or ``"auto"`` to fit the schedule to the budget.
    min_temp
        Reheat trigger, scaled with the starting temperature; must be below
        ``initial_temp``.
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

    def _starting_temp(self, largest_move: float, max_evals: int, n_groups: int) -> float:
        """Never hotter than the largest move, and colder still on a tight budget."""
        tight = max_evals < _TIGHT_MOVES_PER_GROUP * n_groups
        return min(self.initial_temp, largest_move / (_TIGHT_COOLING if tight else 1.0))

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
        needs_calibration = budget.max_evals is None and time_limit is not None

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

        largest = _largest_move(problem, assignment, counts, cost)
        initial_temp = self._starting_temp(largest, max_evals, n_groups)
        min_temp = self.min_temp * initial_temp / self.initial_temp

        best_assignment = assignment.copy()
        best_counts = counts.copy()
        best_cost = cost

        cost_history: list[tuple[int, float]] = [(1, best_cost)]
        temp = initial_temp
        n_reheats = 0
        iteration = 0

        while n_evals < max_evals:
            j = iteration % _DRAW_BATCH
            if j == 0:
                groups = rng.integers(n_groups, size=_DRAW_BATCH).tolist()
                shifts = rng.integers(1, k, size=_DRAW_BATCH).tolist()
            iteration += 1
            n_evals += 1

            g = groups[j]
            old_s = int(assignment[g])
            new_s = (old_s + shifts[j]) % k

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

            if temp < min_temp:
                temp = initial_temp
                assignment = best_assignment.copy()
                counts = best_counts.copy()
                cost = best_cost
                n_reheats += 1

            if n_evals % _TIME_CHECK_INTERVAL == 0:
                budget.report(n_evals, best_cost)
                now = time.perf_counter()
                if needs_calibration and time_limit is not None:
                    projected = max(1, int(n_evals / max(now - t_start, 1e-9) * time_limit))
                    cooling_rate = self._resolve_cooling_rate(projected)
                    rescale = self._starting_temp(largest, projected, n_groups) / initial_temp
                    initial_temp, min_temp, temp = (
                        initial_temp * rescale, min_temp * rescale, temp * rescale
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


def _largest_move(
    problem: SplitProblem, assignment: np.ndarray, counts: np.ndarray, cost: float
) -> float:
    """Largest cost change over single moves of evenly spaced groups (no RNG draws)."""
    k, total = problem.n_splits, problem.prepared.total
    largest = 0.0
    for g in np.linspace(0, problem.n_groups - 1, min(problem.n_groups, _PROBE_MOVES)).astype(int):
        old_s = int(assignment[g])
        moved = counts.copy()
        moved[old_s] -= problem.vectors[g]
        moved[(old_s + 1) % k] += problem.vectors[g]
        largest = max(largest, abs(total(moved) - cost))
    return largest if largest > 0 else float("inf")
