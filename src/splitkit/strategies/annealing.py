"""Simulated Annealing optimizer for group-aware stratified data splitting."""

import math
import time

import numpy as np

from .base import N_SPLITS, Optimizer, SplitResult


class SimulatedAnnealing(Optimizer):
    """Simulated Annealing for group-aware stratified train/val/test splitting.

    State: integer assignment vector, one entry per group in {0, 1, 2}.
    Neighbor: move one random group to a different split.
    Acceptance: Metropolis criterion, exp(-delta/T).
    Cooling: T *= cooling_rate each step; reheat to initial_temp when T < min_temp.
    Cost update: O(n_classes) incremental update per step.

    Params
    ------
    data : DatasetGroups
    ratios : tuple of float        target split fractions, must sum to 1
    max_evals : int                total FFE budget (shared by all algorithms)
    initial_temp : float           starting temperature
    cooling_rate : float           in (0, 1), applied each step
    min_temp : float               reheating trigger, must be < initial_temp
    seed : int | None
    """

    def __init__(
        self,
        data,
        ratios: tuple[float, ...] = (0.70, 0.15, 0.15),
        max_evals: int = 300_000,
        initial_temp: float = 10.0,
        cooling_rate: float = 0.9999,
        min_temp: float = 1e-4,
        seed: int | None = None,
    ) -> None:
        super().__init__(data, ratios, max_evals=max_evals, seed=seed)

        if not 0.0 < cooling_rate < 1.0:
            raise ValueError(
                f"cooling_rate must be in the open interval (0, 1), "
                f"got {cooling_rate}."
            )
        if initial_temp <= 0.0:
            raise ValueError(
                f"initial_temp must be positive, got {initial_temp}."
            )
        if min_temp <= 0.0 or min_temp >= initial_temp:
            raise ValueError(
                f"min_temp must satisfy 0 < min_temp < initial_temp, "
                f"got min_temp={min_temp}, initial_temp={initial_temp}."
            )

        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

    # Core algorithm
    def optimize(
        self,
        verbose: bool = True,
        log_interval: int = 10_000,
    ) -> SplitResult:
        """Run SA and return the best split found."""
        rng = np.random.default_rng(self.seed)
        t_start = time.perf_counter()

        # Initial evaluation (FFE #1)
        assignment = rng.choice(N_SPLITS, size=self.data.n_groups, p=self.ratios)
        actual = self._count_matrix(assignment)
        cost = self._mape_cost(actual)
        n_evals = 1

        best_assignment = assignment.copy()
        best_actual = actual.copy()
        best_cost = cost

        temp = self.initial_temp
        cost_history: list[tuple[int, float]] = []
        n_reheats = 0

        if verbose:
            print(
                f"[SA] {self.data.dataset_name}"
                f"  groups={self.data.n_groups}"
                f"  classes={self.data.n_classes}"
                f"  budget={self.max_evals:,} FFEs"
                f"  initial_cost={cost:.4f}"
            )

        # Main loop: FFEs #2 .. max_evals
        # 1 initial FFE the total is exactly max_evals evaluations.
        iteration = 0
        for iteration in range(1, self.max_evals):
            n_evals = iteration + 1

            g = int(rng.integers(self.data.n_groups))
            old_s = int(assignment[g])

            # Pick any split other than the current one, uniformly.
            new_s = int(rng.integers(N_SPLITS - 1))
            if new_s >= old_s:
                new_s += 1

            # Incremental O(n_classes) cost update
            vec = self.data.group_vectors[g].astype(np.float64)
            actual[old_s] -= vec
            actual[new_s] += vec
            new_cost = self._mape_cost(actual)

            # Metropolis acceptance
            delta = new_cost - cost
            accepted = (
                delta < 0
                or rng.random() < math.exp(-delta / max(temp, 1e-300))
            )

            if accepted:
                assignment[g] = new_s
                cost = new_cost
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = assignment.copy()
                    best_actual = actual.copy()
                    cost_history.append((n_evals, best_cost))
            else:
                actual[new_s] -= vec
                actual[old_s] += vec

            # Geometric cooling
            temp *= self.cooling_rate

            # Reheating: restore best state when temperature collapses
            if temp < self.min_temp:
                temp = self.initial_temp
                assignment = best_assignment.copy()
                actual = best_actual.copy()
                cost = best_cost
                n_reheats += 1

            if n_evals % log_interval == 0:
                elapsed = time.perf_counter() - t_start
                if verbose:
                    print(
                        f"  evals {n_evals:>7,}"
                        f"  best_cost={best_cost:.4f}"
                        f"  temp={temp:.5f}"
                        f"  reheats={n_reheats}"
                        f"  elapsed={elapsed:.1f}s"
                    )

        elapsed = time.perf_counter() - t_start
        return SplitResult(
            assignment=best_assignment,
            cost=best_cost,
            n_evals=n_evals,
            n_iterations=iteration,
            converged=False,
            elapsed_time=elapsed,
            cost_history=cost_history,
            target_counts=self._target.copy(),
            actual_counts=best_actual,
        )
