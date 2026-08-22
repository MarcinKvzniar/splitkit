"""Random Search optimizer (stochastic best-of-N baseline)."""

import time

import numpy as np

from .base import N_SPLITS, Optimizer, SplitResult


class RandomSearch(Optimizer):
    """Random Search: evaluate max_evals i.i.d. random assignments, keep the best.

    Params
    ------
    data : DatasetGroups
    ratios : tuple of float  target split fractions, must sum to 1
    max_evals : int          total FFE budget (same as SA for fair comparison)
    seed : int | None
    """

    def __init__(
        self,
        data,
        ratios: tuple[float, ...] = (0.70, 0.15, 0.15),
        max_evals: int = 300_000,
        seed: int | None = None,
    ) -> None:
        super().__init__(data, ratios, max_evals=max_evals, seed=seed)

    # Core algorithm
    def optimize(
        self,
        verbose: bool = True,
        log_interval: int = 10_000,
    ) -> SplitResult:
        """Run Random Search and return the best split found."""
        rng = np.random.default_rng(self.seed)
        t_start = time.perf_counter()

        best_assignment: np.ndarray | None = None
        best_actual:     np.ndarray | None = None
        best_cost = float("inf")
        cost_history: list[tuple[int, float]] = []
        n_evals = 0

        if verbose:
            print(
                f"[RS] {self.data.dataset_name}"
                f"  groups={self.data.n_groups}"
                f"  classes={self.data.n_classes}"
                f"  budget={self.max_evals:,} FFEs"
            )

        # Main loop: exactly max_evals FFEs
        for n_evals in range(1, self.max_evals + 1):

            assignment = rng.choice(
                N_SPLITS,
                size=self.data.n_groups,
                p=self.ratios,
            )

            actual = self._count_matrix(assignment)
            cost = self._mape_cost(actual)

            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment.copy()
                best_actual = actual.copy()
                cost_history.append((n_evals, best_cost))

            if n_evals % log_interval == 0:
                elapsed = time.perf_counter() - t_start
                if verbose:
                    print(
                        f"  evals {n_evals:>7,}"
                        f"  best_cost={best_cost:.4f}"
                        f"  elapsed={elapsed:.1f}s"
                    )

        elapsed = time.perf_counter() - t_start

        # Fallback: if max_evals == 0 or data is empty, return a trivial result
        if best_assignment is None:
            best_assignment = np.zeros(self.data.n_groups, dtype=np.intp)
            best_actual = self._count_matrix(best_assignment)
            best_cost = self._mape_cost(best_actual)

        return SplitResult(
            assignment=best_assignment,
            cost=best_cost,
            n_evals=n_evals,
            n_iterations=n_evals,
            converged=False,
            elapsed_time=elapsed,
            cost_history=cost_history,
            target_counts=self._target.copy(),
            actual_counts=best_actual,
        )
