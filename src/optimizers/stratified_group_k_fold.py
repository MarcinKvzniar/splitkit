"""StratifiedGroupKFold baseline optimizer."""

import time

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .base import Optimizer, SplitResult


class SGKFBaseline(Optimizer):
    """Scikit-Learn StratifiedGroupKFold baseline.

    Because SGKF creates K equally-sized folds, this wrapper over-splits
    using K=20 (if possible) and groups the resulting folds to approximate
    the target ratios (e.g., 14 folds for 70%, 3 for 15%, 3 for 15%).
    """

    def __init__(
        self,
        data,
        ratios: tuple[float, ...] = (0.70, 0.15, 0.15),
        max_evals: int = 1,
        seed: int | None = None,
    ) -> None:
        super().__init__(data, ratios, max_evals=max_evals, seed=seed)

    def optimize(
        self,
        verbose: bool = True,
        log_interval: int = 10_000,
    ) -> SplitResult:
        t_start = time.perf_counter()

        y_list = []
        groups_list = []

        total_counts = self.data.group_vectors.sum()
        scale = 1.0
        if total_counts > 300_000:
            scale = 300_000 / total_counts

        for g_idx in range(self.data.n_groups):
            for c_idx in range(self.data.n_classes):
                raw_count = self.data.group_vectors[g_idx, c_idx]
                count = int(round(raw_count * scale))
                if count > 0:
                    y_list.extend([c_idx] * count)
                    groups_list.extend([g_idx] * count)

        y_arr = np.array(y_list)
        groups_arr = np.array(groups_list)

        X_dummy = np.zeros((len(y_arr), 1))

        class_counts = np.bincount(y_arr)
        min_class_count = class_counts[class_counts > 0].min()

        n_splits = min(20, max(2, min_class_count))

        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        group_to_fold = np.zeros(self.data.n_groups, dtype=int)
        for fold_idx, (_, test_idx) in enumerate(cv.split(X_dummy, y_arr, groups=groups_arr)):
            fold_groups = np.unique(groups_arr[test_idx])
            group_to_fold[fold_groups] = fold_idx

        target_folds = [int(round(r * n_splits)) for r in self.ratios]

        while sum(target_folds) < n_splits:
            target_folds[np.argmax(self.ratios)] += 1
        while sum(target_folds) > n_splits:
            target_folds[np.argmax(self.ratios)] -= 1

        fold_to_split = {}
        current_split = 0
        folds_assigned = 0

        for fold in range(n_splits):
            fold_to_split[fold] = current_split
            folds_assigned += 1
            if folds_assigned >= target_folds[current_split]:
                current_split += 1
                folds_assigned = 0
                if current_split >= len(self.ratios):
                    current_split = len(self.ratios) - 1

        assignment = np.array([fold_to_split[group_to_fold[g]] for g in range(self.data.n_groups)])

        actual = self._count_matrix(assignment)
        cost = self._mape_cost(actual)

        elapsed = time.perf_counter() - t_start

        if verbose:
            print(f"  [SGKF] heuristic finished with cost={cost:.4f} in {elapsed:.3f}s")

        return SplitResult(
            assignment=assignment,
            cost=cost,
            n_evals=1,
            n_iterations=1,
            converged=True,
            elapsed_time=elapsed,
            cost_history=[(1, cost), (self.max_evals, cost)] if self.max_evals > 0 else [(1, cost)],
            target_counts=self._target.copy(),
            actual_counts=actual,
        )
