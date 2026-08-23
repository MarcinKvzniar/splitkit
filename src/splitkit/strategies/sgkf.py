"""scikit-learn StratifiedGroupKFold reference baseline."""

from __future__ import annotations

import numpy as np

from ..problem import Budget, SplitProblem
from .base import Outcome, Strategy
from .registry import register_strategy

__all__ = ["SGKFBaseline", "largest_remainder"]

#: Cap on the synthetic item table handed to scikit-learn.
_MAX_MATERIALIZED_ITEMS = 300_000


def largest_remainder(ratios: np.ndarray, total: int) -> list[int]:
    """Apportion ``total`` indivisible units across ``ratios``.

    Rounds down, then hands the remaining units to the largest fractional parts.
    Unlike repeatedly nudging the single largest ratio, this cannot starve a split
    of its share, and it stays correct for any number of splits.
    """
    exact = ratios * total
    base = np.floor(exact).astype(int)
    remainder = total - int(base.sum())
    if remainder > 0:
        order = np.argsort(-(exact - base))
        base[order[:remainder]] += 1
    return base.tolist()


@register_strategy
class SGKFBaseline(Strategy):
    """Wraps scikit-learn's ``StratifiedGroupKFold`` as a comparison baseline.

    scikit-learn produces equally sized folds, so this over-splits into up to 20
    folds and then bins them to approximate the requested ratios.

    Kept for benchmark comparability only. It is slow (it must materialise a
    synthetic per-item table to call scikit-learn at all) and, above the item cap,
    it downscales the counts -- which quietly changes the problem being solved.
    A count-matrix-native reimplementation supersedes it.
    """

    name = "sgkf"
    deterministic = False
    requires = ("sklearn",)

    def __init__(self, max_folds: int = 20) -> None:
        self.max_folds = max_folds

    def run(
        self,
        problem: SplitProblem,
        budget: Budget,
        rng: np.random.Generator,
        warm_start: np.ndarray | None = None,
    ) -> Outcome:
        try:
            from sklearn.model_selection import StratifiedGroupKFold
        except ImportError as exc:  # pragma: no cover - exercised via extras
            raise ImportError(
                "The SGKF baseline requires scikit-learn. "
                "Install it with: pip install 'splitkit[sklearn]'"
            ) from exc

        data = problem.data
        k = problem.n_splits

        # scikit-learn needs per-item labels, so rebuild an item table from counts.
        total_counts = data.group_vectors.sum()
        scale = 1.0
        if total_counts > _MAX_MATERIALIZED_ITEMS:
            scale = _MAX_MATERIALIZED_ITEMS / total_counts

        y_list: list[int] = []
        groups_list: list[int] = []
        for g_idx in range(data.n_groups):
            for c_idx in range(data.n_classes):
                count = round(data.group_vectors[g_idx, c_idx] * scale)
                if count > 0:
                    y_list.extend([c_idx] * count)
                    groups_list.extend([g_idx] * count)

        y_arr = np.array(y_list)
        groups_arr = np.array(groups_list)

        class_counts = np.bincount(y_arr)
        min_class_count = int(class_counts[class_counts > 0].min())
        n_folds = min(self.max_folds, max(k, min_class_count))

        seed = int(rng.integers(np.iinfo(np.int32).max))
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        group_to_fold = np.zeros(data.n_groups, dtype=int)
        for fold_idx, (_, test_idx) in enumerate(
            cv.split(np.zeros((len(y_arr), 1)), y_arr, groups=groups_arr)
        ):
            group_to_fold[np.unique(groups_arr[test_idx])] = fold_idx

        # Bin folds into splits proportionally, guaranteeing each split gets >= 1.
        folds_per_split = largest_remainder(problem.ratios, n_folds)
        fold_to_split = np.empty(n_folds, dtype=int)
        fold = 0
        for s, n_take in enumerate(folds_per_split):
            for _ in range(n_take):
                fold_to_split[fold] = s
                fold += 1
        fold_to_split[fold:] = k - 1  # any residue from a zero-share split

        assignment = fold_to_split[group_to_fold]
        cost = problem.evaluate(assignment)

        return Outcome(
            assignment=assignment,
            cost=cost,
            n_evals=1,
            n_iterations=1,
            converged=True,
            cost_history=[(1, cost)],
        )
