"""Abstract base for stratified split optimizers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..preprocessing.common import DatasetGroups

# Shared constants
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
N_SPLITS: int = len(SPLIT_NAMES)


@dataclass
class SplitResult:
    """Standard result returned by every optimizer.

    assignment    : ndarray (n_groups,)           split index {0,1,2} per group
    cost          : float                         objective value (lower=better)
    n_evals       : int                           FFEs consumed
    n_iterations  : int                           algorithmic iterations
    converged     : bool                          True if target_cost reached
    elapsed_time  : float                         wall-clock seconds
    cost_history  : list[(n_evals, cost)]         snapshots on FFE axis
    target_counts : ndarray (N_SPLITS, n_classes)
    actual_counts : ndarray (N_SPLITS, n_classes)
    """

    assignment:    np.ndarray
    cost:          float
    n_evals:       int
    n_iterations:  int
    converged:     bool
    elapsed_time:  float
    cost_history:  list = field(default_factory=list)
    target_counts: np.ndarray = field(default_factory=lambda: np.empty(0))
    actual_counts: np.ndarray = field(default_factory=lambda: np.empty(0))


class Optimizer(ABC):
    """Abstract base for group-aware stratified train/val/test optimizers.

    All subclasses use the same _mape_cost() objective and receive the same
    max_evals FFE budget, guaranteeing fair comparison.

    Params: data, ratios=(0.70,0.15,0.15), max_evals=300_000, seed=None
    Pre-built: self._weights (n_classes,), self._target (N_SPLITS, n_classes)
    """

    def __init__(
        self,
        data: DatasetGroups,
        ratios: tuple[float, ...] = (0.70, 0.15, 0.15),
        max_evals: int = 300_000,
        seed: int | None = None,
    ) -> None:
        if len(ratios) != N_SPLITS:
            raise ValueError(f"Expected {N_SPLITS} split ratios, got {len(ratios)}.")
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {sum(ratios):.8f}."
            )
        if max_evals < 1:
            raise ValueError(f"max_evals must be at least 1, got {max_evals}.")

        self.data = data
        self.ratios = np.asarray(ratios, dtype=np.float64)
        self.max_evals = max_evals
        self.seed = seed
        self._weights: np.ndarray = self._build_weights()
        self._target:  np.ndarray = self._build_target_counts()

    # Public interface
    def evaluate(self, assignment: np.ndarray, eps: float = 1.0) -> float:
        """Compute and return the weighted MAPE cost for the given assignment."""
        actual = self._count_matrix(assignment)
        return self._mape_cost(actual, eps)

    @abstractmethod
    def optimize(
        self,
        verbose: bool = True,
        log_interval: int = 10_000,
    ) -> SplitResult:
        """Run the optimizer and return the best SplitResult found."""

    # Protected cost utilities
    def _build_weights(self) -> np.ndarray:
        """Inverse-frequency class weights, mean-normalised to 1.
        Classes that appear in fewer than N_SPLITS groups are unstratifiable,
        so they are excluded from the cost.
        """
        counts = self.data.global_class_counts.astype(np.float64)
        inv_freq = counts.sum() / (self.data.n_classes * counts + 1e-6)
        n_groups_per_class = (self.data.group_vectors > 0).sum(axis=0)
        stratifiable = (n_groups_per_class >= N_SPLITS).astype(np.float64)
        return inv_freq * stratifiable

    def _build_target_counts(self) -> np.ndarray:
        """Ideal item counts: target[s, c] = global_count[c] * ratio[s]."""
        return (
            self.data.global_class_counts[np.newaxis, :].astype(np.float64)
            * self.ratios[:, np.newaxis]
        )

    def _count_matrix(self, assignment: np.ndarray) -> np.ndarray:
        """Return (N_SPLITS x n_classes) item count matrix for the given assignment."""
        counts = np.zeros((N_SPLITS, self.data.n_classes), dtype=np.float64)
        for s in range(N_SPLITS):
            mask = assignment == s
            if mask.any():
                counts[s] = self.data.group_vectors[mask].sum(axis=0)
        return counts

    def _mape_cost(self, actual: np.ndarray, eps: float = 1.0) -> float:
        """Weighted MAPE: sum_{s,c} w_c * |actual-target| / (target + eps)."""
        rel_err = np.abs(actual - self._target) / (self._target + eps)
        return float((self._weights * rel_err).sum())


# Standalone helper for scoring external baselines (e.g. sklearn splitters)
def evaluate_assignment(
    data: DatasetGroups,
    assignment: np.ndarray,
    ratios: tuple[float, ...] = (0.70, 0.15, 0.15),
    eps: float = 1.0,
) -> float:
    """Score an externally produced assignment with the shared cost function."""
    ratios_ = np.asarray(ratios, dtype=np.float64)
    counts = data.global_class_counts.astype(np.float64)
    weights = counts.sum() / (data.n_classes * counts + 1e-6)
    target = counts[np.newaxis, :] * ratios_[:, np.newaxis]

    actual = np.zeros((N_SPLITS, data.n_classes), dtype=np.float64)
    for s in range(N_SPLITS):
        mask = assignment == s
        if mask.any():
            actual[s] = data.group_vectors[mask].sum(axis=0)

    rel_err = np.abs(actual - target) / (target + eps)
    return float((weights * rel_err).sum())
