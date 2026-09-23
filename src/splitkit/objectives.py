"""Split objectives. Every objective is minimised."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "OBJECTIVES",
    "LinearObjective",
    "Objective",
    "PreparedObjective",
    "WeightedMAPE",
    "get_objective",
]


@runtime_checkable
class PreparedObjective(Protocol):
    """An objective bound to a fixed target and weight vector."""

    separable: bool

    def total(self, counts: np.ndarray) -> float:
        """Cost of a full ``(K, C)`` count matrix."""
        ...

    def rows(self, counts: np.ndarray) -> np.ndarray:
        """Per-split costs, shape ``(K,)``."""
        ...

    def row(self, split_counts: np.ndarray, s: int) -> float:
        """Cost of split ``s`` alone; only meaningful when ``separable``."""
        ...


@runtime_checkable
class LinearObjective(PreparedObjective, Protocol):
    """A prepared objective equal to ``sum(cell_weights * |counts - target|)``."""

    cell_weights: np.ndarray
    target: np.ndarray


@runtime_checkable
class Objective(Protocol):
    """Factory for a :class:`PreparedObjective`."""

    name: str
    separable: bool
    linearizable: bool

    def prepare(self, target: np.ndarray, weights: np.ndarray) -> PreparedObjective: ...


class _PreparedWeightedMAPE:
    # Weights are deliberately not folded into the denominator: w/(t+eps) rounds
    # differently from (1/(t+eps))*w, and annealing turns a one-ulp difference into a
    # different accept/reject decision. A fixed operation order keeps runs reproducible.

    separable = True

    __slots__ = ("_denom", "_target", "_weights")

    def __init__(self, target: np.ndarray, weights: np.ndarray, eps: float) -> None:
        self._target = target
        self._weights = weights
        self._denom = target + eps

    @property
    def target(self) -> np.ndarray:
        return self._target

    @property
    def cell_weights(self) -> np.ndarray:
        return np.asarray(self._weights / self._denom)

    def total(self, counts: np.ndarray) -> float:
        rel_err = np.abs(counts - self._target) / self._denom
        return float((self._weights * rel_err).sum())

    def rows(self, counts: np.ndarray) -> np.ndarray:
        rel_err = np.abs(counts - self._target) / self._denom
        return np.asarray((self._weights * rel_err).sum(axis=1))

    def row(self, split_counts: np.ndarray, s: int) -> float:
        rel_err = np.abs(split_counts - self._target[s]) / self._denom[s]
        return float((self._weights * rel_err).sum())


class WeightedMAPE:
    """``sum_{s,c} w_c * |actual - target| / (target + eps)``, linear in the counts."""

    name = "wmape"
    separable = True
    linearizable = True

    def __init__(self, eps: float = 1.0) -> None:
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.eps = eps

    def prepare(self, target: np.ndarray, weights: np.ndarray) -> _PreparedWeightedMAPE:
        return _PreparedWeightedMAPE(target, weights, self.eps)

    def __repr__(self) -> str:  # pragma: no cover
        return f"WeightedMAPE(eps={self.eps})"


OBJECTIVES: dict[str, type[Objective]] = {
    "wmape": WeightedMAPE,
}


def get_objective(objective: str | Objective) -> Objective:
    """Resolve a name or an instance into an :class:`Objective`."""
    if isinstance(objective, str):
        try:
            return OBJECTIVES[objective]()
        except KeyError:
            raise KeyError(
                f"Unknown objective {objective!r}. Available: {', '.join(sorted(OBJECTIVES))}"
            ) from None
    return objective
