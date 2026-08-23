"""Pluggable split objectives.

An objective scores a ``(K, C)`` matrix of per-split, per-class counts against the
ideal target matrix. Every objective is minimised, so lower is always better.

Objectives are *prepared* against a fixed target and weight vector, which lets the
constant part of the formula be computed once and reused across the hundreds of
thousands of evaluations a search performs.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "OBJECTIVES",
    "Objective",
    "PreparedObjective",
    "WeightedMAPE",
    "get_objective",
]


@runtime_checkable
class PreparedObjective(Protocol):
    """An objective bound to a specific target/weight pair."""

    separable: bool

    def total(self, counts: np.ndarray) -> float:
        """Total cost of a full ``(K, C)`` count matrix."""
        ...

    def rows(self, counts: np.ndarray) -> np.ndarray:
        """Per-split costs, shape ``(K,)``. Sums to :meth:`total` when separable."""
        ...

    def row(self, split_counts: np.ndarray, s: int) -> float:
        """Cost of a single split's count vector.

        Only meaningful when ``separable`` is True; it is what allows a move to be
        scored in O(C) by touching just the two splits it changes.
        """
        ...


@runtime_checkable
class Objective(Protocol):
    """Factory for a :class:`PreparedObjective`."""

    name: str
    separable: bool
    linearizable: bool

    def prepare(self, target: np.ndarray, weights: np.ndarray) -> PreparedObjective:
        ...


class _PreparedWeightedMAPE:
    """Weighted mean absolute percentage error, bound to a target.

    The denominator is precomputed, but the weights are deliberately *not* folded
    into it. Multiplying by ``w/(t+eps)`` is algebraically identical to dividing by
    ``(t+eps)`` and then scaling by ``w``, yet it rounds differently -- and simulated
    annealing turns a one-ulp difference in the cost into a different accept/reject
    decision and thence a different trajectory. Keeping the operation order fixed
    keeps runs reproducible across refactors.
    """

    separable = True

    __slots__ = ("_denom", "_target", "_weights")

    def __init__(self, target: np.ndarray, weights: np.ndarray, eps: float) -> None:
        self._target = target
        self._weights = weights
        self._denom = target + eps

    def total(self, counts: np.ndarray) -> float:
        rel_err = np.abs(counts - self._target) / self._denom
        return float((self._weights * rel_err).sum())

    def rows(self, counts: np.ndarray) -> np.ndarray:
        rel_err = np.abs(counts - self._target) / self._denom
        return (self._weights * rel_err).sum(axis=1)

    def row(self, split_counts: np.ndarray, s: int) -> float:
        rel_err = np.abs(split_counts - self._target[s]) / self._denom[s]
        return float((self._weights * rel_err).sum())


class WeightedMAPE:
    """``sum_{s,c} w_c * |actual - target| / (target + eps)``.

    Scale-free: every ``(split, class)`` cell contributes a relative error, so rare
    classes are not drowned out by frequent ones. Exactly linear in the counts,
    which is what lets the exact solver optimise it directly.
    """

    name = "wmape"
    separable = True
    linearizable = True

    def __init__(self, eps: float = 1.0) -> None:
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.eps = eps

    def prepare(
        self, target: np.ndarray, weights: np.ndarray
    ) -> _PreparedWeightedMAPE:
        return _PreparedWeightedMAPE(target, weights, self.eps)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"WeightedMAPE(eps={self.eps})"


#: Built-in objectives by name.
OBJECTIVES: dict[str, type] = {
    "wmape": WeightedMAPE,
}


def get_objective(objective: str | Objective) -> Objective:
    """Resolve a name or a ready-made objective into an :class:`Objective`."""
    if isinstance(objective, str):
        try:
            return OBJECTIVES[objective]()
        except KeyError:
            raise KeyError(
                f"Unknown objective {objective!r}. "
                f"Available: {', '.join(sorted(OBJECTIVES))}"
            ) from None
    return objective
