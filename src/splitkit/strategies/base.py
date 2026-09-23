"""Strategy interface shared by every splitting algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ..problem import Budget, SplitProblem

__all__ = ["DEFAULT_MAX_EVALS", "Outcome", "Strategy", "resolve_max_evals"]

DEFAULT_MAX_EVALS = 300_000

_EFFECTIVELY_UNBOUNDED = 1 << 62


def resolve_max_evals(budget: Budget) -> int:
    """Evaluation ceiling to loop against; unbounded when only a time limit is set."""
    if budget.max_evals is not None:
        return budget.max_evals
    if budget.time_limit is not None:
        return _EFFECTIVELY_UNBOUNDED
    return DEFAULT_MAX_EVALS


@dataclass
class Outcome:
    """What a strategy returns; wrapped into ``SplitResult`` by :func:`splitkit.split`."""

    assignment:     np.ndarray               # (G,) split index per group
    cost:           float
    n_evals:        int = 0
    n_iterations:   int = 0
    converged:      bool = False
    cost_history:   list[tuple[int, float]] = field(default_factory=list)
    lower_bound:    float | None = None
    proved_optimal: bool = False


class Strategy(ABC):
    """Base class for splitting strategies; instances hold only hyper-parameters."""

    name: ClassVar[str] = ""
    deterministic: ClassVar[bool] = False
    supports_warm_start: ClassVar[bool] = False
    requires: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def run(
        self,
        problem: SplitProblem,
        budget: Budget,
        rng: np.random.Generator,
        warm_start: np.ndarray | None = None,
    ) -> Outcome:
        """Search for a low-cost assignment within ``budget``."""

    def params(self) -> dict[str, Any]:
        """Public hyper-parameters, for reporting and reproducibility."""
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def __repr__(self) -> str:  # pragma: no cover
        args = ", ".join(f"{k}={v!r}" for k, v in self.params().items())
        return f"{type(self).__name__}({args})"
