"""Strategy interface shared by every splitting algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from ..problem import Budget, SplitProblem

__all__ = ["DEFAULT_MAX_EVALS", "Outcome", "Strategy", "resolve_max_evals"]

#: Evaluation budget assumed when the caller gives no stop condition at all.
DEFAULT_MAX_EVALS = 300_000

#: Stand-in for "unbounded" when wall-clock time is the real constraint.
_EFFECTIVELY_UNBOUNDED = 1 << 62


def resolve_max_evals(budget: Budget) -> int:
    """The evaluation ceiling a strategy should loop against.

    When only a time limit is given the evaluation count must not cap the run,
    or a fast machine would stop early with budget left on the clock.
    """
    if budget.max_evals is not None:
        return budget.max_evals
    if budget.time_limit is not None:
        return _EFFECTIVELY_UNBOUNDED
    return DEFAULT_MAX_EVALS


@dataclass
class Outcome:
    """What a strategy returns.

    Internal: :func:`splitkit.split` wraps this into the public ``SplitResult``,
    recomputing the count matrix from ``assignment`` so that no incrementally
    maintained state can drift into the reported numbers.

    assignment    : (G,) split index per group
    cost          : objective value (lower is better)
    n_evals       : objective evaluations consumed
    n_iterations  : algorithmic iterations
    converged     : stopped because the target cost was reached
    cost_history  : (n_evals, best_cost) snapshots
    lower_bound   : proven bound on the optimum, when the strategy can supply one
    proved_optimal: the returned assignment is provably optimal
    """

    assignment:     np.ndarray
    cost:           float
    n_evals:        int = 0
    n_iterations:   int = 0
    converged:      bool = False
    cost_history:   list[tuple[int, float]] = field(default_factory=list)
    lower_bound:    float | None = None
    proved_optimal: bool = False


class Strategy(ABC):
    """Base class for splitting strategies.

    Subclasses hold only their own hyper-parameters. Everything about the data,
    the objective and the stop conditions arrives through :meth:`run`, which keeps
    strategies stateless and reusable across problems.
    """

    #: Registry key, e.g. ``"annealing"``.
    name: ClassVar[str] = ""
    #: Same input always yields the same output; seeds are irrelevant.
    deterministic: ClassVar[bool] = False
    #: Honours the ``warm_start`` argument to :meth:`run`.
    supports_warm_start: ClassVar[bool] = False
    #: Optional distribution extras required, e.g. ``("ortools",)``.
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

    def params(self) -> dict:
        """Hyper-parameters, for reporting and reproducibility."""
        return {
            k: v for k, v in vars(self).items() if not k.startswith("_")
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        args = ", ".join(f"{k}={v!r}" for k, v in self.params().items())
        return f"{type(self).__name__}({args})"
