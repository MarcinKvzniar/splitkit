"""Split strategies and their registry.

Every strategy implements :class:`Strategy`, receives the same
:class:`~splitkit.problem.SplitProblem` and :class:`~splitkit.problem.Budget`, and
returns an :class:`Outcome`, so results are directly comparable.
"""

from __future__ import annotations

from .annealing import SimulatedAnnealing
from .base import Outcome, Strategy
from .evolution import DifferentialEvolution
from .random_search import RandomSearch
from .registry import get_strategy, list_strategies, register_strategy
from .sgkf import SGKFBaseline

__all__ = [
    "DifferentialEvolution",
    "Outcome",
    "RandomSearch",
    "SGKFBaseline",
    "SimulatedAnnealing",
    "Strategy",
    "get_strategy",
    "list_strategies",
    "register_strategy",
]
