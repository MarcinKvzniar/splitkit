"""Split strategies and their registry."""

from __future__ import annotations

from .annealing import SimulatedAnnealing
from .base import Outcome, Strategy
from .evolution import DifferentialEvolution
from .exact import ExactMILP
from .random_search import RandomSearch
from .registry import get_strategy, list_strategies, register_strategy
from .sgkf import SGKFBaseline

__all__ = [
    "DifferentialEvolution",
    "ExactMILP",
    "Outcome",
    "RandomSearch",
    "SGKFBaseline",
    "SimulatedAnnealing",
    "Strategy",
    "get_strategy",
    "list_strategies",
    "register_strategy",
]
