"""Split strategies.

All stochastic strategies share the same evaluation budget and cost function, so
their results are directly comparable.
"""

from .annealing import SimulatedAnnealing
from .base import N_SPLITS, SPLIT_NAMES, Optimizer, SplitResult
from .evolution import DifferentialEvolution
from .random_search import RandomSearch
from .sgkf import SGKFBaseline

__all__ = [
    "N_SPLITS",
    "SPLIT_NAMES",
    "DifferentialEvolution",
    "Optimizer",
    "RandomSearch",
    "SGKFBaseline",
    "SimulatedAnnealing",
    "SplitResult",
]
