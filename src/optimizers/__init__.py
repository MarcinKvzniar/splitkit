"""Split optimizers package.

All stochastic algorithms share the same max_evals FFE budget and cost function
for direct comparison.
"""

from .base import (
    N_SPLITS,
    SPLIT_NAMES,
    Optimizer,
    SplitResult,
    evaluate_assignment,
)
from .de import DifferentialEvolution
from .random_search import RandomSearch
from .sa import SimulatedAnnealing
from .stratified_group_k_fold import SGKFBaseline

__all__ = [
    "Optimizer",
    "SplitResult",
    "SimulatedAnnealing",
    "RandomSearch",
    "SGKFBaseline",
    "DifferentialEvolution",
    "SPLIT_NAMES",
    "N_SPLITS",
    "evaluate_assignment",
]
