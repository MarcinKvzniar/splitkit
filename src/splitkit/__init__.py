"""splitkit — group-aware stratified dataset splitting.

Assign indivisible groups (patients, slides, identities, protein families) wholly to
train/val/test so that no group leaks across splits *and* each split mirrors the
global class distribution.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

from .api import evaluate, split
from .dataset import GroupedDataset
from .io import load_npz, save_npz
from .objectives import get_objective
from .problem import Budget, SplitProblem
from .result import SplitMapping, SplitResult
from .strategies import get_strategy, list_strategies
from .synthetic import from_preset, list_presets, make_synthetic

__all__ = [
    "Budget",
    "GroupedDataset",
    "SplitMapping",
    "SplitProblem",
    "SplitResult",
    "__version__",
    "evaluate",
    "from_preset",
    "get_objective",
    "get_strategy",
    "list_presets",
    "list_strategies",
    "load_npz",
    "make_synthetic",
    "save_npz",
    "split",
]
