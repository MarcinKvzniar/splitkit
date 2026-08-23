"""splitkit — group-aware stratified dataset splitting.

Assign indivisible groups (patients, slides, identities, protein families) wholly to
train/val/test so that no group leaks across splits *and* each split mirrors the
global class distribution.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

from .dataset import GroupedDataset
from .io import load_npz, save_npz
from .synthetic import from_preset, list_presets, make_synthetic

__all__ = [
    "GroupedDataset",
    "__version__",
    "from_preset",
    "list_presets",
    "load_npz",
    "make_synthetic",
    "save_npz",
]
