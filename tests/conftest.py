"""Shared fixtures.

The datasets here are deliberately tiny and hand-written: their expected costs can
be computed by hand or by exhaustive enumeration, which is what makes them useful
as ground truth rather than as regression snapshots.
"""

from __future__ import annotations

import numpy as np
import pytest

from _helpers import make_dataset
from splitkit.dataset import GroupedDataset
from splitkit.problem import SplitProblem


@pytest.fixture
def tiny() -> GroupedDataset:
    """6 groups, 3 classes, one-hot. Small enough to enumerate exhaustively."""
    return make_dataset(
        [
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10],
            [5, 5, 0],
            [0, 5, 5],
            [5, 0, 5],
        ],
        name="tiny",
    )


@pytest.fixture
def uniform() -> GroupedDataset:
    """8 identical groups: every assignment with the same split sizes ties."""
    return make_dataset([[4, 4]] * 8, name="uniform")


@pytest.fixture
def dense() -> GroupedDataset:
    """10 groups, every class present in every group: stratifiable for any K<=10."""
    return make_dataset(
        [[6, 3, 1], [5, 4, 2], [7, 2, 1], [4, 5, 3], [6, 4, 2],
         [5, 3, 1], [8, 2, 2], [3, 6, 1], [5, 5, 2], [7, 3, 3]],
        name="dense",
    )


@pytest.fixture
def multilabel() -> GroupedDataset:
    """Class mass exceeds item count, as with multi-label targets."""
    return make_dataset(
        [[3, 2, 1], [1, 1, 1], [4, 4, 2], [2, 0, 2], [1, 3, 0], [0, 2, 2]],
        sizes=[2, 1, 3, 2, 2, 1],
        name="multilabel",
    )


@pytest.fixture
def soft_counts() -> GroupedDataset:
    """Vectors in different units from sizes, as with BCSS pixel counts."""
    return make_dataset(
        [[1000, 200], [50, 4000], [700, 700], [2500, 100], [80, 900], [300, 300]],
        sizes=[4, 12, 6, 9, 3, 2],
        name="soft_counts",
    )


@pytest.fixture
def unstratifiable() -> GroupedDataset:
    """Class 2 occurs in only two groups, so it cannot span three splits."""
    return make_dataset(
        [
            [10, 5, 0],
            [8, 6, 0],
            [9, 4, 0],
            [7, 7, 0],
            [6, 3, 4],
            [5, 8, 3],
        ],
        name="unstratifiable",
    )


@pytest.fixture
def tiny_problem(tiny) -> SplitProblem:
    return SplitProblem.build(tiny, (0.5, 0.25, 0.25))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260823)
