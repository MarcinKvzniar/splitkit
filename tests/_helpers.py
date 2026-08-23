"""Construction helpers shared by the test modules."""

from __future__ import annotations

import numpy as np

from splitkit.dataset import GroupedDataset


def make_dataset(
    vectors,
    *,
    sizes=None,
    class_names=None,
    name="test",
    item_group_index=None,
) -> GroupedDataset:
    """Build a GroupedDataset from a plain nested list of counts."""
    arr = np.asarray(vectors, dtype=np.float64)
    g, c = arr.shape
    return GroupedDataset(
        group_ids=np.array([f"g{i}" for i in range(g)], dtype=np.str_),
        group_vectors=arr,
        group_sizes=(
            arr.sum(axis=1) if sizes is None else np.asarray(sizes, dtype=np.float64)
        ),
        class_names=(
            tuple(f"c{j}" for j in range(c)) if class_names is None
            else tuple(class_names)
        ),
        name=name,
        item_group_index=item_group_index,
    )
