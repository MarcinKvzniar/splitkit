"""The grouped-dataset container shared by every strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

__all__ = ["GroupedDataset"]


@dataclass(frozen=True)
class GroupedDataset:
    """A dataset summarised as per-group class counts.

    This is the only representation the optimizers ever see, which is what makes
    splitkit domain-agnostic: image tiles per slide, images per patient, sequences
    per protein family and rows per customer all reduce to the same count matrix.

    group_vectors[i, c] = total count of class c across all items in group i
    group_sizes[i]      = number of items in group i
    """

    group_ids:     np.ndarray            # (G,) unicode
    group_vectors: np.ndarray            # (G, C) float64
    group_sizes:   np.ndarray            # (G,) float64
    class_names:   tuple[str, ...]
    name:          str = "dataset"

    # Item-level provenance. Present only when the dataset was built from item-level
    # input; without it a split can name groups but cannot produce item indices.
    item_group_index: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        g, c = self.group_vectors.shape
        if self.group_ids.shape != (g,):
            raise ValueError(
                f"group_ids has length {self.group_ids.shape[0]}, "
                f"but group_vectors has {g} rows."
            )
        if self.group_sizes.shape != (g,):
            raise ValueError(
                f"group_sizes has length {self.group_sizes.shape[0]}, "
                f"but group_vectors has {g} rows."
            )
        if len(self.class_names) != c:
            raise ValueError(
                f"class_names has {len(self.class_names)} entries, "
                f"but group_vectors has {c} columns."
            )
        if np.any(self.group_vectors < 0):
            raise ValueError("group_vectors must be non-negative.")

    # Shape
    @property
    def n_groups(self) -> int:
        return int(self.group_vectors.shape[0])

    @property
    def n_classes(self) -> int:
        return int(self.group_vectors.shape[1])

    @property
    def total_items(self) -> int:
        return int(self.group_sizes.sum())

    # Class statistics
    @cached_property
    def global_class_counts(self) -> np.ndarray:
        return self.group_vectors.sum(axis=0)

    @property
    def global_class_frequencies(self) -> np.ndarray:
        counts = self.global_class_counts
        total = counts.sum()
        return counts / total if total > 0 else np.zeros_like(counts, dtype=float)

    @cached_property
    def class_group_counts(self) -> np.ndarray:
        """(C,) number of groups in which each class occurs at all.

        A class occurring in fewer than K groups cannot be spread across K splits,
        so this drives the unstratifiable-class mask.
        """
        return (self.group_vectors > 0).sum(axis=0)

    @cached_property
    def is_onehot(self) -> bool:
        """True when every item contributes exactly one unit of class mass.

        For such data, matching class counts implies matching item counts. For
        multi-label or soft/pixel counts it does not, and the split objective needs
        an explicit item-count term to avoid silently unbalanced splits.
        """
        return bool(np.allclose(self.group_vectors.sum(axis=1), self.group_sizes))

    def summary(self) -> str:
        sizes = self.group_sizes
        lines = [
            "=" * 62,
            f"Dataset : {self.name}",
            f"Groups  : {self.n_groups:,}",
            f"Items   : {self.total_items:,}",
            f"Classes : {self.n_classes}",
            f"Group sizes - min:{sizes.min():g}  max:{sizes.max():g}  "
            f"mean:{sizes.mean():.1f}  median:{np.median(sizes):.1f}",
            "=" * 62,
            f"{'Class':<40} {'Frequency':>10}  {'Count':>10}",
            "-" * 62,
        ]
        for name, freq, count in zip(
            self.class_names, self.global_class_frequencies, self.global_class_counts
        ):
            lines.append(f"{name:<40} {freq * 100:>9.3f}%  {int(count):>10,}")
        lines.append("=" * 62)
        return "\n".join(lines)
