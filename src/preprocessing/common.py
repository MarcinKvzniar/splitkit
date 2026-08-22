"""Shared data format used by preprocessors and optimizers."""

import os
import pickle
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class DatasetGroups:
    """Grouped dataset container.

    group_vectors[i, c] = total count of class c across all items in group i
    group_sizes[i]      = number of items in group i
    """
    dataset_name: str
    group_ids: List[str]
    group_vectors: np.ndarray
    group_sizes: np.ndarray
    class_names: List[str]

    @property
    def n_groups(self) -> int:
        return len(self.group_ids)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def global_class_counts(self) -> np.ndarray:
        return self.group_vectors.sum(axis=0)

    @property
    def global_class_frequencies(self) -> np.ndarray:
        counts = self.global_class_counts
        total = counts.sum()
        return counts / total if total > 0 else np.zeros_like(counts, dtype=float)

    @property
    def total_items(self) -> int:
        return int(self.group_sizes.sum())

    def summary(self) -> str:
        sizes = self.group_sizes
        freqs = self.global_class_frequencies
        counts = self.global_class_counts

        lines = [
            f"{'=' * 60}",
            f"Dataset : {self.dataset_name}",
            f"Groups  : {self.n_groups}",
            f"Items   : {self.total_items}",
            f"Classes : {self.n_classes}",
            f"Group sizes - min:{sizes.min()}  max:{sizes.max()}  "
            f"mean:{sizes.mean():.1f}  median:{np.median(sizes):.1f}",
            f"{'=' * 60}",
            f"{'Class':<40} {'Frequency':>10}  {'Count':>10}",
            f"{'-' * 62}",
        ]
        for name, freq, count in zip(self.class_names, freqs, counts):
            lines.append(f"{name:<40} {freq * 100:>9.3f}%  {int(count):>10,}")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


def save_dataset(data: DatasetGroups, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"[saved] {data.dataset_name} -> {path}")


def load_dataset(path: str) -> DatasetGroups:
    with open(path, "rb") as f:
        return pickle.load(f)
