"""The grouped-dataset container every strategy operates on."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["GroupedDataset"]


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Building a dataset from a DataFrame requires pandas. "
            "Install it with: pip install 'splitkit[pandas]'"
        ) from exc
    return pd


def _factorize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous integer codes plus the sorted distinct values (deterministic order)."""
    try:
        uniques, codes = np.unique(values, return_inverse=True)
    except TypeError:  # unorderable mixed types, e.g. ints and strings
        keys = np.array([f"{type(v).__name__}:{v}" for v in values])
        _, first, codes = np.unique(keys, return_index=True, return_inverse=True)
        uniques = values[first]
    return codes.ravel(), uniques


def _is_missing(value: object) -> bool:
    """None, NaN, or pandas NA (whose comparisons are not booleans)."""
    try:
        return value is None or bool(value != value)
    except TypeError:
        return True


def _has_missing(values: np.ndarray) -> bool:
    """True if a 1-D array holds NaN, None or pandas NA."""
    if values.dtype.kind == "f":
        return bool(np.isnan(values).any())
    if values.dtype.kind == "O":
        return any(_is_missing(v) for v in values)
    return False


def _aggregate(codes: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum item-level ``values`` (N, C) into per-group totals (G, C)."""
    n_cols = values.shape[1]
    out = np.zeros((n_groups, n_cols), dtype=np.float64)
    for j in range(n_cols):
        out[:, j] = np.bincount(codes, weights=values[:, j], minlength=n_groups)
    return out


def _onehot_aggregate(
    codes: np.ndarray, label_codes: np.ndarray, n_groups: int, n_classes: int
) -> np.ndarray:
    """Count (group, class) co-occurrences without a one-hot matrix."""
    flat = codes * n_classes + label_codes
    counts = np.bincount(flat, minlength=n_groups * n_classes)
    return counts.reshape(n_groups, n_classes).astype(np.float64)


def _decode_binary(values: np.ndarray) -> np.ndarray:
    """Map a +/-1 indicator encoding onto 0/1, leaving other encodings alone."""
    finite = values[np.isfinite(values)]
    if finite.size and np.isin(finite, (-1, 1)).all() and (finite == -1).any():
        return (values + 1) / 2
    return values


@dataclass(frozen=True)
class GroupedDataset:
    """A dataset summarised as per-group class counts.

    ``group_vectors[i, c]`` is the count of class ``c`` in group ``i``;
    ``group_sizes[i]`` is the number of items in group ``i``.
    """

    group_ids:     np.ndarray            # (G,) unicode
    group_vectors: np.ndarray            # (G, C) float64
    group_sizes:   np.ndarray            # (G,) float64
    class_names:   tuple[str, ...]
    name:          str = "dataset"
    # (N,) group index per item; only set when built from item-level input.
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

    @property
    def n_groups(self) -> int:
        return int(self.group_vectors.shape[0])

    @property
    def n_classes(self) -> int:
        return int(self.group_vectors.shape[1])

    @property
    def total_items(self) -> int:
        return int(self.group_sizes.sum())

    @cached_property
    def global_class_counts(self) -> np.ndarray:
        return np.asarray(self.group_vectors.sum(axis=0))

    @property
    def global_class_frequencies(self) -> np.ndarray:
        counts = self.global_class_counts
        total = counts.sum()
        return counts / total if total > 0 else np.zeros_like(counts, dtype=float)

    @cached_property
    def class_group_counts(self) -> np.ndarray:
        """(C,) number of groups containing each class."""
        return np.asarray((self.group_vectors > 0).sum(axis=0))

    @cached_property
    def is_onehot(self) -> bool:
        """True when every item carries exactly one unit of class mass."""
        return bool(np.allclose(self.group_vectors.sum(axis=1), self.group_sizes))

    @property
    def n_items(self) -> int:
        """Number of items; exact row count when item provenance is tracked."""
        if self.item_group_index is None:
            return self.total_items
        return int(self.item_group_index.size)

    @property
    def has_items(self) -> bool:
        """Whether per-item indices can be recovered from a split."""
        return self.item_group_index is not None

    def subset(self, mask: np.ndarray) -> GroupedDataset:
        """Restrict to a selection of groups; item provenance is dropped."""
        mask = np.asarray(mask)
        return replace(
            self,
            group_ids=self.group_ids[mask],
            group_vectors=self.group_vectors[mask],
            group_sizes=self.group_sizes[mask],
            item_group_index=None,
        )

    @classmethod
    def from_counts(
        cls,
        group_vectors: ArrayLike,
        *,
        group_ids: Sequence[str] | None = None,
        class_names: Sequence[str] | None = None,
        group_sizes: ArrayLike | None = None,
        name: str = "dataset",
    ) -> GroupedDataset:
        """Build from an aggregated ``(n_groups, n_classes)`` count matrix."""
        vectors = np.ascontiguousarray(group_vectors, dtype=np.float64)
        if vectors.ndim != 2:
            raise ValueError(
                f"group_vectors must be 2-D (n_groups, n_classes), got shape "
                f"{vectors.shape}."
            )
        g, c = vectors.shape
        return cls(
            group_ids=(
                np.array([f"g{i}" for i in range(g)], dtype=np.str_)
                if group_ids is None
                else np.asarray(group_ids, dtype=np.str_)
            ),
            group_vectors=vectors,
            group_sizes=(
                vectors.sum(axis=1)
                if group_sizes is None
                else np.asarray(group_sizes, dtype=np.float64)
            ),
            class_names=(
                tuple(f"class_{j}" for j in range(c))
                if class_names is None
                else tuple(str(x) for x in class_names)
            ),
            name=name,
        )

    @classmethod
    def from_arrays(
        cls,
        groups: ArrayLike,
        y: ArrayLike,
        *,
        classes: Sequence[str] | None = None,
        name: str = "dataset",
        track_items: bool = True,
    ) -> GroupedDataset:
        """Build from item-level arrays.

        Parameters
        ----------
        groups
            ``(N,)`` group key per item; items sharing a key are never separated.
        y
            ``(N,)`` class labels, or ``(N, C)`` per-item counts or indicators.
        classes
            Column names when ``y`` is 2-D.
        track_items
            Keep per-item group indices so a split can return item indices.
        """
        groups = np.asarray(groups)
        y = np.asarray(y)
        if groups.ndim != 1:
            raise ValueError(f"groups must be 1-D, got shape {groups.shape}.")
        if len(groups) != len(y):
            raise ValueError(
                f"groups has {len(groups)} entries but y has {len(y)}; "
                f"they must align item for item."
            )
        if len(groups) == 0:
            raise ValueError("Cannot build a dataset from zero items.")
        if _has_missing(groups):
            raise ValueError("groups contains missing values; every item needs a group.")

        codes, group_ids = _factorize(groups)
        n_groups = len(group_ids)
        sizes = np.bincount(codes, minlength=n_groups).astype(np.float64)

        if y.ndim == 1:
            if _has_missing(y):
                raise ValueError("y contains NaN or None; every item needs a label.")
            label_codes, label_values = _factorize(y)
            vectors = _onehot_aggregate(
                codes, label_codes, n_groups, len(label_values)
            )
            class_names = tuple(str(v) for v in label_values)
        elif y.ndim == 2:
            values = _decode_binary(np.asarray(y, dtype=np.float64))
            if np.isnan(values).any():
                raise ValueError("y contains NaN; every item needs a value.")
            vectors = _aggregate(codes, values, n_groups)
            class_names = (
                tuple(f"class_{j}" for j in range(values.shape[1]))
                if classes is None
                else tuple(str(x) for x in classes)
            )
            if len(class_names) != values.shape[1]:
                raise ValueError(
                    f"classes has {len(class_names)} entries but y has "
                    f"{values.shape[1]} columns."
                )
        else:
            raise ValueError(f"y must be 1-D or 2-D, got shape {y.shape}.")

        return cls(
            group_ids=np.asarray(group_ids, dtype=np.str_),
            group_vectors=np.ascontiguousarray(vectors),
            group_sizes=sizes,
            class_names=class_names,
            name=name,
            item_group_index=(
                codes.astype(np.int32) if track_items else None
            ),
        )

    @classmethod
    def from_labels(cls, y: ArrayLike, groups: ArrayLike, **kwargs: Any) -> GroupedDataset:
        """``from_arrays`` with scikit-learn's ``(y, groups)`` argument order."""
        return cls.from_arrays(groups, y, **kwargs)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        group_col: str,
        label_col: str | None = None,
        label_cols: Sequence[str] | None = None,
        count_cols: Sequence[str] | None = None,
        size_col: str | None = None,
        name: str = "dataset",
        track_items: bool = True,
    ) -> GroupedDataset:
        """Build from a pandas DataFrame, with exactly one label mode.

        Parameters
        ----------
        label_col
            One categorical column; each row is an item of one class.
        label_cols
            Indicator columns for multi-label items (+/-1 is mapped to 0/1).
        count_cols
            Columns that already hold per-class counts; ``size_col`` gives the
            number of items a row stands for.
        """
        pd = _require_pandas()
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")

        if isinstance(label_cols, str):
            label_cols = [label_cols]
        if isinstance(count_cols, str):
            count_cols = [count_cols]
        chosen = [
            n
            for n, v in (
                ("label_col", label_col),
                ("label_cols", label_cols),
                ("count_cols", count_cols),
            )
            if v is not None
        ]
        if len(chosen) != 1:
            raise ValueError(
                "Provide exactly one of label_col, label_cols or count_cols; "
                f"got {chosen or 'none'}."
            )

        missing = [
            c
            for c in [group_col, label_col, size_col, *(label_cols or ()), *(count_cols or ())]
            if c is not None and c not in df.columns
        ]
        if missing:
            raise KeyError(
                f"Column(s) not found in the DataFrame: {missing}. "
                f"Available: {list(df.columns)}."
            )

        if df.empty:
            raise ValueError("Cannot build a dataset from an empty DataFrame.")
        if df[group_col].isna().any():
            raise ValueError(f"Column {group_col!r} contains missing group keys.")

        if label_col is not None:
            return cls.from_arrays(
                df[group_col].to_numpy(),
                df[label_col].to_numpy(),
                name=name,
                track_items=track_items,
            )

        cols = list(label_cols or count_cols or ())
        values = df[cols].to_numpy(dtype=np.float64)
        if np.isnan(values).any():
            raise ValueError(f"Columns {cols} contain missing values.")
        if label_cols is not None:
            values = _decode_binary(values)

        codes, group_ids = _factorize(df[group_col].to_numpy())
        n_groups = len(group_ids)
        vectors = _aggregate(codes, values, n_groups)

        if size_col is not None:
            sizes = np.bincount(
                codes, weights=df[size_col].to_numpy(dtype=np.float64),
                minlength=n_groups,
            )
        else:
            sizes = np.bincount(codes, minlength=n_groups).astype(np.float64)

        return cls(
            group_ids=np.asarray(group_ids, dtype=np.str_),
            group_vectors=np.ascontiguousarray(vectors),
            group_sizes=sizes,
            class_names=tuple(str(c) for c in cols),
            name=name,
            item_group_index=codes.astype(np.int32) if track_items else None,
        )

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
            self.class_names,
            self.global_class_frequencies,
            self.global_class_counts,
            strict=True,
        ):
            lines.append(f"{name:<40} {freq * 100:>9.3f}%  {int(count):>10,}")
        lines.append("=" * 62)
        return "\n".join(lines)
