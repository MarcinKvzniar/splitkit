"""What a split returns."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from .dataset import GroupedDataset

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["SplitMapping", "SplitResult"]


class SplitMapping(Mapping[str, np.ndarray]):
    """Ordered split-name -> array mapping; use :meth:`astuple` to unpack arrays."""

    __slots__ = ("_data", "_names")

    def __init__(self, names: tuple[str, ...], arrays: tuple[np.ndarray, ...]) -> None:
        self._names = names
        self._data = dict(zip(names, arrays, strict=True))

    def __getitem__(self, key: str) -> np.ndarray:
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(
                f"No split named {key!r}. Available: {', '.join(self._names)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __getattr__(self, name: str) -> np.ndarray:
        try:
            return self.__getitem__(name)
        except KeyError as exc:
            raise AttributeError(str(exc)) from None

    def astuple(self) -> tuple[np.ndarray, ...]:
        """Arrays in split order, for ``train, val, test = ...``."""
        return tuple(self._data[n] for n in self._names)

    def __repr__(self) -> str:  # pragma: no cover
        sizes = ", ".join(f"{n}={len(self._data[n])}" for n in self._names)
        return f"SplitMapping({sizes})"


@dataclass(frozen=True)
class SplitResult:
    """A completed split; ``actual_counts`` is always recomputed from ``assignment``."""

    names:           tuple[str, ...]
    assignment:      np.ndarray            # (G,) split index per group
    ratios:          np.ndarray            # (K,) requested, normalised
    target_counts:   np.ndarray            # (K, C')
    actual_counts:   np.ndarray            # (K, C')
    cost:            float
    strategy:        str
    dataset:         GroupedDataset = field(repr=False)
    strategy_params: dict[str, Any] = field(default_factory=dict, repr=False)
    seed:            int | None = None
    n_evals:         int = 0
    n_iterations:    int = 0
    elapsed_time:    float = 0.0
    converged:       bool = False
    cost_history:    list[tuple[int, float]] = field(default_factory=list, repr=False)
    lower_bound:     float | None = None
    proved_optimal:  bool = False
    dropped_classes: tuple[str, ...] = ()

    @property
    def n_splits(self) -> int:
        return len(self.names)

    def mask(self, name: str) -> np.ndarray:
        """(G,) boolean mask selecting the groups in ``name``."""
        return np.asarray(self.assignment == self._index(name))

    @cached_property
    def groups(self) -> SplitMapping:
        """Split name -> array of group ids."""
        ids = self.dataset.group_ids
        return SplitMapping(
            self.names,
            tuple(ids[self.assignment == s] for s in range(self.n_splits)),
        )

    def item_mask(self, name: str) -> np.ndarray:
        """(N,) boolean mask selecting the items in ``name``."""
        index = self._index(name)  # validate the name before the provenance check
        return np.asarray(self._item_split() == index)

    @cached_property
    def indices(self) -> SplitMapping:
        """Split name -> array of item indices into your original data."""
        item_split = self._item_split()
        return SplitMapping(
            self.names,
            tuple(np.flatnonzero(item_split == s) for s in range(self.n_splits)),
        )

    def to_group_ids(self) -> dict[str, np.ndarray]:
        """Plain dict of split name -> group ids."""
        return dict(self.groups)

    def to_indices(self) -> dict[str, np.ndarray]:
        """Plain dict of split name -> item indices."""
        return dict(self.indices)

    @property
    def item_counts(self) -> np.ndarray:
        """(K,) number of items in each split."""
        sizes = self.dataset.group_sizes
        return np.array(
            [sizes[self.assignment == s].sum() for s in range(self.n_splits)]
        )

    @property
    def achieved_ratios(self) -> np.ndarray:
        """(K,) realised share of *items* per split."""
        counts = self.item_counts
        total = counts.sum()
        return counts / total if total else np.zeros_like(counts)

    @property
    def relative_error(self) -> np.ndarray:
        """(K, C') per-cell relative deviation from target."""
        return np.asarray(
            np.abs(self.actual_counts - self.target_counts) / (self.target_counts + 1.0)
        )

    def worst_cell(self) -> tuple[str, str, float]:
        """The ``(split, class, relative error)`` furthest from target."""
        err = self.relative_error
        s, c = np.unravel_index(int(np.argmax(err)), err.shape)
        class_names = self._column_names()
        return self.names[s], class_names[c], float(err[s, c])

    def empty_classes(self) -> dict[str, tuple[str, ...]]:
        """Classes entirely absent from a split."""
        class_names = self._column_names()
        out: dict[str, tuple[str, ...]] = {}
        for s, name in enumerate(self.names):
            missing = tuple(
                class_names[c]
                for c in range(self.actual_counts.shape[1])
                if self.actual_counts[s, c] == 0
                and self.target_counts[s, c] > 0
            )
            if missing:
                out[name] = missing
        return out

    def gap(self) -> float | None:
        """Relative distance to the proven lower bound, if any."""
        if self.lower_bound is None:
            return None
        return (self.cost - self.lower_bound) / max(abs(self.lower_bound), 1e-12)

    def to_frame(self) -> pd.DataFrame:
        """One row per group: id, split, size and per-class counts."""
        pd = _require_pandas()
        frame = pd.DataFrame(
            {
                "group_id": self.dataset.group_ids,
                "split": [self.names[s] for s in self.assignment],
                "size": self.dataset.group_sizes,
            }
        )
        for j, cls in enumerate(self.dataset.class_names):
            frame[cls] = self.dataset.group_vectors[:, j]
        return frame

    def counts_frame(self) -> pd.DataFrame:
        """Target vs actual per (split, class), with the relative error."""
        pd = _require_pandas()
        class_names = self._column_names()
        rows = []
        for s, name in enumerate(self.names):
            for c, cls in enumerate(class_names):
                rows.append(
                    {
                        "split": name,
                        "class": cls,
                        "target": self.target_counts[s, c],
                        "actual": self.actual_counts[s, c],
                        "rel_error": self.relative_error[s, c],
                    }
                )
        return pd.DataFrame(rows).set_index(["split", "class"])

    def assign_column(
        self,
        df: pd.DataFrame,
        group_col: str,
        column: str = "split",
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Label each row of ``df`` with the split its group landed in."""
        _require_pandas()
        mapping: dict[str, str] = {}
        for s, name in enumerate(self.names):
            for gid in self.dataset.group_ids[self.assignment == s]:
                mapping[gid] = name

        # Stringify keys exactly as the dataset builders did, so dates etc. match.
        keys = np.asarray(df[group_col].to_numpy()).astype(np.str_)
        target = df if inplace else df.copy()
        target[column] = [mapping.get(k) for k in keys]
        return target

    def summary(self) -> str:
        """Human-readable report of split quality."""
        lines = [
            "=" * 68,
            f"Split: {self.dataset.name}  ({self.n_splits} splits, "
            f"{self.dataset.n_groups:,} groups, {self.dataset.n_classes} classes)",
            f"Strategy: {self.strategy}   cost: {self.cost:.6g}"
            + ("   [PROVEN OPTIMAL]" if self.proved_optimal else ""),
        ]
        if self.lower_bound is not None and not self.proved_optimal:
            lines.append(
                f"Lower bound: {self.lower_bound:.6g}   gap: {self.gap():.2%}"
            )
        lines += [
            f"Evaluations: {self.n_evals:,}   time: {self.elapsed_time:.2f}s",
            "-" * 68,
            f"{'split':<12}{'groups':>10}{'items':>12}"
            f"{'requested':>12}{'achieved':>12}",
        ]

        achieved = self.achieved_ratios
        items = self.item_counts
        for s, name in enumerate(self.names):
            lines.append(
                f"{name:<12}{int((self.assignment == s).sum()):>10,}"
                f"{int(items[s]):>12,}{self.ratios[s]:>11.1%}{achieved[s]:>12.1%}"
            )

        lines.append("-" * 68)
        split_name, class_name, err = self.worst_cell()
        lines.append(
            f"Worst class balance: {class_name!r} in {split_name!r} "
            f"off target by {err:.1%}"
        )

        tags = {"warning": "WARNING  ", "note": "NOTE     "}
        for level, message in self._issues():
            lines.append(tags[level] + message.replace("\n", "\n         "))
        lines.append("=" * 68)
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()

    def __rich__(self) -> Any:
        """Styled report for ``rich.print(result)``."""
        from ._console import render_report

        return render_report(self)

    def _issues(self) -> list[tuple[str, str]]:
        """``(level, message)`` pairs for empty and unstratifiable classes."""
        issues = []
        for name, classes in self.empty_classes().items():
            issues.append(("warning", f"{name!r} contains no: {_shorten(classes)}"))
        if self.dropped_classes:
            issues.append((
                "note",
                f"{len(self.dropped_classes)} class(es) excluded from the objective as "
                f"unstratifiable: {_shorten(self.dropped_classes)}\n"
                "They occur in too few groups to appear in every split.",
            ))
        return issues

    def _index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(
                f"No split named {name!r}. Available: {', '.join(self.names)}"
            ) from None

    def _column_names(self) -> tuple[str, ...]:
        """Class names, plus the item-count pseudo-class if present."""
        names = tuple(self.dataset.class_names)
        if self.actual_counts.shape[1] == len(names) + 1:
            return (*names, "<item count>")
        return names

    def _item_split(self) -> np.ndarray:
        index = self.dataset.item_group_index
        if index is None:
            raise ValueError(
                f"{self.dataset.name!r} was built from aggregated counts, so item "
                f"indices are unavailable. Rebuild it with GroupedDataset."
                f"from_arrays/from_dataframe to keep item provenance, or use "
                f".groups to get group ids instead."
            )
        return np.asarray(self.assignment[index])


def _shorten(names: tuple[str, ...], limit: int = 5) -> str:
    more = f" (+{len(names) - limit} more)" if len(names) > limit else ""
    return ", ".join(names[:limit]) + more


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "This method requires pandas. "
            "Install it with: pip install 'splitkit[pandas]'"
        ) from exc
    return pd
