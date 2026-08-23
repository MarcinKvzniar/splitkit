"""The immutable optimisation problem handed to every strategy."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .dataset import GroupedDataset
from .objectives import Objective, PreparedObjective, get_objective

__all__ = ["Budget", "SplitProblem", "default_names"]


def default_names(k: int) -> tuple[str, ...]:
    """Conventional split names for ``k`` splits."""
    if k == 2:
        return ("train", "test")
    if k == 3:
        return ("train", "val", "test")
    return tuple(f"split_{i}" for i in range(k))


def normalize_ratios(ratios: Sequence[float]) -> np.ndarray:
    """Validate and rescale split ratios to sum to 1.

    ``(7, 1.5, 1.5)`` and ``(0.7, 0.15, 0.15)`` express the same intent, so ratios
    are normalised rather than rejected for not summing to one.
    """
    arr = np.asarray(ratios, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("ratios must be a non-empty 1-D sequence.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"ratios must all be finite, got {list(ratios)}.")
    if np.any(arr <= 0):
        raise ValueError(f"ratios must all be positive, got {list(ratios)}.")
    return arr / arr.sum()


@dataclass(frozen=True)
class Budget:
    """Stop conditions shared by every strategy.

    A strategy must respect whichever of these are set; ``None`` means unbounded.
    """

    max_evals: int | None = None
    time_limit: float | None = None
    target_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.max_evals is not None and self.max_evals < 1:
            raise ValueError(f"max_evals must be at least 1, got {self.max_evals}.")
        if self.time_limit is not None and self.time_limit <= 0:
            raise ValueError(f"time_limit must be positive, got {self.time_limit}.")

    def exhausted(self, n_evals: int, elapsed: float, cost: float) -> bool:
        """True when the search should stop."""
        if self.max_evals is not None and n_evals >= self.max_evals:
            return True
        if self.time_limit is not None and elapsed >= self.time_limit:
            return True
        return cost <= self.target_cost

    def deadline_from(self, t_start: float) -> float:
        """Absolute perf-counter time at which ``time_limit`` expires."""
        if self.time_limit is None:
            return float("inf")
        return t_start + self.time_limit


@dataclass(frozen=True)
class SplitProblem:
    """A fully specified grouped-splitting problem.

    Holds everything constant across an optimisation run — the count matrix, the
    class weights, the ideal target counts and the prepared objective — so that
    strategies stay stateless and directly comparable.

    ``vectors`` may carry one extra trailing pseudo-class column tracking item
    counts (see ``size_weight``), which is why it is used in preference to
    ``data.group_vectors`` everywhere in the search.
    """

    data:            GroupedDataset
    names:           tuple[str, ...]
    ratios:          np.ndarray                  # (K,)
    vectors:         np.ndarray                  # (G, C') C-contiguous float64
    weights:         np.ndarray                  # (C',)
    target:          np.ndarray                  # (K, C')
    objective:       Objective
    prepared:        PreparedObjective = field(repr=False)
    dropped_classes: tuple[str, ...] = ()
    has_size_column: bool = False

    # Shape
    @property
    def n_splits(self) -> int:
        return len(self.names)

    @property
    def n_groups(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def n_columns(self) -> int:
        """Number of objective columns, including the size pseudo-class if present."""
        return int(self.vectors.shape[1])

    # Evaluation
    def count_matrix(self, assignment: np.ndarray) -> np.ndarray:
        """(K, C') item counts per split for ``assignment``."""
        k, g = self.n_splits, self.n_groups
        if k * g < 50_000:
            counts = np.zeros((k, self.n_columns), dtype=np.float64)
            for s in range(k):
                mask = assignment == s
                if mask.any():
                    counts[s] = self.vectors[mask].sum(axis=0)
            return counts
        # BLAS-backed scatter; faster than masking once the problem is large.
        onehot = np.zeros((k, g), dtype=np.float64)
        onehot[assignment, np.arange(g)] = 1.0
        return onehot @ self.vectors

    def evaluate(self, assignment: np.ndarray) -> float:
        """Objective value for a group->split assignment. Lower is better."""
        return self.prepared.total(self.count_matrix(assignment))

    def row_costs(self, counts: np.ndarray) -> np.ndarray:
        """Per-split costs, shape ``(K,)``."""
        return self.prepared.rows(counts)

    def random_assignment(self, rng: np.random.Generator) -> np.ndarray:
        """Draw an assignment with each group placed according to ``ratios``."""
        return rng.choice(self.n_splits, size=self.n_groups, p=self.ratios)

    # Construction
    @classmethod
    def build(
        cls,
        data: GroupedDataset,
        ratios: Sequence[float] = (0.70, 0.15, 0.15),
        *,
        names: Sequence[str] | None = None,
        objective: str | Objective = "wmape",
        class_weights: str | np.ndarray = "inverse_frequency",
        size_weight: float | str = 0.0,
        unstratifiable: str = "drop",
        weight_normalize: bool = False,
        weight_clip: float | None = None,
        min_groups_per_class: int | None = None,
    ) -> SplitProblem:
        """Assemble a problem from a dataset and a target ratio vector.

        Parameters
        ----------
        ratios
            Target fraction per split; rescaled to sum to 1.
        names
            Split names; defaults to train/val/test conventions for K in (2, 3).
        class_weights
            ``"inverse_frequency"`` (rare classes penalised harder),
            ``"uniform"``, or an explicit ``(C,)`` array.
        size_weight
            Weight of an extra item-count column. ``"auto"`` enables it (weight 1)
            only when the data is not one-hot, where matching class counts does
            *not* imply matching item counts. ``0`` disables it.
        unstratifiable
            ``"drop"`` zero-weights classes present in fewer than K groups (they
            cannot be spread across K splits); ``"keep"`` scores them anyway.
        weight_normalize
            Rescale weights to mean 1 over the retained classes, making costs
            comparable across datasets.
        weight_clip
            Optional upper percentile (0-100) at which to clip weights, stopping a
            single ultra-rare class from dominating the objective.
        """
        ratio_arr = normalize_ratios(ratios)
        k = int(ratio_arr.size)

        split_names = tuple(names) if names is not None else default_names(k)
        if len(split_names) != k:
            raise ValueError(
                f"Got {len(split_names)} split names for {k} ratios: {split_names}."
            )
        if len(set(split_names)) != k:
            raise ValueError(f"Split names must be unique, got {split_names}.")
        if k > data.n_groups:
            raise ValueError(
                f"Cannot make {k} non-empty splits from {data.n_groups} group(s)."
            )

        obj = get_objective(objective)

        # --- class weights -------------------------------------------------
        counts = data.global_class_counts.astype(np.float64)
        if isinstance(class_weights, str):
            if class_weights == "inverse_frequency":
                weights = counts.sum() / (data.n_classes * counts + 1e-6)
            elif class_weights == "uniform":
                weights = np.ones(data.n_classes, dtype=np.float64)
            else:
                raise ValueError(
                    f"Unknown class_weights {class_weights!r}; expected "
                    f"'inverse_frequency', 'uniform', or an array."
                )
        else:
            weights = np.asarray(class_weights, dtype=np.float64)
            if weights.shape != (data.n_classes,):
                raise ValueError(
                    f"class_weights must have shape ({data.n_classes},), "
                    f"got {weights.shape}."
                )
            if np.any(weights < 0):
                raise ValueError("class_weights must be non-negative.")

        if weight_clip is not None:
            if not 0 < weight_clip <= 100:
                raise ValueError(
                    f"weight_clip must be a percentile in (0, 100], got {weight_clip}."
                )
            weights = np.minimum(weights, np.percentile(weights, weight_clip))

        # --- unstratifiable classes ----------------------------------------
        threshold = k if min_groups_per_class is None else min_groups_per_class
        keep = data.class_group_counts >= threshold
        if unstratifiable == "drop":
            dropped = tuple(
                name for name, ok in zip(data.class_names, keep, strict=True) if not ok
            )
            weights = weights * keep
        elif unstratifiable == "keep":
            dropped = ()
        else:
            raise ValueError(
                f"unstratifiable must be 'drop' or 'keep', got {unstratifiable!r}."
            )

        retained = weights > 0
        if not retained.any():
            raise ValueError(
                f"No class occurs in at least {threshold} groups, so no split can be "
                f"stratified. Reduce the number of splits or pass unstratifiable='keep'."
            )

        if weight_normalize:
            weights = weights / weights[retained].mean()

        # --- optional item-count pseudo-class -------------------------------
        vectors = data.group_vectors
        if size_weight == "auto":
            size_weight = 0.0 if data.is_onehot else 1.0
        size_weight = float(size_weight)
        if size_weight < 0:
            raise ValueError(f"size_weight must be non-negative, got {size_weight}.")

        has_size = size_weight > 0
        if has_size:
            vectors = np.hstack([vectors, data.group_sizes[:, np.newaxis]])
            weights = np.append(weights, size_weight * weights[retained].mean())

        vectors = np.ascontiguousarray(vectors, dtype=np.float64)
        target = vectors.sum(axis=0)[np.newaxis, :] * ratio_arr[:, np.newaxis]

        return cls(
            data=data,
            names=split_names,
            ratios=ratio_arr,
            vectors=vectors,
            weights=weights,
            target=target,
            objective=obj,
            prepared=obj.prepare(target, weights),
            dropped_classes=dropped,
            has_size_column=has_size,
        )


def elapsed_since(t_start: float) -> float:
    """Seconds since a ``time.perf_counter()`` reading."""
    return time.perf_counter() - t_start
