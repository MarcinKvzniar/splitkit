"""The functional entry points: :func:`split` and :func:`evaluate`."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .dataset import GroupedDataset
from .objectives import Objective
from .problem import Budget, SplitProblem, default_names
from .result import SplitResult
from .strategies import get_strategy
from .strategies.base import Strategy

__all__ = ["evaluate", "split"]

#: Used when neither an evaluation budget nor a time budget is given.
_DEFAULT_MAX_EVALS = 300_000


def _coerce_ratios(
    ratios: Mapping[str, float] | Sequence[float],
    names: Sequence[str] | None,
) -> tuple[Sequence[float], Sequence[str] | None]:
    """Accept either ``{"train": 0.8, "test": 0.2}`` or ``(0.8, 0.2)``."""
    if isinstance(ratios, Mapping):
        if names is not None:
            raise ValueError(
                "Pass names either as the keys of `ratios` or as `names`, not both."
            )
        return list(ratios.values()), list(ratios.keys())
    return ratios, names


def _as_dataset(
    data: Any,
    *,
    group_col: str | None,
    label_col: str | None,
    label_cols: Sequence[str] | None,
    count_cols: Sequence[str] | None,
    groups: Any,
    y: Any,
    name: str | None,
) -> GroupedDataset:
    """Build a dataset from whatever the caller passed."""
    if isinstance(data, GroupedDataset):
        return data

    if groups is not None or y is not None:
        if groups is None or y is None:
            raise ValueError("Pass both `groups` and `y`, or neither.")
        return GroupedDataset.from_arrays(groups, y, name=name or "dataset")

    if group_col is not None:
        return GroupedDataset.from_dataframe(
            data,
            group_col=group_col,
            label_col=label_col,
            label_cols=label_cols,
            count_cols=count_cols,
            name=name or "dataset",
        )

    raise TypeError(
        "Cannot build a dataset from the given input. Pass a GroupedDataset, or a "
        "DataFrame together with `group_col` and one of `label_col`/`label_cols`/"
        "`count_cols`, or arrays via `groups=` and `y=`."
    )


def split(
    data: Any,
    ratios: Mapping[str, float] | Sequence[float] = (0.70, 0.15, 0.15),
    *,
    names: Sequence[str] | None = None,
    strategy: str | Strategy = "annealing",
    objective: str | Objective = "wmape",
    class_weights: str | np.ndarray = "inverse_frequency",
    size_weight: float | str = 0.0,
    unstratifiable: str = "drop",
    weight_normalize: bool = False,
    weight_clip: float | None = None,
    max_evals: int | None = None,
    time_budget: float | None = None,
    target_cost: float = 0.0,
    seed: int | None = None,
    warm_start: np.ndarray | None = None,
    name: str | None = None,
    group_col: str | None = None,
    label_col: str | None = None,
    label_cols: Sequence[str] | None = None,
    count_cols: Sequence[str] | None = None,
    groups: Any = None,
    y: Any = None,
    **strategy_params: Any,
) -> SplitResult:
    """Split grouped data into train/validation/test (or any K parts).

    Groups are indivisible: every item sharing a group key lands in the same
    split, which is what prevents leakage between related samples. Subject to
    that, the split is chosen to match the global class distribution.

    Parameters
    ----------
    data
        A :class:`GroupedDataset`, or a DataFrame (with ``group_col`` and one of
        ``label_col`` / ``label_cols`` / ``count_cols``), or anything at all when
        passing ``groups=`` and ``y=`` directly.
    ratios
        Either a mapping of split name to share (``{"train": 0.8, "test": 0.2}``)
        or a bare sequence (``(0.8, 0.2)``). Rescaled to sum to 1, so ``(8, 2)``
        and ``(0.8, 0.2)`` mean the same thing.
    strategy
        Name from the registry, or a configured :class:`Strategy` instance.
    size_weight
        Weight on an item-count term. ``"auto"`` enables it only when the data is
        not one-hot, where matching class counts does *not* imply matching item
        counts.
    max_evals, time_budget
        Stop conditions. Given neither, a default evaluation budget applies.
    seed
        Seeds the strategy's random generator, making the result reproducible.

    Returns
    -------
    SplitResult
        Use ``.indices`` for item positions, ``.groups`` for group ids, and
        ``.summary()`` for a quality report.

    Examples
    --------
    >>> result = split(df, group_col="patient_id", label_col="diagnosis")
    >>> train, val, test = result.indices.astuple()
    >>> X_train, y_train = X[train], y[train]
    """
    ratio_values, split_names = _coerce_ratios(ratios, names)
    dataset = _as_dataset(
        data,
        group_col=group_col,
        label_col=label_col,
        label_cols=label_cols,
        count_cols=count_cols,
        groups=groups,
        y=y,
        name=name,
    )

    problem = SplitProblem.build(
        dataset,
        ratio_values,
        names=split_names,
        objective=objective,
        class_weights=class_weights,
        size_weight=size_weight,
        unstratifiable=unstratifiable,
        weight_normalize=weight_normalize,
        weight_clip=weight_clip,
    )

    if max_evals is None and time_budget is None:
        max_evals = _DEFAULT_MAX_EVALS
    budget = Budget(
        max_evals=max_evals, time_limit=time_budget, target_cost=target_cost
    )

    engine = get_strategy(strategy, **strategy_params)
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    outcome = engine.run(problem, budget, rng, warm_start=warm_start)
    elapsed = time.perf_counter() - t0

    # Recompute from the returned assignment: a search maintains its counts
    # incrementally, and a drifted accumulator must never reach the caller.
    assignment = np.asarray(outcome.assignment)
    actual = problem.count_matrix(assignment)

    return SplitResult(
        names=problem.names,
        assignment=assignment,
        ratios=problem.ratios,
        target_counts=problem.target,
        actual_counts=actual,
        cost=problem.prepared.total(actual),
        strategy=engine.name,
        strategy_params=engine.params(),
        dataset=dataset,
        seed=seed,
        n_evals=outcome.n_evals,
        n_iterations=outcome.n_iterations,
        elapsed_time=elapsed,
        converged=outcome.converged,
        cost_history=outcome.cost_history,
        lower_bound=outcome.lower_bound,
        proved_optimal=outcome.proved_optimal,
        dropped_classes=problem.dropped_classes,
    )


def evaluate(
    data: GroupedDataset,
    assignment: np.ndarray,
    ratios: Mapping[str, float] | Sequence[float] = (0.70, 0.15, 0.15),
    *,
    names: Sequence[str] | None = None,
    objective: str | Objective = "wmape",
    class_weights: str | np.ndarray = "inverse_frequency",
    size_weight: float | str = 0.0,
    unstratifiable: str = "drop",
    weight_normalize: bool = False,
    weight_clip: float | None = None,
) -> float:
    """Score an existing assignment on the same objective :func:`split` minimises.

    Use it to compare a split produced elsewhere -- a manual one, or another
    library's -- against splitkit's, on equal terms.
    """
    ratio_values, split_names = _coerce_ratios(ratios, names)
    problem = SplitProblem.build(
        data,
        ratio_values,
        names=split_names,
        objective=objective,
        class_weights=class_weights,
        size_weight=size_weight,
        unstratifiable=unstratifiable,
        weight_normalize=weight_normalize,
        weight_clip=weight_clip,
    )
    assignment = np.asarray(assignment)
    if assignment.shape != (data.n_groups,):
        raise ValueError(
            f"assignment has shape {assignment.shape}, expected ({data.n_groups},)."
        )
    if assignment.min() < 0 or assignment.max() >= problem.n_splits:
        raise ValueError(
            f"assignment contains split indices outside "
            f"[0, {problem.n_splits - 1}]."
        )
    return problem.evaluate(assignment)


# Re-exported for callers that want the default names without building a problem.
__all__ += ["default_names"]
