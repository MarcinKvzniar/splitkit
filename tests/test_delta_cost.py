"""The incremental-update invariant.

Annealing maintains the count matrix incrementally: a move subtracts a group's
vector from one split and adds it to another, rather than recomputing from
scratch. Every bug in that scheme -- a missed undo on a rejected move, a stale
row, accumulated float drift over hundreds of thousands of updates -- shows up as
a divergence between the incrementally maintained cost and a fresh recomputation.

These tests pin that invariant down. They are what make it safe to replace the
full recomputation with true O(C) row deltas: the optimisation is only valid if it
agrees with the definition.
"""

from __future__ import annotations

import numpy as np
import pytest

from _helpers import make_dataset
from splitkit.problem import Budget, SplitProblem
from splitkit.strategies import get_strategy


@pytest.fixture
def problem() -> SplitProblem:
    rng = np.random.default_rng(11)
    return SplitProblem.build(
        make_dataset(rng.integers(0, 40, size=(60, 8)).astype(float)),
        (0.7, 0.15, 0.15),
    )


class TestIncrementalCounts:
    def test_move_matches_recomputation(self, problem):
        """Applying a move incrementally must equal recounting from scratch."""
        rng = np.random.default_rng(0)
        assignment = problem.random_assignment(rng)
        counts = problem.count_matrix(assignment)

        for _ in range(2000):
            g = int(rng.integers(problem.n_groups))
            old_s = int(assignment[g])
            new_s = int(rng.integers(problem.n_splits - 1))
            if new_s >= old_s:
                new_s += 1

            vec = problem.vectors[g]
            counts[old_s] -= vec
            counts[new_s] += vec
            assignment[g] = new_s

            np.testing.assert_allclose(
                counts, problem.count_matrix(assignment), rtol=1e-12, atol=1e-9
            )

    def test_rejected_move_is_fully_undone(self, problem):
        """The undo path is only exercised on rejection, so test it directly."""
        rng = np.random.default_rng(1)
        assignment = problem.random_assignment(rng)
        counts = problem.count_matrix(assignment)
        pristine = counts.copy()

        for _ in range(500):
            g = int(rng.integers(problem.n_groups))
            old_s = int(assignment[g])
            new_s = (old_s + 1) % problem.n_splits

            vec = problem.vectors[g]
            counts[old_s] -= vec
            counts[new_s] += vec
            # reject
            counts[new_s] -= vec
            counts[old_s] += vec

        np.testing.assert_allclose(counts, pristine, rtol=1e-12, atol=1e-9)


class TestDeltaCost:
    def test_two_row_delta_matches_total(self, problem):
        """Only two splits change on a move, so the cost delta needs only those
        two rows. This is the identity Phase 7's fast path depends on."""
        rng = np.random.default_rng(2)
        assignment = problem.random_assignment(rng)
        counts = problem.count_matrix(assignment)
        prepared = problem.prepared

        for _ in range(5000):
            g = int(rng.integers(problem.n_groups))
            old_s = int(assignment[g])
            new_s = int(rng.integers(problem.n_splits - 1))
            if new_s >= old_s:
                new_s += 1

            vec = problem.vectors[g]
            before = prepared.total(counts)
            row_before = prepared.row(counts[old_s], old_s) + prepared.row(
                counts[new_s], new_s
            )

            counts[old_s] -= vec
            counts[new_s] += vec

            after = prepared.total(counts)
            row_after = prepared.row(counts[old_s], old_s) + prepared.row(
                counts[new_s], new_s
            )

            assert (row_after - row_before) == pytest.approx(
                after - before, rel=1e-9, abs=1e-9
            )
            assignment[g] = new_s

    def test_untouched_rows_keep_their_cost(self, problem):
        """A move must not change the cost of any split it did not touch."""
        rng = np.random.default_rng(3)
        assignment = problem.random_assignment(rng)
        counts = problem.count_matrix(assignment)
        prepared = problem.prepared

        for _ in range(300):
            g = int(rng.integers(problem.n_groups))
            old_s = int(assignment[g])
            new_s = int(rng.integers(problem.n_splits - 1))
            if new_s >= old_s:
                new_s += 1

            rows_before = prepared.rows(counts).copy()
            counts[old_s] -= problem.vectors[g]
            counts[new_s] += problem.vectors[g]
            rows_after = prepared.rows(counts)

            for s in range(problem.n_splits):
                if s not in (old_s, new_s):
                    assert rows_after[s] == rows_before[s]
            assignment[g] = new_s


class TestNoDriftInStrategies:
    """A long run must not report a cost that has drifted from its assignment."""

    @pytest.mark.parametrize("name", ["annealing", "evolution", "random"])
    @pytest.mark.parametrize("evals", [1000, 25_000])
    def test_reported_cost_survives_long_runs(self, problem, name, evals):
        out = get_strategy(name).run(
            problem, Budget(max_evals=evals), np.random.default_rng(4)
        )
        assert out.cost == pytest.approx(
            problem.evaluate(out.assignment), rel=1e-9, abs=1e-9
        )

    def test_annealing_survives_many_reheats(self, problem):
        """Reheating restores a saved snapshot; a stale snapshot would drift."""
        strategy = get_strategy(
            "annealing", initial_temp=1.0, cooling_rate=0.99, min_temp=1e-3
        )
        out = strategy.run(
            problem, Budget(max_evals=30_000), np.random.default_rng(5)
        )
        assert out.cost == pytest.approx(
            problem.evaluate(out.assignment), rel=1e-9, abs=1e-9
        )
