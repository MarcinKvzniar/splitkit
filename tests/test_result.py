"""SplitResult paths not reached by the main API tests."""

from __future__ import annotations

import numpy as np
import pytest

from _helpers import make_dataset
from splitkit import split
from splitkit.problem import SplitProblem
from splitkit.result import SplitMapping


class TestPlainDictAccessors:
    def test_to_group_ids(self, tiny):
        r = split(tiny, (0.5, 0.25, 0.25), max_evals=500, seed=0)
        d = r.to_group_ids()
        assert isinstance(d, dict)
        assert set(d) == set(r.names)

    def test_to_indices(self):
        from splitkit.dataset import GroupedDataset

        built = GroupedDataset.from_arrays(
            np.repeat(np.arange(8), 3), np.tile([0, 1, 1], 8)
        )
        r = split(built, (0.5, 0.5), max_evals=500, seed=0)
        out = r.to_indices()
        assert isinstance(out, dict)
        assert sum(len(v) for v in out.values()) == built.n_items


class TestLowerBoundReporting:
    def test_gap_when_a_bound_is_known(self, tiny):
        """A strategy that proves a bound gets it surfaced as a relative gap."""
        r = split(tiny, (0.5, 0.25, 0.25), max_evals=500, seed=0)
        from dataclasses import replace

        with_bound = replace(r, lower_bound=r.cost / 2)
        assert with_bound.gap() == pytest.approx(1.0)

    def test_proved_optimal_shown_in_summary(self, tiny):
        from dataclasses import replace

        r = split(tiny, (0.5, 0.25, 0.25), max_evals=500, seed=0)
        assert "PROVEN OPTIMAL" in replace(r, proved_optimal=True).summary()

    def test_bound_without_proof_shows_gap(self, tiny):
        from dataclasses import replace

        r = split(tiny, (0.5, 0.25, 0.25), max_evals=500, seed=0)
        text = replace(r, lower_bound=r.cost * 0.9).summary()
        assert "Lower bound" in text
        assert "gap" in text


class TestSizeColumnLabelling:
    def test_pseudo_class_named_in_reports(self, soft_counts):
        """The item-count column is not a real class and must not look like one."""
        r = split(soft_counts, (0.5, 0.5), size_weight=1.0, max_evals=500, seed=0)
        assert r.actual_counts.shape[1] == soft_counts.n_classes + 1

        frame_classes = set(r.counts_frame().index.get_level_values("class"))
        assert "<item count>" in frame_classes

    def test_absent_when_disabled(self, soft_counts):
        r = split(soft_counts, (0.5, 0.5), size_weight=0, max_evals=500, seed=0)
        _, class_name, _ = r.worst_cell()
        assert class_name in soft_counts.class_names

    def test_worst_cell_can_name_the_size_column(self, soft_counts):
        r = split(soft_counts, (0.5, 0.5), size_weight=1.0, max_evals=500, seed=0)
        _, class_name, _ = r.worst_cell()
        assert class_name in {*soft_counts.class_names, "<item count>"}


class TestEmptySplitHandling:
    def test_summary_survives_an_empty_split(self):
        """A degenerate assignment must still report, not crash."""
        data = make_dataset([[5, 5]] * 4)
        problem = SplitProblem.build(data, (0.5, 0.5))
        r = split(data, (0.5, 0.5), max_evals=500, seed=0)

        from dataclasses import replace

        empty = np.zeros(data.n_groups, dtype=np.intp)
        degenerate = replace(
            r, assignment=empty, actual_counts=problem.count_matrix(empty)
        )
        text = degenerate.summary()
        assert "test" in text
        assert degenerate.achieved_ratios[1] == 0.0

    def test_achieved_ratios_with_no_items(self):
        """Zero total size must not divide by zero.

        Unreachable through split() -- a dataset with no mass is rejected earlier --
        so the guard is exercised on the result type directly.
        """
        from splitkit.result import SplitResult

        data = make_dataset([[1.0, 1.0], [1.0, 1.0]], sizes=[0.0, 0.0])
        problem = SplitProblem.build(data, (0.5, 0.5))
        assignment = np.array([0, 1])

        r = SplitResult(
            names=problem.names,
            assignment=assignment,
            ratios=problem.ratios,
            target_counts=problem.target,
            actual_counts=problem.count_matrix(assignment),
            cost=problem.evaluate(assignment),
            strategy="manual",
            dataset=data,
        )
        np.testing.assert_array_equal(r.achieved_ratios, [0.0, 0.0])
        assert r.item_counts.sum() == 0.0


class TestSplitMappingDirect:
    def test_repr_lists_sizes(self):
        m = SplitMapping(("a", "b"), (np.arange(3), np.arange(5)))
        assert "a=3" in repr(m)
        assert "b=5" in repr(m)

    def test_values_and_items(self):
        m = SplitMapping(("a", "b"), (np.arange(2), np.arange(4)))
        assert [len(v) for v in m.values()] == [2, 4]
        assert [k for k, _ in m.items()] == ["a", "b"]

    def test_str_dunder_is_the_summary(self, tiny):
        r = split(tiny, (0.5, 0.25, 0.25), max_evals=500, seed=0)
        assert str(r) == r.summary()


class TestManyDroppedClasses:
    def test_summary_truncates_long_lists(self):
        """A wide dataset can drop many classes; the report must stay readable."""
        rng = np.random.default_rng(0)
        vectors = np.zeros((6, 12))
        vectors[:, 0] = rng.integers(5, 20, size=6)
        # classes 1..11 each occur in a single group -> all unstratifiable at K=3
        for j in range(1, 12):
            vectors[j % 6, j] = 3
        data = make_dataset(vectors)

        r = split(data, (0.5, 0.25, 0.25), max_evals=500, seed=0)
        assert len(r.dropped_classes) > 5
        text = r.summary()
        assert "more)" in text


class TestUnknownSplitName:
    @pytest.mark.parametrize("method", ["mask", "item_mask"])
    def test_lists_available_names(self, tiny, method):
        r = split(tiny, (0.5, 0.25, 0.25), max_evals=500, seed=0)
        with pytest.raises(KeyError, match="No split named"):
            getattr(r, method)("nope")
