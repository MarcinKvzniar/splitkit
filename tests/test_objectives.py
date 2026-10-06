"""Objective correctness, against values computed by hand."""

from __future__ import annotations

import numpy as np
import pytest

from splitkit.objectives import OBJECTIVES, WeightedMAPE, get_objective


class TestWeightedMAPE:
    def test_known_value(self):
        """Hand-computed: 2 splits x 3 classes, eps=1, unit weights.

        |a-t|      = [[1, 0, 2], [1, 0, 2]]
        (t+eps)    = [[6, 3, 3], [5, 2, 6]]
        ratio      = [[1/6, 0, 2/3], [1/5, 0, 1/3]]
        total      = 1/6 + 2/3 + 1/5 + 1/3 = 1.3666666...
        """
        target = np.array([[5.0, 2.0, 2.0], [4.0, 1.0, 5.0]])
        counts = np.array([[6.0, 2.0, 4.0], [3.0, 1.0, 3.0]])
        weights = np.ones(3)

        prepared = WeightedMAPE(eps=1.0).prepare(target, weights)
        expected = 1 / 6 + 2 / 3 + 1 / 5 + 1 / 3
        assert prepared.total(counts) == pytest.approx(expected)

    def test_weights_scale_per_class(self):
        target = np.array([[5.0, 2.0]])
        counts = np.array([[6.0, 4.0]])
        weights = np.array([2.0, 10.0])
        # 2 * (1/6) + 10 * (2/3)
        expected = 2 * (1 / 6) + 10 * (2 / 3)
        prepared = WeightedMAPE(eps=1.0).prepare(target, weights)
        assert prepared.total(counts) == pytest.approx(expected)

    def test_zero_when_exact(self):
        target = np.array([[5.0, 2.0], [1.0, 3.0]])
        prepared = WeightedMAPE().prepare(target, np.ones(2))
        assert prepared.total(target.copy()) == 0.0

    def test_zero_weight_class_ignored(self):
        """A zero-weighted class must not influence the cost at all."""
        target = np.array([[5.0, 2.0]])
        counts = np.array([[5.0, 99.0]])
        prepared = WeightedMAPE().prepare(target, np.array([1.0, 0.0]))
        assert prepared.total(counts) == 0.0

    def test_rows_sum_to_total(self):
        rng = np.random.default_rng(0)
        target = rng.uniform(1, 100, size=(4, 6))
        weights = rng.uniform(0.1, 3, size=6)
        prepared = WeightedMAPE().prepare(target, weights)

        for _ in range(50):
            counts = rng.uniform(0, 120, size=(4, 6))
            assert prepared.rows(counts).sum() == pytest.approx(
                prepared.total(counts), rel=1e-12
            )

    def test_row_matches_rows(self):
        """Single-split cost must agree with the vectorised per-split costs."""
        rng = np.random.default_rng(1)
        target = rng.uniform(1, 50, size=(3, 5))
        weights = rng.uniform(0.1, 2, size=5)
        prepared = WeightedMAPE().prepare(target, weights)

        counts = rng.uniform(0, 60, size=(3, 5))
        rows = prepared.rows(counts)
        for s in range(3):
            assert prepared.row(counts[s], s) == pytest.approx(rows[s], rel=1e-12)

    def test_monotone_in_deviation(self):
        target = np.array([[10.0]])
        prepared = WeightedMAPE().prepare(target, np.ones(1))
        costs = [prepared.total(np.array([[10.0 + d]])) for d in (0, 1, 5, 20)]
        assert costs == sorted(costs)
        assert costs[0] == 0.0

    def test_symmetric_over_and_under(self):
        target = np.array([[10.0]])
        prepared = WeightedMAPE().prepare(target, np.ones(1))
        assert prepared.total(np.array([[7.0]])) == pytest.approx(
            prepared.total(np.array([[13.0]]))
        )

    def test_eps_must_be_positive(self):
        with pytest.raises(ValueError, match="eps must be positive"):
            WeightedMAPE(eps=0.0)
        with pytest.raises(ValueError, match="eps must be positive"):
            WeightedMAPE(eps=-1.0)

    def test_eps_guards_zero_target(self):
        """A class absent from the data must not divide by zero."""
        target = np.array([[0.0, 5.0]])
        counts = np.array([[3.0, 5.0]])
        prepared = WeightedMAPE(eps=1.0).prepare(target, np.ones(2))
        assert prepared.total(counts) == pytest.approx(3.0)

    def test_declared_flags(self):
        obj = WeightedMAPE()
        assert obj.separable is True
        assert obj.linearizable is True
        assert obj.prepare(np.ones((2, 2)), np.ones(2)).separable is True


class TestGetObjective:
    def test_by_name(self):
        assert isinstance(get_objective("wmape"), WeightedMAPE)

    def test_instance_passes_through(self):
        obj = WeightedMAPE(eps=2.0)
        assert get_objective(obj) is obj

    def test_unknown_name_lists_options(self):
        with pytest.raises(KeyError, match="Unknown objective"):
            get_objective("no_such_objective")

    def test_registry_names_match_classes(self):
        for name, cls in OBJECTIVES.items():
            assert cls.name == name
