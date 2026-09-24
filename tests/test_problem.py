"""SplitProblem construction, weighting and evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from _helpers import make_dataset
from splitkit.problem import Budget, SplitProblem, default_names, normalize_ratios


class TestNormalizeRatios:
    def test_already_normalized_unchanged(self):
        np.testing.assert_allclose(
            normalize_ratios((0.7, 0.15, 0.15)), [0.7, 0.15, 0.15]
        )

    def test_rescales_to_one(self):
        """Ratios express proportions, so any positive scaling is equivalent."""
        np.testing.assert_allclose(
            normalize_ratios((7, 1.5, 1.5)), [0.7, 0.15, 0.15]
        )
        np.testing.assert_allclose(normalize_ratios((1, 1, 1, 1)), [0.25] * 4)

    @pytest.mark.parametrize(
        "bad, match",
        [
            ((0.7, -0.2, 0.5), "positive"),
            ((0.7, 0.0, 0.3), "positive"),
            ((0.7, float("nan"), 0.3), "finite"),
            ((0.7, float("inf")), "finite"),
            ((), "at least 2"),
            ((1.0,), "at least 2"),
        ],
    )
    def test_rejects_invalid(self, bad, match):
        with pytest.raises(ValueError, match=match):
            normalize_ratios(bad)


class TestDefaultNames:
    def test_conventional_names(self):
        assert default_names(2) == ("train", "test")
        assert default_names(3) == ("train", "val", "test")

    def test_generic_beyond_three(self):
        assert default_names(5) == tuple(f"split_{i}" for i in range(5))


class TestTargets:
    def test_target_is_counts_times_ratio(self, tiny):
        p = SplitProblem.build(tiny, (0.5, 0.25, 0.25), size_weight=0)
        totals = tiny.global_class_counts
        for s, r in enumerate((0.5, 0.25, 0.25)):
            np.testing.assert_allclose(p.target[s], totals * r)

    def test_target_columns_sum_to_totals(self, tiny):
        p = SplitProblem.build(tiny, (0.7, 0.15, 0.15), size_weight=0)
        np.testing.assert_allclose(p.target.sum(axis=0), tiny.global_class_counts)

    @pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 8])
    def test_generalizes_to_k_splits(self, dense, k):
        p = SplitProblem.build(dense, [1 / k] * k, size_weight=0)
        assert p.n_splits == k
        assert p.target.shape == (k, dense.n_classes)
        assert len(p.names) == k
        np.testing.assert_allclose(
            p.target.sum(axis=0), dense.global_class_counts
        )

    def test_refuses_k_beyond_class_spread(self, tiny):
        """Each class in `tiny` spans 3 groups, so 4 splits cannot be stratified."""
        assert tiny.class_group_counts.max() == 3
        with pytest.raises(ValueError, match="no split can be stratified"):
            SplitProblem.build(tiny, [0.25] * 4)


class TestCountMatrix:
    def test_partitions_all_mass(self, tiny_problem):
        a = np.array([0, 1, 2, 0, 1, 2])
        counts = tiny_problem.count_matrix(a)
        np.testing.assert_allclose(
            counts.sum(axis=0), tiny_problem.vectors.sum(axis=0)
        )

    def test_rows_match_membership(self, tiny_problem):
        a = np.array([0, 0, 1, 1, 2, 2])
        counts = tiny_problem.count_matrix(a)
        for s in range(3):
            np.testing.assert_allclose(
                counts[s], tiny_problem.vectors[a == s].sum(axis=0)
            )

    def test_empty_split_is_zero(self, tiny_problem):
        counts = tiny_problem.count_matrix(np.zeros(6, dtype=int))
        np.testing.assert_allclose(counts[1], 0)
        np.testing.assert_allclose(counts[2], 0)

    def test_matmul_and_mask_paths_agree(self):
        """The large-problem BLAS path must match the small-problem mask path."""
        rng = np.random.default_rng(3)
        data = make_dataset(rng.integers(0, 20, size=(400, 6)).astype(float))
        p = SplitProblem.build(data, (0.6, 0.2, 0.2))
        a = rng.integers(0, 3, size=400)

        masked = p.count_matrix(a)
        onehot = np.zeros((3, 400))
        onehot[a, np.arange(400)] = 1.0
        np.testing.assert_allclose(masked, onehot @ p.vectors)


class TestWeights:
    def test_inverse_frequency_favours_rare_classes(self):
        data = make_dataset([[100, 1], [100, 1], [100, 1], [100, 1]])
        p = SplitProblem.build(data, (0.5, 0.5))
        assert p.weights[1] > p.weights[0]

    def test_uniform_weights_are_equal(self, tiny):
        p = SplitProblem.build(tiny, (0.5, 0.5), class_weights="uniform")
        np.testing.assert_allclose(p.weights, 1.0)

    def test_explicit_weights_used(self, tiny):
        w = np.array([1.0, 2.0, 3.0])
        p = SplitProblem.build(tiny, (0.5, 0.5), class_weights=w, weight_normalize=False, size_weight=0)
        np.testing.assert_allclose(p.weights, w)

    def test_normalized_by_default(self, tiny):
        w = np.array([1.0, 2.0, 3.0])
        p = SplitProblem.build(tiny, (0.5, 0.5), class_weights=w, size_weight=0)
        np.testing.assert_allclose(p.weights, w / w.mean())

    def test_explicit_weights_shape_checked(self, tiny):
        with pytest.raises(ValueError, match="shape"):
            SplitProblem.build(tiny, (0.5, 0.5), class_weights=np.ones(99))

    def test_negative_weights_rejected(self, tiny):
        with pytest.raises(ValueError, match="non-negative"):
            SplitProblem.build(tiny, (0.5, 0.5), class_weights=np.array([1.0, -1, 1]))

    def test_normalize_gives_mean_one(self, tiny):
        p = SplitProblem.build(tiny, (0.5, 0.5), weight_normalize=True)
        assert p.weights[p.weights > 0].mean() == pytest.approx(1.0)

    def test_normalize_makes_datasets_comparable(self):
        """Unnormalised inverse-frequency weights scale with the data itself,
        which is what made costs incomparable across datasets."""
        small = make_dataset([[10, 5], [8, 6], [9, 4], [7, 7]])
        big = make_dataset([[10000, 5000], [8000, 6000], [9000, 4000], [7000, 7000]])
        ps = SplitProblem.build(small, (0.5, 0.5), weight_normalize=True)
        pb = SplitProblem.build(big, (0.5, 0.5), weight_normalize=True)
        assert ps.weights[ps.weights > 0].mean() == pytest.approx(
            pb.weights[pb.weights > 0].mean()
        )

    def test_clip_caps_extreme_weights(self):
        data = make_dataset([[1000, 1000, 1]] * 5)
        base = SplitProblem.build(data, (0.5, 0.5))
        clipped = SplitProblem.build(data, (0.5, 0.5), weight_clip=50.0)
        assert clipped.weights.max() < base.weights.max()

    @pytest.mark.parametrize("bad", [0, -5, 101])
    def test_clip_percentile_validated(self, tiny, bad):
        with pytest.raises(ValueError, match="percentile"):
            SplitProblem.build(tiny, (0.5, 0.5), weight_clip=bad)


class TestUnstratifiable:
    def test_rare_class_dropped_and_reported(self, unstratifiable):
        """Class c2 occurs in 2 groups, so it cannot span 3 splits."""
        p = SplitProblem.build(unstratifiable, (0.5, 0.25, 0.25))
        assert p.dropped_classes == ("c2",)
        assert p.weights[2] == 0.0

    def test_kept_when_requested(self, unstratifiable):
        p = SplitProblem.build(
            unstratifiable, (0.5, 0.25, 0.25), unstratifiable="keep"
        )
        assert p.dropped_classes == ()
        assert p.weights[2] > 0.0

    def test_threshold_follows_split_count(self, unstratifiable):
        """With only 2 splits, a class in 2 groups is stratifiable again."""
        p2 = SplitProblem.build(unstratifiable, (0.5, 0.5))
        assert p2.dropped_classes == ()
        p4 = SplitProblem.build(unstratifiable, [0.25] * 4)
        assert "c2" in p4.dropped_classes

    def test_all_dropped_raises(self):
        data = make_dataset([[5, 0], [0, 5], [3, 0], [0, 3]])
        with pytest.raises(ValueError, match="no split can be stratified"):
            SplitProblem.build(data, [0.25] * 4)

    def test_invalid_mode(self, tiny):
        with pytest.raises(ValueError, match=r"drop.*keep"):
            SplitProblem.build(tiny, (0.5, 0.5), unstratifiable="sometimes")


class TestSizeWeight:
    def test_on_by_default(self, tiny):
        assert SplitProblem.build(tiny, (0.5, 0.5)).has_size_column is True

    def test_zero_disables(self, multilabel):
        p = SplitProblem.build(multilabel, (0.5, 0.5), size_weight=0)
        assert p.has_size_column is False
        assert p.n_columns == multilabel.n_classes

    def test_adds_pseudo_class_column(self, multilabel):
        p = SplitProblem.build(multilabel, (0.5, 0.5), size_weight=1.0)
        assert p.has_size_column is True
        assert p.n_columns == multilabel.n_classes + 1
        np.testing.assert_allclose(p.vectors[:, -1], multilabel.group_sizes)

    def test_negative_rejected(self, tiny):
        with pytest.raises(ValueError, match="non-negative"):
            SplitProblem.build(tiny, (0.5, 0.5), size_weight=-1.0)


class TestBuildValidation:
    def test_more_splits_than_groups(self, tiny):
        with pytest.raises(ValueError, match="Cannot make"):
            SplitProblem.build(tiny, [1 / 10] * 10)

    def test_name_count_must_match(self, tiny):
        with pytest.raises(ValueError, match="split names"):
            SplitProblem.build(tiny, (0.5, 0.5), names=("a", "b", "c"))

    def test_names_must_be_unique(self, tiny):
        with pytest.raises(ValueError, match="unique"):
            SplitProblem.build(tiny, (0.5, 0.5), names=("a", "a"))

    def test_custom_names_used(self, tiny):
        p = SplitProblem.build(tiny, (0.5, 0.5), names=("fit", "holdout"))
        assert p.names == ("fit", "holdout")

    def test_unknown_class_weights(self, tiny):
        with pytest.raises(ValueError, match="Unknown class_weights"):
            SplitProblem.build(tiny, (0.5, 0.5), class_weights="magic")

    def test_vectors_are_contiguous_float64(self, tiny):
        p = SplitProblem.build(tiny, (0.5, 0.5))
        assert p.vectors.dtype == np.float64
        assert p.vectors.flags["C_CONTIGUOUS"]


class TestEvaluate:
    def test_matches_prepared_total(self, tiny_problem, rng):
        for _ in range(20):
            a = tiny_problem.random_assignment(rng)
            assert tiny_problem.evaluate(a) == tiny_problem.prepared.total(
                tiny_problem.count_matrix(a)
            )

    def test_nonnegative(self, tiny_problem, rng):
        for _ in range(50):
            assert tiny_problem.evaluate(tiny_problem.random_assignment(rng)) >= 0.0

    def test_random_assignment_in_range(self, tiny_problem, rng):
        a = tiny_problem.random_assignment(rng)
        assert a.shape == (tiny_problem.n_groups,)
        assert set(np.unique(a)) <= {0, 1, 2}

    def test_perfectly_balanced_costs_zero(self):
        """Four identical groups split 50/50 can match the target exactly."""
        data = make_dataset([[4, 4]] * 4)
        p = SplitProblem.build(data, (0.5, 0.5), class_weights="uniform")
        assert p.evaluate(np.array([0, 0, 1, 1])) == 0.0


class TestBudget:
    def test_max_evals_stops(self):
        assert Budget(max_evals=100).exhausted(100, 0.0, 1.0)
        assert not Budget(max_evals=100).exhausted(99, 0.0, 1.0)

    def test_time_limit_stops(self):
        assert Budget(time_limit=5.0).exhausted(0, 5.1, 1.0)
        assert not Budget(time_limit=5.0).exhausted(0, 4.9, 1.0)

    def test_target_cost_stops(self):
        assert Budget().exhausted(0, 0.0, 0.0)
        assert Budget(target_cost=0.5).exhausted(0, 0.0, 0.4)
        assert not Budget(target_cost=0.5).exhausted(0, 0.0, 0.6)

    def test_unbounded_by_default(self):
        assert not Budget().exhausted(10**9, 10**9, 1.0)

    def test_deadline(self):
        assert Budget().deadline_from(100.0) == float("inf")
        assert Budget(time_limit=5.0).deadline_from(100.0) == 105.0

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"max_evals": 0}, "at least 1"),
            ({"max_evals": -5}, "at least 1"),
            ({"time_limit": 0}, "positive"),
            ({"time_limit": -1}, "positive"),
        ],
    )
    def test_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            Budget(**kwargs)
