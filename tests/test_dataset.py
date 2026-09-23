"""GroupedDataset validation and derived statistics."""

from __future__ import annotations

import numpy as np
import pytest

from _helpers import make_dataset
from splitkit.dataset import GroupedDataset


class TestValidation:
    def test_group_ids_length_checked(self):
        with pytest.raises(ValueError, match="group_ids"):
            GroupedDataset(
                group_ids=np.array(["a", "b"], dtype=np.str_),
                group_vectors=np.ones((3, 2)),
                group_sizes=np.ones(3),
                class_names=("x", "y"),
            )

    def test_group_sizes_length_checked(self):
        with pytest.raises(ValueError, match="group_sizes"):
            GroupedDataset(
                group_ids=np.array(["a", "b", "c"], dtype=np.str_),
                group_vectors=np.ones((3, 2)),
                group_sizes=np.ones(5),
                class_names=("x", "y"),
            )

    def test_class_names_length_checked(self):
        with pytest.raises(ValueError, match="class_names"):
            GroupedDataset(
                group_ids=np.array(["a"], dtype=np.str_),
                group_vectors=np.ones((1, 2)),
                group_sizes=np.ones(1),
                class_names=("only_one",),
            )

    def test_negative_counts_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            make_dataset([[1.0, -2.0], [3.0, 4.0]])

    def test_is_frozen(self, tiny):
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            tiny.name = "renamed"


class TestShape:
    def test_dimensions(self, tiny):
        assert tiny.n_groups == 6
        assert tiny.n_classes == 3

    def test_total_items(self, tiny):
        assert tiny.total_items == 60

    def test_single_group_allowed(self):
        d = make_dataset([[5.0, 5.0]])
        assert d.n_groups == 1
        assert d.total_items == 10

    def test_single_class_allowed(self):
        d = make_dataset([[5.0], [3.0]])
        assert d.n_classes == 1


class TestStatistics:
    def test_global_counts(self, tiny):
        np.testing.assert_allclose(tiny.global_class_counts, [20, 20, 20])

    def test_frequencies_sum_to_one(self, tiny):
        assert tiny.global_class_frequencies.sum() == pytest.approx(1.0)

    def test_frequencies_zero_for_empty_dataset(self):
        d = make_dataset([[0.0, 0.0], [0.0, 0.0]])
        np.testing.assert_allclose(d.global_class_frequencies, [0.0, 0.0])

    def test_class_group_counts(self, tiny):
        """Each class in `tiny` occurs in exactly three of the six groups."""
        np.testing.assert_array_equal(tiny.class_group_counts, [3, 3, 3])

    def test_class_group_counts_ignores_magnitude(self):
        d = make_dataset([[1, 0], [1000, 0], [0, 5]])
        np.testing.assert_array_equal(d.class_group_counts, [2, 1])


class TestIsOnehot:
    def test_true_when_mass_equals_items(self, tiny):
        assert tiny.is_onehot is True

    def test_false_for_multilabel(self, multilabel):
        """Each item carries several labels, so class mass exceeds item count."""
        assert multilabel.is_onehot is False

    def test_false_when_units_differ(self, soft_counts):
        """Pixel counts against tile counts: different units entirely."""
        assert soft_counts.is_onehot is False


class TestSummary:
    def test_mentions_key_facts(self, tiny):
        text = tiny.summary()
        assert "tiny" in text
        assert "6" in text
        assert all(name in text for name in tiny.class_names)

    def test_handles_zero_counts(self):
        make_dataset([[0.0, 0.0], [0.0, 0.0]]).summary()
