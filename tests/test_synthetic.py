"""Synthetic dataset generation."""

from __future__ import annotations

import numpy as np
import pytest

from splitkit.synthetic import PRESETS, from_preset, list_presets, make_synthetic


class TestMakeSynthetic:
    def test_shape_and_determinism(self):
        a = make_synthetic(n_groups=50, n_classes=4, total_items=1000, seed=1)
        b = make_synthetic(n_groups=50, n_classes=4, total_items=1000, seed=1)
        assert a.n_classes == 4
        assert a.n_groups <= 50
        np.testing.assert_array_equal(a.group_vectors, b.group_vectors)

    def test_different_seeds_differ(self):
        a = make_synthetic(n_groups=50, n_classes=4, seed=1)
        b = make_synthetic(n_groups=50, n_classes=4, seed=2)
        assert not np.array_equal(a.group_vectors, b.group_vectors)

    def test_total_items_respected(self):
        d = make_synthetic(n_groups=40, n_classes=5, total_items=2000, seed=0)
        assert d.total_items == pytest.approx(2000, rel=0.02)

    def test_output_is_onehot(self):
        """Generated vectors are item counts, so mass equals size by construction."""
        assert make_synthetic(n_groups=30, n_classes=3, seed=0).is_onehot

    def test_no_empty_groups(self):
        d = make_synthetic(n_groups=200, n_classes=8, groups_per_class=5, seed=0)
        assert (d.group_sizes > 0).all()

    def test_imbalance_increases_spread(self):
        balanced = make_synthetic(n_classes=10, imbalance=0.0, seed=0)
        skewed = make_synthetic(n_classes=10, imbalance=2.5, seed=0)

        def ratio(d):
            c = d.global_class_counts
            return c.max() / max(c.min(), 1)

        assert ratio(skewed) > ratio(balanced)

    def test_groups_per_class_controls_concentration(self):
        spread = make_synthetic(n_groups=200, n_classes=6, groups_per_class=150, seed=0)
        tight = make_synthetic(n_groups=200, n_classes=6, groups_per_class=3, seed=0)
        assert spread.class_group_counts.mean() > tight.class_group_counts.mean()

    @pytest.mark.parametrize("kwargs", [{"n_groups": 0}, {"n_classes": 0}])
    def test_validation(self, kwargs):
        with pytest.raises(ValueError, match="must be positive"):
            make_synthetic(**kwargs)

    def test_groups_per_class_capped_at_n_groups(self):
        d = make_synthetic(n_groups=10, n_classes=3, groups_per_class=999, seed=0)
        assert d.n_groups <= 10


class TestPresets:
    def test_all_presets_build(self):
        for name in list_presets():
            d = from_preset(name)
            assert d.n_groups > 0
            assert d.name == f"synth_{name}"

    def test_listed_names_match_registry(self):
        assert set(list_presets()) == set(PRESETS)

    def test_unknown_preset_lists_options(self):
        with pytest.raises(KeyError, match="Unknown preset"):
            from_preset("not_a_preset")

    def test_presets_are_distinguishable(self):
        """Presets exist to cover different topologies, not to be near-duplicates."""
        shapes = {
            name: (from_preset(name).n_groups, from_preset(name).n_classes)
            for name in list_presets()
        }
        assert len(set(shapes.values())) == len(shapes)
