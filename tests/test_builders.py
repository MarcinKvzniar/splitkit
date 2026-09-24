"""Dataset builders.

The builders are what make splitkit usable outside the benchmark suite: a user
arrives with a DataFrame or a pair of arrays, not a count matrix. They are also
where item provenance is captured, without which a split can name groups but
cannot tell you which rows to actually train on.
"""

from __future__ import annotations

import numpy as np
import pytest

from splitkit.dataset import GroupedDataset

pd = pytest.importorskip("pandas")


class TestFromCounts:
    def test_minimal(self):
        d = GroupedDataset.from_counts([[5, 5], [3, 7]])
        assert d.n_groups == 2
        assert d.n_classes == 2
        np.testing.assert_allclose(d.group_sizes, [10, 10])

    def test_explicit_names(self):
        d = GroupedDataset.from_counts(
            [[1, 2]], group_ids=["slide7"], class_names=["tumor", "stroma"]
        )
        assert d.group_ids.tolist() == ["slide7"]
        assert d.class_names == ("tumor", "stroma")

    def test_sizes_may_differ_from_mass(self):
        """Pixel counts against tile counts: the units are unrelated."""
        d = GroupedDataset.from_counts([[900, 100]], group_sizes=[4])
        assert d.total_items == 4
        assert d.is_onehot is False

    def test_has_no_item_provenance(self):
        assert GroupedDataset.from_counts([[1, 1]]).has_items is False

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            GroupedDataset.from_counts([1, 2, 3])


class TestFromArrays:
    @pytest.fixture
    def arrays(self):
        groups = np.array(["b", "a", "a", "c", "b"])
        y = np.array(["x", "x", "y", "x", "y"])
        return groups, y

    def test_groups_sorted_deterministically(self, arrays):
        d = GroupedDataset.from_arrays(*arrays)
        assert d.group_ids.tolist() == ["a", "b", "c"]
        assert d.class_names == ("x", "y")

    def test_counts_are_correct(self, arrays):
        d = GroupedDataset.from_arrays(*arrays)
        # a: x,y | b: x,y | c: x
        np.testing.assert_allclose(
            d.group_vectors, [[1, 1], [1, 1], [1, 0]]
        )
        np.testing.assert_allclose(d.group_sizes, [2, 2, 1])

    def test_single_label_is_onehot(self, arrays):
        assert GroupedDataset.from_arrays(*arrays).is_onehot

    def test_two_dimensional_y(self):
        groups = np.array([0, 0, 1])
        y = np.array([[1.0, 2.0], [3.0, 0.0], [0.0, 5.0]])
        d = GroupedDataset.from_arrays(groups, y, classes=["p", "q"])
        np.testing.assert_allclose(d.group_vectors, [[4.0, 2.0], [0.0, 5.0]])
        assert d.class_names == ("p", "q")

    def test_from_labels_matches_sklearn_order(self, arrays):
        groups, y = arrays
        a = GroupedDataset.from_arrays(groups, y)
        b = GroupedDataset.from_labels(y, groups)
        np.testing.assert_array_equal(a.group_vectors, b.group_vectors)

    def test_numeric_group_keys(self):
        d = GroupedDataset.from_arrays(np.array([10, 2, 2]), np.array([0, 0, 1]))
        assert d.group_ids.tolist() == ["2", "10"]  # sorted numerically, then named

    @pytest.mark.parametrize(
        "groups, y, match",
        [
            (np.ones((2, 2)), np.ones(2), "1-D"),
            (np.array([1, 2]), np.array([1]), "align"),
            (np.array([]), np.array([]), "zero items"),
            (np.array([1, 2]), np.ones((2, 2, 2)), "1-D or 2-D"),
        ],
    )
    def test_validation(self, groups, y, match):
        with pytest.raises(ValueError, match=match):
            GroupedDataset.from_arrays(groups, y)

    def test_rejects_nan_labels(self):
        with pytest.raises(ValueError, match="NaN"):
            GroupedDataset.from_arrays(np.array([1, 2]), np.array([1.0, np.nan]))

    def test_rejects_missing_text_labels(self):
        y = np.array(["a", None, float("nan")], dtype=object)
        with pytest.raises(ValueError, match="NaN or None"):
            GroupedDataset.from_arrays(np.array([1, 2, 3]), y)

    def test_mixed_type_labels_and_groups(self):
        ds = GroupedDataset.from_arrays(
            np.array([1, "a", 1, "a"], dtype=object),
            np.array([0, "x", "x", 0], dtype=object),
        )
        assert ds.n_groups == 2
        assert ds.n_classes == 2

    @pytest.mark.parametrize(
        "groups", [np.array([1.0, np.nan]), np.array(["a", None], dtype=object)]
    )
    def test_rejects_missing_groups(self, groups):
        with pytest.raises(ValueError, match="missing values"):
            GroupedDataset.from_arrays(groups, np.array([0, 1]))

    def test_rejects_nan_in_2d(self):
        with pytest.raises(ValueError, match="NaN"):
            GroupedDataset.from_arrays(
                np.array([1, 2]), np.array([[1.0, 2.0], [np.nan, 1.0]])
            )

    def test_class_count_mismatch(self):
        with pytest.raises(ValueError, match="classes has"):
            GroupedDataset.from_arrays(
                np.array([1, 2]), np.ones((2, 3)), classes=["only", "two"]
            )


class TestItemProvenance:
    """item_group_index is what lets a split return usable row indices."""

    def test_maps_each_item_to_its_group(self):
        groups = np.array(["b", "a", "a", "c", "b"])
        y = np.array(["x", "x", "y", "x", "y"])
        d = GroupedDataset.from_arrays(groups, y)

        assert d.item_group_index.tolist() == [1, 0, 0, 2, 1]
        assert d.n_items == 5

    def test_reconstructs_group_membership(self):
        """Selecting a group's items must recover exactly that group's counts."""
        rng = np.random.default_rng(0)
        groups = rng.integers(0, 8, size=200)
        y = rng.integers(0, 4, size=200)
        d = GroupedDataset.from_arrays(groups, y)

        for g in range(d.n_groups):
            rows = np.flatnonzero(d.item_group_index == g)
            counts = np.bincount(y[rows], minlength=d.n_classes)
            np.testing.assert_allclose(counts, d.group_vectors[g])

    def test_partitions_every_item(self):
        rng = np.random.default_rng(1)
        groups = rng.integers(0, 5, size=50)
        d = GroupedDataset.from_arrays(groups, rng.integers(0, 3, size=50))

        assert d.item_group_index.shape == (50,)
        assert d.item_group_index.min() >= 0
        assert d.item_group_index.max() == d.n_groups - 1
        # every group is represented, and group sizes account for every item
        assert set(d.item_group_index.tolist()) == set(range(d.n_groups))
        np.testing.assert_allclose(
            np.bincount(d.item_group_index, minlength=d.n_groups), d.group_sizes
        )

    def test_can_be_disabled(self):
        d = GroupedDataset.from_arrays(
            np.array([1, 1, 2]), np.array([0, 1, 0]), track_items=False
        )
        assert d.has_items is False
        assert d.n_items == d.total_items

    def test_dtype_is_compact(self):
        d = GroupedDataset.from_arrays(np.array([1, 2]), np.array([0, 1]))
        assert d.item_group_index.dtype == np.int32

    def test_subset_drops_stale_provenance(self):
        """Retained item indices would point at the old group numbering."""
        d = GroupedDataset.from_arrays(
            np.array([1, 1, 2, 3]), np.array([0, 1, 0, 1])
        )
        assert d.has_items
        s = d.subset(np.array([True, False, True]))
        assert s.n_groups == 2
        assert s.has_items is False


class TestFromDataFrame:
    @pytest.fixture
    def clinical(self):
        return pd.DataFrame(
            {
                "patient": ["p2", "p1", "p1", "p3", "p2", "p1"],
                "dx": ["mel", "nev", "mel", "nev", "nev", "nev"],
            }
        )

    def test_single_label(self, clinical):
        d = GroupedDataset.from_dataframe(
            clinical, group_col="patient", label_col="dx"
        )
        assert d.group_ids.tolist() == ["p1", "p2", "p3"]
        assert d.class_names == ("mel", "nev")
        np.testing.assert_allclose(d.group_vectors, [[1, 2], [1, 1], [0, 1]])
        np.testing.assert_allclose(d.group_sizes, [3, 2, 1])

    def test_matches_from_arrays(self, clinical):
        a = GroupedDataset.from_dataframe(
            clinical, group_col="patient", label_col="dx"
        )
        b = GroupedDataset.from_arrays(
            clinical["patient"].to_numpy(), clinical["dx"].to_numpy()
        )
        np.testing.assert_array_equal(a.group_vectors, b.group_vectors)
        np.testing.assert_array_equal(a.item_group_index, b.item_group_index)

    def test_multilabel(self):
        df = pd.DataFrame({"id": [1, 1, 2], "a": [1, 0, 1], "b": [0, 1, 1]})
        d = GroupedDataset.from_dataframe(
            df, group_col="id", label_cols=["a", "b"]
        )
        np.testing.assert_allclose(d.group_vectors, [[1, 1], [1, 1]])
        assert d.class_names == ("a", "b")

    def test_plus_minus_one_encoding_decoded(self):
        """A -1/+1 attribute table would otherwise cancel instead of counting."""
        df = pd.DataFrame({"id": [1, 1], "smiling": [1, -1], "male": [-1, -1]})
        d = GroupedDataset.from_dataframe(
            df, group_col="id", label_cols=["smiling", "male"]
        )
        np.testing.assert_allclose(d.group_vectors, [[1, 0]])

    def test_zero_one_encoding_left_alone(self):
        df = pd.DataFrame({"id": [1, 1], "a": [1, 0], "b": [0, 0]})
        d = GroupedDataset.from_dataframe(df, group_col="id", label_cols=["a", "b"])
        np.testing.assert_allclose(d.group_vectors, [[1, 0]])

    def test_count_cols_with_size(self):
        """A row can be a count vector standing for many items."""
        df = pd.DataFrame(
            {
                "wsi": ["s1", "s1", "s2"],
                "tumor": [900, 100, 50],
                "stroma": [100, 900, 950],
                "tiles": [4, 4, 9],
            }
        )
        d = GroupedDataset.from_dataframe(
            df, group_col="wsi", count_cols=["tumor", "stroma"], size_col="tiles"
        )
        np.testing.assert_allclose(d.group_vectors, [[1000, 1000], [50, 950]])
        np.testing.assert_allclose(d.group_sizes, [8, 9])
        assert d.is_onehot is False

    def test_count_cols_without_size_counts_rows(self):
        df = pd.DataFrame({"g": ["a", "a"], "x": [5, 5]})
        d = GroupedDataset.from_dataframe(df, group_col="g", count_cols=["x"])
        np.testing.assert_allclose(d.group_sizes, [2])

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({}, "exactly one"),
            ({"label_col": "dx", "label_cols": ["a"]}, "exactly one"),
            ({"label_col": "nope"}, "not found"),
            ({"label_cols": ["nope"]}, "not found"),
        ],
    )
    def test_label_mode_validation(self, clinical, kwargs, match):
        with pytest.raises((ValueError, KeyError), match=match):
            GroupedDataset.from_dataframe(clinical, group_col="patient", **kwargs)

    def test_missing_group_column(self, clinical):
        with pytest.raises(KeyError, match="not found"):
            GroupedDataset.from_dataframe(clinical, group_col="nope", label_col="dx")

    def test_empty_frame(self):
        df = pd.DataFrame({"g": [], "y": []})
        with pytest.raises(ValueError, match="empty DataFrame"):
            GroupedDataset.from_dataframe(df, group_col="g", label_col="y")

    def test_missing_group_keys_rejected(self):
        df = pd.DataFrame({"g": ["a", None], "y": ["x", "x"]})
        with pytest.raises(ValueError, match="missing group keys"):
            GroupedDataset.from_dataframe(df, group_col="g", label_col="y")

    def test_missing_pandas_na_labels_rejected(self, clinical):
        clinical["dx"] = pd.array(["mel", None] * 3, dtype="string")
        with pytest.raises(ValueError, match="NaN or None"):
            GroupedDataset.from_dataframe(clinical, group_col="patient", label_col="dx")

    def test_missing_counts_rejected(self):
        df = pd.DataFrame({"g": ["a", "b"], "x": [1.0, np.nan]})
        with pytest.raises(ValueError, match="missing values"):
            GroupedDataset.from_dataframe(df, group_col="g", count_cols=["x"])

    def test_single_column_name_as_string(self):
        df = pd.DataFrame({"g": ["a", "b"], "tumor": [1, 0]})
        d = GroupedDataset.from_dataframe(df, group_col="g", label_cols="tumor")
        assert d.class_names == ("tumor",)
        d = GroupedDataset.from_dataframe(df, group_col="g", count_cols="tumor")
        assert d.class_names == ("tumor",)

    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError, match="DataFrame"):
            GroupedDataset.from_dataframe(
                {"g": [1]}, group_col="g", label_col="y"
            )


class TestBuilderEquivalence:
    def test_all_paths_agree(self):
        """The same data through three builders must produce one dataset."""
        groups = np.array(["a", "a", "b", "b", "c"])
        y = np.array(["x", "y", "x", "x", "y"])

        from_arr = GroupedDataset.from_arrays(groups, y)
        from_df = GroupedDataset.from_dataframe(
            pd.DataFrame({"g": groups, "y": y}), group_col="g", label_col="y"
        )
        from_cnt = GroupedDataset.from_counts(
            from_arr.group_vectors,
            group_ids=from_arr.group_ids,
            class_names=from_arr.class_names,
        )

        for other in (from_df, from_cnt):
            np.testing.assert_array_equal(
                from_arr.group_vectors, other.group_vectors
            )
            np.testing.assert_array_equal(from_arr.group_ids, other.group_ids)
            assert from_arr.class_names == other.class_names

    def test_survives_a_round_trip_through_disk(self, tmp_path):
        from splitkit.io import load_npz, save_npz

        d = GroupedDataset.from_arrays(
            np.array(["a", "a", "b"]), np.array([0, 1, 0])
        )
        back = load_npz(save_npz(d, tmp_path / "d.npz"))
        np.testing.assert_array_equal(back.group_vectors, d.group_vectors)
        np.testing.assert_array_equal(back.item_group_index, d.item_group_index)


class TestEndToEnd:
    def test_built_dataset_splits(self):
        """A dataset built from raw arrays must feed straight into a strategy."""
        from splitkit.problem import Budget, SplitProblem
        from splitkit.strategies import get_strategy

        rng = np.random.default_rng(0)
        groups = rng.integers(0, 40, size=500)
        y = rng.integers(0, 3, size=500)
        d = GroupedDataset.from_arrays(groups, y, name="built")

        problem = SplitProblem.build(d, (0.7, 0.15, 0.15))
        out = get_strategy("annealing").run(
            problem, Budget(max_evals=2000), np.random.default_rng(0)
        )

        # THE guarantee: every item of a group lands in the same split, so no
        # group leaks across the train/val/test boundary.
        item_split = out.assignment[d.item_group_index]
        for g in range(d.n_groups):
            rows = item_split[d.item_group_index == g]
            assert len(np.unique(rows)) == 1, f"group {g} leaked across splits"

        # and the item indices partition the dataset exactly
        per_split = [np.flatnonzero(item_split == s) for s in range(3)]
        assert sum(len(ix) for ix in per_split) == d.n_items
        np.testing.assert_array_equal(
            np.sort(np.concatenate(per_split)), np.arange(d.n_items)
        )
        assert out.cost == pytest.approx(problem.evaluate(out.assignment), rel=1e-9)
