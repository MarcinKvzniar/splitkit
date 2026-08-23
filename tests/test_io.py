"""Dataset persistence round-trips.

The format must stay pickle-free: loading a dataset file should never be able to
execute code, which rules out object arrays and `allow_pickle=True`.
"""

from __future__ import annotations

import numpy as np
import pytest

from _helpers import make_dataset
from splitkit.io import FORMAT_VERSION, load_npz, save_npz


class TestRoundTrip:
    def test_preserves_everything(self, tiny, tmp_path):
        path = save_npz(tiny, tmp_path / "d.npz")
        back = load_npz(path)

        assert back.name == tiny.name
        assert back.class_names == tiny.class_names
        np.testing.assert_array_equal(back.group_ids, tiny.group_ids)
        np.testing.assert_allclose(back.group_vectors, tiny.group_vectors)
        np.testing.assert_allclose(back.group_sizes, tiny.group_sizes)

    def test_preserves_distinct_sizes(self, soft_counts, tmp_path):
        """Sizes are independent of the vectors and must not be recomputed."""
        back = load_npz(save_npz(soft_counts, tmp_path / "s.npz"))
        np.testing.assert_allclose(back.group_sizes, soft_counts.group_sizes)
        assert back.is_onehot is False

    def test_unicode_group_ids(self, tmp_path):
        d = make_dataset([[1.0, 2.0], [3.0, 4.0]])
        d = type(d)(
            group_ids=np.array(["Kuźniar", "日本語"], dtype=np.str_),
            group_vectors=d.group_vectors,
            group_sizes=d.group_sizes,
            class_names=("ä", "ß"),
            name="ünïcödé",
        )
        back = load_npz(save_npz(d, tmp_path / "u.npz"))
        np.testing.assert_array_equal(back.group_ids, d.group_ids)
        assert back.class_names == ("ä", "ß")
        assert back.name == "ünïcödé"

    def test_item_group_index_optional(self, tiny, tmp_path):
        assert load_npz(save_npz(tiny, tmp_path / "a.npz")).item_group_index is None

        with_items = type(tiny)(
            group_ids=tiny.group_ids,
            group_vectors=tiny.group_vectors,
            group_sizes=tiny.group_sizes,
            class_names=tiny.class_names,
            name=tiny.name,
            item_group_index=np.array([0, 0, 1, 2, 3, 4, 5], dtype=np.int32),
        )
        back = load_npz(save_npz(with_items, tmp_path / "b.npz"))
        np.testing.assert_array_equal(
            back.item_group_index, with_items.item_group_index
        )


class TestFileHandling:
    def test_suffix_added(self, tiny, tmp_path):
        path = save_npz(tiny, tmp_path / "noext")
        assert path.suffix == ".npz"
        assert path.exists()

    def test_creates_parent_directories(self, tiny, tmp_path):
        path = save_npz(tiny, tmp_path / "deep" / "nested" / "d.npz")
        assert path.exists()

    def test_rejects_unknown_format_version(self, tiny, tmp_path):
        import json

        path = tmp_path / "bad.npz"
        np.savez_compressed(
            path,
            group_vectors=tiny.group_vectors,
            group_sizes=tiny.group_sizes,
            group_ids=tiny.group_ids,
            class_names=np.asarray(tiny.class_names, dtype=np.str_),
            meta=np.array(json.dumps({"name": "x", "format": 999}), dtype=np.str_),
        )
        with pytest.raises(ValueError, match="format 999"):
            load_npz(path)

    def test_rejects_foreign_npz(self, tmp_path):
        path = tmp_path / "foreign.npz"
        np.savez_compressed(path, something_else=np.ones(3))
        with pytest.raises(ValueError, match="not a splitkit dataset"):
            load_npz(path)


class TestPickleFree:
    def test_loads_without_allow_pickle(self, tiny, tmp_path):
        """The whole point of the format: no code execution on load."""
        path = save_npz(tiny, tmp_path / "d.npz")
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) >= {
                "group_vectors", "group_sizes", "group_ids", "class_names", "meta",
            }

    def test_no_object_arrays(self, tiny, tmp_path):
        """Object dtypes would force allow_pickle=True on the reader."""
        path = save_npz(tiny, tmp_path / "d.npz")
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                assert z[key].dtype != np.dtype("O"), key

    def test_current_version_recorded(self, tiny, tmp_path):
        import json

        path = save_npz(tiny, tmp_path / "d.npz")
        with np.load(path, allow_pickle=False) as z:
            assert json.loads(str(z["meta"]))["format"] == FORMAT_VERSION


class TestFixturesOnDisk:
    """The committed benchmark fixtures must stay loadable."""

    @pytest.mark.parametrize(
        "path, groups, classes",
        [
            ("datasets/bcss/preprocessed/groups.npz", 151, 21),
            ("datasets/isic2020/preprocessed/groups.npz", 2056, 9),
            ("datasets/celeb-faces/preprocessed/groups.npz", 10177, 40),
        ],
    )
    def test_real_fixtures(self, path, groups, classes):
        import pathlib

        p = pathlib.Path(path)
        if not p.exists():
            pytest.skip(f"{path} not present")
        d = load_npz(p)
        assert d.n_groups == groups
        assert d.n_classes == classes
