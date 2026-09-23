"""Pickle-free ``.npz`` persistence; loading never enables ``allow_pickle``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import GroupedDataset

__all__ = ["load_npz", "save_npz"]

FORMAT_VERSION = 1


def save_npz(data: GroupedDataset, path: str | Path) -> Path:
    """Write ``data`` to ``path`` as a compressed, pickle-free ``.npz``."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {
        "group_vectors": np.ascontiguousarray(data.group_vectors, dtype=np.float64),
        "group_sizes": np.ascontiguousarray(data.group_sizes, dtype=np.float64),
        "group_ids": np.asarray(data.group_ids, dtype=np.str_),
        "class_names": np.asarray(data.class_names, dtype=np.str_),
        "meta": np.array(
            json.dumps({"name": data.name, "format": FORMAT_VERSION}), dtype=np.str_
        ),
    }
    if data.item_group_index is not None:
        arrays["item_group_index"] = np.ascontiguousarray(
            data.item_group_index, dtype=np.int32
        )

    np.savez_compressed(path, **arrays)
    return path


def load_npz(path: str | Path) -> GroupedDataset:
    """Read a dataset written by :func:`save_npz`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        try:
            meta = json.loads(str(z["meta"]))
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"{path} is not a splitkit dataset file.") from exc

        version = meta.get("format")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{path} uses dataset format {version}, but this version of splitkit "
                f"reads format {FORMAT_VERSION}."
            )

        item_group_index = (
            z["item_group_index"] if "item_group_index" in z.files else None
        )
        return GroupedDataset(
            group_ids=z["group_ids"],
            group_vectors=z["group_vectors"],
            group_sizes=z["group_sizes"],
            class_names=tuple(str(c) for c in z["class_names"]),
            name=meta.get("name", path.stem),
            item_group_index=item_group_index,
        )
