"""CelebA: identity + attribute CSVs -> GroupedDataset.

Groups: 10,177 celebrity identities. Items: face images.
Vector: sum of 40 binary attributes per identity (CSV encodes -1/+1 -> 0/1).

    uv run python -m benchmarks.etl.celeba --data-dir datasets/celeb-faces
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from splitkit import GroupedDataset
from splitkit.io import save_npz


def preprocess(data_dir: Path, output: Path | None = None) -> GroupedDataset:
    print("[CelebA] Loading identity file...")
    identity_df = pd.read_csv(
        data_dir / "identity_CelebA.txt",
        sep=" ",
        header=None,
        names=["image_id", "identity_id"],
    )

    print("[CelebA] Loading attribute file...")
    attr_df = pd.read_csv(data_dir / "list_attr_celeba.csv")

    attr_cols = [c for c in attr_df.columns if c != "image_id"]
    attr_df[attr_cols] = ((attr_df[attr_cols] + 1) // 2).astype(np.int8)

    print("[CelebA] Merging attributes with identities...")
    merged = attr_df.merge(identity_df, on="image_id")
    print(
        f"[CelebA] {len(merged):,} images, "
        f"{merged['identity_id'].nunique():,} identities"
    )

    print("[CelebA] Aggregating group feature vectors...")
    grouped = merged.groupby("identity_id")
    vectors = grouped[attr_cols].sum().sort_index()
    sizes = grouped.size().sort_index()

    data = GroupedDataset(
        group_ids=np.asarray(vectors.index.astype(str), dtype=np.str_),
        group_vectors=vectors.to_numpy(dtype=np.float64),
        group_sizes=sizes.to_numpy(dtype=np.float64),
        class_names=tuple(attr_cols),
        name="CelebA",
    )

    out = output or data_dir / "preprocessed" / "groups.npz"
    save_npz(data, out)
    print(f"[saved] {data.name} -> {out}")
    print(data.summary())
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("datasets/celeb-faces"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    preprocess(args.data_dir, args.out)


if __name__ == "__main__":
    main()
