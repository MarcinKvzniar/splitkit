"""ISIC 2020: train.csv -> GroupedDataset.

Groups: 2,056 patients. Items: dermoscopy images.
Vector: image counts per diagnosis class (9 classes, sorted alphabetically).

    uv run python -m benchmarks.etl.isic --data-dir datasets/isic2020
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from splitkit import GroupedDataset
from splitkit.io import save_npz


def preprocess(data_dir: Path, output: Path | None = None) -> GroupedDataset:
    csv_path = data_dir / "train.csv"
    print(f"[ISIC2020] Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    print(
        f"[ISIC2020] {len(df):,} images, "
        f"{df['patient_id'].nunique():,} patients, "
        f"{df['diagnosis'].nunique()} diagnosis classes"
    )

    table = pd.crosstab(df["patient_id"], df["diagnosis"]).sort_index()
    table = table.reindex(columns=sorted(table.columns))

    data = GroupedDataset(
        group_ids=np.asarray(table.index.astype(str), dtype=np.str_),
        group_vectors=table.to_numpy(dtype=np.float64),
        group_sizes=table.to_numpy(dtype=np.float64).sum(axis=1),
        class_names=tuple(str(c) for c in table.columns),
        name="ISIC2020",
    )

    out = output or data_dir / "preprocessed" / "groups.npz"
    save_npz(data, out)
    print(f"[saved] {data.name} -> {out}")
    print(data.summary())
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("datasets/isic2020"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    preprocess(args.data_dir, args.out)


if __name__ == "__main__":
    main()
