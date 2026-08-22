"""
CelebA preprocessing: identity + attribute CSVs -> DatasetGroups.

Groups: 10,177 celebrity identities. Items: face images.
Feature vector: sum of 40 binary attributes per identity (CSV +1->1, -1->0).
"""

import os

import numpy as np
import pandas as pd

from .common import DatasetGroups, save_dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

IDENTITY_PATH = os.path.join(_ROOT, "datasets", "celeb-faces", "identity_CelebA.txt")
ATTR_PATH = os.path.join(_ROOT, "datasets", "celeb-faces", "list_attr_celeba.csv")
OUTPUT_PATH = os.path.join(_ROOT, "datasets", "celeb-faces", "preprocessed", "groups.pkl")


def preprocess() -> DatasetGroups:
    print("[CelebA] Loading identity file...")
    identity_df = pd.read_csv(
        IDENTITY_PATH, sep=" ", header=None, names=["image_id", "identity_id"]
    )

    print("[CelebA] Loading attribute file...")
    attr_df = pd.read_csv(ATTR_PATH)

    attr_cols = [c for c in attr_df.columns if c != "image_id"]

    attr_df[attr_cols] = ((attr_df[attr_cols] + 1) // 2).astype(np.int8)

    print("[CelebA] Merging attributes with identities...")
    merged = attr_df.merge(identity_df, on="image_id")

    print(f"[CelebA] {len(merged)} images, "
          f"{merged['identity_id'].nunique()} identities")

    print("[CelebA] Aggregating group feature vectors...")
    grouped = merged.groupby("identity_id")

    group_ids: list[str] = []
    group_vectors: list = []
    group_sizes: list[int] = []

    for identity_id, group in grouped:
        vectors = group[attr_cols].values
        group_ids.append(str(identity_id))
        group_vectors.append(vectors.sum(axis=0))
        group_sizes.append(len(group))

    data = DatasetGroups(
        dataset_name="CelebA",
        group_ids=group_ids,
        group_vectors=np.array(group_vectors, dtype=np.int64),
        group_sizes=np.array(group_sizes, dtype=np.int32),
        class_names=attr_cols,
    )
    save_dataset(data, OUTPUT_PATH)
    print(data.summary())
    return data


if __name__ == "__main__":
    preprocess()
