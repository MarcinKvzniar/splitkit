"""
ISIC 2020 preprocessing: train.csv -> DatasetGroups.

Groups: 2,056 patients. Items: dermoscopy images.
Feature vector: image counts per diagnosis class (9 classes, sorted alphabetically).
"""

import os

import numpy as np
import pandas as pd

from .common import DatasetGroups, save_dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

CSV_PATH = os.path.join(_ROOT, "datasets", "isic2020", "train.csv")
OUTPUT_PATH = os.path.join(_ROOT, "datasets", "isic2020", "preprocessed", "groups.pkl")


def preprocess() -> DatasetGroups:
    print("[ISIC2020] Loading train.csv...")
    df = pd.read_csv(CSV_PATH)

    class_names = sorted(df["diagnosis"].unique().tolist())
    class_index = {name: i for i, name in enumerate(class_names)}
    n_classes = len(class_names)

    print(f"[ISIC2020] {len(df)} images, "
          f"{df['patient_id'].nunique()} patients, "
          f"{n_classes} diagnosis classes")

    group_ids: list[str] = []
    group_vectors: list = []
    group_sizes: list[int] = []

    for patient_id, group in df.groupby("patient_id"):
        vec = np.zeros(n_classes, dtype=np.int64)
        for diag in group["diagnosis"]:
            vec[class_index[diag]] += 1

        group_ids.append(patient_id)
        group_vectors.append(vec)
        group_sizes.append(len(group))

    data = DatasetGroups(
        dataset_name="ISIC2020",
        group_ids=group_ids,
        group_vectors=np.array(group_vectors, dtype=np.int64),
        group_sizes=np.array(group_sizes, dtype=np.int32),
        class_names=class_names,
    )
    save_dataset(data, OUTPUT_PATH)
    print(data.summary())
    return data


if __name__ == "__main__":
    preprocess()
