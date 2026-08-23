"""
BCSS preprocessing: 151 WSI mask PNGs -> DatasetGroups.

Groups: WSIs. Items: non-overlapping PATCH_SIZExPATCH_SIZE tiles.
Feature vector: pixel counts for classes 1-21 (class 0 = outside ROI, excluded).
Group size = floor(H/PATCH_SIZE) x floor(W/PATCH_SIZE).
"""

import os

import numpy as np
from PIL import Image

from .common import DatasetGroups, save_dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

MASK_DIR = os.path.join(_ROOT, "datasets", "bcss", "mask")
OUTPUT_PATH = os.path.join(_ROOT, "datasets", "bcss", "preprocessed", "groups.pkl")

PATCH_SIZE = 512

# GT codes 1-21 from gtruth_codes.tsv (code 0 = outside_roi, excluded)
CLASS_NAMES = [
    "tumor",                    # 1
    "stroma",                   # 2
    "lymphocytic_infiltrate",   # 3
    "necrosis_or_debris",       # 4
    "glandular_secretions",     # 5
    "blood",                    # 6
    "exclude",                  # 7
    "metaplasia_NOS",           # 8
    "fat",                      # 9
    "plasma_cells",             # 10
    "other_immune_infiltrate",  # 11
    "mucoid_material",          # 12
    "normal_acinus_or_duct",    # 13
    "lymphatics",               # 14
    "undetermined",             # 15
    "nerve",                    # 16
    "skin_adnexa",              # 17
    "blood_vessel",             # 18
    "angioinvasion",            # 19
    "dcis",                     # 20
    "other",                    # 21
]
N_CLASSES = len(CLASS_NAMES)  # 21


def preprocess(patch_size: int = PATCH_SIZE) -> DatasetGroups:
    mask_files = sorted(f for f in os.listdir(MASK_DIR) if f.endswith(".png"))
    if not mask_files:
        raise FileNotFoundError(f"No mask PNG files found in {MASK_DIR}")

    group_ids: list[str] = []
    group_vectors: list = []
    group_sizes: list[int] = []

    print(f"[BCSS] Processing {len(mask_files)} masks (patch_size={patch_size})...")
    for idx, fname in enumerate(mask_files, 1):
        wsi_id = fname.split("_xmin")[0]
        mask_path = os.path.join(MASK_DIR, fname)

        mask = np.array(Image.open(mask_path))
        h, w = mask.shape

        pixel_counts = np.bincount(mask.ravel(), minlength=N_CLASSES + 1)
        counts = pixel_counts[1: N_CLASSES + 1].astype(np.int64)

        n_patches = max(1, (h // patch_size) * (w // patch_size))

        group_ids.append(wsi_id)
        group_vectors.append(counts)
        group_sizes.append(n_patches)

        if idx % 25 == 0 or idx == len(mask_files):
            print(f"  {idx}/{len(mask_files)}  {fname}")

    data = DatasetGroups(
        dataset_name="BCSS",
        group_ids=group_ids,
        group_vectors=np.array(group_vectors, dtype=np.int64),
        group_sizes=np.array(group_sizes, dtype=np.int32),
        class_names=CLASS_NAMES,
    )
    save_dataset(data, OUTPUT_PATH)
    print(data.summary())
    return data


if __name__ == "__main__":
    preprocess()
