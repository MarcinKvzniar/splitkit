"""BCSS: WSI mask PNGs -> GroupedDataset.

Groups: whole-slide images. Items: non-overlapping PATCH_SIZExPATCH_SIZE tiles.
Vector: pixel counts for classes 1-21 (class 0 = outside ROI, excluded).
Group size = floor(H/PATCH_SIZE) x floor(W/PATCH_SIZE).

    uv run python -m benchmarks.etl.bcss --data-dir datasets/bcss
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from splitkit import GroupedDataset
from splitkit.io import save_npz

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


def preprocess(
    data_dir: Path,
    output: Path | None = None,
    patch_size: int = PATCH_SIZE,
) -> GroupedDataset:
    mask_dir = data_dir / "mask"
    mask_files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".png"))
    if not mask_files:
        raise FileNotFoundError(f"No mask PNG files found in {mask_dir}")

    group_ids: list[str] = []
    group_vectors: list = []
    group_sizes: list[int] = []

    print(f"[BCSS] Processing {len(mask_files)} masks (patch_size={patch_size})...")
    for idx, fname in enumerate(mask_files, 1):
        wsi_id = fname.split("_xmin")[0]
        mask_path = mask_dir / fname

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

    data = GroupedDataset.from_counts(
        np.array(group_vectors, dtype=np.float64),
        group_ids=group_ids,
        class_names=CLASS_NAMES,
        group_sizes=np.array(group_sizes, dtype=np.float64),
        name="BCSS",
    )

    out = output or data_dir / "preprocessed" / "groups.npz"
    save_npz(data, out)
    print(f"[saved] {data.name} -> {out}")
    print(data.summary())
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("datasets/bcss"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    args = ap.parse_args()
    preprocess(args.data_dir, args.out, args.patch_size)


if __name__ == "__main__":
    main()
