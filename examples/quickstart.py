"""Split patient-grouped images into train/val/test without leakage."""

import numpy as np

import splitkit

rng = np.random.default_rng(0)
n_images = 2_000
patients = rng.integers(0, 300, n_images)
diagnosis = rng.choice(["benign", "nevus", "melanoma"], n_images, p=[0.80, 0.15, 0.05])

result = splitkit.split(groups=patients, y=diagnosis, seed=0)
print(result.summary())

train, val, test = result.indices.astuple()
assert not set(patients[train]) & (set(patients[val]) | set(patients[test]))
print(f"{len(train)} train, {len(val)} val, {len(test)} test images; no patient is shared.")
