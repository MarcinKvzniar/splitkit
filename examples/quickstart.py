"""Split patient-grouped images into train/val/test without leakage."""

import numpy as np
import pandas as pd

import splitkit

# Example data: 2,000 images from 300 patients; melanoma is rare.
rng = np.random.default_rng(0)
df = pd.DataFrame({
    "image": [f"img_{i:04d}.jpg" for i in range(2_000)],
    "patient_id": rng.integers(0, 300, 2_000),
    "diagnosis": rng.choice(["benign", "nevus", "melanoma"], 2_000, p=[0.80, 0.15, 0.05]),
})

result = splitkit.split(df, group_col="patient_id", label_col="diagnosis", seed=0)
print(result.summary())

train, val, test = result.indices.astuple()     # row positions into df
df_train, df_val, df_test = df.iloc[train], df.iloc[val], df.iloc[test]

assert not set(df_train.patient_id) & (set(df_val.patient_id) | set(df_test.patient_id))
print(f"{len(df_train)} train, {len(df_val)} val, {len(df_test)} test images; no patient is shared.")
