# splitkit

**Leakage-free train/val/test splits that keep every class balanced.**

Many datasets are grouped: several images per patient, many tiles per slide, many
photos per person. Every item of a group must land in the same split, or the model
is tested on data it has effectively seen. Keeping groups whole makes class balance
hard, especially with rare classes and uneven group sizes, and scikit-learn's
`StratifiedGroupKFold` only offers equal folds, single labels, and a greedy
heuristic. splitkit treats the split as an optimisation problem and solves it
directly.

- Groups never cross splits, by construction.
- Any ratios and any number of splits: `{"train": 0.7, "val": 0.15, "test": 0.15}`.
- Single-label, multi-label, and count data (such as pixel counts per class).
- A quality report on every split, and a proven optimum for problems up to a few
  thousand distinct groups.
- Works on DataFrames, plain arrays, or CSV files from the command line.

## Install

```bash
pip install splitkit
```

This covers everything except two optional extras:

| extra | adds | for |
|---|---|---|
| `splitkit[viz]` | matplotlib | `splitkit.viz` plots |
| `splitkit[sklearn]` | scikit-learn | the `sgkf` baseline strategy |
| `splitkit[all]` | both | |

## Quickstart

```python
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
```

```text
====================================================================
Split: dataset  (3 splits, 299 groups, 3 classes)
Strategy: annealing   cost: 0.0917532
Evaluations: 300,000   time: 1.34s
--------------------------------------------------------------------
split           groups       items   requested    achieved
train              197       1,400      70.0%       70.0%
val                 51         300      15.0%       15.0%
test                51         300      15.0%       15.0%
--------------------------------------------------------------------
Worst class balance: 'melanoma' in 'val' off target by 1.6%
====================================================================
```

For your own data, replace the example `df` with your table. The next section
explains what it needs.

## Preparing your data

splitkit never reads your images or files. It needs a table with **one row per item**
(an image, a tile, a visit) and two kinds of columns:

- **A group column**: the unit that must stay whole, such as a patient, slide or
  case. Choose it carefully. If one patient appears under two IDs (for example the
  folders `case7_0` and `case7_1`), map both to one ID first, or that patient can
  leak across splits.
- **Labels**, in one of three forms:

| your labels | argument | example |
|---|---|---|
| one class per item | `label_col="diagnosis"` | classification |
| several 0/1 (or -1/1) indicators per item | `label_cols=["smiling", "male"]` | multi-label attributes |
| an amount of each class per item | `count_cols=["tumor", "stroma"]` | pixel counts from segmentation masks |

```python
splitkit.split(df, group_col="identity", label_cols=attributes)
splitkit.split(df, group_col="slide", count_cols=["tumor", "stroma"])
splitkit.split(groups=patient_ids, y=labels)                  # plain arrays, no DataFrame
splitkit.split(df, {"train": 0.8, "test": 0.2}, group_col=..., label_col=...)
```

Group IDs and labels must not be missing; drop or fill those rows first. Other
columns, such as file paths, are carried along untouched.

### Segmentation masks

Count the pixels of each class in every mask, then split on those counts. Each class
then gets the same share of its pixels in every split, so every split keeps the
overall class proportions. This example reads RGB masks stored as
`masks/<case>/<tile>.png` (it needs Pillow):

```python
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import splitkit

PALETTE = {"background": (0, 0, 0), "healthy": (66, 135, 245), "cancer": (245, 66, 66)}
colours = np.array(list(PALETTE.values()))

rows = []
for path in sorted(Path("masks").glob("*/*.png")):
    pixels = np.asarray(Image.open(path).convert("RGB")).reshape(-1, 3).astype(int)
    nearest = ((pixels[:, None] - colours) ** 2).sum(axis=2).argmin(axis=1)
    counts = np.bincount(nearest, minlength=len(colours))
    rows.append({"case": path.parent.name, "mask": str(path), **dict(zip(PALETTE, counts))})
df = pd.DataFrame(rows)

result = splitkit.split(df, group_col="case", count_cols=["healthy", "cancer"])
df = result.assign_column(df, "case")       # adds a "split" column next to each path
```

Matching each pixel to the nearest palette colour also handles blended colours at
region edges, which resized masks often have. For masks that store class indices
(0, 1, 2, ...) instead of colours, use
`np.bincount(np.asarray(Image.open(path)).ravel(), minlength=n_classes)`. Leave
background out of `count_cols` unless its share matters to you.

## Working with the result

`result.groups` gives group ids per split, `result.assign_column(df, "patient_id")`
adds a split column, and `splitkit.evaluate(dataset, assignment)` scores a split
made elsewhere on the same objective.

Pass `progress=True` for a live progress bar, and use `rich.print(result)` to print
the report as a formatted panel.

## Command line

```bash
splitkit data.csv --group-col patient_id --label-col diagnosis --seed 0 -o split.csv
```

This writes `data.csv` back unchanged plus a `split` column. In a terminal it shows a
progress bar and then the quality report on stderr; `-q` silences both. Run
`splitkit --help` for all options.

## Strategies

| strategy | when to use it |
|---|---|
| `annealing` (default) | Any size. Simulated annealing whose schedule adapts to the budget. |
| `exact` | Up to a few thousand *distinct* groups. Mixed-integer programming via HiGHS: a proven optimum, or the best split found plus a lower bound when time runs out. Identical groups are merged, so 70k single-visit patients may be only a few hundred types. Above 10,000 types it warns and uses annealing. |
| `evolution` | Very large datasets, when minutes are acceptable. Differential evolution; it was best on CelebA's 10k groups. |
| `random`, `sgkf` | Baselines: random search and scikit-learn's `StratifiedGroupKFold` (`sgkf` needs `splitkit[sklearn]`). |

```python
splitkit.split(df, ..., strategy="exact", time_budget=30)
splitkit.split(df, ..., max_evals=50_000)       # any strategy also accepts time_budget=
```

## How it works

Each group is summarised as a vector of class counts. For every split `s` and class
`c` there is a target count, the class total times the split's ratio. splitkit
minimises the weighted relative error

```
cost = Σ_s Σ_c  w_c · |actual[s,c] − target[s,c]| / (target[s,c] + 1)
```

The weights `w_c` are inverse class frequencies, normalised to mean 1, so rare
classes count as much as common ones. One extra column tracks item counts, so the
split sizes stay on target too. Classes present in fewer groups than there are
splits cannot appear in every split; they are left out of the cost and listed in
the report.

The objective balances every class *on average*. When a class is concentrated in a
few large groups, a perfect ratio may not exist. Use
`splitkit.viz.plot_distribution(result)` to see where each class went.

## Benchmarks

Cost on a 70/15/15 split (lower is better), averaged over 10 seeds with 300k
evaluations each. Exact gets 20 seconds.

| dataset | groups | annealing | exact | evolution | `StratifiedGroupKFold` |
|---|---|---|---|---|---|
| CelebA | 10,177 | 0.215 | 0.215\* | **0.154** | 1.278 |
| ISIC 2020 | 2,056 | 0.589 | **0.583** | **0.583** | 11.286 |
| BCSS | 151 | 7.49 | **7.20** | 7.48 | 44.27 |
| synth_large_complex | 763 | 0.420 | **0.410** | 0.575 | 10.34 |
| synth_heavy_imbalance | 158 | 2.34 | **1.91** | 2.52 | 28.24 |
| synth_few_groups | 96 | 1.61 | **1.01** | 2.27 | 17.61 |
| typical time | | 1.3 s | 20 s | 12–200 s | < 5 s |

\* More than 10,000 distinct groups, so the exact strategy hands over to annealing.

scikit-learn's `StratifiedGroupKFold` is 3–25 times worse than annealing on every
dataset here. It comes close only on data made of many small single-label groups,
such as patients with one or two visits each. The exact solver is best wherever it
can run, and differential evolution wins on the largest dataset if you can spend
minutes rather than seconds.

Details, datasets and how to reproduce them are in
[`benchmarks/`](https://github.com/MarcinKvzniar/splitkit/tree/main/benchmarks).

## License

MIT
