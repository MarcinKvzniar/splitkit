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
  thousand groups.
- Only numpy is required.

## Install

```bash
pip install splitkit                 # numpy only
pip install 'splitkit[pandas]'       # DataFrame input and the command line
pip install 'splitkit[all]'          # plus exact solver (SciPy), plots, SGKF baseline
```

## Quickstart

```python
import splitkit

result = splitkit.split(df, group_col="patient_id", label_col="diagnosis", seed=0)

train, val, test = result.indices.astuple()     # row positions into df
df_train = df.iloc[train]
print(result.summary())
```

```text
====================================================================
Split: dataset  (3 splits, 299 groups, 3 classes)
Strategy: annealing   cost: 0.0917532
Evaluations: 300,000   time: 1.33s
--------------------------------------------------------------------
split           groups       items   requested    achieved
train              194       1,400      70.0%       70.0%
val                 54         300      15.0%       15.0%
test                51         300      15.0%       15.0%
--------------------------------------------------------------------
Worst class balance: 'melanoma' in 'val' off target by 1.6%
====================================================================
```

Other input forms:

```python
splitkit.split(groups=patient_ids, y=labels)                       # plain arrays
splitkit.split(df, group_col="identity", label_cols=attributes)   # multi-label indicators
splitkit.split(df, group_col="slide", count_cols=pixel_counts)    # per-class counts
splitkit.split(df, {"train": 0.8, "test": 0.2}, group_col=..., label_col=...)
```

`result.groups` gives group ids per split, `result.assign_column(df, "patient_id")`
adds a split column, and `splitkit.evaluate(dataset, assignment)` scores a split
made elsewhere on the same objective.

## Command line

```bash
splitkit data.csv --group-col patient_id --label-col diagnosis --seed 0 -o split.csv
```

This writes `data.csv` back with a `split` column and prints the quality report to
stderr. Run `splitkit --help` for all options.

## Strategies

| strategy | when to use it |
|---|---|
| `annealing` (default) | Any size. Simulated annealing whose schedule adapts to the budget. |
| `exact` | Up to a few thousand groups, with `pip install 'splitkit[exact]'`. Mixed-integer programming via HiGHS: a proven optimum, or the best split found plus a lower bound when time runs out. |
| `evolution` | Very large datasets, when minutes are acceptable. Differential evolution; it was best on CelebA's 10k groups. |
| `random`, `sgkf` | Baselines: random search and scikit-learn's `StratifiedGroupKFold`. |

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
| CelebA | 10,177 | 0.231 | 0.235\* | **0.154** | 1.278 |
| ISIC 2020 | 2,056 | 0.589 | **0.583** | **0.583** | 11.286 |
| BCSS | 151 | 7.49 | **7.20** | 7.48 | 44.27 |
| synth_large_complex | 763 | 0.420 | **0.410** | 0.575 | 10.34 |
| synth_heavy_imbalance | 158 | 2.34 | **1.91** | 2.52 | 28.24 |
| synth_few_groups | 96 | 1.61 | **1.01** | 2.27 | 17.61 |
| typical time | | 1.3 s | 20 s | 12–200 s | < 5 s |

\* No solution within 20 s, so it fell back to annealing.

scikit-learn's `StratifiedGroupKFold` is 5–70 times worse than annealing on every
dataset. The exact solver is best wherever it can run, and differential evolution
wins on the largest dataset if you can spend minutes rather than seconds.

Details, datasets and how to reproduce them are in
[`benchmarks/`](https://github.com/MarcinKvzniar/splitkit/tree/main/benchmarks).

## License

MIT
