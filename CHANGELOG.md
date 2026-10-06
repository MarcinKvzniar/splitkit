# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-24

Initial release.

### Added

- `splitkit.split()`: group-aware stratified splitting into any number of parts
  with any ratios. Groups never cross splits, and class balance is optimised
  directly. Accepts a DataFrame (`label_col`, multi-label `label_cols`, or
  per-class `count_cols`), plain `groups=`/`y=` arrays, or a `GroupedDataset`.
- `SplitResult` with item indices (`.indices`), group ids (`.groups`), pandas
  helpers (`.assign_column()`, `.to_frame()`, `.counts_frame()`) and a quality
  report (`.summary()`), including classes too rare to stratify.
- `splitkit.evaluate()` to score a split made elsewhere on the same objective.
- Strategies: simulated annealing (default), an exact MILP solver that reports a
  proven optimum or lower bound, differential evolution for very large datasets,
  and random search and `StratifiedGroupKFold` (`[sklearn]` extra) baselines.
- Budgets by evaluation count or wall-clock time, and reproducible results
  with `seed=`.
- `splitkit` command line for CSV/TSV files.
- `splitkit.viz` plots of per-class allocation and convergence (`[viz]` extra).
- A live progress bar (`progress=True`, on by default in the command line) and a
  styled report via `rich.print(result)`.
- `GroupedDataset` builders, pickle-free `.npz` I/O, and synthetic datasets for
  experiments.
- Requires numpy, pandas, SciPy and rich; plots and the SGKF baseline are
  optional (`[all]`). Python 3.10–3.14, fully typed.

[Unreleased]: https://github.com/MarcinKvzniar/splitkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MarcinKvzniar/splitkit/releases/tag/v0.1.0
