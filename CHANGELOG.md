# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The project was a university coursework repository (metaheuristic optimization of
group-aware stratified splitting) and is being rebuilt as an installable package.

### Added

- Installable `splitkit` package under `src/splitkit/`, MIT licensed, typed
  (`py.typed`), with **numpy as the only required dependency**. Everything else
  (pandas, scikit-learn, matplotlib, SciPy) is an optional extra.
- `GroupedDataset`: validated, immutable dataset container replacing
  `DatasetGroups`, with `class_group_counts` and `is_onehot`.
- `splitkit.io`: pickle-free `.npz` persistence. Loading never enables
  `allow_pickle`, so opening a dataset file cannot execute code.
- `splitkit.synthetic`: `make_synthetic()`, `from_preset()` and six named presets,
  promoted from a benchmark script into the public API.
- **Dataset builders.** `GroupedDataset.from_arrays()`, `.from_labels()`
  (scikit-learn's `(y, groups)` argument order), `.from_counts()` and
  `.from_dataframe()`. The DataFrame builder covers the three shapes real data
  arrives in — one categorical column (`label_col`), several indicator columns
  (`label_cols`), or rows that already *are* count vectors (`count_cols`, with an
  optional `size_col`). A ±1 indicator encoding is detected and mapped to 0/1
  rather than cancelling presence against absence. Only `from_dataframe` needs
  pandas, and it raises an actionable install message when absent.
- **Item provenance** (`item_group_index`). Datasets built from item-level input
  record which group each row belongs to, so a split can return the row indices
  a user actually trains on rather than only naming groups. Costs 4 bytes per
  item and can be disabled with `track_items=False`. `subset()` deliberately
  drops it, since retained indices would point at the old group numbering.
- `SplitProblem`: immutable problem object holding the count matrix, class
  weights, target counts and prepared objective. Strategies are now stateless.
- `Budget`: `max_evals`, `time_limit` and `target_cost` stop conditions.
  **Wall-clock limits are new** — the previous implementation could not stop
  early under any circumstances.
- `splitkit.objectives`: pluggable objective interface, with `wmape` as the
  default. Objectives are *prepared* against a fixed target so the constant part
  of the formula is computed once instead of per evaluation.
- **Exact MILP strategy** (`strategy="exact"`, `pip install 'splitkit[exact]'`).
  Solves the split with SciPy's HiGHS solver and reports a proven optimum, or the
  best solution plus a lower bound when the time limit runs out. Count constraints
  are scaled by their targets, so pixel-scale counts stay within solver
  tolerances. With a 20 s limit it beat 300k-evaluation annealing on 7 of 8
  benchmark datasets (for example 1.01 vs 1.81 on `synth_few_groups`, and
  proven optimal on `synth_concentrated` in 0.2 s). Practical up to a few
  thousand groups; beyond that it warns and falls back to annealing.
- **`splitkit` command line** (also `python -m splitkit`). Reads a CSV or TSV
  (or stdin) and writes it back with a split column, printing the quality report
  to stderr so the output pipes cleanly:
  `splitkit data.csv --group-col patient --label-col diagnosis -o split.csv`.
  Requires the `pandas` extra.
- **Plots** (`splitkit.viz`, `pip install 'splitkit[viz]'`).
  `plot_distribution()` draws, for every class, the fraction of its items in
  each split against the target ratios, so rare classes are as visible as common
  ones. `plot_convergence()` draws best cost against evaluations, with the
  proven lower bound when the exact strategy supplies one. Both accept and
  return a matplotlib `Axes`.
- Strategy registry: `get_strategy()`, `list_strategies()`, `register_strategy()`.
- Warm-start support (`Strategy.run(..., warm_start=...)`).
- Arbitrary **K splits**. Split counts are no longer hardcoded to three; ratios
  may be any length, with conventional names for K=2/K=3 and `split_i` beyond.
- **`splitkit.split()` and `splitkit.evaluate()`**, the public entry points.
  `split()` accepts a `GroupedDataset`, a DataFrame plus column names, or raw
  `groups=`/`y=` arrays, and takes ratios either as a mapping
  (`{"train": 0.8, "test": 0.2}`) or a bare sequence. `evaluate()` scores an
  externally produced assignment on the same objective, so a split from anywhere
  else can be compared on equal terms.
- **`SplitResult`**, with `.indices` for item positions, `.groups` for group ids,
  `.to_frame()` / `.counts_frame()` / `.assign_column()` for pandas users, and
  `.summary()` for a quality report. `actual_counts` is always recomputed from the
  returned assignment, so no incrementally maintained accumulator can drift into
  the reported numbers.
- **`SplitMapping`**, an ordered name-keyed view. A plain dict would unpack to its
  *keys*, so `train, val, test = result.groups` would silently yield three
  strings; `.astuple()` gives the arrays people mean.
- Split quality reporting: `achieved_ratios` (the realised **item** share, which
  differs from the class-count objective on non-one-hot data), `worst_cell()`,
  `empty_classes()`, and the list of classes excluded as unstratifiable — which
  the previous implementation silently zero-weighted without telling anyone.

### Changed

- **Ratios are normalised rather than rejected.** `(7, 1.5, 1.5)` and
  `(0.7, 0.15, 0.15)` express the same intent. Non-finite, zero and negative
  ratios still raise.
- `Optimizer` (ABC) replaced by `Strategy`, returning an internal `Outcome`.
- **Class weights are normalised to mean 1 by default** (`weight_normalize=True`).
  The original inverse-frequency weights were documented as mean-normalised but
  never were, so costs were incomparable across datasets and the annealing
  temperature meant something different on every dataset.
- **Item counts are part of the objective by default** (`size_weight=1.0`).
  Inverse-frequency weights make a majority class nearly free to move, so class
  counts alone let item ratios drift badly even on one-hot data. `size_weight=0`
  restores class-only matching.

  Worst per-split item-ratio error against a 70/15/15 target (annealing, 40k
  evaluations, mean of 3 seeds), old defaults vs new:

  | dataset | old | new |
  |---|---|---|
  | ISIC 2020 (98% one class) | 0.81 pp | 0.00 pp |
  | BCSS (pixel vs tile counts) | 4.42 pp | 0.02 pp |
  | synth_heavy_imbalance | 13.23 pp | 0.30 pp |
  | synth_concentrated | 0.85 pp | 0.49 pp |

  Class-balance cost is equal or better on 6 of 9 benchmark datasets, and drops
  from 4.78 to 0.98 on CelebA.
- Simulated annealing's default `initial_temp` is now `1.0` (was `10.0`), the scale
  of a typical move under normalised weights. At 300k evaluations it matches or
  beats `100.0` on 7 of 9 datasets.
- **Simulated annealing's cooling schedule now fits the budget** (`cooling_rate="auto"`,
  the new default). A fixed rate silently assumes a particular budget: 0.9999
  needs roughly 300k steps to anneal, so a shorter run never left its exploration
  phase and returned something close to random. Passing an explicit
  `cooling_rate` keeps the old geometric behaviour exactly.

  Measured on a 60-group, 2-class patient dataset (seed 42):

  | evaluations | `"auto"` | fixed `0.9999` | random search |
  |---|---|---|---|
  | 2,000 | **0.062** | 2.435 | 0.078 |
  | 10,000 | **0.000** | 2.435 | 0.031 |
  | 50,000 | **0.000** | 0.450 | 0.016 |

  On a 1,000-group instance the same change takes the cost at 20k evaluations
  from 0.960 to 0.021. With only a wall-clock budget the schedule is calibrated
  once from observed throughput, since there is no evaluation count to fit to.
- SGKF fold-to-split apportionment now uses the largest-remainder method. The
  previous loop repeatedly adjusted the same index and could leave a split with
  no folds at all for skewed ratios.
- SGKF now derives scikit-learn's `random_state` from the strategy's random
  generator instead of receiving a raw seed, so every strategy draws from one
  RNG. Its fold assignment therefore differs from previous releases for the same
  nominal seed; the other strategies are unaffected.
- Dataset fixtures converted from `.pkl` to `.npz` (3.8 MB to ~386 KiB).
- Benchmark and tuning scripts moved to `benchmarks/`, no longer shipped in the
  wheel. Their broken `sys.path` manipulation is gone, and the duplicated
  `_unpack_history` helper is shared.
- Dataset ETL scripts take `--data-dir` instead of walking up from `__file__`,
  which broke as soon as directory depth changed and was meaningless once
  installed. They now build datasets through the public builders, which removed
  their hand-rolled aggregation loops (ISIC looped per row) and the hardcoded
  CelebA ±1 conversion. All three reproduce the committed fixtures byte for byte.
- Simulated annealing draws its random moves in blocks rather than one at a time,
  raising throughput by 36% (162k to 221k evaluations/s on `synth_large_complex`).
  Search quality is statistically unchanged, but trajectories for a given seed
  differ from earlier versions. Scoring a move from its two changed rows was
  measured as well and rejected: with numpy's per-call overhead it is slower than
  a full recomputation until K x C reaches the thousands.
- The package passes `mypy --strict`, and docstrings were trimmed to the essentials.

### Removed

- `evaluate_assignment()`. It was dead code, and it scored assignments with a
  *different* objective than the optimizers minimised (it omitted the
  unstratifiable-class mask), so it would have silently disagreed with
  `Optimizer.evaluate` had anything called it.
- Pickle loading. `.npz` is the only supported dataset format.

### Fixed

- **Differential evolution could loop forever.** An evaluation is only spent when
  a trial's discretised assignment differs from its parent, so once the population
  collapses no trial differs, the evaluation counter stops advancing, and
  `while n_evals < max_evals` never terminates. Reproducible on small problems
  with small populations (`DE/best/1/bin`, `pop_size=12`); large benchmark runs
  masked it. A generation that spends no evaluations now ends the run and reports
  `converged`.
- **SGKF could request more folds than there are groups.** The fold count was
  capped by the rarest class and by 20, but never by the number of groups, so a
  6-group dataset asked scikit-learn for 20 folds and raised. It is now capped by
  group count as well.
- `splitkit.strategies` no longer requires scikit-learn to import; the dependency
  is resolved lazily when the SGKF baseline actually runs, and the error names
  the extra to install.

### Testing

- First test suite for the project: 281 tests, 100% statement coverage of
  `src/splitkit`, running in about 8 seconds.
- Strategy tests are parametrized over the registry, so a newly registered
  strategy is held to the full contract automatically.
- A brute-force oracle enumerates every assignment for tiny instances, giving
  incontestable ground truth for optimality claims.
- The incremental-update invariant is pinned down explicitly: incremental counts,
  rejected-move undo, and two-row cost deltas are each checked against a fresh
  recomputation.
- Builder tests assert the no-leakage guarantee end to end: every item of a group
  lands in one split, and the returned item indices partition the dataset exactly.
