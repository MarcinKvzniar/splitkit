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
  (pandas, scikit-learn, matplotlib, OR-Tools, SciPy, PuLP) is an optional extra.
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
- Simulated annealing's default `initial_temp` is now `100.0` (was `10.0`),
  matching the value the grid search selected.
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
  recomputation. This is the precondition for replacing the full cost
  recomputation with true O(C) deltas.
- Builder tests assert the no-leakage guarantee end to end: every item of a group
  lands in one split, and the returned item indices partition the dataset exactly.

### Notes on correctness

Two behaviour switches are implemented but **not yet enabled by default**, so
that the refactor could be verified as behaviour-preserving against the original
benchmark numbers. They become defaults in a later step:

- `weight_normalize` — the original inverse-frequency weights were documented as
  "mean-normalised to 1" but never normalised, leaving costs incomparable across
  datasets.
- `size_weight` — matching per-class counts implies matching *item* counts only
  for one-hot single-label data.

Measured effect of `size_weight` on the achieved train/val/test **item** ratio
against a 70/15/15 target (annealing, 40k evaluations, seed 42):

| dataset | `size_weight=0` | `size_weight="auto"` |
|---|---|---|
| BCSS (pixel counts vs tile counts) | 0.551 / 0.203 / 0.245 — **14.9 pp error** | 0.696 / 0.152 / 0.153 — 0.4 pp error |
| CelebA (40 binary attributes) | 0.703 / 0.149 / 0.148 — 0.25 pp error | 0.703 / 0.149 / 0.148 — 0.25 pp error |

The failure mode is severe but driven by **unit mismatch** (pixels vs tiles),
not by multi-label targets as such: CelebA's attribute mass tracks image count
closely enough (r = 0.90) that class counts alone pin the item counts. BCSS
correlates similarly (r = 0.86) yet still fails badly, so the correlation is not
by itself a safe predictor — which is why `"auto"` keys off `is_onehot` rather
than a heuristic threshold.
