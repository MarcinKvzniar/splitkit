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

### Changed

- **Ratios are normalised rather than rejected.** `(7, 1.5, 1.5)` and
  `(0.7, 0.15, 0.15)` express the same intent. Non-finite, zero and negative
  ratios still raise.
- `Optimizer` (ABC) replaced by `Strategy`, returning an internal `Outcome`.
- Simulated annealing's default `initial_temp` is now `100.0` (was `10.0`),
  matching the value the grid search selected.
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
  installed.

### Removed

- `evaluate_assignment()`. It was dead code, and it scored assignments with a
  *different* objective than the optimizers minimised (it omitted the
  unstratifiable-class mask), so it would have silently disagreed with
  `Optimizer.evaluate` had anything called it.
- Pickle loading. `.npz` is the only supported dataset format.

### Fixed

- `splitkit.strategies` no longer requires scikit-learn to import; the dependency
  is resolved lazily when the SGKF baseline actually runs, and the error names
  the extra to install.

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
