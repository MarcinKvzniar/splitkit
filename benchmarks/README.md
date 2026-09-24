# Benchmarks

Scripts that compare splitkit's strategies on three real datasets and six synthetic
ones. They live outside the package and are not shipped in the wheel.

## Running

```bash
uv sync --all-extras
uv run python -m benchmarks.run_benchmark              # every dataset
uv run python -m benchmarks.run_benchmark isic bcss    # a subset
```

Each stochastic strategy runs with 10 seeds (42–51) under a 300,000-evaluation
budget and a 70/15/15 target. SGKF and the exact solver are deterministic and run
once; the exact solver gets 20 seconds. Reports and convergence plots go to
`results/`.

## Results

Mean cost ± standard deviation (lower is better). Exact and SGKF run once.

| dataset | annealing | exact (20 s) | evolution | SGKF | random |
|---|---|---|---|---|---|
| CelebA | 0.231 ± 0.018 | 0.235\* | **0.154 ± 0.011** | 1.278 | 2.098 ± 0.107 |
| ISIC 2020 | 0.589 ± 0.001 | **0.583** | **0.583 ± 0.000** | 11.286 | 0.659 ± 0.019 |
| BCSS | 7.49 ± 0.17 | **7.20** | 7.48 ± 0.04 | 44.27 | 10.74 ± 0.32 |
| synth_large_complex | 0.420 ± 0.027 | **0.410** | 0.575 ± 0.044 | 10.34 | 12.42 ± 0.39 |
| synth_easy_balanced | 0.040 ± 0.006 | **0.008** | 0.011 ± 0.002 | 0.120 | 0.097 ± 0.011 |
| synth_mild_imbalance | **0.128 ± 0.018** | 0.143 | 0.146 ± 0.019 | 0.692 | 1.785 ± 0.140 |
| synth_heavy_imbalance | 2.34 ± 0.11 | **1.91** | 2.52 ± 0.07 | 28.24 | 11.67 ± 0.58 |
| synth_few_groups | 1.61 ± 0.19 | **1.01** | 2.27 ± 0.19 | 17.61 | 9.79 ± 0.69 |
| synth_concentrated | 7.76 ± 0.09 | **7.64** (proven optimal, 0.2 s) | 7.74 ± 0.07 | 29.97 | 15.09 ± 0.55 |
| mean time | 1.3 s | 0.2–37 s | 12–204 s | 0.05–4.7 s | 7–159 s |

\* The solver found no solution within 20 s on CelebA and fell back to annealing.

Costs are comparable within a row, not across datasets.

## Datasets

The grouped count matrices are committed under `datasets/*/preprocessed/` as
`.npz` files (load them with `splitkit.load_npz`), so the raw data is not needed
to run the benchmark.

| dataset | groups | classes | items | grouping |
|---|---|---|---|---|
| CelebA | 10,177 | 40 | 202,599 | face images per identity; 40 binary attributes |
| ISIC 2020 | 2,056 | 9 | 33,126 | dermoscopy images per patient; 1.8% melanoma |
| BCSS | 151 | 21 | 8,768 | 512×512 tiles per whole-slide image; pixel counts per tissue class |
| synth_large_complex | 763 | 25 | 100,002 | large search space, many classes |
| synth_easy_balanced | 500 | 5 | 10,005 | balanced classes |
| synth_mild_imbalance | 297 | 10 | 15,000 | mild power-law imbalance |
| synth_heavy_imbalance | 158 | 20 | 50,001 | majority class holds ~75% of items |
| synth_few_groups | 96 | 15 | 8,001 | few, large groups |
| synth_concentrated | 56 | 12 | 20,000 | each class in only a few groups |

The synthetic sets come from `splitkit.from_preset(name)`.

### Origin

The three real datasets are publicly available. The committed files contain only
per-group class counts derived from them, with no images or metadata. Download the
originals from their sources, which also give their terms of use:

- **BCSS:** [github.com/PathologyDataScience/BCSS](https://github.com/PathologyDataScience/BCSS)
- **CelebA:** [mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **ISIC 2020:** [challenge2020.isic-archive.com](https://challenge2020.isic-archive.com/)

To rebuild the real fixtures from the original downloads:

```bash
uv run python -m benchmarks.etl.bcss   --data-dir datasets/bcss
uv run python -m benchmarks.etl.celeba --data-dir datasets/celeb-faces
uv run python -m benchmarks.etl.isic   --data-dir datasets/isic2020
```

### Citations

- **BCSS:** Amgad M, Elfandy H, Hussein H, Atteya LA, Elsebaie MAT, Gutman DA,
  Cooper LAD. Structured crowdsourcing enables convolutional segmentation of
  histology images. *Bioinformatics* 35(18):3461–3467, 2019.
- **CelebA:** Liu Z, Luo P, Wang X, Tang X. Deep learning face attributes in the
  wild. *ICCV*, 2015.
- **ISIC 2020:** Rotemberg V, Kurtansky N, Betz-Stablein B, et al. A patient-centric
  dataset of images and metadata for identifying melanomas using clinical context.
  *Scientific Data* 8(1):34, 2021.
