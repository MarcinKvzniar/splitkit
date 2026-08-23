# Metaheuristic Optimization of Group-Aware Stratified Splitting

> **Course:** Optimization Methods: Theory and Applications - Final Project

> **Authors:** Marcin Kuźniar
---

## Problem Description

Many real-world datasets are naturally **hierarchical**: multiple images may belong to one patient, or many patches to a single whole-slide image (WSI). Splitting such data into Train / Validation / Test sets creates two conflicting constraints:

- **Group-level isolation** -- all items from the same group must land in the same split (no data leakage).
- **Class balance** -- each split should mirror the global class distribution, which is hard when groups vary in size and rare classes may be concentrated in just a few groups.

Standard scikit-learn splitters handle one constraint or the other, never both simultaneously. This project formulates the problem as a variant of the **multidimensional Multiple Knapsack Problem** and evaluates metaheuristic optimizers against deterministic baselines.

### Cost function

Weighted MAPE summed over all *(split x class)* pairs:

$$
\mathcal{L} = \sum_{s,\,c} w_c \cdot \frac{|\text{actual}_{s,c} - \text{target}_{s,c}|}{\text{target}_{s,c} + \varepsilon}
$$

where $w_c \propto 1/f_c$ gives higher penalty to rare classes.

---

## Implemented Optimizers

The project implements and evaluates four split strategies under a strict, shared budget of **300,000 Fitness Function Evaluations (FFEs)** using a target split ratio of **70% Train / 15% Validation / 15% Test**:

1. **Simulated Annealing (SA)**: A single-trajectory metaheuristic configured with a geometric cooling schedule (`initial_temp=100.0`, `cooling_rate=0.9999`, `min_temp=1e-4`).
2. **Differential Evolution (DE)**: A population-based optimization algorithm tuned for localized exploitation using the `DE/best/2/exp` strategy (`pop_size=50`, `f_weight=0.9`, `crossover_prob=0.5`).
3. **Random Search (RS)**: A baseline random sampler used to evaluate search space complexity.
4. **Stratified Group K-Fold (SGKF)**: The standard deterministic baseline from Scikit-Learn.

---

## Datasets

*No download needed - for optimization all datasets are summarized in groups.pkl representation.*

### Real-World Benchmarks
* **BCSS - Breast Cancer Semantic Segmentation**: 151 whole-slide images tiled into 512 x 512 patches (8 768 tiles total), 21 tissue classes.
  * Citation: Mohamed Amgad, Habiba Elfandy, Hagar Hussein, Lamees A Atteya, Maha A T Elsebaie, David A Gutman, and Lee A D Cooper. Structured crowdsourcing enables convolutional segmentation of histology images. Bioinformatics, 35(18):3461–3467, 2019
* **CelebA - Large-Scale Face Attributes**: 202 599 celebrity face images grouped by identity (10 177 unique persons), 40 binary attributes.
  * Citation: Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), December 2015.
* **ISIC 2020 - Melanoma Classification**: 33 126 dermoscopy images from 2 056 patients, 9 diagnosis classes (1.76 % melanoma).
  * Citation: Veronica Rotemberg, Nicholas Kurtansky, Brigid Betz-Stablein, Liam Caffery, Emmanouil Chousakos, Noel Codella, Marc Combalia, Stephen Dusza, Pascale Guitera, David Gutman, Allan Halpern, Brian Helba, Harald Kittler, Kivanc Kose, Stefan Langer, Konstantinos Liopyris, Josep Malvehy, Shaikhah Musthaq, Jashan Nanda, Ofer Reiter, George Shih, Alexander Stratigos, Philipp Tschandl, James Weber, and H. Peter Soyer. A patient-centric dataset of images and metadata for identifying melanomas using clinical context. Scientific Data, 8(1):34, 2021.

### Synthetic Benchmarks
The repository also includes six pre-generated synthetic datasets (`synth_*`) designed to test specific edge-case topologies:
* `synth_concentrated`
* `synth_easy_balanced`
* `synth_few_groups`
* `synth_heavy_imbalance`
* `synth_large_complex`
* `synth_mild_imbalance`

---

## Usage & Benchmarking

The benchmarking suite runs each stochastic optimizer across **10 independent random seeds** to gather mean costs and standard deviations. It automatically generates per-dataset reports and saves convergence curve plots (showing the mean curve and a shaded $\pm\text{Std Dev}$ region) inside the `results/` folder.

To evaluate all real-world and synthetic datasets sequentially:
```bash
uv run python run_benchmark.py
