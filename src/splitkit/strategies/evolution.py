"""Differential Evolution optimizer with Latent Continuous Encoding."""

import time

import numpy as np

from .base import N_SPLITS, Optimizer, SplitResult


class DifferentialEvolution(Optimizer):
    def __init__(self, data, ratios=(0.70, 0.15, 0.15), max_evals=300_000,
                 pop_size=50, f_weight=0.5, crossover_prob=0.7, strategy="DE/rand/1/bin", seed=None):
        super().__init__(data, ratios, max_evals=max_evals, seed=seed)

        self.pop_size = pop_size
        self.f_weight = f_weight
        self.crossover_prob = crossover_prob
        self.strategy = strategy

        _, self._base, _diffs, self._cross = strategy.split("/")
        self._n_diffs = int(_diffs)

        min_pop = (2 * self._n_diffs) + (1 if self._base == "rand" else 0) + 1
        if pop_size < min_pop:
            raise ValueError(f"Population size must be at least {min_pop}.")

    def optimize(self, verbose: bool = False, log_interval: int = 10_000) -> SplitResult:
        rng = np.random.default_rng(self.seed)
        t_start = time.perf_counter()

        assignments = rng.choice(
            N_SPLITS,
            size=(self.pop_size, self.data.n_groups),
            p=self.ratios,
        )
        pop_idx = np.arange(self.pop_size)[:, np.newaxis]
        grp_idx = np.arange(self.data.n_groups)[np.newaxis, :]
        population = rng.uniform(-1.0, 0.0, size=(self.pop_size, self.data.n_groups, N_SPLITS))
        population[pop_idx, grp_idx, assignments] = rng.uniform(
            0.0, 1.0, size=(self.pop_size, self.data.n_groups)
        )
        costs = np.array([self.evaluate(ind) for ind in assignments])

        best_idx = np.argmin(costs)
        best_cost = costs[best_idx]
        best_latent = population[best_idx].copy()
        best_assignment = assignments[best_idx].copy()

        n_evals = self.pop_size
        iteration = 0
        cost_history = [(n_evals, best_cost)]

        while n_evals < self.max_evals and best_cost > 0.0:
            iteration += 1

            for i in range(self.pop_size):
                if n_evals >= self.max_evals:
                    break

                # Continuous Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                n_needed = (2 * self._n_diffs) + (1 if self._base == "rand" else 0)
                selected = rng.choice(idxs, n_needed, replace=False)

                if self._base == "rand":
                    base_vec = population[selected[0]]
                    diff_idxs = selected[1:]
                else:
                    base_vec = best_latent
                    diff_idxs = selected

                mutant = base_vec.copy()
                for d in range(self._n_diffs):
                    idx1, idx2 = diff_idxs[d * 2], diff_idxs[d * 2 + 1]
                    mutant += self.f_weight * (population[idx1] - population[idx2])

                # Continuous Crossover
                if self._cross == "bin":
                    cross_points = rng.random(self.data.n_groups) < self.crossover_prob
                    cross_points[rng.integers(self.data.n_groups)] = True
                    cross_points = cross_points[:, np.newaxis]
                    trial = np.where(cross_points, mutant, population[i])
                else:
                    trial = population[i].copy()
                    start_idx = rng.integers(self.data.n_groups)
                    curr_idx = start_idx
                    while True:
                        trial[curr_idx] = mutant[curr_idx]
                        curr_idx = (curr_idx + 1) % self.data.n_groups
                        if rng.random() >= self.crossover_prob or curr_idx == start_idx:
                            break

                # prevent exploding values
                trial = np.clip(trial, -1.0, 1.0)

                # Evaluation & Selection
                # Discretize the continuous trial vector to get the actual assignment
                trial_assignment = np.argmax(trial, axis=1)

                # if the discrete assignment didn't change, do not waste FFE
                if np.array_equal(trial_assignment, assignments[i]):
                    trial_cost = costs[i]
                else:
                    trial_cost = self.evaluate(trial_assignment)
                    n_evals += 1

                if trial_cost <= costs[i]:
                    population[i] = trial
                    costs[i] = trial_cost
                    assignments[i] = trial_assignment

                    if trial_cost < best_cost:
                        best_cost = trial_cost
                        best_latent = trial.copy()
                        best_assignment = trial_assignment.copy()

            if n_evals % log_interval < self.pop_size:
                cost_history.append((n_evals, best_cost))

        elapsed = time.perf_counter() - t_start
        best_actual = self._count_matrix(best_assignment)

        return SplitResult(
            assignment=best_assignment,
            cost=best_cost,
            n_evals=n_evals,
            n_iterations=iteration,
            converged=(best_cost == 0.0),
            elapsed_time=elapsed,
            cost_history=cost_history,
            target_counts=self._target.copy(),
            actual_counts=best_actual,
        )
