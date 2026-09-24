"""Differential evolution over a latent continuous encoding of the assignment."""

from __future__ import annotations

import time

import numpy as np

from ..problem import Budget, SplitProblem
from .base import Outcome, Strategy, resolve_max_evals
from .registry import register_strategy


@register_strategy
class DifferentialEvolution(Strategy):
    """Differential evolution on a ``(G, K)`` latent whose row-wise argmax is the assignment.

    Memory is ``O(pop_size * n_groups * n_splits)``; prefer annealing at scale.

    Parameters
    ----------
    variant
        ``DE/{rand,best}/{1,2}/{bin,exp}``.
    pop_size
        Population size; must exceed the number of individuals each step samples.
    f_weight
        Differential weight applied to each difference vector.
    crossover_prob
        Per-group crossover probability.
    """

    name = "evolution"

    def __init__(
        self,
        pop_size: int = 50,
        f_weight: float = 0.9,
        crossover_prob: float = 0.5,
        variant: str = "DE/best/2/exp",
        history_interval: int = 10_000,
    ) -> None:
        parts = variant.split("/")
        if len(parts) != 4 or parts[0] != "DE":
            raise ValueError(f"variant must look like 'DE/rand/1/bin', got {variant!r}.")
        _, base, n_diffs, cross = parts
        if base not in ("rand", "best"):
            raise ValueError(f"variant base must be 'rand' or 'best', got {base!r}.")
        if cross not in ("bin", "exp"):
            raise ValueError(f"variant crossover must be 'bin' or 'exp', got {cross!r}.")

        self.variant = variant
        self.pop_size = pop_size
        self.f_weight = f_weight
        self.crossover_prob = crossover_prob
        self.history_interval = history_interval

        self._base = base
        self._cross = cross
        self._n_diffs = int(n_diffs)

        min_pop = (2 * self._n_diffs) + (1 if base == "rand" else 0) + 1
        if pop_size < min_pop:
            raise ValueError(
                f"pop_size must be at least {min_pop} for {variant}, got {pop_size}."
            )

    def run(
        self,
        problem: SplitProblem,
        budget: Budget,
        rng: np.random.Generator,
        warm_start: np.ndarray | None = None,
    ) -> Outcome:
        t_start = time.perf_counter()
        deadline = budget.deadline_from(t_start)
        max_evals = resolve_max_evals(budget)

        k = problem.n_splits
        n_groups = problem.n_groups
        pop_size = self.pop_size

        # Each individual's argmax reproduces a ratio-sampled assignment.
        assignments = rng.choice(k, size=(pop_size, n_groups), p=problem.ratios)
        pop_idx = np.arange(pop_size)[:, np.newaxis]
        grp_idx = np.arange(n_groups)[np.newaxis, :]
        population = rng.uniform(-1.0, 0.0, size=(pop_size, n_groups, k))
        population[pop_idx, grp_idx, assignments] = rng.uniform(
            0.0, 1.0, size=(pop_size, n_groups)
        )

        costs = np.array([problem.evaluate(ind) for ind in assignments])
        best_idx = int(np.argmin(costs))
        best_cost = float(costs[best_idx])
        best_latent = population[best_idx].copy()
        best_assignment = assignments[best_idx].copy()

        n_evals = pop_size
        iteration = 0
        cost_history: list[tuple[int, float]] = [(n_evals, best_cost)]
        stop = False
        converged_population = False

        while n_evals < max_evals and best_cost > budget.target_cost and not stop:
            iteration += 1
            evals_at_generation_start = n_evals

            for i in range(pop_size):
                if n_evals >= max_evals:
                    break

                idxs = [idx for idx in range(pop_size) if idx != i]
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
                    i1, i2 = diff_idxs[d * 2], diff_idxs[d * 2 + 1]
                    mutant += self.f_weight * (population[i1] - population[i2])

                if self._cross == "bin":
                    cross_points = rng.random(n_groups) < self.crossover_prob
                    cross_points[rng.integers(n_groups)] = True
                    trial = np.where(cross_points[:, np.newaxis], mutant, population[i])
                else:
                    trial = population[i].copy()
                    start_idx = rng.integers(n_groups)
                    curr_idx = start_idx
                    while True:
                        trial[curr_idx] = mutant[curr_idx]
                        curr_idx = (curr_idx + 1) % n_groups
                        if rng.random() >= self.crossover_prob or curr_idx == start_idx:
                            break

                np.clip(trial, -1.0, 1.0, out=trial)
                trial_assignment = np.argmax(trial, axis=1)

                # An unchanged argmax carries no new information; skip the eval.
                if np.array_equal(trial_assignment, assignments[i]):
                    trial_cost = costs[i]
                else:
                    trial_cost = problem.evaluate(trial_assignment)
                    n_evals += 1

                if trial_cost <= costs[i]:
                    population[i] = trial
                    costs[i] = trial_cost
                    assignments[i] = trial_assignment

                    if trial_cost < best_cost:
                        best_cost = trial_cost
                        best_latent = trial.copy()
                        best_assignment = trial_assignment.copy()

            budget.report(n_evals, best_cost)
            if n_evals % self.history_interval < pop_size:
                cost_history.append((n_evals, best_cost))

            # No trial changed its assignment: the population has collapsed and no
            # evaluation can ever be spent again, so stop instead of looping forever.
            if n_evals == evals_at_generation_start:
                converged_population = True
                break

            if time.perf_counter() >= deadline:
                stop = True

        return Outcome(
            assignment=best_assignment,
            cost=best_cost,
            n_evals=n_evals,
            n_iterations=iteration,
            converged=best_cost <= budget.target_cost or converged_population,
            cost_history=cost_history,
        )
