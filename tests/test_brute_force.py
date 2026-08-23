"""Ground truth from exhaustive enumeration.

On instances small enough to enumerate every assignment, the optimum is not a
matter of opinion. These tests use that to check the objective actually ranks
splits the way the problem statement claims, and to give later work (the exact
solver, greedy) something incontestable to be measured against.
"""

from __future__ import annotations

import numpy as np
import pytest

from _brute import MAX_STATES, all_costs, brute_force
from _helpers import make_dataset
from splitkit.problem import SplitProblem


class TestOracle:
    def test_finds_the_true_minimum(self):
        data = make_dataset([[6, 2], [2, 6], [4, 4], [5, 3], [3, 5]])
        problem = SplitProblem.build(data, (0.6, 0.4))

        assignment, cost = brute_force(problem)
        costs = all_costs(problem)

        assert cost == costs.min()
        assert problem.evaluate(assignment) == pytest.approx(cost)

    def test_refuses_intractable_instances(self):
        """The oracle must fail loudly rather than hang."""
        data = make_dataset(np.ones((40, 2)))
        problem = SplitProblem.build(data, (0.5, 0.5))
        with pytest.raises(ValueError, match="Refusing to enumerate"):
            brute_force(problem)

    def test_state_count_is_k_to_the_g(self):
        data = make_dataset([[1, 1]] * 5)
        problem = SplitProblem.build(data, (1 / 3, 1 / 3, 1 / 3))
        assert len(all_costs(problem)) == 3**5
        assert MAX_STATES > 3**5


class TestObjectiveRanksSplitsCorrectly:
    def test_perfect_split_is_the_unique_optimum(self):
        """Four identical groups at 50/50: any 2-2 split is exactly on target."""
        data = make_dataset([[4, 4]] * 4)
        problem = SplitProblem.build(data, (0.5, 0.5), class_weights="uniform")

        _, cost = brute_force(problem)
        assert cost == 0.0

        # every balanced arrangement ties at zero, unbalanced ones do not
        assert problem.evaluate(np.array([0, 0, 1, 1])) == 0.0
        assert problem.evaluate(np.array([0, 1, 0, 1])) == 0.0
        assert problem.evaluate(np.array([0, 0, 0, 1])) > 0.0

    def test_optimum_respects_group_indivisibility(self):
        """One group holds all of a class, so that class cannot be spread.

        The optimum is therefore strictly positive: no assignment can put half of
        an indivisible group in each split.
        """
        data = make_dataset([[10, 0], [0, 2], [0, 2], [0, 2], [0, 2]])
        problem = SplitProblem.build(data, (0.5, 0.5), unstratifiable="keep")
        _, cost = brute_force(problem)
        assert cost > 0.0

    def test_optimum_places_the_heavy_group_in_the_large_split(self):
        data = make_dataset([[20, 20], [2, 2], [2, 2], [2, 2]])
        problem = SplitProblem.build(data, (0.8, 0.2), class_weights="uniform")
        assignment, _ = brute_force(problem)
        assert assignment[0] == 0  # the 20/20 group belongs in the 80% split

    def test_rare_class_weighting_changes_the_optimum(self):
        """Inverse-frequency weighting should prioritise balancing the rare class."""
        data = make_dataset([[50, 1], [50, 1], [50, 0], [50, 0]])
        weighted = SplitProblem.build(data, (0.5, 0.5))
        uniform = SplitProblem.build(data, (0.5, 0.5), class_weights="uniform")

        assignment, _ = brute_force(weighted)
        counts = weighted.count_matrix(assignment)
        # the two rare-class items end up one per split
        assert counts[0][1] == pytest.approx(1.0)
        assert counts[1][1] == pytest.approx(1.0)
        assert weighted.weights[1] > uniform.weights[1]


class TestStrategiesAgainstOracle:
    @pytest.mark.parametrize("name", ["annealing", "evolution", "random"])
    def test_never_beats_the_optimum(self, name):
        """A strategy reporting a cost below the true optimum is reporting a lie."""
        from splitkit.problem import Budget
        from splitkit.strategies import get_strategy

        data = make_dataset([[7, 3], [3, 7], [5, 5], [6, 4], [4, 6], [8, 2]])
        problem = SplitProblem.build(data, (0.5, 0.5))
        _, optimal = brute_force(problem)

        out = get_strategy(name).run(
            problem, Budget(max_evals=3000), np.random.default_rng(0)
        )
        assert out.cost >= optimal - 1e-9
