"""Exact MILP strategy, checked against the brute-force oracle."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from _brute import brute_force
from _helpers import make_dataset
from splitkit import split
from splitkit.problem import Budget, SplitProblem
from splitkit.strategies import Outcome, get_strategy
from splitkit.strategies.exact import _expand

pytest.importorskip("scipy")
pytestmark = pytest.mark.exact


def solve(problem: SplitProblem, **budget: float) -> Outcome:
    return get_strategy("exact").run(problem, Budget(**budget), np.random.default_rng(0))


class TestOptimality:
    @pytest.mark.parametrize("ratios", [(0.5, 0.5), (0.6, 0.4), (0.5, 0.25, 0.25)])
    def test_matches_brute_force(self, tiny, ratios):
        problem = SplitProblem.build(tiny, ratios)
        _, optimal = brute_force(problem)
        out = solve(problem)
        assert out.proved_optimal
        assert out.cost == pytest.approx(optimal, rel=1e-9)
        assert out.lower_bound == pytest.approx(optimal, rel=1e-6)

    @pytest.mark.parametrize("fixture", ["multilabel", "soft_counts", "unstratifiable"])
    def test_matches_brute_force_on_awkward_data(self, request, fixture):
        problem = SplitProblem.build(request.getfixturevalue(fixture), (0.5, 0.25, 0.25))
        _, optimal = brute_force(problem)
        assert solve(problem).cost == pytest.approx(optimal, rel=1e-9)

    def test_large_counts_stay_well_conditioned(self, tiny):
        """Pixel-scale counts once pushed coefficients below solver tolerance."""
        huge = make_dataset(tiny.group_vectors * 1e6, sizes=tiny.group_sizes)
        problem = SplitProblem.build(huge, (0.5, 0.25, 0.25))
        _, optimal = brute_force(problem)
        assert solve(problem).cost == pytest.approx(optimal, rel=1e-9)

    def test_identical_groups_are_merged_without_losing_optimality(self):
        data = make_dataset([[3, 1], [3, 1], [3, 1], [1, 2], [1, 2], [0, 4], [2, 2], [2, 2]])
        problem = SplitProblem.build(data, (0.5, 0.25, 0.25))
        _, optimal = brute_force(problem)
        out = solve(problem)
        assert out.proved_optimal
        assert out.cost == pytest.approx(optimal, rel=1e-9)
        assert np.bincount(out.assignment, minlength=3).min() >= 1

    def test_many_duplicates_are_merged_without_losing_optimality(self, monkeypatch):
        data = make_dataset([[3, 1]] * 6 + [[1, 2]] * 4 + [[0, 4], [2, 2]])
        problem = SplitProblem.build(data, (0.5, 0.5))
        _, optimal = brute_force(problem)
        seen = []
        import scipy.optimize

        milp = scipy.optimize.milp
        monkeypatch.setattr("scipy.optimize.milp", lambda **kw: seen.append(kw) or milp(**kw))
        out = solve(problem)
        assert len(seen[0]["c"]) == 4 * 2 + 2 * 3  # 4 types x 2 splits, plus error terms
        assert out.proved_optimal
        assert out.cost == pytest.approx(optimal, rel=1e-9)

    def test_expand_hands_out_each_types_groups(self):
        type_of = np.array([0, 1, 0, 0, 1])
        assignment = _expand(np.array([[2, 1], [0, 2]]), type_of)
        np.testing.assert_array_equal(np.bincount(assignment[type_of == 0], minlength=2), [2, 1])
        np.testing.assert_array_equal(assignment[type_of == 1], [1, 1])

    def test_summary_reports_the_proof(self, tiny):
        assert "PROVEN OPTIMAL" in split(tiny, (0.5, 0.5), strategy="exact").summary()


class TestLimits:
    def test_timeout_still_reports_a_valid_bound(self):
        data = make_dataset(np.random.default_rng(0).integers(1, 50, (300, 10)))
        problem = SplitProblem.build(data, (0.7, 0.15, 0.15))
        out = solve(problem, time_limit=0.5)
        assert out.cost == pytest.approx(problem.evaluate(out.assignment), rel=1e-9)
        assert out.lower_bound is not None
        assert out.lower_bound <= out.cost

    def test_falls_back_to_annealing_without_a_solution(self, tiny_problem, monkeypatch):
        no_solution = SimpleNamespace(x=None, status=1, mip_dual_bound=float("nan"))
        monkeypatch.setattr("scipy.optimize.milp", lambda **_: no_solution)
        with pytest.warns(RuntimeWarning, match="no solution found"):
            out = solve(tiny_problem, max_evals=500)
        assert out.lower_bound is None
        assert not out.proved_optimal
        assert out.cost == pytest.approx(tiny_problem.evaluate(out.assignment), rel=1e-9)

    def test_too_many_types_skips_the_solver(self, tiny_problem, monkeypatch):
        def never_called(**_):
            raise AssertionError("the MILP should not be built")

        monkeypatch.setattr("scipy.optimize.milp", never_called)
        engine = get_strategy("exact", max_types=2)
        with pytest.warns(RuntimeWarning, match="max_types=2"):
            out = engine.run(tiny_problem, Budget(max_evals=500), np.random.default_rng(0))
        assert out.cost == pytest.approx(tiny_problem.evaluate(out.assignment), rel=1e-9)
        assert out.lower_bound is None

    def test_rejects_nonlinear_objectives(self, tiny):
        class Squared:
            name, separable, linearizable = "squared", True, False

            def prepare(self, target, weights):
                return SimpleNamespace(
                    separable=True,
                    total=lambda c: float((weights * (c - target) ** 2).sum()),
                    rows=lambda c: (weights * (c - target) ** 2).sum(axis=1),
                    row=lambda c, s: float((weights * (c - target[s]) ** 2).sum()),
                )

        problem = SplitProblem.build(tiny, (0.5, 0.5), objective=Squared())
        with pytest.raises(TypeError, match="linear objective"):
            solve(problem)

    @pytest.mark.parametrize(
        ("params", "match"),
        [
            ({"default_time_limit": 0}, "default_time_limit"),
            ({"mip_rel_gap": -1}, "mip_rel_gap"),
            ({"max_types": 0}, "max_types"),
        ],
    )
    def test_validates_parameters(self, params, match):
        with pytest.raises(ValueError, match=match):
            get_strategy("exact", **params)
