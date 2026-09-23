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
        with pytest.warns(RuntimeWarning, match="falling back"):
            out = solve(tiny_problem, max_evals=500)
        assert out.lower_bound is None
        assert not out.proved_optimal
        assert out.cost == pytest.approx(tiny_problem.evaluate(out.assignment), rel=1e-9)

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
        [({"default_time_limit": 0}, "default_time_limit"), ({"mip_rel_gap": -1}, "mip_rel_gap")],
    )
    def test_validates_parameters(self, params, match):
        with pytest.raises(ValueError, match=match):
            get_strategy("exact", **params)
