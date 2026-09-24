"""Contract every strategy must satisfy.

These tests are parametrized over the registry, so a newly registered strategy is
held to the same contract automatically rather than needing its own suite.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from _brute import brute_force
from _helpers import make_dataset
from splitkit.problem import Budget, SplitProblem
from splitkit.strategies import Strategy, get_strategy, list_strategies
from splitkit.strategies.registry import _REGISTRY

#: Strategies that need an optional dependency are skipped when it is absent.
_EXTRA_MODULES = {"sklearn": "sklearn", "exact": "scipy"}


def needs_skip(name: str) -> str | None:
    for extra in _REGISTRY[name].requires:
        module = _EXTRA_MODULES.get(extra, extra)
        if not pytest.importorskip.__module__:  # pragma: no cover
            return None
        try:
            __import__(module)
        except ImportError:
            return f"requires {extra}"
    return None


@pytest.fixture(params=list_strategies())
def strategy_name(request) -> str:
    skip = needs_skip(request.param)
    if skip:
        pytest.skip(skip)
    return request.param


def budget_for(name: str, evals: int = 2000) -> Budget:
    """SGKF is a one-shot heuristic; the rest consume an evaluation budget."""
    return Budget(max_evals=1 if name == "sgkf" else evals)


class TestContract:
    def test_returns_valid_assignment(self, strategy_name, tiny_problem):
        out = get_strategy(strategy_name).run(
            tiny_problem, budget_for(strategy_name), np.random.default_rng(0)
        )
        a = out.assignment
        assert a.shape == (tiny_problem.n_groups,)
        assert np.issubdtype(a.dtype, np.integer)
        assert a.min() >= 0
        assert a.max() < tiny_problem.n_splits

    def test_reported_cost_matches_assignment(self, strategy_name, tiny_problem):
        """The cost a strategy reports must be the cost of what it returns.

        Strategies maintain counts incrementally, so a drifted accumulator would
        otherwise be reported to the caller as fact.
        """
        out = get_strategy(strategy_name).run(
            tiny_problem, budget_for(strategy_name), np.random.default_rng(1)
        )
        assert out.cost == pytest.approx(
            tiny_problem.evaluate(out.assignment), rel=1e-9
        )

    def test_deterministic_for_a_seed(self, strategy_name, tiny_problem):
        runs = [
            get_strategy(strategy_name)
            .run(tiny_problem, budget_for(strategy_name), np.random.default_rng(7))
            .cost
            for _ in range(2)
        ]
        assert runs[0] == runs[1]

    def test_respects_max_evals(self, strategy_name, tiny_problem):
        budget = budget_for(strategy_name, evals=500)
        out = get_strategy(strategy_name).run(
            tiny_problem, budget, np.random.default_rng(2)
        )
        assert out.n_evals <= budget.max_evals

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_handles_k_splits(self, strategy_name, dense, k):
        problem = SplitProblem.build(dense, [1 / k] * k)
        out = get_strategy(strategy_name).run(
            problem, budget_for(strategy_name), np.random.default_rng(3)
        )
        assert out.assignment.max() < k
        assert out.cost >= 0.0

    def test_cost_history_is_monotone(self, strategy_name, tiny_problem):
        out = get_strategy(strategy_name).run(
            tiny_problem, budget_for(strategy_name), np.random.default_rng(4)
        )
        costs = [c for _, c in out.cost_history]
        assert costs == sorted(costs, reverse=True)
        evals = [e for e, _ in out.cost_history]
        assert evals == sorted(evals)

    def test_final_cost_matches_history(self, strategy_name, tiny_problem):
        out = get_strategy(strategy_name).run(
            tiny_problem, budget_for(strategy_name), np.random.default_rng(5)
        )
        if out.cost_history:
            assert out.cost == pytest.approx(out.cost_history[-1][1], rel=1e-9)

    def test_does_not_mutate_problem(self, strategy_name, tiny_problem):
        before = tiny_problem.vectors.copy(), tiny_problem.target.copy()
        get_strategy(strategy_name).run(
            tiny_problem, budget_for(strategy_name), np.random.default_rng(6)
        )
        np.testing.assert_array_equal(tiny_problem.vectors, before[0])
        np.testing.assert_array_equal(tiny_problem.target, before[1])

    def test_registered_name_matches_class(self, strategy_name):
        assert _REGISTRY[strategy_name].name == strategy_name
        assert issubclass(_REGISTRY[strategy_name], Strategy)


class TestSearchQuality:
    """Directed strategies must actually beat undirected sampling."""

    @pytest.fixture
    def searchable(self) -> SplitProblem:
        """Large enough that sampling cannot cover it, with budget to search it.

        The claim "directed beats random" is only well posed in that regime: on a
        10-group instance the whole space fits in a few thousand draws, and with
        only a handful of moves per group a local search has not organised
        anything yet.
        """
        rng = np.random.default_rng(1)
        data = make_dataset(rng.integers(1, 30, size=(100, 6)).astype(float))
        return SplitProblem.build(data, (0.6, 0.2, 0.2))

    @pytest.mark.parametrize("name", ["annealing", "evolution"])
    def test_beats_random_search(self, name, searchable):
        budget = Budget(max_evals=8000)
        directed = get_strategy(name).run(searchable, budget, np.random.default_rng(0))
        undirected = get_strategy("random").run(
            searchable, budget, np.random.default_rng(0)
        )
        assert directed.cost < undirected.cost

    def test_annealing_margin_is_substantial(self, searchable):
        """Not merely better -- decisively so, or the search is not earning its cost."""
        budget = Budget(max_evals=8000)
        sa = get_strategy("annealing").run(searchable, budget, np.random.default_rng(0))
        rs = get_strategy("random").run(searchable, budget, np.random.default_rng(0))
        assert sa.cost < rs.cost / 1.5

    @pytest.mark.parametrize("name", ["annealing", "evolution", "random"])
    def test_more_budget_never_hurts(self, name, dense):
        """Best-so-far tracking means a longer run cannot return a worse result."""
        problem = SplitProblem.build(dense, (0.6, 0.2, 0.2))
        short = get_strategy(name).run(
            problem, Budget(max_evals=300), np.random.default_rng(0)
        )
        long = get_strategy(name).run(
            problem, Budget(max_evals=6000), np.random.default_rng(0)
        )
        assert long.cost <= short.cost

    @pytest.mark.parametrize("name", ["annealing", "evolution"])
    def test_finds_optimum_on_tiny_instance(self, name):
        """On an instance small enough to enumerate, a real search should find it."""
        data = make_dataset([[8, 2], [2, 8], [5, 5], [4, 6], [6, 4], [3, 7]])
        problem = SplitProblem.build(data, (0.5, 0.5))
        _, optimal = brute_force(problem)

        out = get_strategy(name).run(
            problem, Budget(max_evals=5000), np.random.default_rng(0)
        )
        assert out.cost == pytest.approx(optimal, rel=1e-9)


class TestTimeLimit:
    @pytest.mark.parametrize("name", ["annealing", "random"])
    def test_stops_near_the_deadline(self, name):
        """Wall-clock limits are the stop condition a library user reaches for."""
        data = make_dataset(np.random.default_rng(0).integers(1, 50, (600, 12)))
        problem = SplitProblem.build(data, (0.7, 0.15, 0.15))

        limit = 0.5
        t0 = time.perf_counter()
        get_strategy(name).run(
            problem,
            Budget(max_evals=10**9, time_limit=limit),
            np.random.default_rng(0),
        )
        elapsed = time.perf_counter() - t0
        assert limit <= elapsed < limit * 2.0


class TestWarmStart:
    def test_declared_support_is_honoured(self, tiny_problem):
        """A warm start must actually change the trajectory."""
        name = "annealing"
        assert _REGISTRY[name].supports_warm_start

        warm = np.zeros(tiny_problem.n_groups, dtype=np.intp)
        cold_out = get_strategy(name).run(
            tiny_problem, Budget(max_evals=20), np.random.default_rng(0)
        )
        warm_out = get_strategy(name).run(
            tiny_problem, Budget(max_evals=20), np.random.default_rng(0),
            warm_start=warm,
        )
        assert cold_out.assignment.tolist() != warm_out.assignment.tolist()

    def test_rejects_wrong_shape(self, tiny_problem):
        with pytest.raises(ValueError, match="warm_start"):
            get_strategy("annealing").run(
                tiny_problem,
                Budget(max_evals=10),
                np.random.default_rng(0),
                warm_start=np.zeros(99, dtype=np.intp),
            )


class TestRegistry:
    def test_lists_expected_strategies(self):
        assert set(list_strategies()) >= {"annealing", "evolution", "random", "sgkf"}

    def test_instance_passes_through(self):
        from splitkit.strategies import RandomSearch

        s = RandomSearch()
        assert get_strategy(s) is s

    def test_unknown_name_lists_options(self):
        with pytest.raises(KeyError, match="Unknown strategy"):
            get_strategy("does_not_exist")

    def test_variant_is_configurable_through_split(self, tiny):
        from splitkit import split

        r = split(tiny, (0.5, 0.5), strategy="evolution", variant="DE/best/2/exp", max_evals=200)
        assert r.strategy_params["variant"] == "DE/best/2/exp"

    def test_params_are_reported(self):
        s = get_strategy("annealing", initial_temp=50.0)
        assert s.params()["initial_temp"] == 50.0


class TestStrategyValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"cooling_rate": 0.0}, "cooling_rate"),
            ({"cooling_rate": 1.0}, "cooling_rate"),
            ({"initial_temp": -1.0}, "initial_temp"),
            ({"min_temp": 0.0}, "min_temp"),
            ({"min_temp": 1e9}, "min_temp"),
        ],
    )
    def test_annealing_rejects_bad_params(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            get_strategy("annealing", **kwargs)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"variant": "nonsense"}, "DE/rand/1/bin"),
            ({"variant": "DE/other/1/bin"}, "base"),
            ({"variant": "DE/rand/1/zzz"}, "crossover"),
            ({"pop_size": 2}, "pop_size must be at least"),
        ],
    )
    def test_evolution_rejects_bad_params(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            get_strategy("evolution", **kwargs)
