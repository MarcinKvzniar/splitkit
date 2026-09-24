"""Paths that only trigger on larger inputs or non-default configuration.

Each of these is reachable in real use -- the tuned DE variant, the large-problem
count path, the SGKF downscale -- but not on the small fixtures the rest of the
suite uses, so they need to be reached deliberately.
"""

from __future__ import annotations

import numpy as np
import pytest

from _helpers import make_dataset
from splitkit.problem import Budget, SplitProblem, elapsed_since
from splitkit.strategies import Strategy, get_strategy
from splitkit.strategies.registry import register_strategy


class TestLargeProblemCountPath:
    """count_matrix switches to a BLAS matmul once K*G gets large."""

    @pytest.fixture
    def big_problem(self) -> SplitProblem:
        rng = np.random.default_rng(0)
        data = make_dataset(rng.integers(0, 30, size=(20_000, 5)).astype(float))
        return SplitProblem.build(data, (0.7, 0.15, 0.15))

    def test_large_path_is_taken(self, big_problem):
        assert big_problem.n_splits * big_problem.n_groups >= 50_000

    def test_matches_explicit_masking(self, big_problem):
        rng = np.random.default_rng(1)
        a = rng.integers(0, 3, size=big_problem.n_groups)
        counts = big_problem.count_matrix(a)

        for s in range(3):
            np.testing.assert_allclose(
                counts[s], big_problem.vectors[a == s].sum(axis=0), rtol=1e-9
            )

    def test_conserves_total_mass(self, big_problem):
        rng = np.random.default_rng(2)
        a = rng.integers(0, 3, size=big_problem.n_groups)
        np.testing.assert_allclose(
            big_problem.count_matrix(a).sum(axis=0),
            big_problem.vectors.sum(axis=0),
            rtol=1e-9,
        )

    def test_empty_split_handled(self, big_problem):
        a = np.zeros(big_problem.n_groups, dtype=int)
        counts = big_problem.count_matrix(a)
        np.testing.assert_allclose(counts[1], 0.0)
        np.testing.assert_allclose(counts[2], 0.0)


class TestEvolutionVariants:
    """All eight DE variants must work, including the tuned DE/best/2/exp."""

    @pytest.fixture
    def problem(self) -> SplitProblem:
        rng = np.random.default_rng(5)
        data = make_dataset(rng.integers(1, 25, size=(40, 5)).astype(float))
        return SplitProblem.build(data, (0.6, 0.2, 0.2))

    @pytest.mark.parametrize("base", ["rand", "best"])
    @pytest.mark.parametrize("n_diffs", [1, 2])
    @pytest.mark.parametrize("cross", ["bin", "exp"])
    def test_variant_runs_and_is_valid(self, problem, base, n_diffs, cross):
        variant = f"DE/{base}/{n_diffs}/{cross}"
        out = get_strategy("evolution", variant=variant, pop_size=12).run(
            problem, Budget(max_evals=800), np.random.default_rng(0)
        )
        assert out.assignment.max() < problem.n_splits
        assert out.cost == pytest.approx(problem.evaluate(out.assignment), rel=1e-9)

    def test_benchmark_variant_is_deterministic(self, problem):
        """DE/best/2/exp with the tuned hyper-parameters, as the benchmark runs it."""
        kw = dict(
            variant="DE/best/2/exp", pop_size=50, f_weight=0.9, crossover_prob=0.5
        )
        runs = [
            get_strategy("evolution", **kw)
            .run(problem, Budget(max_evals=2000), np.random.default_rng(42))
            .cost
            for _ in range(2)
        ]
        assert runs[0] == runs[1]

    def test_min_pop_depends_on_variant(self):
        """DE/rand/2 needs more distinct individuals per step than DE/best/1."""
        get_strategy("evolution", variant="DE/best/1/bin", pop_size=3)
        with pytest.raises(ValueError, match="at least"):
            get_strategy("evolution", variant="DE/rand/2/bin", pop_size=5)

    def test_population_collapse_terminates(self, problem):
        """A collapsed population must end the run, not spin forever.

        Evaluations are only spent when a trial's argmax differs from its parent.
        Once the population converges no trial differs, so `n_evals` stops rising
        and a `while n_evals < max_evals` loop never exits on its own.
        """
        out = get_strategy("evolution", variant="DE/best/1/bin", pop_size=12).run(
            problem, Budget(max_evals=10**7), np.random.default_rng(0)
        )
        assert out.n_evals < 10**7
        assert out.converged
        assert out.cost == pytest.approx(problem.evaluate(out.assignment), rel=1e-9)

    def test_stops_on_time_limit(self):
        data = make_dataset(np.random.default_rng(0).integers(1, 50, (600, 12)))
        out = get_strategy("evolution", pop_size=10).run(
            SplitProblem.build(data, (0.7, 0.15, 0.15)),
            Budget(max_evals=10**9, time_limit=0.3),
            np.random.default_rng(0),
        )
        assert out.n_evals < 10**9
        assert not out.converged


class TestSGKFDownscale:
    """Above the item cap, SGKF rescales counts to keep the item table tractable."""

    def test_downscales_without_error(self):
        pytest.importorskip("sklearn")
        rng = np.random.default_rng(0)
        # ~2M items, comfortably over the 300k materialisation cap
        data = make_dataset(rng.integers(3000, 5000, size=(60, 8)).astype(float))
        assert data.group_vectors.sum() > 300_000

        problem = SplitProblem.build(data, (0.7, 0.15, 0.15))
        out = get_strategy("sgkf").run(
            problem, Budget(max_evals=1), np.random.default_rng(0)
        )
        assert out.assignment.max() < 3
        assert out.cost == pytest.approx(problem.evaluate(out.assignment), rel=1e-9)


class TestLargestRemainder:
    def test_apportions_exactly(self):
        from splitkit.strategies.sgkf import largest_remainder

        for ratios in ([0.7, 0.15, 0.15], [0.5, 0.5], [0.34, 0.33, 0.33]):
            for total in (2, 3, 7, 20):
                got = largest_remainder(np.asarray(ratios), total)
                assert sum(got) == total
                assert all(v >= 0 for v in got)

    def test_never_starves_a_split(self):
        """The previous rounding loop could leave a split with zero folds."""
        from splitkit.strategies.sgkf import largest_remainder

        got = largest_remainder(np.asarray([0.8, 0.1, 0.1]), 10)
        assert sum(got) == 10
        assert all(v >= 1 for v in got)

    def test_matches_proportions(self):
        from splitkit.strategies.sgkf import largest_remainder

        got = largest_remainder(np.asarray([0.7, 0.15, 0.15]), 20)
        assert got == [14, 3, 3]


class TestRegistryValidation:
    def test_rejects_unnamed_strategy(self):
        class Unnamed(Strategy):
            def run(self, problem, budget, rng, warm_start=None):  # pragma: no cover
                raise NotImplementedError

        with pytest.raises(ValueError, match="non-empty"):
            register_strategy(Unnamed)


class TestRowCosts:
    def test_sums_to_total(self, tiny_problem, rng):
        for _ in range(20):
            a = tiny_problem.random_assignment(rng)
            counts = tiny_problem.count_matrix(a)
            assert tiny_problem.row_costs(counts).sum() == pytest.approx(
                tiny_problem.evaluate(a), rel=1e-12
            )

    def test_shape_is_per_split(self, tiny_problem, rng):
        counts = tiny_problem.count_matrix(tiny_problem.random_assignment(rng))
        assert tiny_problem.row_costs(counts).shape == (tiny_problem.n_splits,)


class TestElapsedSince:
    def test_returns_positive_duration(self):
        import time

        t0 = time.perf_counter()
        assert elapsed_since(t0) >= 0.0


class TestResolveMaxEvals:
    def test_explicit_evals_win(self):
        from splitkit.strategies.base import resolve_max_evals

        assert resolve_max_evals(Budget(max_evals=123)) == 123

    def test_time_only_is_effectively_unbounded(self):
        """A fast machine must not stop early with time left on the clock."""
        from splitkit.strategies.base import DEFAULT_MAX_EVALS, resolve_max_evals

        assert resolve_max_evals(Budget(time_limit=1.0)) > DEFAULT_MAX_EVALS

    def test_falls_back_to_the_default(self):
        from splitkit.strategies.base import DEFAULT_MAX_EVALS, resolve_max_evals

        assert resolve_max_evals(Budget()) == DEFAULT_MAX_EVALS


class TestAnnealingScheduleValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"cooling_rate": "fast"}, "or 'auto'"),
            ({"anneal_cycles": 0}, "anneal_cycles"),
            ({"anneal_cycles": -1}, "anneal_cycles"),
        ],
    )
    def test_rejects_bad_schedule(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            get_strategy("annealing", **kwargs)

    def test_auto_rate_shrinks_with_budget(self):
        """A smaller budget must cool faster, or it never leaves exploration."""
        sa = get_strategy("annealing")
        assert sa._resolve_cooling_rate(2_000) < sa._resolve_cooling_rate(300_000)

    def test_explicit_rate_is_used_verbatim(self):
        sa = get_strategy("annealing", cooling_rate=0.9999)
        assert sa._resolve_cooling_rate(2_000) == 0.9999
