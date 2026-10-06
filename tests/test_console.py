"""Progress reporting and the styled report."""

from __future__ import annotations

import io
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from rich.console import Console

from _helpers import make_dataset
from splitkit import _console, split
from splitkit.problem import Budget
from splitkit.strategies import get_strategy


def collect(problem, strategy: str, budget: Budget, rng) -> list[tuple[int, float]]:
    calls: list[tuple[int, float]] = []
    budget = replace(budget, on_progress=lambda n, c: calls.append((n, c)))
    get_strategy(strategy).run(problem, budget, rng)
    return calls


class TestBudgetHook:
    def test_report_without_callback_is_a_no_op(self):
        Budget(max_evals=10).report(5, 1.0)

    @pytest.mark.parametrize("strategy", ["annealing", "random", "evolution"])
    def test_iterative_strategies_report(self, tiny_problem, rng, strategy):
        calls = collect(tiny_problem, strategy, Budget(max_evals=10_000), rng)
        evals = [n for n, _ in calls]
        costs = [c for _, c in calls]
        assert calls
        assert evals == sorted(evals)
        assert costs == sorted(costs, reverse=True)  # best cost never worsens

    @pytest.mark.exact
    def test_exact_fallback_forwards_the_callback(self, tiny_problem, rng, monkeypatch):
        no_solution = SimpleNamespace(x=None, status=1, mip_dual_bound=float("nan"))
        monkeypatch.setattr("scipy.optimize.milp", lambda **_: no_solution)
        with pytest.warns(RuntimeWarning):
            calls = collect(tiny_problem, "exact", Budget(max_evals=10_000), rng)
        assert calls

    def test_progress_does_not_change_the_result(self, dense):
        quiet = split(dense, max_evals=10_000, seed=0)
        shown = split(dense, max_evals=10_000, seed=0, progress=True)
        np.testing.assert_array_equal(quiet.assignment, shown.assignment)


class TestFractionDone:
    @pytest.mark.parametrize(
        "budget, expected",
        [
            (Budget(max_evals=100), 0.5),
            (Budget(time_limit=4.0), 0.25),
            (Budget(), None),
            (Budget(max_evals=10), 1.0),
        ],
    )
    def test_prefers_evals_then_time(self, budget, expected):
        assert _console._fraction_done(budget, 50, 1.0) == expected


class TestReport:
    def render(self, result) -> str:
        console = Console(record=True, width=120, file=io.StringIO())
        console.print(result)
        return console.export_text()

    def test_progress_bar_runs(self, dense):
        with _console.progress_display("annealing", Budget(max_evals=100)) as report:
            report(50, 0.25)
        split(dense, max_evals=10_000, seed=0, progress=True)

    def test_report_contents(self, tiny):
        text = self.render(split(tiny, max_evals=500, seed=0))
        for token in ("train", "val", "test", "requested", "achieved", "annealing",
                      "Worst class balance", "6 groups"):
            assert token in text

    def test_report_marks_optimum_and_gap(self, tiny):
        r = split(tiny, max_evals=500, seed=0)
        assert "PROVEN OPTIMAL" in self.render(replace(r, proved_optimal=True))
        assert "gap" in self.render(replace(r, lower_bound=r.cost * 0.9))

    def test_report_lists_issues(self, unstratifiable):
        data = make_dataset([[10, 0], [10, 0], [10, 1], [10, 0], [10, 0], [10, 1]])
        kept = split(data, max_evals=500, seed=0, unstratifiable="keep")
        assert "contains no" in self.render(kept)
        dropped = split(unstratifiable, max_evals=500, seed=0)
        assert "unstratifiable" in self.render(dropped)

    def test_print_report_goes_to_stderr(self, capsys, tiny):
        _console.print_report(split(tiny, max_evals=500, seed=0))
        out, err = capsys.readouterr()
        assert out == ""
        assert "Worst class balance" in err
