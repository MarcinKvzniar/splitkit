"""Plot helpers render without error and draw what they claim to."""

from __future__ import annotations

import pytest

from splitkit import split

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

from splitkit.viz import plot_convergence, plot_distribution  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def result(dense):
    return split(dense, max_evals=2000, seed=0)


class TestDistribution:
    def test_one_stacked_bar_per_class_plus_all_items(self, result, dense):
        ax = plot_distribution(result)
        assert len(ax.patches) == result.n_splits * (dense.n_classes + 1)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["all items", *dense.class_names]

    def test_bars_stack_to_one(self, result, dense):
        ax = plot_distribution(result)
        tops = [p.get_y() + p.get_height() for p in ax.patches[-(dense.n_classes + 1):]]
        assert tops == pytest.approx([1.0] * (dense.n_classes + 1))

    def test_target_boundaries_are_drawn(self, result):
        ax = plot_distribution(result)
        assert len(ax.get_lines()) == result.n_splits - 1

    def test_draws_on_given_axes(self, result):
        _, ax = plt.subplots()
        assert plot_distribution(result, ax=ax) is ax

    def test_size_column_is_not_plotted_as_a_class(self, soft_counts):
        r = split(soft_counts, (0.5, 0.5), max_evals=500, seed=0)
        assert len(plot_distribution(r).patches) == 2 * (soft_counts.n_classes + 1)


class TestConvergence:
    def test_history_is_a_step_curve_to_the_last_evaluation(self, result):
        line = plot_convergence(result).get_lines()[0]
        assert line.get_xdata()[-1] == result.n_evals

    def test_marks_the_lower_bound(self, tiny):
        pytest.importorskip("scipy")
        r = split(tiny, (0.5, 0.5), strategy="exact")
        labels = [line.get_label() for line in plot_convergence(r).get_lines()]
        assert "lower bound" in labels

    def test_draws_on_given_axes(self, result):
        _, ax = plt.subplots()
        assert plot_convergence(result, ax=ax) is ax
