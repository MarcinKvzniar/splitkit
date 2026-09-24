"""The public API: split(), evaluate(), and the SplitResult it returns."""

from __future__ import annotations

import numpy as np
import pytest

import splitkit
from _helpers import make_dataset
from splitkit import GroupedDataset, SplitResult, evaluate, split

pd = pytest.importorskip("pandas")


@pytest.fixture
def clinical_df():
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame(
        {
            "patient_id": rng.integers(0, 60, size=n),
            "diagnosis": rng.choice(["benign", "malignant"], size=n, p=[0.8, 0.2]),
        }
    )


@pytest.fixture
def result(clinical_df) -> SplitResult:
    return split(
        clinical_df,
        group_col="patient_id",
        label_col="diagnosis",
        max_evals=5000,
        seed=42,
    )


class TestInputForms:
    def test_from_dataframe(self, clinical_df):
        r = split(
            clinical_df, group_col="patient_id", label_col="diagnosis",
            max_evals=1000, seed=0,
        )
        assert r.n_splits == 3

    def test_from_arrays(self):
        rng = np.random.default_rng(0)
        groups = rng.integers(0, 30, size=200)
        y = rng.integers(0, 3, size=200)
        r = split(groups=groups, y=y, max_evals=1000, seed=0)
        assert r.dataset.n_groups == len(np.unique(groups))

    def test_from_grouped_dataset(self, tiny):
        r = split(tiny, (0.5, 0.25, 0.25), max_evals=500, seed=0)
        assert r.dataset is tiny

    def test_requires_both_groups_and_y(self):
        with pytest.raises(ValueError, match="both"):
            split(groups=np.array([1, 2]), max_evals=10)

    def test_unusable_input_explains_options(self):
        with pytest.raises(TypeError, match="Cannot build a dataset"):
            split(object(), max_evals=10)


class TestRatios:
    def test_mapping_sets_names(self, tiny):
        r = split(tiny, {"fit": 0.6, "holdout": 0.4}, max_evals=500, seed=0)
        assert r.names == ("fit", "holdout")

    def test_sequence_uses_conventional_names(self, tiny):
        assert split(tiny, (0.5, 0.5), max_evals=500, seed=0).names == (
            "train", "test",
        )
        assert split(tiny, (0.5, 0.25, 0.25), max_evals=500, seed=0).names == (
            "train", "val", "test",
        )

    def test_unnormalised_ratios_accepted(self, tiny):
        a = split(tiny, (7, 1.5, 1.5), max_evals=500, seed=0)
        b = split(tiny, (0.7, 0.15, 0.15), max_evals=500, seed=0)
        np.testing.assert_allclose(a.ratios, b.ratios)
        np.testing.assert_array_equal(a.assignment, b.assignment)

    def test_explicit_names_override(self, tiny):
        r = split(tiny, (0.5, 0.5), names=["a", "b"], max_evals=500, seed=0)
        assert r.names == ("a", "b")

    def test_names_given_twice_rejected(self, tiny):
        with pytest.raises(ValueError, match="not both"):
            split(tiny, {"a": 0.5, "b": 0.5}, names=["c", "d"], max_evals=500)

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_k_splits(self, dense, k):
        r = split(dense, [1 / k] * k, max_evals=1000, seed=0)
        assert r.n_splits == k
        assert set(np.unique(r.assignment)) <= set(range(k))


class TestGuarantees:
    def test_every_group_in_exactly_one_split(self, result):
        counted = sum(len(g) for g in result.groups.values())
        assert counted == result.dataset.n_groups
        seen = np.concatenate(list(result.groups.values()))
        assert len(np.unique(seen)) == result.dataset.n_groups

    def test_no_group_leaks_across_splits(self, result):
        """The core promise: related items never straddle a split boundary."""
        item_split = result.assignment[result.dataset.item_group_index]
        for g in range(result.dataset.n_groups):
            rows = item_split[result.dataset.item_group_index == g]
            assert len(np.unique(rows)) == 1

    def test_indices_partition_the_dataset(self, result):
        parts = result.indices.astuple()
        allix = np.sort(np.concatenate(parts))
        np.testing.assert_array_equal(allix, np.arange(result.dataset.n_items))

    def test_reported_cost_matches_assignment(self, result):
        recomputed = evaluate(
            result.dataset, result.assignment, result.ratios.tolist()
        )
        assert result.cost == pytest.approx(recomputed, rel=1e-9)

    def test_actual_counts_recomputed_not_carried(self, result):
        """Reported counts must describe the returned assignment, not a
        search's incrementally maintained accumulator."""
        for s in range(result.n_splits):
            expected = result.dataset.group_vectors[result.assignment == s].sum(axis=0)
            np.testing.assert_allclose(
                result.actual_counts[s][: result.dataset.n_classes], expected
            )

    def test_deterministic_for_a_seed(self, clinical_df):
        kw = dict(
            group_col="patient_id", label_col="diagnosis", max_evals=2000, seed=7
        )
        a = split(clinical_df, **kw)
        b = split(clinical_df, **kw)
        np.testing.assert_array_equal(a.assignment, b.assignment)
        assert a.cost == b.cost


class TestSplitMapping:
    def test_astuple_gives_arrays_not_keys(self, result):
        """A dict would unpack to its keys; this is why astuple exists."""
        train, val, test = result.indices.astuple()
        for part in (train, val, test):
            assert isinstance(part, np.ndarray)

    def test_attribute_access(self, result):
        np.testing.assert_array_equal(result.groups.train, result.groups["train"])

    def test_mapping_protocol(self, result):
        assert list(result.groups) == list(result.names)
        assert len(result.groups) == result.n_splits
        assert "train" in result.groups

    def test_unknown_split_lists_options(self, result):
        with pytest.raises(KeyError, match="No split named"):
            _ = result.groups["nope"]
        with pytest.raises(AttributeError, match="No split named"):
            _ = result.groups.nope


class TestItemAccess:
    def test_indices_index_real_data(self, clinical_df):
        X = np.random.default_rng(0).normal(size=(len(clinical_df), 3))
        r = split(
            clinical_df, group_col="patient_id", label_col="diagnosis",
            max_evals=2000, seed=0,
        )
        train, val, test = r.indices.astuple()
        assert X[train].shape[0] + X[val].shape[0] + X[test].shape[0] == len(X)

    def test_masks_agree_with_indices(self, result):
        for name in result.names:
            np.testing.assert_array_equal(
                np.flatnonzero(result.item_mask(name)), result.indices[name]
            )
            np.testing.assert_array_equal(
                result.dataset.group_ids[result.mask(name)], result.groups[name]
            )

    def test_aggregated_data_explains_the_limitation(self, tiny):
        """Count-only datasets cannot produce item indices; say so precisely."""
        r = split(tiny, (0.5, 0.5), max_evals=500, seed=0)
        assert not tiny.has_items
        with pytest.raises(ValueError, match="aggregated counts"):
            _ = r.indices


class TestQualityReporting:
    def test_achieved_ratios_are_reported(self, result):
        assert result.achieved_ratios.sum() == pytest.approx(1.0)
        assert len(result.achieved_ratios) == result.n_splits

    def test_item_counts_sum_to_total(self, result):
        assert result.item_counts.sum() == pytest.approx(result.dataset.total_items)

    def test_worst_cell_is_identified(self, result):
        split_name, _class_name, err = result.worst_cell()
        assert split_name in result.names
        assert err == pytest.approx(result.relative_error.max())

    def test_dropped_classes_surfaced(self, unstratifiable):
        r = split(unstratifiable, (0.5, 0.25, 0.25), max_evals=500, seed=0)
        assert "c2" in r.dropped_classes
        assert "unstratifiable" in r.summary()

    def test_empty_classes_detected(self):
        """A class missing from a split breaks stratified evaluation downstream."""
        data = make_dataset([[10, 0], [10, 0], [10, 1], [10, 0], [10, 0], [10, 1]])
        r = split(data, (0.5, 0.25, 0.25), max_evals=500, seed=0,
                  unstratifiable="keep")
        empty = r.empty_classes()
        assert any("c1" in v for v in empty.values())

    def test_summary_mentions_the_essentials(self, result):
        text = result.summary()
        for token in ("train", "val", "test", "requested", "achieved", "annealing"):
            assert token in text

    def test_gap_is_none_without_a_bound(self, result):
        assert result.gap() is None


class TestDataFrameHelpers:
    def test_to_frame(self, result):
        frame = result.to_frame()
        assert len(frame) == result.dataset.n_groups
        assert set(frame["split"]) <= set(result.names)

    def test_counts_frame(self, result):
        frame = result.counts_frame()
        assert len(frame) == result.n_splits * result.actual_counts.shape[1]
        assert {"target", "actual", "rel_error"} <= set(frame.columns)

    def test_assign_column(self, clinical_df, result):
        out = result.assign_column(clinical_df, "patient_id")
        assert "split" not in clinical_df.columns  # not mutated by default
        assert out["split"].notna().all()
        # every row of a patient carries the same label
        assert (out.groupby("patient_id")["split"].nunique() == 1).all()

    def test_assign_column_inplace(self, clinical_df, result):
        copy = clinical_df.copy()
        result.assign_column(copy, "patient_id", column="fold", inplace=True)
        assert "fold" in copy.columns


class TestEvaluate:
    def test_scores_an_external_assignment(self, tiny):
        a = np.array([0, 0, 1, 1, 2, 2])
        assert evaluate(tiny, a, (0.5, 0.25, 0.25)) > 0

    def test_agrees_with_split(self, tiny):
        r = split(tiny, (0.5, 0.25, 0.25), max_evals=1000, seed=0)
        assert evaluate(tiny, r.assignment, (0.5, 0.25, 0.25)) == pytest.approx(
            r.cost, rel=1e-12
        )

    def test_lets_you_compare_against_another_splitter(self, tiny):
        """The point of a public evaluate(): compare on equal terms."""
        ours = split(tiny, (0.5, 0.25, 0.25), max_evals=2000, seed=0)
        naive = np.array([0, 1, 2, 0, 1, 2])
        assert ours.cost <= evaluate(tiny, naive, (0.5, 0.25, 0.25))

    @pytest.mark.parametrize(
        "bad, match",
        [
            (np.zeros(99), "shape"),
            (np.array([0, 0, 0, 0, 0, 9]), "outside"),
            (np.array([0, 0, 0, 0, 0, -1]), "outside"),
        ],
    )
    def test_validation(self, tiny, bad, match):
        with pytest.raises(ValueError, match=match):
            evaluate(tiny, bad, (0.5, 0.25, 0.25))


class TestStrategySelection:
    @pytest.mark.parametrize("name", ["annealing", "evolution", "random"])
    def test_by_name(self, tiny, name):
        assert split(tiny, (0.5, 0.5), strategy=name, max_evals=500, seed=0).strategy == name

    def test_params_forwarded_and_recorded(self, tiny):
        r = split(
            tiny, (0.5, 0.5), strategy="annealing", initial_temp=5.0,
            max_evals=500, seed=0,
        )
        assert r.strategy_params["initial_temp"] == 5.0

    def test_instance_accepted(self, tiny):
        from splitkit.strategies import RandomSearch

        r = split(tiny, (0.5, 0.5), strategy=RandomSearch(), max_evals=500, seed=0)
        assert r.strategy == "random"


class TestBudgets:
    def test_max_evals_respected(self, tiny):
        assert split(tiny, (0.5, 0.5), max_evals=250, seed=0).n_evals <= 250

    def test_time_budget_respected(self, dense):
        import time

        t0 = time.perf_counter()
        split(dense, (0.5, 0.5), time_budget=0.3, seed=0)
        assert time.perf_counter() - t0 < 1.0

    def test_schedule_fits_the_budget(self, clinical_df):
        """The default cooling schedule must fit the budget it is given.

        A fixed rate silently assumes a particular budget: 0.9999 needs roughly
        300k steps to anneal, so on a small run the temperature has barely moved
        and the search never leaves its exploration phase.
        """
        kw = dict(group_col="patient_id", label_col="diagnosis", seed=42)
        adaptive = split(clinical_df, max_evals=2_000, **kw).cost
        assumes_300k = split(
            clinical_df, max_evals=2_000, cooling_rate=0.9999, **kw
        ).cost
        assert adaptive < assumes_300k / 10

    def test_more_budget_still_helps(self, clinical_df):
        """Quality must improve monotonically with budget, never degrade."""
        kw = dict(group_col="patient_id", label_col="diagnosis", seed=42)
        costs = [
            split(clinical_df, max_evals=ev, **kw).cost
            for ev in (2_000, 10_000, 50_000)
        ]
        assert costs == sorted(costs, reverse=True)


class TestTopLevelNamespace:
    def test_exports_are_importable(self):
        for name in splitkit.__all__:
            assert hasattr(splitkit, name), name

    def test_split_and_evaluate_exported(self):
        assert splitkit.split is split
        assert splitkit.evaluate is evaluate
        assert splitkit.GroupedDataset is GroupedDataset


class TestDefaultObjective:
    def test_item_ratios_held_under_heavy_imbalance(self):
        """Inverse-frequency weights make the majority class nearly free to move."""
        r = split(splitkit.from_preset("heavy_imbalance"), max_evals=20_000, seed=0)
        np.testing.assert_allclose(r.achieved_ratios, r.ratios, atol=0.02)

    def test_without_size_term_item_ratios_drift(self):
        r = split(
            splitkit.from_preset("heavy_imbalance"), size_weight=0, max_evals=20_000, seed=0
        )
        assert np.abs(r.achieved_ratios - r.ratios).max() > 0.02


class TestDefaultBudget:
    def test_applies_when_no_stop_condition_given(self, tiny):
        """With neither max_evals nor time_budget, a default budget applies."""
        from splitkit.strategies.base import DEFAULT_MAX_EVALS

        r = split(tiny, (0.5, 0.25, 0.25), seed=0)
        assert r.n_evals == DEFAULT_MAX_EVALS
