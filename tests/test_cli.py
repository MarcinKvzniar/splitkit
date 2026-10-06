"""The ``splitkit`` command line."""

from __future__ import annotations

import io
import runpy
import sys

import numpy as np
import pandas as pd
import pytest

from splitkit.cli import main

FAST = ["--max-evals", "2000", "--seed", "0"]


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    patients = np.repeat([f"p{i:02d}" for i in range(30)], rng.integers(1, 6, 30))
    return pd.DataFrame(
        {
            "patient": patients,
            "diagnosis": rng.choice(["benign", "malignant", "other"], len(patients)),
            "a": rng.integers(0, 2, len(patients)),
            "b": rng.integers(0, 2, len(patients)),
        }
    )


@pytest.fixture
def csv(tmp_path, frame) -> str:
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    return str(path)


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def read_output(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text))


class TestSplitting:
    def test_labels_every_row_without_leakage(self, capsys, csv, frame):
        code, out, err = run(capsys, csv, "--group-col", "patient", "--label-col", "diagnosis", *FAST)
        labelled = read_output(out)
        assert code == 0
        assert len(labelled) == len(frame)
        assert set(labelled["split"]) == {"train", "val", "test"}
        assert (labelled.groupby("patient")["split"].nunique() == 1).all()
        assert "annealing" in err

    def test_writes_output_file(self, capsys, csv, tmp_path):
        target = tmp_path / "out.csv"
        code, out, _ = run(
            capsys, csv, "--group-col", "patient", "--label-col", "diagnosis",
            "-o", str(target), "--column", "fold", *FAST,
        )
        assert code == 0
        assert out == ""
        assert "fold" in pd.read_csv(target).columns

    @pytest.mark.parametrize("mode", ["--label-cols", "--count-cols"])
    def test_multi_column_label_modes(self, capsys, csv, mode):
        code, out, _ = run(capsys, csv, "--group-col", "patient", mode, "a,b", *FAST)
        assert code == 0
        assert "split" in read_output(out).columns

    def test_custom_ratios_and_names(self, capsys, csv):
        code, out, _ = run(
            capsys, csv, "--group-col", "patient", "--label-col", "diagnosis",
            "--ratios", "0.8,0.2", "--names", "fit,holdout", *FAST,
        )
        assert code == 0
        assert set(read_output(out)["split"]) == {"fit", "holdout"}

    def test_seed_makes_output_reproducible(self, capsys, csv):
        argv = [csv, "--group-col", "patient", "--label-col", "diagnosis", *FAST]
        assert run(capsys, *argv)[1] == run(capsys, *argv)[1]

    def test_reads_stdin_and_tsv(self, capsys, frame, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "stdin", io.StringIO(frame.to_csv(index=False)))
        assert run(capsys, "-", "--group-col", "patient", "--label-col", "diagnosis", *FAST)[0] == 0

        tsv = tmp_path / "data.tsv"
        frame.to_csv(tsv, sep="\t", index=False)
        assert run(capsys, str(tsv), "--group-col", "patient", "--label-col", "diagnosis", *FAST)[0] == 0

    def test_preserves_input_values(self, capsys, tmp_path):
        path = tmp_path / "ids.csv"
        path.write_text("g,y,v\n007,a,1.50\n7,b,NA\n8,a,\n9,b,2\n10,a,3\n")
        code, out, _ = run(capsys, str(path), "--group-col", "g", "--label-col", "y", *FAST)
        raw = pd.read_csv(io.StringIO(out), dtype=str, keep_default_na=False)
        assert code == 0
        assert raw["g"].tolist() == ["007", "7", "8", "9", "10"]
        assert raw["v"].tolist() == ["1.50", "NA", "", "2", "3"]
        assert raw["split"].ne("").all()

    def test_quiet_suppresses_the_report(self, capsys, csv):
        code, out, err = run(
            capsys, csv, "--group-col", "patient", "--label-col", "diagnosis", "-q", *FAST
        )
        assert code == 0
        assert err == ""
        assert "split" in read_output(out).columns


class TestErrors:
    def test_missing_column_is_reported_cleanly(self, capsys, csv):
        code, _, err = run(capsys, csv, "--group-col", "nope", "--label-col", "diagnosis")
        assert code == 1
        assert err.startswith("splitkit: error:")
        assert "nope" in err

    def test_refuses_to_overwrite_existing_column(self, capsys, tmp_path, frame):
        path = tmp_path / "data.csv"
        frame.assign(split="old").to_csv(path, index=False)
        code, _, err = run(capsys, str(path), "--group-col", "patient", "--label-col", "diagnosis")
        assert code == 1
        assert "already exists" in err

    def test_unwritable_output_is_reported_cleanly(self, capsys, csv, tmp_path):
        target = tmp_path / "missing_dir" / "out.csv"
        code, _, err = run(
            capsys, csv, "--group-col", "patient", "--label-col", "diagnosis",
            "-o", str(target), *FAST,
        )
        assert code == 1
        assert "splitkit: error:" in err

    def test_missing_file_is_reported_cleanly(self, capsys, tmp_path):
        code, _, err = run(capsys, str(tmp_path / "absent.csv"), "--group-col", "g", "--label-col", "y")
        assert code == 1
        assert "splitkit: error:" in err

    @pytest.mark.parametrize(
        "argv",
        [
            ["--label-col", "diagnosis"],
            ["--group-col", "patient"],
            ["--group-col", "patient", "--label-col", "diagnosis", "--ratios", "a,b"],
            ["--group-col", "patient", "--label-col", "d", "--label-cols", "a,b"],
        ],
    )
    def test_invalid_arguments_exit_with_usage(self, csv, argv):
        with pytest.raises(SystemExit) as exc:
            main([csv, *argv])
        assert exc.value.code == 2

    def test_input_is_required(self):
        with pytest.raises(SystemExit):
            main(["--group-col", "patient", "--label-col", "diagnosis"])


class TestInformational:
    def test_lists_strategies(self, capsys):
        code, out, _ = run(capsys, "--list-strategies")
        assert code == 0
        assert "annealing" in out.split()

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert "splitkit" in capsys.readouterr().out

    def test_runs_as_module(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["splitkit", "--list-strategies"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("splitkit", run_name="__main__")
        assert exc.value.code == 0
