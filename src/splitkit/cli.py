"""Command line: ``splitkit data.csv --group-col patient --label-col diagnosis``."""

from __future__ import annotations

import argparse
import io
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from ._console import print_report
from .api import split
from .dataset import _require_pandas
from .strategies import list_strategies


def _numbers(text: str) -> list[float]:
    try:
        return [float(x) for x in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated numbers, got {text!r}"
        ) from None


def _names(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="splitkit",
        description="Split grouped data into leakage-free, class-balanced parts. "
        "Writes the input with an added split column; the quality report goes to stderr.",
    )
    parser.add_argument("input", nargs="?", help="CSV file (tab-separated if .tsv), or '-' for stdin")
    parser.add_argument("-o", "--output", default="-", help="output file, same format as the input (default: stdout)")
    parser.add_argument("--column", default="split", help="name of the added column")
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress bar or report")

    columns = parser.add_argument_group("columns")
    columns.add_argument("--group-col", help="items sharing a value stay in one split")
    labels = columns.add_mutually_exclusive_group()
    labels.add_argument("--label-col", help="one categorical label per row")
    labels.add_argument("--label-cols", type=_names, help="comma-separated multi-label indicators")
    labels.add_argument("--count-cols", type=_names, help="comma-separated per-class counts")

    search = parser.add_argument_group("split")
    search.add_argument("--ratios", type=_numbers, default=[0.7, 0.15, 0.15],
                        help="comma-separated shares (default: 0.7,0.15,0.15)")
    search.add_argument("--names", type=_names, help="comma-separated split names")
    search.add_argument("--strategy", default="annealing", choices=list_strategies())
    search.add_argument("--seed", type=int)
    search.add_argument("--max-evals", type=int)
    search.add_argument("--time-budget", type=float, help="seconds")

    parser.add_argument("--list-strategies", action="store_true", help="print strategies and exit")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list_strategies:
        print("\n".join(list_strategies()))
        return 0
    if args.input is None or args.group_col is None:
        parser.error("an input file and --group-col are required")
    if not (args.label_col or args.label_cols or args.count_cols):
        parser.error("one of --label-col, --label-cols or --count-cols is required")

    try:
        pd = _require_pandas()
        text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text("utf-8-sig")
        sep = "\t" if args.input.endswith(".tsv") else ","
        # Keys and labels stay text, so IDs like "007" and "7" remain distinct.
        as_text = {c: str for c in (args.group_col, args.label_col) if c}
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=as_text)
        if args.column in df.columns:
            raise ValueError(
                f"column {args.column!r} already exists; choose another with --column"
            )
        result = split(
            df,
            args.ratios,
            names=args.names,
            strategy=args.strategy,
            seed=args.seed,
            max_evals=args.max_evals,
            time_budget=args.time_budget,
            group_col=args.group_col,
            label_col=args.label_col,
            label_cols=args.label_cols,
            count_cols=args.count_cols,
            name=Path(args.input).stem if args.input != "-" else "stdin",
            progress=not args.quiet and sys.stderr.isatty(),
        )
        if not args.quiet:
            print_report(result)

        # Write the input back verbatim (no re-typed values) plus the split column.
        out = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
        out[args.column] = result.assign_column(df, args.group_col)["split"].to_numpy()
        out.to_csv(sys.stdout if args.output == "-" else args.output, sep=sep, index=False)
    except (ImportError, OSError, KeyError, ValueError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) else exc
        print(f"splitkit: error: {message}", file=sys.stderr)
        return 1
    return 0
