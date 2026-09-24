"""Terminal output: a live progress display and a styled report, via rich if installed."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .problem import Budget
    from .result import SplitResult

ProgressCallback = Callable[[int, float], None]

_PLAIN_REFRESH = 0.1  # seconds between plain-text redraws


def _rich_available() -> bool:
    try:
        import rich  # noqa: F401
    except ImportError:
        return False
    return True


@contextmanager
def progress_display(strategy: str, budget: Budget) -> Iterator[ProgressCallback]:
    """Yield a ``(n_evals, best_cost)`` callback that draws progress on stderr."""
    display = _rich_progress if _rich_available() else _plain_progress
    with display(strategy, budget) as report:
        yield report


def _fraction_done(budget: Budget, n_evals: int, elapsed: float) -> float | None:
    if budget.max_evals is not None:
        return min(n_evals / budget.max_evals, 1.0)
    if budget.time_limit is not None:
        return min(elapsed / budget.time_limit, 1.0)
    return None


@contextmanager
def _rich_progress(strategy: str, budget: Budget) -> Iterator[ProgressCallback]:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("best cost [cyan]{task.fields[cost]}"),
        TimeElapsedColumn(),
        console=Console(stderr=True),
        transient=True,
    )
    # Starts indeterminate; strategies that never report (exact, sgkf) keep a spinner.
    task = progress.add_task(strategy, total=None, cost="-")
    t_start = time.perf_counter()

    def report(n_evals: int, best_cost: float) -> None:
        done = _fraction_done(budget, n_evals, time.perf_counter() - t_start)
        total = None if done is None else 1.0
        progress.update(task, total=total, completed=done or 0.0, cost=f"{best_cost:.4g}")

    with progress:
        yield report


@contextmanager
def _plain_progress(strategy: str, budget: Budget) -> Iterator[ProgressCallback]:
    t_start = last_draw = time.perf_counter()
    drawn = False

    def report(n_evals: int, best_cost: float) -> None:
        nonlocal last_draw, drawn
        now = time.perf_counter()
        if now - last_draw < _PLAIN_REFRESH:
            return
        last_draw, drawn = now, True
        done = _fraction_done(budget, n_evals, now - t_start)
        percent = "" if done is None else f"{done:4.0%}  "
        sys.stderr.write(
            f"\r{strategy}: {percent}{n_evals:,} evals  best cost {best_cost:.4g}  "
            f"{now - t_start:.1f}s"
        )
        sys.stderr.flush()

    try:
        yield report
    finally:
        if drawn:
            sys.stderr.write("\n")


def print_report(result: SplitResult) -> None:
    """Print the quality report to stderr, styled when rich is installed."""
    if _rich_available():
        from rich.console import Console

        Console(stderr=True).print(result)
    else:
        print(result.summary(), file=sys.stderr)


def render_report(result: SplitResult) -> Any:
    """The quality report as a rich renderable."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    header = Text.assemble(
        ("Strategy ", "dim"), (result.strategy, "bold"),
        ("   cost ", "dim"), (f"{result.cost:.6g}", "bold cyan"),
    )
    if result.proved_optimal:
        header.append("   PROVEN OPTIMAL", style="bold green")
    elif result.lower_bound is not None:
        header.append(f"   lower bound {result.lower_bound:.6g}  gap {result.gap():.2%}")
    header.append(
        f"\n{result.n_evals:,} evaluations in {result.elapsed_time:.2f}s", style="dim"
    )

    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("split")
    for column in ("groups", "items", "requested", "achieved"):
        table.add_column(column, justify="right")
    items = result.item_counts
    for s, name in enumerate(result.names):
        table.add_row(
            name,
            f"{int((result.assignment == s).sum()):,}",
            f"{int(items[s]):,}",
            f"{result.ratios[s]:.1%}",
            f"{result.achieved_ratios[s]:.1%}",
        )

    split_name, class_name, err = result.worst_cell()
    colour = "green" if err < 0.05 else "yellow" if err < 0.2 else "red"
    lines = [
        Text.assemble(
            "Worst class balance: ", (class_name, "bold"), " in ", (split_name, "bold"),
            " off target by ", (f"{err:.1%}", f"bold {colour}"),
        )
    ]
    styles = {"warning": "bold red", "note": "yellow"}
    lines += [Text(message, style=styles[level]) for level, message in result._issues()]

    dataset = result.dataset
    title = (
        f"[bold]{dataset.name}[/]  {result.n_splits} splits, "
        f"{dataset.n_groups:,} groups, {dataset.n_classes} classes"
    )
    return Panel(
        Group(header, Text(), table, Text(), *lines),
        title=title,
        title_align="left",
        expand=False,
    )
