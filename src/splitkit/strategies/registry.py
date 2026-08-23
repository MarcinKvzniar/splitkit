"""Name-to-strategy registry.

Keeps strategy lookup in one place so that ``strategy="annealing"`` works from the
public API, the CLI and the benchmarks alike, and so third-party strategies can
register themselves.
"""

from __future__ import annotations

from .base import Strategy

__all__ = ["get_strategy", "list_strategies", "register_strategy"]

_REGISTRY: dict[str, type[Strategy]] = {}


def register_strategy(cls: type[Strategy]) -> type[Strategy]:
    """Register a strategy class under its ``name`` attribute.

    Returns the class, so it can be used as a decorator.
    """
    if not getattr(cls, "name", ""):
        raise ValueError(f"{cls.__name__} must define a non-empty `name`.")
    _REGISTRY[cls.name] = cls
    return cls


def list_strategies() -> tuple[str, ...]:
    """Names accepted by :func:`get_strategy`."""
    return tuple(sorted(_REGISTRY))


def get_strategy(strategy: str | Strategy, /, **params) -> Strategy:
    """Resolve a name into a strategy instance.

    An already-constructed :class:`Strategy` passes through unchanged, so callers
    can accept either a convenient name or a fully configured object.

    The selector is positional-only because strategies may legitimately take a
    parameter of their own called ``strategy`` -- differential evolution's variant
    string, for instance -- which would otherwise collide here.
    """
    if isinstance(strategy, Strategy):
        return strategy
    try:
        cls = _REGISTRY[strategy]
    except KeyError:
        raise KeyError(
            f"Unknown strategy {strategy!r}. Available: {', '.join(list_strategies())}"
        ) from None
    return cls(**params)
