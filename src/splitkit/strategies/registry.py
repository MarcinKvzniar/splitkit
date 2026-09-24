"""Name-to-strategy registry, open to third-party strategies."""

from __future__ import annotations

from typing import Any

from .base import Strategy

__all__ = ["get_strategy", "list_strategies", "register_strategy"]

_REGISTRY: dict[str, type[Strategy]] = {}


def register_strategy(cls: type[Strategy]) -> type[Strategy]:
    """Class decorator registering a strategy under its ``name``."""
    if not getattr(cls, "name", ""):
        raise ValueError(f"{cls.__name__} must define a non-empty `name`.")
    _REGISTRY[cls.name] = cls
    return cls


def list_strategies() -> tuple[str, ...]:
    """Names accepted by :func:`get_strategy`."""
    return tuple(sorted(_REGISTRY))


def get_strategy(strategy: str | Strategy, /, **params: Any) -> Strategy:
    """Resolve a name into a strategy instance; instances pass through unchanged."""
    if isinstance(strategy, Strategy):
        return strategy
    try:
        cls = _REGISTRY[strategy]
    except KeyError:
        raise KeyError(
            f"Unknown strategy {strategy!r}. Available: {', '.join(list_strategies())}"
        ) from None
    return cls(**params)
