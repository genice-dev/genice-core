"""
Backward compatibility utilities (explicit parameter alias mapping).
"""

import functools
import logging
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable)


def accept_aliases(**alias_map: str) -> Callable[[F], F]:
    """Decorator that maps deprecated keyword argument names to the current names.

    Use when renaming parameters: list old_name -> new_name explicitly.
    This allows you to choose any new names, not only snake_case.
    Logs a warning when a deprecated (old) name is used.

    Example:
        @accept_aliases(
            vertexPositions="positions",
            dipoleOptimizationCycles="max_cycles",
            isPeriodicBoundary="pbc",
            targetPol="target",
        )
        def optimize(paths, positions, max_cycles=2000, pbc=False, target=None):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            normalized = {}
            for key, value in kwargs.items():
                if key in alias_map:
                    logger.warning(
                        "Parameter %r is deprecated; use %r instead.",
                        key,
                        alias_map[key],
                    )
                    normalized[alias_map[key]] = value
                else:
                    normalized[key] = value
            return func(*args, **normalized)
        return wrapper  # type: ignore[return-value]
    return decorator
