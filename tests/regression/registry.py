"""Registry of the end-to-end tests."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from .canon import Raised

CASES: dict[str, Case] = {}


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    fn: Callable[[Any], Any]
    tier: str = "exact"
    frozen: bool = False


def case(name: str, *, tier: str = "exact", frozen: bool = False):
    """Register ``fn(world)`` as the end-to-end test ``name``.

    ``tier`` is the comparison mode used unless the environment forces one:
    ``exact`` needs identical values, ``tolerance`` allows rounding-level
    differences (see ``canon.py``). A ``frozen`` test has a stored reference
    output that did not come from the code under test and is maintained by hand
    (see the module of the test); the command that updates the stored outputs
    leaves it alone.
    """

    def register(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if name in CASES:
            raise ValueError(f"duplicate end-to-end test {name!r}")
        CASES[name] = Case(name=name, fn=fn, tier=tier, frozen=frozen)
        return fn

    return register


def outcome(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn``; return its value, or a ``Raised`` record of the exception."""

    try:
        return fn(*args, **kwargs)
    except Exception as error:
        return Raised(type=type(error).__name__, message=str(error))


def outcome_type(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Like ``outcome`` but records only the exception type.

    For failures raised inside a third-party library, whose wording is not
    this repository's behaviour and changes with a dependency upgrade.
    """

    try:
        return fn(*args, **kwargs)
    except Exception as error:
        return Raised(type=type(error).__name__, message="<third-party message not pinned>")
