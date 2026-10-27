"""Every public name of the package is in the table of library functions the tests call, or is exempted."""

from __future__ import annotations

import importlib
import inspect

import pytest

from . import entrypoints

PACKAGES = (
    "impacts",
    "impacts.climatecore",
    "impacts.buildings",
    "impacts.thermoelectric",
    "impacts.thermoelectric.siting",
    "impacts.engine",
)

# Names that are typing or language helpers imported into a module namespace, not
# behaviour of this package.
EXEMPT = {
    ("impacts.engine", name)
    for name in (
        "Callable",
        "Duration",
        "Literal",
        "Mapping",
        "NormalizationSpec",
        "NormalizeFn",
        "PhysicsFn",
        "TypeAlias",
        "annotations",
        "dataclass",
    )
}


def _public_names(module) -> list[str]:
    declared = getattr(module, "__all__", None)
    if declared is not None:
        return list(declared)
    return [name for name in dir(module) if not name.startswith("_")]


def _adapted_objects() -> set[int]:
    return {
        id(getattr(importlib.import_module(module), attribute))
        for module, attribute in entrypoints.TABLE.values()
    }


@pytest.mark.parametrize("package", PACKAGES)
def test_public_names_are_adapted(package):
    module = importlib.import_module(package)
    adapted = _adapted_objects()
    unadapted = []
    for name in _public_names(module):
        if (package, name) in EXEMPT:
            continue
        value = getattr(module, name, None)
        if value is None or inspect.ismodule(value):
            continue
        if id(value) not in adapted:
            unadapted.append(name)
    assert not unadapted, f"public names of {package} missing from entrypoints.TABLE: {unadapted}"


def test_table_entries_resolve():
    for name in entrypoints.TABLE:
        entrypoints.get(name)
