from __future__ import annotations

from collections.abc import Mapping

import xarray as xr


def resolve_unit(
    data: xr.DataArray,
    *,
    variable_units: Mapping[str, str] | None = None,
) -> str:
    """Resolve units from explicit overrides or DataArray metadata."""

    explicit_unit = None if variable_units is None else variable_units.get(str(data.name))
    unit = explicit_unit or data.attrs.get("units")
    if unit is None:
        raise ValueError(f"{data.name!r} is missing units metadata; pass variable_units explicitly")
    return str(unit).strip()


_CELSIUS = frozenset({"c", "degc", "degree_celsius", "degrees_celsius", "celsius"})
_KELVIN = frozenset({"k", "kelvin"})
_PASCAL = frozenset({"pa", "pascal", "pascals"})
_KILOPASCAL = frozenset({"kpa", "kilopascal", "kilopascals"})


def convert_units(
    data: xr.DataArray,
    *,
    target_unit: str,
    unit: str,
) -> xr.DataArray:
    """Convert ``data`` from ``unit`` to ``target_unit``.

    Dispatches on the kind of ``target_unit``: temperature (Celsius, Kelvin) or
    pressure (Pa, kPa).
    """

    target = target_unit.lower()
    if target in _CELSIUS | _KELVIN:
        return normalize_temperature(data, target_unit=target_unit, unit=unit)
    if target in _PASCAL | _KILOPASCAL:
        return normalize_pressure(data, target_unit=target_unit, unit=unit)
    raise ValueError(f"unsupported unit conversion for {data.name!r}: {unit} -> {target_unit}")


def normalize_temperature(
    data: xr.DataArray,
    *,
    target_unit: str,
    unit: str,
) -> xr.DataArray:
    """Convert a temperature array between Celsius and Kelvin."""

    normalized_source = unit.lower()
    normalized_target = target_unit.lower()

    if normalized_target in _CELSIUS:
        if normalized_source in _KELVIN:
            return _with_units(data - 273.15, data=data, units="C")
        if normalized_source in _CELSIUS:
            return data
    if normalized_target in _KELVIN:
        if normalized_source in _KELVIN:
            return data
        if normalized_source in _CELSIUS:
            return _with_units(data + 273.15, data=data, units="K")

    raise ValueError(f"unsupported temperature unit conversion for {data.name!r}: {unit} -> {target_unit}")


def normalize_pressure(
    data: xr.DataArray,
    *,
    target_unit: str,
    unit: str,
) -> xr.DataArray:
    """Convert a pressure array between Pa and kPa."""

    normalized_source = unit.lower()
    normalized_target = target_unit.lower()

    if normalized_target in _KILOPASCAL:
        if normalized_source in _PASCAL:
            return _with_units(data / 1_000.0, data=data, units="kPa")
        if normalized_source in _KILOPASCAL:
            return data
    if normalized_target in _PASCAL:
        if normalized_source in _PASCAL:
            return data
        if normalized_source in _KILOPASCAL:
            return _with_units(data * 1_000.0, data=data, units="Pa")

    raise ValueError(f"unsupported pressure unit conversion for {data.name!r}: {unit} -> {target_unit}")


def _with_units(converted: xr.DataArray, *, data: xr.DataArray, units: str) -> xr.DataArray:
    converted.attrs = data.attrs
    converted.attrs["units"] = units
    return converted
