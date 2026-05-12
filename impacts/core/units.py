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


def normalize_temperature(
    data: xr.DataArray,
    *,
    target_unit: str,
    unit: str,
) -> xr.DataArray:
    """Convert a temperature array between Celsius and Kelvin."""

    normalized_source = unit.lower()
    normalized_target = target_unit.lower()

    if normalized_target in {"c", "degc", "degree_celsius", "degrees_celsius", "celsius"}:
        if normalized_source in {"k", "kelvin"}:
            return _with_units(data - 273.15, data=data, units="C")
        if normalized_source in {"c", "degc", "degree_celsius", "degrees_celsius", "celsius"}:
            return data
    if normalized_target in {"k", "kelvin"}:
        if normalized_source in {"k", "kelvin"}:
            return data
        if normalized_source in {"c", "degc", "degree_celsius", "degrees_celsius", "celsius"}:
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

    if normalized_target in {"kpa", "kilopascal", "kilopascals"}:
        if normalized_source in {"pa", "pascal", "pascals"}:
            return _with_units(data / 1_000.0, data=data, units="kPa")
        if normalized_source in {"kpa", "kilopascal", "kilopascals"}:
            return data
    if normalized_target in {"pa", "pascal", "pascals"}:
        if normalized_source in {"pa", "pascal", "pascals"}:
            return data
        if normalized_source in {"kpa", "kilopascal", "kilopascals"}:
            return _with_units(data * 1_000.0, data=data, units="Pa")

    raise ValueError(f"unsupported pressure unit conversion for {data.name!r}: {unit} -> {target_unit}")


def _with_units(converted: xr.DataArray, *, data: xr.DataArray, units: str) -> xr.DataArray:
    converted.attrs = data.attrs
    converted.attrs["units"] = units
    return converted
