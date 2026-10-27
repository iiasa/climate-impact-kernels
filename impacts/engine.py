"""The ``Kernel`` class and the functions that evaluate a kernel.

A kernel is one impact calculation: the climate inputs it reads, its constants and
its equations. It declares what it needs and states its physics; the engine does
the rest. In prose, a site is one row of the mapping table (or one grid cell); the
code calls the site dimension ``entity`` and the mapping column that labels sites
``site_id_column``.

- ``Kernel.inputs`` names each climate variable the physics reads, the source
  variable it comes from by default, and the unit the physics receives. The
  engine selects, renames and converts them (``prepare_climate``).
- ``Kernel.site_columns`` names the per-site columns a mapped evaluation carries
  from the mapping table. The engine samples climate at each site, hands the
  physics a ``(time, entity)`` climate Dataset and a Dataset of site values indexed
  by ``entity``, and attaches the site columns to the output as coordinates.
- ``Kernel.physics(climate, sites, assumptions)`` returns one Dataset of output
  variables. The engine writes the assumptions and the kernel's ``attrs`` onto
  it, drops coordinates no output uses, and resamples it to the requested
  ``duration`` (``native``, ``monthly`` or ``seasonal``).
- An assumption whose field metadata names a ``site`` value is overridden per
  site whenever the sites carry that value; the output's attributes then say so
  rather than report the scalar the physics did not use.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
import xarray as xr

from impacts.climatecore import (
    aggregate_monthly,
    aggregate_seasonal,
    attach_gridded_dataset,
    convert_units,
    resolve_unit,
)

__all__ = [
    "ClimateInput",
    "Duration",
    "Kernel",
    "evaluate_gridded",
    "evaluate_mapped",
    "prepare_climate",
]

Duration: TypeAlias = Literal["native", "monthly", "seasonal"]
PhysicsFn: TypeAlias = Callable[[xr.Dataset, xr.Dataset, Any], xr.Dataset]


@dataclass(frozen=True)
class ClimateInput:
    """One climate variable a kernel's physics reads.

    ``source`` is the default source variable name (an ISIMIP short name);
    callers override it with ``variable_names``. ``unit`` is the unit the physics
    receives: the engine converts to it, and ``None`` passes the variable through
    unconverted and unchecked. ``point`` names the pair of mapping columns holding
    a second sampling point per site (for example a saline unit's sea intake);
    a mapped evaluation reads the variable there, NaN where a site has no such
    point. An ``optional`` input is used when present in the climate data and
    otherwise left out; the physics decides whether it needs it.
    """

    source: str
    unit: str | None = None
    point: tuple[str, str] | None = None
    optional: bool = False


@dataclass(frozen=True)
class Kernel:
    """A pickle-safe kernel: climate inputs, site columns, physics and assumptions.

    ``assumptions`` is the kernel's default assumptions dataclass instance; an
    evaluation can pass another. The output records each assumption's value under
    its field name, and the ``unit`` and ``source`` of its field metadata under
    ``<name>_unit`` and ``<name>_source``. The metadata ``source`` describes the
    field's default: for a value that differs from it, ``<name>_source`` reads
    "set by the caller; default <default> from <source>". ``attrs`` are fixed dataset
    attributes (citations, policy notes) written onto every output after the assumptions.

    An assumption field whose metadata has ``"site": <site variable>`` is one the
    physics replaces per site with that site value where a site has one. When the
    sites carry it, the output records ``<name>`` as a description of the per-site
    values and ``<name>_source`` as their source, read from the site variable's
    ``description`` and ``source`` attributes (a mapping table carries them in
    ``mapping.attrs["site_value_attrs"][column]``), followed by the scalar and
    its source as the fallback for sites without a value. The recorded source counts
    only while it still describes the values the engine sees, so values edited
    after it was recorded are reported as of unrecorded source. It is tied to
    them in one of two ways, both indifferent to the order and subset of sites:

    - ``values_by_key``, ``{"key": <site column>, "values": {key: value or None}}``: every
      site's value equals the map's value for its key, or is missing where the
      key is not in the map (the scalar applies there);
    - ``values_digest`` with ``values_order``: ``site_values_digest`` of the
      values sorted by the named coordinates (for a grid, ``"lat lon"``).

    ``chunks`` is a speed setting, not physics: the chunk widths an
    evaluation rechunks the sampled ``(time, entity)`` data to when the caller
    passes ``chunks=None``. ``None`` keeps the loader's chunks.
    """

    name: str
    inputs: Mapping[str, ClimateInput]
    physics: PhysicsFn
    site_columns: tuple[str, ...] = ()
    assumptions: Any = None
    attrs: Mapping[str, object] = field(default_factory=dict)
    chunks: Mapping[str, int] | None = None
    # Output coordinate order: False keeps the order of the sampled data (time
    # first); True puts the ``entity`` coordinates first and time last. Both exist so
    # that the output layout of existing kernels does not change.
    entity_coords_first: bool = False


def prepare_climate(
    kernel: Kernel,
    climate: xr.Dataset,
    *,
    variable_names: Mapping[str, str] | None = None,
    variable_units: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Select, rename and unit-convert the climate variables a kernel reads.

    ``variable_names`` maps a kernel input name to the source variable to read
    instead of its default. ``variable_units`` gives the unit of an input, keyed by
    the input name; it takes precedence over the source variable's ``units``
    attribute, so it both fills a missing attribute and corrects a wrong one.
    A required input missing from ``climate`` raises; an optional one is left out.
    """

    names = dict(variable_names or {})
    optional = {
        name: names.pop(name, spec.source) for name, spec in kernel.inputs.items() if spec.optional
    }
    variables = {name: spec.source for name, spec in kernel.inputs.items() if not spec.optional}
    variables.update(names)
    variables.update({name: source for name, source in optional.items() if source in climate.data_vars})

    missing = [
        source
        for name, source in variables.items()
        if name not in optional and source not in climate.data_vars
    ]
    if missing:
        raise ValueError(f"dataset missing required climate variables: {', '.join(sorted(missing))}")

    prepared = climate[list(variables.values())].rename(
        {source: target for target, source in variables.items()}
    )
    for name, spec in kernel.inputs.items():
        if spec.unit is None or name not in prepared.data_vars:
            continue
        prepared[name] = convert_units(
            prepared[name],
            target_unit=spec.unit,
            unit=resolve_unit(prepared[name], variable_units=variable_units),
        )
    return prepared


def evaluate_mapped(
    kernel: Kernel,
    mapping: pd.DataFrame,
    climate: xr.Dataset,
    duration: Duration,
    *,
    site_id_column: str,
    assumptions: Any = None,
    gridded_coords: xr.Dataset | None = None,
    seasons: Mapping[str, tuple[int, ...]] | None = None,
    chunks: Mapping[str, int] | None = None,
    variable_names: Mapping[str, str] | None = None,
    variable_units: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Evaluate a kernel over the sites of a mapping table.

    Each row of ``mapping`` is a site with a ``grid_lat``/``grid_lon`` cell and the
    kernel's ``site_columns``. ``site_id_column`` names the column of ``mapping``
    whose unique values label the sites. ``gridded_coords`` carries extra gridded fields
    (for example a monthly solar climatology) sampled at the same cells and handed
    to the physics as site values. ``assumptions`` defaults to the kernel's own.
    ``chunks`` rechunks the sampled ``(time, entity)`` data before the physics runs,
    for example ``{"time": 365, "entity": 10_000}``; ``None`` uses the kernel's
    ``chunks`` and ``{}`` keeps the loader's chunks.
    """

    binding = _MappedBinding(
        mapping,
        site_id_column=site_id_column,
        site_columns=kernel.site_columns,
        gridded_coords=gridded_coords,
        points={name: spec.point for name, spec in kernel.inputs.items() if spec.point is not None},
    )
    return _evaluate(
        kernel,
        binding,
        climate,
        duration,
        assumptions=assumptions,
        seasons=seasons,
        chunks=chunks,
        variable_names=variable_names,
        variable_units=variable_units,
    )


def evaluate_gridded(
    kernel: Kernel,
    climate: xr.Dataset,
    parameters: xr.Dataset,
    duration: Duration,
    *,
    assumptions: Any = None,
    spatial_dims: tuple[str, ...] = ("lat", "lon"),
    seasons: Mapping[str, tuple[int, ...]] | None = None,
    chunks: Mapping[str, int] | None = None,
    variable_names: Mapping[str, str] | None = None,
    variable_units: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Evaluate a kernel on every cell of a grid shared by climate and parameters.

    The grid is stacked to ``entity`` (one site per cell); every variable of ``parameters`` becomes a
    site value (spatially varying ones stacked with the grid, constant ones as
    they are) and an output coordinate. ``chunks`` is as for ``evaluate_mapped``.
    """

    binding = _GriddedBinding(parameters, spatial_dims=spatial_dims)
    return _evaluate(
        kernel,
        binding,
        climate,
        duration,
        assumptions=assumptions,
        seasons=seasons,
        chunks=chunks,
        variable_names=variable_names,
        variable_units=variable_units,
    )


def _evaluate(
    kernel: Kernel,
    binding: _MappedBinding | _GriddedBinding,
    climate: xr.Dataset,
    duration: Duration,
    *,
    assumptions: Any,
    seasons: Mapping[str, tuple[int, ...]] | None,
    chunks: Mapping[str, int] | None,
    variable_names: Mapping[str, str] | None,
    variable_units: Mapping[str, str] | None,
) -> xr.Dataset:
    assumptions = kernel.assumptions if assumptions is None else assumptions
    prepared = prepare_climate(
        kernel, climate, variable_names=variable_names, variable_units=variable_units
    )
    attached = _rechunk(binding.attach(prepared), kernel.chunks if chunks is None else chunks)

    # Hand the physics plain data variables: climate on (time, entity), site values
    # on entity. Coordinates go back on once the physics has run.
    flat = attached.reset_coords()
    climate_vars = list(attached.data_vars)
    site_vars = [name for name in flat.data_vars if name not in attached.data_vars]
    overrides = _site_overrides(assumptions, flat[site_vars])
    # The source attributes describe the site values to the output attributes only; they
    # are not handed to the physics or carried on the output coordinates.
    for site_name in {site_name for site_name, _ in overrides.values()}:
        bare = flat[site_name].variable.copy(deep=False)
        bare.attrs = {
            key: value
            for key, value in bare.attrs.items()
            if key not in (*SITE_VALUE_ATTRS, *SITE_VALUE_BINDING_ATTRS)
        }
        flat[site_name] = bare
        attached = attached.assign_coords({site_name: bare})
    result = kernel.physics(flat[climate_vars], flat[site_vars], assumptions)

    coords = attached.coords
    if kernel.entity_coords_first:
        order = ["entity", *(name for name in coords if name not in ("entity", "time")), "time"]
        coords = xr.Coordinates(
            {name: coords.variables[name] for name in order if name in coords},
            indexes=dict(coords.xindexes),
        )
    # The outputs are on the sampled (time, entity) grid, so they take its coordinates.
    evaluated = xr.Dataset(
        {name: output.variable for name, output in result.data_vars.items()},
        coords=coords,
        attrs=_output_attrs(kernel, assumptions, overrides),
    )
    evaluated = _drop_unused_dim_coords(evaluated)
    if all(dim in evaluated.dims for dim in ("time", "entity")):
        evaluated = evaluated.transpose("time", "entity")
    return _apply_duration(evaluated, duration, seasons=seasons)


# Attributes of a site variable that describe where its values came from.
SITE_VALUE_ATTRS = ("description", "source")
# Attributes tying that source to the values it describes. These and
# ``site_values_digest`` are for the builders of site values in the domain
# packages, not part of the evaluation API.
SITE_VALUES_BY_KEY = "values_by_key"
SITE_VALUES_DIGEST = "values_digest"
SITE_VALUES_ORDER = "values_order"
SITE_VALUE_BINDING_ATTRS = (SITE_VALUES_BY_KEY, SITE_VALUES_DIGEST, SITE_VALUES_ORDER)


def site_values_digest(values: xr.DataArray, order_by: Iterable[str] = ()) -> str:
    """A digest of site values in a fixed order.

    The values are flattened and sorted by the coordinates ``order_by`` (the
    first named varies slowest), so a grid and its stacked or flipped copies give
    the same digest. Numbers are compared as float64 with every NaN alike.
    """

    order_by = tuple(order_by)
    if values.ndim > 1:
        values = values.stack({"_site": values.dims})
    if order_by:
        keys = [np.asarray(values[name].values) for name in reversed(order_by)]
        values = values.isel({values.dims[0]: np.lexsort(keys)})
    floats = np.asarray(values.values, dtype=np.float64).ravel().copy()
    floats[np.isnan(floats)] = np.nan
    return hashlib.sha256(floats.tobytes()).hexdigest()


def _source_is_current(site: xr.DataArray, sites: xr.Dataset) -> bool:
    """Whether a site variable's recorded source still describes its values."""

    by_key = site.attrs.get(SITE_VALUES_BY_KEY)
    if by_key is not None:
        key = by_key["key"]
        if key not in sites.variables:
            return False
        recorded = by_key["values"]
        values = np.asarray(site.values, dtype=np.float64)
        for site_key, value in zip(np.asarray(sites[key].values).tolist(), values.tolist(), strict=True):
            if site_key in recorded:
                expected = np.nan if recorded[site_key] is None else float(recorded[site_key])
                if not (value == expected or (np.isnan(value) and np.isnan(expected))):
                    return False
            elif not np.isnan(value):
                return False
        return True
    digest = site.attrs.get(SITE_VALUES_DIGEST)
    if digest is not None:
        order_by = str(site.attrs.get(SITE_VALUES_ORDER, "")).split()
        if not all(name in site.coords for name in order_by):
            return False
        return digest == site_values_digest(site, order_by)
    return False


def _site_overrides(assumptions: Any, sites: xr.Dataset) -> dict[str, tuple[str, dict[str, str]]]:
    """Assumptions the sites override: name -> (site variable, its source attributes)."""

    if not dataclasses.is_dataclass(assumptions):
        return {}
    overrides = {}
    for assumption in dataclasses.fields(assumptions):
        site_name = assumption.metadata.get("site")
        if site_name is not None and site_name in sites.data_vars:
            site = sites[site_name]
            # A source not tied to these very values (edited since, or never
            # tied) does not describe them.
            current = _source_is_current(site, sites)
            provenance = (
                {key: str(site.attrs[key]) for key in SITE_VALUE_ATTRS if key in site.attrs}
                if current
                else {}
            )
            overrides[assumption.name] = (site_name, provenance)
    return overrides


def _output_attrs(
    kernel: Kernel,
    assumptions: Any,
    overrides: Mapping[str, tuple[str, Mapping[str, str]]],
) -> dict[str, object]:
    attrs: dict[str, object] = {}
    if dataclasses.is_dataclass(assumptions):
        for assumption in dataclasses.fields(assumptions):
            value = getattr(assumptions, assumption.name)
            attrs[assumption.name] = value
            if "unit" in assumption.metadata:
                attrs[f"{assumption.name}_unit"] = assumption.metadata["unit"]
            source = _value_source(assumption, value)
            if source is not None:
                attrs[f"{assumption.name}_source"] = source
            if assumption.name in overrides:
                site_name, provenance = overrides[assumption.name]
                description = provenance.get("description", f"per-site values of {site_name!r}")
                attrs[assumption.name] = f"{description}; {value!r} where a site has none"
                site_source = provenance.get("source", "not recorded with the site values")
                attrs[f"{assumption.name}_source"] = (
                    site_source if source is None else f"{site_source}. Fallback value: {source}"
                )
    attrs.update(kernel.attrs)
    return attrs


def _value_source(assumption: dataclasses.Field, value: Any) -> str | None:
    """The source of an assumption's value.

    The field's ``source`` metadata describes its default, so a value the caller
    changed is reported as theirs, with the default and its source for reference.
    """

    source = assumption.metadata.get("source")
    if assumption.default is not dataclasses.MISSING:
        default = assumption.default
    elif assumption.default_factory is not dataclasses.MISSING:
        default = assumption.default_factory()
    else:
        return source
    if _same_value(value, default):
        return source
    if source is None:
        return f"set by the caller; default {default!r}"
    return f"set by the caller; default {default!r} from {source}"


def _same_value(value: Any, default: Any) -> bool:
    """Whether an assumption still has its default: NaN matches NaN, arrays by content."""

    if isinstance(value, np.ndarray) or isinstance(default, np.ndarray):
        try:
            return bool(np.array_equal(value, default, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(value, default))
    if isinstance(value, float) and isinstance(default, float) and value != value and default != default:
        return True
    try:
        return bool(value == default)
    except (TypeError, ValueError):
        return False


def _apply_duration(
    result: xr.Dataset,
    duration: Duration,
    *,
    seasons: Mapping[str, tuple[int, ...]] | None,
) -> xr.Dataset:
    match duration:
        case "native":
            return result
        case "monthly":
            return aggregate_monthly(result)
        case "seasonal":
            return aggregate_seasonal(result, seasons=seasons)
        case _:
            raise ValueError(f"unsupported duration: {duration!r}")


@dataclass(frozen=True)
class _MappedBinding:
    """Build a ``(time, entity)`` dataset by sampling climate at each site's cell.

    ``points`` reads named variables at a second point per site instead of
    ``grid_lat``/``grid_lon``: it maps a variable to the pair of mapping columns
    holding that point's cell. A site with no second point (NaN in those
    columns) gets NaN for the variable. A variable absent from the climate dataset
    is skipped. When the mapping lacks a variable's point columns, the variable is
    dropped, so the physics sees it as missing rather than read at the wrong point.
    """

    mapping: pd.DataFrame
    site_id_column: str
    site_columns: Iterable[str] | None = None
    gridded_coords: xr.Dataset | None = None
    points: Mapping[str, tuple[str, str]] | None = None

    def attach(self, prepared: xr.Dataset) -> xr.Dataset:
        attached = attach_gridded_dataset(
            self.mapping,
            prepared,
            site_id_column=self.site_id_column,
            metadata_columns=self.site_columns,
        )
        # A table cannot carry attributes per column, so the mapping's own ``attrs``
        # hold the source of its site values.
        for column, provenance in self.mapping.attrs.get("site_value_attrs", {}).items():
            if column in attached.coords:
                described = attached[column].variable.copy(deep=False)
                described.attrs = {**described.attrs, **provenance}
                attached = attached.assign_coords({column: described})
        for name, (lat_column, lon_column) in (self.points or {}).items():
            if name not in prepared.data_vars:
                continue
            if not {lat_column, lon_column}.issubset(self.mapping.columns):
                attached = attached.drop_vars(name)
                continue
            has_point = self.mapping[lat_column].notna() & self.mapping[lon_column].notna()
            points = self.mapping.assign(
                grid_lat=self.mapping[lat_column].where(has_point, self.mapping["grid_lat"]),
                grid_lon=self.mapping[lon_column].where(has_point, self.mapping["grid_lon"]),
            )
            sampled = attach_gridded_dataset(
                points, prepared[[name]], site_id_column=self.site_id_column, metadata_columns=()
            )[name]
            sampled = sampled.reset_coords(drop=True)
            has_point_da = xr.DataArray(
                has_point.to_numpy(), dims=("entity",), coords={"entity": sampled["entity"].values}
            )
            attached[name] = sampled.where(has_point_da)
        if self.gridded_coords is not None:
            sampled = attach_gridded_dataset(
                self.mapping,
                self.gridded_coords,
                site_id_column=self.site_id_column,
                metadata_columns=(),
            )
            attached = attached.assign_coords(
                {name: sampled[name] for name in self.gridded_coords.data_vars}
            )
        return attached


@dataclass(frozen=True)
class _GriddedBinding:
    """Build a ``(time, entity)`` dataset by stacking the spatial grid to ``entity``.

    Parameter variables that carry the spatial dims (e.g. a ``(month, lat, lon)``
    solar field) are stacked to ``(..., entity)`` and attached as coordinates;
    spatially-constant parameters are attached as-is. This keeps the dense grid
    on the cheap xarray stack path rather than going through a join.
    """

    parameters: xr.Dataset
    spatial_dims: tuple[str, ...] = ("lat", "lon")
    site_dim: str = "entity"

    def attach(self, prepared: xr.Dataset) -> xr.Dataset:
        stacked = prepared.stack({self.site_dim: self.spatial_dims})

        param_coords: dict[str, xr.DataArray] = {}
        for name in self.parameters.data_vars:
            param = self.parameters[name]
            if set(self.spatial_dims).issubset(param.dims):
                param_coords[name] = param.stack({self.site_dim: self.spatial_dims})
            else:
                param_coords[name] = param
        return stacked.assign_coords(param_coords)


def _rechunk(dataset: xr.Dataset, chunks: Mapping[str, int] | None) -> xr.Dataset:
    """Rechunk so the physics sees the requested chunk widths, not the loader's."""

    if not chunks:
        return dataset
    sizes = {dim: min(width, dataset.sizes[dim]) for dim, width in chunks.items() if dim in dataset.sizes}
    return dataset.chunk(sizes) if sizes else dataset


def _drop_unused_dim_coords(dataset: xr.Dataset) -> xr.Dataset:
    """Drop coordinates whose dimensions no output variable uses.

    A gridded field attached for the kernel to read — a ``(month, entity)`` solar
    climatology — would otherwise leave a stray ``month`` dimension on the result.
    """

    if not dataset.data_vars:
        return dataset
    data_dims = set().union(*(set(var.dims) for var in dataset.data_vars.values()))
    unused = [name for name, coord in dataset.coords.items() if not set(coord.dims) <= data_dims]
    return dataset.drop_vars(unused) if unused else dataset
