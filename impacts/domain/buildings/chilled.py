"""CHILLED building-cooling kernel.

Owns only the kernel contract: the energy-intensity physics, the climate-input
normalizer, the assumptions dataclass, and the kernel factory. File I/O,
validators, and grid/mapping builders live in
``impacts.domain.buildings.chilled_io``; path defaults live in
``impacts.domain.buildings.chilled_config``.

Callers compose the kernel with ``impacts.kernel.evaluate_kernel`` (mapped
DataFrame inputs) or ``impacts.kernel.evaluate_gridded_kernel`` (gridded
parameter Dataset inputs); both reach the same physics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr
from impacts.climatecore import EvaluationResult, normalize_temperature

from impacts.kernel import ImpactKernel, normalize_climate_inputs

DEFAULT_CHILLED_VARIABLE_NAMES = {
    "air_temperature_c": "tas",
}

REQUIRED_CHILLED_CLIMATE_VARIABLES = frozenset({"air_temperature_c"})

MONTHS = tuple(range(1, 13))
ARCHETYPE_PARAMETER_COLUMNS = (
    "area_env",
    "gl_perc",
    "vol",
    "u_val",
    "ach_cl",
    "ach_op",
    "gn_int",
    "gl_g",
    "gl_sh",
    "roof_area",
    "roof_abs",
    "u_roof",
)
CHILLED_METADATA_COLUMNS = (
    "spatial_unit_id",
    "region",
    "name",
    "urt",
    "arch",
    "permanent",
    *ARCHETYPE_PARAMETER_COLUMNS,
)

# Air density × specific heat / seconds-per-hour for the ventilation
# coefficient `h_v = ρ·c_p·V·ach / 3600`. 1200 J/(m³·K) covers ρ·c_p for air
# at typical indoor conditions.
_AIR_VOLUMETRIC_HEAT_CAPACITY = 1200.0
# Roof absorptance vs vertical-aperture solar gain weighting in the
# CHILLED archetype model.
_ROOF_HORIZONTAL_WEIGHT = 0.04
# Seconds per day, divided by MJ-per-Joule, for converting instantaneous
# power (W = J/s) into a daily energy total in MJ.
_DAILY_POWER_TO_MJ = 86_400.0 / 1e6


@dataclass(frozen=True)
class ChilledEnergyAssumptions:
    """Equipment and setpoint parameters for the CHILLED kernel."""

    cooling_setpoint_c: float = 26.0
    cooling_max_c: float = 28.0
    heating_setpoint_c: float = 20.0
    cooling_hours_fraction: float = 0.333
    fan_hours_fraction: float = 0.2
    heating_hours_fraction: float = 1.0
    ac_cop: float = 3.2
    heating_efficiency: float = 0.9
    fan_power_w: float = 55.0
    area_per_fan_m2: float = 25.0

    @classmethod
    def from_parametric_row(cls, row: pd.Series | Mapping[str, object]) -> ChilledEnergyAssumptions:
        values = dict(row)
        return cls(
            cooling_setpoint_c=float(values["t_sp_c"]),
            cooling_max_c=float(values["t_sp_c_max"]),
            heating_setpoint_c=float(values["t_sp_h"]),
            cooling_hours_fraction=float(values["f_c"]),
            fan_hours_fraction=float(values["f_f"]),
            heating_hours_fraction=float(values["f_h"]),
            ac_cop=float(values["cop"]),
            heating_efficiency=float(values["eff"]),
        )


CHILLED_NORMALIZATIONS = (
    ("air_temperature_c", "C", normalize_temperature),
)


def normalize_chilled_climate_inputs(
    dataset: xr.Dataset,
    *,
    variable_names: Mapping[str, str] | None = None,
    variable_units: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Select, rename, and normalize the daily climate inputs required by CHILLED."""

    return normalize_climate_inputs(
        dataset,
        default_variable_names=DEFAULT_CHILLED_VARIABLE_NAMES,
        normalizations=CHILLED_NORMALIZATIONS,
        variable_names=variable_names,
        variable_units=variable_units,
    )


def create_chilled_kernel(
    assumptions: ChilledEnergyAssumptions | None = None,
    *,
    entity_chunk: int = 10_000,
    time_chunk: int = 365,
) -> ImpactKernel:
    """Build the CHILLED production kernel on the shared mapped-kernel contract.

    ``entity_chunk`` and ``time_chunk`` control the dask rechunk at kernel
    entry. They are surfaced as explicit args so a calling script can tune
    them per-run (e.g. via env vars) without mutating module state.
    """

    params = assumptions or ChilledEnergyAssumptions()
    return ImpactKernel(
        name="chilled_buildings",
        normalize=normalize_chilled_climate_inputs,
        physics=partial(_chilled_physics, assumptions=params),
        metadata_columns=CHILLED_METADATA_COLUMNS,
        entry_chunks={"time": time_chunk, "entity": entity_chunk},
    )


def _chilled_physics(
    inputs: xr.Dataset,
    *,
    assumptions: ChilledEnergyAssumptions,
) -> EvaluationResult:
    if "time" not in inputs.sizes:
        raise ValueError("climate inputs must include a time dimension")

    air_temperature = inputs["air_temperature_c"].transpose("time", "entity")
    solar_vertical = inputs.coords["solar_vertical"]
    solar_horizontal = inputs.coords["solar_horizontal"]

    gl_perc = _entity_coord(inputs, "gl_perc")
    gl_g = _entity_coord(inputs, "gl_g")
    gl_sh = _entity_coord(inputs, "gl_sh")
    roof_area = _entity_coord(inputs, "roof_area")
    roof_abs = _entity_coord(inputs, "roof_abs")
    u_roof = _entity_coord(inputs, "u_roof")
    u_val = _entity_coord(inputs, "u_val")
    area_env = _entity_coord(inputs, "area_env")
    vol = _entity_coord(inputs, "vol")
    ach_cl = _entity_coord(inputs, "ach_cl")
    ach_op = _entity_coord(inputs, "ach_op")
    gn_int = _entity_coord(inputs, "gn_int")

    gn_sol = _solar_gain(
        solar_vertical, solar_horizontal,
        gl_perc=gl_perc, gl_g=gl_g, gl_sh=gl_sh,
        roof_area=roof_area, roof_abs=roof_abs, u_roof=u_roof,
    )
    h_tr = _transmission_loss_coefficient(u_val, area_env)
    h_v_cl = _ventilation_loss_coefficient(vol, ach_cl)
    h_v_op = _ventilation_loss_coefficient(vol, ach_op)

    gn_total = gn_int + gn_sol
    t_bal_h = _balance_temperature(assumptions.heating_setpoint_c, gn_total, h_tr + h_v_cl)
    t_bal_c = _balance_temperature(assumptions.cooling_setpoint_c, gn_total, h_tr + h_v_cl)
    t_max_c = _balance_temperature(assumptions.cooling_max_c, gn_total, h_tr + h_v_op)

    t_bal_h_daily = _project_monthly_to_daily(t_bal_h, air_temperature)
    t_bal_c_daily = _project_monthly_to_daily(t_bal_c, air_temperature)
    t_max_c_daily = _project_monthly_to_daily(t_max_c, air_temperature)

    vdd_h = _degree_days(t_bal_h_daily, air_temperature)
    vdd_tmax_c = _degree_days(air_temperature, t_max_c_daily)
    nd = (air_temperature - t_max_c_daily > 0.0).astype(float)
    nf = (air_temperature > t_bal_c_daily).astype(float)

    q_c = (
        (h_tr + h_v_cl)
        * assumptions.cooling_hours_fraction
        * (vdd_tmax_c + ((t_max_c_daily - t_bal_c_daily) * nd))
        * _DAILY_POWER_TO_MJ
    )
    q_h = (h_tr + h_v_cl) * assumptions.heating_hours_fraction * vdd_h * _DAILY_POWER_TO_MJ

    cooling_energy_ac = q_c / assumptions.ac_cop
    cooling_energy_fan = (
        assumptions.fan_hours_fraction
        * assumptions.fan_power_w
        * nf
        * _DAILY_POWER_TO_MJ
        / assumptions.area_per_fan_m2
    )
    heating_energy = q_h / assumptions.heating_efficiency

    return EvaluationResult(
        data_vars={
            "vdd_h": vdd_h,
            "vdd_tmax_c": vdd_tmax_c,
            "Nd": nd,
            "Nf": nf,
            "cooling_energy_ac": cooling_energy_ac,
            "cooling_energy_fan": cooling_energy_fan,
            "cooling_energy_total": cooling_energy_ac + cooling_energy_fan,
            "heating_energy": heating_energy,
        },
        attrs={
            "cooling_model": "CHILLED",
            "cooling_model_citation": (
                "Mastrucci, A. et al. Heating and cooling degree-day sensitivity "
                "with CHILLED archetype energy-intensity equations."
            ),
            "climate_contract": "daily tas only; solar climatology carried as static monthly metadata",
            "energy_units": "MJ/m2/day",
        },
    )


def _solar_gain(
    solar_vertical: xr.DataArray,
    solar_horizontal: xr.DataArray,
    *,
    gl_perc: xr.DataArray,
    gl_g: xr.DataArray,
    gl_sh: xr.DataArray,
    roof_area: xr.DataArray,
    roof_abs: xr.DataArray,
    u_roof: xr.DataArray,
) -> xr.DataArray:
    """Combined solar heat gain from vertical apertures and the roof."""

    return (
        solar_vertical * gl_perc * gl_g * gl_sh
        + solar_horizontal * _ROOF_HORIZONTAL_WEIGHT * u_roof * roof_area * roof_abs
    )


def _transmission_loss_coefficient(u_val: xr.DataArray, area_env: xr.DataArray) -> xr.DataArray:
    """Envelope transmission heat-loss coefficient ``U·A``."""

    return u_val * area_env


def _ventilation_loss_coefficient(vol: xr.DataArray, ach: xr.DataArray) -> xr.DataArray:
    """Ventilation heat-loss coefficient ``ρ·c_p·V·ach / 3600``."""

    return _AIR_VOLUMETRIC_HEAT_CAPACITY * vol * ach / 3600.0


def _balance_temperature(
    setpoint_c: float,
    gain_total: xr.DataArray,
    loss_coefficient: xr.DataArray,
) -> xr.DataArray:
    """Outdoor temperature at which internal gains exactly offset losses to the setpoint."""

    return setpoint_c - (gain_total / loss_coefficient)


def _degree_days(reference: xr.DataArray, baseline: xr.DataArray) -> xr.DataArray:
    """Positive part of ``reference - baseline`` (degree-day-style accumulator)."""

    return xr.where(reference >= baseline, reference - baseline, 0.0)


def _project_monthly_to_daily(monthly: xr.DataArray, daily_template: xr.DataArray) -> xr.DataArray:
    """Broadcast a (month, entity) array onto the daily time axis of `daily_template`.

    Chunk on entity before the month→time broadcast so the (time, entity) result
    stays dask-backed, then rechunk on time: the isel propagates the month dim's
    chunk width onto the new time dim, mis-aligning with the daily chunk width and
    forcing an intersection rechunk during xr.where that inflates task count. The
    chunk widths follow `daily_template`, which the kernel-entry rechunk has
    already sized (see ``ImpactKernel.entry_chunks``).
    """

    time_index = daily_template.indexes["time"]
    month_index = np.asarray(time_index.month - 1, dtype=np.int32)
    indexer = xr.DataArray(month_index, dims=("time",))
    entity_chunk = _chunk_width(daily_template, "entity")
    time_chunk = _chunk_width(daily_template, "time")
    return monthly.chunk({"entity": entity_chunk}).isel(month=indexer).chunk({"time": time_chunk})


def _chunk_width(template: xr.DataArray, dim: str) -> int:
    """Single chunk width for `dim`, falling back to the full size when unchunked."""

    sizes = template.chunksizes.get(dim)
    return int(sizes[0]) if sizes else int(template.sizes[dim])


def _entity_coord(inputs: xr.Dataset, name: str) -> xr.DataArray:
    return xr.DataArray(inputs.coords[name].data, dims=("entity",), coords={"entity": inputs.coords["entity"]})
