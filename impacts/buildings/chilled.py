"""CHILLED building-cooling kernel.

The physics is adapted from message-ix-buildings
(https://github.com/iiasa/message-ix-buildings,
``message_ix_buildings/chilled/functions/buildings_funcs_grid.py``), Apache
License 2.0, which implements the method of Mastrucci et al. (2019),
https://doi.org/10.1016/j.enbuild.2019.01.015. Changes from that code: the
equations are evaluated per day rather than per month, so monthly and seasonal
outputs are means of daily values (MJ m-2 day-1) rather than monthly sums, and
cooling setpoints can optionally be conditioned on region, which the source
does not do.

Holds only the kernel itself: the energy-intensity physics, the assumptions
dataclass, and the kernels. File input and output, validators, and grid and
mapping builders live in ``impacts.buildings.archetypes``; input locations are documented in
``impacts.buildings._config``.

Callers compose the kernel with ``impacts.engine.evaluate_mapped`` (mapped
DataFrame inputs) or ``impacts.engine.evaluate_gridded`` (gridded
parameter Dataset inputs); both reach the same physics.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from impacts.engine import ClimateInput, Duration, Kernel, evaluate_mapped

MASTRUCCI_2019_DOI = "doi:10.1016/j.enbuild.2019.01.015"
# Source of the default equipment and setpoint values: the CHILLED constants of
# message-ix-buildings. Mastrucci et al. (2019) is the source of the cooling
# method; these values are cited to the code that sets them.
MESSAGE_IX_BUILDINGS_CONSTANT = (
    "message-ix-buildings CHILLED constant (https://github.com/iiasa/message-ix-buildings, "
    "message_ix_buildings/chilled/functions/buildings_funcs_grid.py)"
)
MESSAGE_IX_BUILDINGS_CONFIG = (
    "message-ix-buildings CHILLED configuration (https://github.com/iiasa/message-ix-buildings, "
    "message_ix_buildings/chilled/util/config.py)"
)
# Heating is message-ix-buildings' extension of the cooling method, so its
# constants are that code's own, not Mastrucci et al.'s.
MESSAGE_IX_BUILDINGS_HEATING_CONSTANT = (
    f"{MESSAGE_IX_BUILDINGS_CONSTANT}; heating extends the cooling method of "
    f"Mastrucci et al. 2019 ({MASTRUCCI_2019_DOI}) and is not part of that paper"
)

MONTHS = tuple(range(1, 13))
# A mapped CHILLED site (``entity`` in the code) is one row of the spatial-unit x
# archetype mapping, which has no identifier of its own: callers of ``evaluate_mapped`` number the rows
# under this name and pass it as ``site_id_column``.
CHILLED_SITE_ID_COLUMN = "entity_id"
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
CHILLED_SITE_COLUMNS = (
    "spatial_unit_id",
    "region",
    "name",
    "urt",
    "arch",
    "permanent",
    *ARCHETYPE_PARAMETER_COLUMNS,
)
# Optional per-site (region-conditioned) cooling setpoints, carried as site
# values like the archetype parameters. Where a site has none, the scalars on
# ``ChilledEnergyAssumptions`` apply. The gridded path reads them whenever the
# parameter grids carry them; the mapped path needs ``CHILLED_REGION_SETPOINT_KERNEL``.
CHILLED_SETPOINT_COLUMNS = (
    "cooling_setpoint_c",
    "cooling_max_c",
)

# Volumetric heat capacity of air, rho c_p, J m-3 K-1, at typical indoor conditions.
AIR_VOLUMETRIC_HEAT_CAPACITY_J_PER_M3_K = 1200.0
SECONDS_PER_HOUR = 3600.0
# External surface resistance R_se, m2 K W-1, the standard value for an outer
# surface (EN ISO 6946), through which absorbed solar radiation reaches the roof.
EXTERNAL_SURFACE_RESISTANCE_M2_K_PER_W = 0.04
# One watt sustained for a day, in MJ.
MJ_PER_W_DAY = 86_400.0 / 1e6


@dataclass(frozen=True)
class ChilledEnergyAssumptions:
    """Equipment and setpoint parameters for the CHILLED kernel.

    The cooling setpoints can be overridden per site (``CHILLED_SETPOINT_COLUMNS``).
    """

    # Cooling balance setpoint.
    cooling_setpoint_c: float = field(default=26.0, metadata={"unit": "degC", "source": MESSAGE_IX_BUILDINGS_CONSTANT, "site": "cooling_setpoint_c"})
    # Maximum comfortable temperature, windows open.
    cooling_max_c: float = field(default=28.0, metadata={"unit": "degC", "source": MESSAGE_IX_BUILDINGS_CONSTANT, "site": "cooling_max_c"})
    # Heating setpoint.
    heating_setpoint_c: float = field(default=20.0, metadata={"unit": "degC", "source": MESSAGE_IX_BUILDINGS_HEATING_CONSTANT})
    # Share of hours the AC runs (f_c).
    cooling_hours_fraction: float = field(default=0.333, metadata={"unit": "1", "source": MESSAGE_IX_BUILDINGS_CONSTANT})
    # Share of hours the fans run (f_f).
    fan_hours_fraction: float = field(default=0.2, metadata={"unit": "1", "source": MESSAGE_IX_BUILDINGS_CONSTANT})
    # Share of hours the heating runs (f_h).
    heating_hours_fraction: float = field(default=1.0, metadata={"unit": "1", "source": MESSAGE_IX_BUILDINGS_HEATING_CONSTANT})
    # AC coefficient of performance.
    ac_cop: float = field(default=3.2, metadata={"unit": "1", "source": MESSAGE_IX_BUILDINGS_CONSTANT})
    # Heating system efficiency.
    heating_efficiency: float = field(default=0.9, metadata={"unit": "1", "source": MESSAGE_IX_BUILDINGS_HEATING_CONSTANT})
    # Power of one fan.
    fan_power_w: float = field(default=55.0, metadata={"unit": "W", "source": MESSAGE_IX_BUILDINGS_CONSTANT})
    # Floor area served by one fan.
    area_per_fan_m2: float = field(default=25.0, metadata={"unit": "m2", "source": MESSAGE_IX_BUILDINGS_CONFIG})

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


def chilled_physics(
    climate: xr.Dataset,
    sites: xr.Dataset,
    assumptions: ChilledEnergyAssumptions,
) -> xr.Dataset:
    """Daily cooling and heating energy intensity, MJ m-2 day-1.

    The variable degree-day method of Mastrucci et al. (2019) for cooling, applied
    to heating as message-ix-buildings does: a monthly balance temperature per archetype,
    from its envelope, ventilation, solar and internal gains, against which daily
    air temperature (C) accumulates degree-days. ``sites`` holds the archetype
    parameters (see ``ARCHETYPE_PARAMETER_COLUMNS``), the ``(month, entity)``
    solar climatology (W m-2) and, optionally, per-site setpoints.
    """

    if "time" not in climate.sizes:
        raise ValueError("climate inputs must include a time dimension")
    a = assumptions
    s = sites
    air_temperature = climate["air_temperature_c"].transpose("time", "entity")

    # Heat-loss coefficients, W K-1: transmission through the envelope, and
    # ventilation with windows closed (cl) and open (op).
    h_transmission = s["u_val"] * s["area_env"]
    h_ventilation_closed = AIR_VOLUMETRIC_HEAT_CAPACITY_J_PER_M3_K * s["vol"] * s["ach_cl"] / SECONDS_PER_HOUR
    h_ventilation_open = AIR_VOLUMETRIC_HEAT_CAPACITY_J_PER_M3_K * s["vol"] * s["ach_op"] / SECONDS_PER_HOUR

    # Monthly solar gain, W: through the glazing, and absorbed by the roof and
    # conducted in through its external surface resistance.
    solar_gain = (
        s["solar_vertical"] * s["gl_perc"] * s["gl_g"] * s["gl_sh"]
        + s["solar_horizontal"] * EXTERNAL_SURFACE_RESISTANCE_M2_K_PER_W * s["u_roof"] * s["roof_area"] * s["roof_abs"]
    )
    gains = s["gn_int"] + solar_gain

    # Balance temperatures, C: the outdoor temperature at which the gains hold the
    # building at the setpoint.
    cooling_setpoint = _site_or_default(s, "cooling_setpoint_c", a.cooling_setpoint_c)
    cooling_max = _site_or_default(s, "cooling_max_c", a.cooling_max_c)
    t_balance_heating = a.heating_setpoint_c - (gains / (h_transmission + h_ventilation_closed))
    t_balance_cooling = cooling_setpoint - (gains / (h_transmission + h_ventilation_closed))
    t_max_cooling = cooling_max - (gains / (h_transmission + h_ventilation_open))
    t_balance_heating = _project_monthly_to_daily(t_balance_heating, air_temperature)
    t_balance_cooling = _project_monthly_to_daily(t_balance_cooling, air_temperature)
    t_max_cooling = _project_monthly_to_daily(t_max_cooling, air_temperature)

    # Daily variable degree-days (K day) and day counts. A day with missing air
    # temperature counts as no demand, as in message-ix-buildings.
    vdd_heating = _positive_part(t_balance_heating, air_temperature)
    vdd_cooling = _positive_part(air_temperature, t_max_cooling)
    cooling_days = (air_temperature - t_max_cooling > 0.0).astype(float)  # Nd: AC on, windows shut
    fan_days = (air_temperature > t_balance_cooling).astype(float)  # Nf: fans on

    # Energy need and delivered energy, MJ m-2 day-1.
    cooling_need = (
        (h_transmission + h_ventilation_closed)
        * a.cooling_hours_fraction
        * (vdd_cooling + ((t_max_cooling - t_balance_cooling) * cooling_days))
        * MJ_PER_W_DAY
    )
    heating_need = (h_transmission + h_ventilation_closed) * a.heating_hours_fraction * vdd_heating * MJ_PER_W_DAY
    cooling_energy_ac = cooling_need / a.ac_cop
    cooling_energy_fan = a.fan_hours_fraction * a.fan_power_w * fan_days * MJ_PER_W_DAY / a.area_per_fan_m2

    return xr.Dataset(
        {
            "vdd_h": vdd_heating,
            "vdd_tmax_c": vdd_cooling,
            "Nd": cooling_days,
            "Nf": fan_days,
            "cooling_energy_ac": cooling_energy_ac,
            "cooling_energy_fan": cooling_energy_fan,
            "cooling_energy_total": cooling_energy_ac + cooling_energy_fan,
            "heating_energy": heating_need / a.heating_efficiency,
        }
    )


# Chunk widths the CHILLED physics is tuned for: a year of days by 10,000 sites.
# A speed setting, not physics: the kernels rechunk to it unless the
# evaluation passes other ``chunks``.
CHILLED_CHUNKS = {"time": 365, "entity": 10_000}

CHILLED_KERNEL = Kernel(
    name="chilled_buildings",
    inputs={"air_temperature_c": ClimateInput("tas", unit="C")},
    physics=chilled_physics,
    site_columns=CHILLED_SITE_COLUMNS,
    assumptions=ChilledEnergyAssumptions(),
    attrs={
        "cooling_model": "CHILLED",
        "cooling_model_citation": (
            "Mastrucci, A., Byers, E., Pachauri, S. & Rao, N. D. Improving the "
            "SDG energy poverty targets: Residential cooling needs in the Global "
            "South. Energy and Buildings 186, 405-415. "
            "https://doi.org/10.1016/j.enbuild.2019.01.015 (variable degree-day "
            "method; heating applies the same balance to heating degree-days)."
        ),
        "climate_contract": "daily tas only; solar climatology carried as static monthly metadata",
        "energy_units": "MJ/m2/day",
    },
    chunks=CHILLED_CHUNKS,
    entity_coords_first=True,
)
# Mapped-path variant that reads the setpoint columns, which the mapping then
# must hold (``build_chilled_archetype_mapping(setpoints=...)``).
CHILLED_REGION_SETPOINT_KERNEL = dataclasses.replace(
    CHILLED_KERNEL, site_columns=(*CHILLED_SITE_COLUMNS, *CHILLED_SETPOINT_COLUMNS)
)


def chilled_energy(
    spatial_units: pd.DataFrame,
    archetypes: pd.DataFrame,
    solar_climatology: xr.Dataset,
    climate: xr.Dataset,
    duration: Duration,
    *,
    setpoints: pd.DataFrame | None = None,
    assumptions: ChilledEnergyAssumptions | None = None,
    seasons: Mapping[str, tuple[int, ...]] | None = None,
    chunks: Mapping[str, int] | None = None,
    variable_names: Mapping[str, str] | None = None,
    variable_units: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """CHILLED energy intensity of spatial units x archetypes, in one call.

    Pairs each spatial unit with every archetype of its region, using the archetype table
    (``build_chilled_archetype_mapping``, which also takes ``setpoints``) and
    evaluates the CHILLED kernel at ``duration`` with the solar climatology
    sampled per site. For the steps one at a time, call that function and
    ``impacts.engine.evaluate_mapped``; for a grid, use
    ``build_chilled_parameter_grids`` and ``impacts.engine.evaluate_gridded``.
    ``chunks=None`` uses ``CHILLED_CHUNKS``.
    """

    # The mapping builder lives with the file I/O, which imports this module.
    from impacts.buildings.archetypes import build_chilled_archetype_mapping

    mapping, solar = build_chilled_archetype_mapping(
        spatial_units, archetypes, solar_climatology, setpoints=setpoints
    )
    return evaluate_mapped(
        CHILLED_KERNEL if setpoints is None else CHILLED_REGION_SETPOINT_KERNEL,
        mapping.assign(**{CHILLED_SITE_ID_COLUMN: np.arange(len(mapping))}),
        climate,
        duration,
        site_id_column=CHILLED_SITE_ID_COLUMN,
        assumptions=assumptions,
        gridded_coords=solar,
        seasons=seasons,
        chunks=chunks,
        variable_names=variable_names,
        variable_units=variable_units,
    )


def _positive_part(reference: xr.DataArray, baseline: xr.DataArray) -> xr.DataArray:
    """``reference - baseline`` where it is positive, else 0 (also 0 where either is NaN)."""

    return xr.where(reference >= baseline, reference - baseline, 0.0)


def _project_monthly_to_daily(monthly: xr.DataArray, daily_template: xr.DataArray) -> xr.DataArray:
    """Broadcast a (month, entity) array onto the daily time axis of `daily_template`.

    Chunk on entity before the month→time broadcast so the (time, entity) result
    stays dask-backed, then rechunk on time: the isel propagates the month dim's
    chunk width onto the new time dim, mis-aligning with the daily chunk width and
    forcing an intersection rechunk during xr.where that inflates task count. The
    chunk widths follow `daily_template`, which the evaluation has already
    rechunked (to ``CHILLED_CHUNKS`` unless the caller passed other ``chunks``).
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


def _site_or_default(sites: xr.Dataset, name: str, default: float) -> float | xr.DataArray:
    """A per-site value, or ``default`` where the site has none.

    Returns the scalar ``default`` when the sites carry no such value. When they
    do, sites with a missing value (e.g. a region absent from the setpoint table)
    take ``default``, so the scalar assumption still applies at those sites.
    """

    if name not in sites.data_vars:
        return default
    return sites[name].where(sites[name].notnull(), default)
