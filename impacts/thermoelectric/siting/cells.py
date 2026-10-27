"""Grid cells for the siting of new plants, and the synthetic plants placed on them.

The site evaluation is dense in space: every potential site is evaluated, so the
distribution of results over cells *is* the distribution over all potential
sites. This module lists the cells considered, the 0.5 degree cells a kernel could
place a plant on, and tags each cell with its R12 region, ISO3 country and an
exact spherical cell area weight (the only weight the site evaluation uses).

A potential site is a grid cell where a new plant could be built. An existing site
is a grid cell where today's fleet has plants. Both hold the same synthetic plants,
not the real units. A synthetic plant is a made-up plant of fixed type and size
(for example coal, 500 MW, once-through; code name ``archetype``), placed on every
cell so that cells can be compared.

The two kernels have different spatial domains, so they consider different cells:

- **Wet** cooling needs a river, so the wet cells are those where the ISIMIP
  hydrology field (``triver``) exists (``require_hydrology=True``).
- **Dry** cooling reads only air temperature (``tas``), which exists on all land,
  so the dry cells are every land cell in an R12 region: the full grid masked to
  land by the region tag (``require_hydrology=False``). Masking on the hydrology
  field would drop hot arid land (no modeled runoff) that dry cooling can use, and
  would bias the hot tail of the dry results low in the most heat-exposed regions.

The module also holds the synthetic plants of the wet and dry cooling kernels
(:class:`SyntheticPlant`), the table that places each of them on each sampled
cell (:func:`build_synthetic_sites`), and the functions that run the kernels on
that table.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from impacts.climatecore import (
    CLIMATOLOGICAL_SEASONS,
    aggregate_seasonal,
    attach_gridded_dataarray,
)
from impacts.engine import evaluate_mapped
from impacts.isimip import ISIMIPPaths, ISIMIPVariable, load_isimip_variable
from impacts.thermoelectric.dry import (
    DRY_COOLING_KERNEL,
    DryCoolingAssumptions,
)
from impacts.thermoelectric.wet import (
    EVAPORATION_COEFFICIENTS,
    HEAT_REJECTION_COEFFICIENTS,
    WET_COOLING_KERNEL,
    WetCoolingAssumptions,
    fuel_efficiency,
    water_limited_capacity_mw,
)

# Field whose grid defines the cells considered. The cells where it exists are the
# cells with hydrology data (the wet cells); the dry cells use only its lat/lon
# grid.
DEFAULT_MASK_VARIABLE = ISIMIPVariable(
    "triver", "watergap", "monthly", output_name="tw"
)

CELLS_CONSIDERED_COLUMNS = (
    "cell_id",
    "region_id",
    "country",
    "grid_lat",
    "grid_lon",
    "cell_area_m2",
)


@dataclass(frozen=True)
class RegionRasterPaths:
    """On-disk inputs for the GAUL -> ISO3 -> R12 region raster."""

    gaul_raster: Path
    gaul_csv: Path
    r12_yaml: Path


def build_region_country_rasters(
    paths: RegionRasterPaths,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return (region, country) string rasters on the GAUL 0.5 degree grid.

    ``region`` carries the bare R12 code (e.g. ``"AFR"``); ``country`` carries the
    ISO3 code. Cells with no R12 membership are the empty string.
    """

    with open(paths.r12_yaml, encoding="utf-8") as stream:
        r12_spec = yaml.safe_load(stream)
    iso_to_region: dict[str, str] = {}
    for node_id, definition in r12_spec.items():
        if (
            node_id == "World"
            or not isinstance(definition, dict)
            or "child" not in definition
        ):
            continue
        region = str(node_id).removeprefix("R12_")
        for iso3 in definition["child"]:
            iso_to_region[str(iso3)] = region

    gaul_csv = pd.read_csv(paths.gaul_csv)
    gaul_to_iso: dict[int, str] = {
        int(row["ID"]): str(row["ISO3"]) for _, row in gaul_csv.iterrows()
    }

    gaul = xr.open_dataarray(paths.gaul_raster).sortby("lat")
    ids = gaul.values
    region_values = np.full(ids.shape, "", dtype="U8")
    country_values = np.full(ids.shape, "", dtype="U8")
    for raster_id, iso3 in gaul_to_iso.items():
        region = iso_to_region.get(iso3)
        if region is None:
            continue
        mask = ids == raster_id
        region_values[mask] = region
        country_values[mask] = iso3

    region = xr.DataArray(
        region_values, dims=gaul.dims, coords=gaul.coords, name="region_id"
    )
    country = xr.DataArray(
        country_values, dims=gaul.dims, coords=gaul.coords, name="country"
    )
    return region, country


def build_valid_cells(
    paths: ISIMIPPaths,
    gcm: str,
    scenario: str | Sequence[str],
    region_paths: RegionRasterPaths,
    *,
    mask_variable: ISIMIPVariable = DEFAULT_MASK_VARIABLE,
    require_hydrology: bool = True,
    require_region: bool = True,
) -> pd.DataFrame:
    """List the cells considered (potential sites), tagged with R12 region, country and area.

    With ``require_hydrology`` true (the wet cells) a cell is kept where
    ``mask_variable`` is finite at any time step in any requested scenario (the
    mask that the ISIMIP hydrology data already carries). With it false (the dry
    cells) every grid cell is kept and the land mask falls out of the region tag. Each cell is tagged by
    sampling the region and country rasters at its centre and weighted by its
    spherical cell area. When ``require_region`` is true, cells with no R12
    membership (open ocean, a few coastal slivers the GAUL raster misses) are
    dropped, since results are reported per R12 region.
    """

    scenarios = _as_scenarios(scenario)
    valid = (
        _hydrology_valid_mask(paths, gcm, scenarios, mask_variable)
        if require_hydrology
        else xr.ones_like(
            load_isimip_variable(
                mask_variable,
                gcm,
                scenarios[0],
                paths,
                target_frequency="monthly",
                chunks={"time": -1},
            ).isel(time=0),
            dtype=bool,
        )
    ).load()

    lat_name = "lat" if "lat" in valid.dims else valid.dims[-2]
    lon_name = "lon" if "lon" in valid.dims else valid.dims[-1]
    stacked = valid.stack(cell=(lat_name, lon_name))
    stacked = stacked[stacked]
    cells = pd.DataFrame(
        {
            "grid_lat": stacked[lat_name].values.astype(float),
            "grid_lon": stacked[lon_name].values.astype(float),
        }
    )
    if cells.empty:
        raise ValueError(f"no valid cells for {mask_variable.name}/{gcm}/{scenario}")

    return tag_cells(cells, region_paths, require_region=require_region)


def _as_scenarios(scenario: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(scenario, str):
        return (scenario,)
    scenarios = tuple(str(item) for item in scenario)
    if not scenarios:
        raise ValueError("scenario sequence must not be empty")
    return scenarios


def _hydrology_valid_mask(
    paths: ISIMIPPaths,
    gcm: str,
    scenarios: Sequence[str],
    mask_variable: ISIMIPVariable,
) -> xr.DataArray:
    masks: list[xr.DataArray] = []
    for scenario in scenarios:
        field = load_isimip_variable(
            mask_variable,
            gcm,
            scenario,
            paths,
            target_frequency="monthly",
            chunks={"time": -1},
        )
        # River temperature is finite only in flowing months for many real wet
        # cells. Using one arbitrary month drops scenario-specific cells; the
        # rule is finite in any month of any configured scenario.
        masks.append(field.notnull().any(dim="time"))
    valid = masks[0]
    for mask in masks[1:]:
        valid = valid | mask
    return valid


def extract_existing_site_cells(
    wet_path: Path,
    dry_path: Path,
    region_paths: RegionRasterPaths,
    *,
    restrict_to: pd.DataFrame | None = None,
    require_region: bool = True,
) -> pd.DataFrame:
    """Return the distinct existing sites (the cells of today's fleet), tagged like the cells considered.

    Reads the per-plant files (wet + dry), takes the union of distinct
    ``(grid_lat, grid_lon)`` plant cells, and tags each with region, country, and
    spherical cell area. When ``restrict_to`` is a tagged table of the cells
    considered, fleet cells are intersected with it so that potential and existing
    sites are compared on the same domain (saline-coastal fleet cells outside the
    hydrology mask are dropped).
    """

    frames: list[pd.DataFrame] = []
    for path in (wet_path, dry_path):
        plants = xr.open_dataset(path, chunks={})
        frames.append(
            pd.DataFrame(
                {
                    "grid_lat": plants["grid_lat"].values.astype(float),
                    "grid_lon": plants["grid_lon"].values.astype(float),
                }
            )
        )
        plants.close()
    cells = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["grid_lat", "grid_lon"]
    )
    tagged = tag_cells(cells, region_paths, require_region=require_region)
    if restrict_to is not None:
        keep = set(restrict_to["cell_id"])
        tagged = tagged.loc[tagged["cell_id"].isin(keep)].reset_index(drop=True)
        if tagged.empty:
            raise ValueError("no fleet cells fall inside the cells considered")
    return tagged


def tag_cells(
    cells: pd.DataFrame,
    region_paths: RegionRasterPaths,
    *,
    resolution_deg: float = 0.5,
    require_region: bool = True,
) -> pd.DataFrame:
    """Attach cell_id, R12 region, ISO3 country, and spherical cell area to cells.

    ``cells`` needs ``grid_lat``/``grid_lon`` columns. Region and country come
    from sampling the GAUL/R12 rasters; the area weight is the analytic spherical
    cell area (identical formula to ``climatecore.grid_cell_area``) so a given
    cell carries the same weight in both sets of cells.
    """

    cells = cells[["grid_lat", "grid_lon"]].copy()
    region_raster, country_raster = build_region_country_rasters(region_paths)
    positions = cells.assign(position=np.arange(len(cells)))
    cells["region_id"] = attach_gridded_dataarray(
        positions, region_raster, site_id_column="position"
    ).values.astype(str)
    cells["country"] = attach_gridded_dataarray(
        positions, country_raster, site_id_column="position"
    ).values.astype(str)
    cells["cell_area_m2"] = _spherical_cell_area(
        cells["grid_lat"].to_numpy(), resolution_deg
    )
    cells["cell_id"] = (
        cells["grid_lat"].map(lambda v: f"{v:+08.3f}")
        + "_"
        + cells["grid_lon"].map(lambda v: f"{v:+09.3f}")
    )
    if require_region:
        cells = cells.loc[cells["region_id"].str.len() > 0].copy()
        if cells.empty:
            raise ValueError("no cells carry an R12 region label")
    cells = cells.sort_values("cell_id", kind="stable").reset_index(drop=True)
    if cells["cell_id"].duplicated().any():
        raise ValueError("cell identifiers are not unique")
    return cells[list(CELLS_CONSIDERED_COLUMNS)]


def _spherical_cell_area(
    grid_lat: np.ndarray,
    resolution_deg: float,
    *,
    earth_radius_m: float = 6_371_000.0,
) -> np.ndarray:
    """Spherical area (m2) of a regular ``resolution_deg`` cell at each latitude."""

    half = np.deg2rad(resolution_deg) / 2.0
    lat_rad = np.deg2rad(np.asarray(grid_lat, dtype=float))
    north = np.clip(lat_rad + half, -np.pi / 2.0, np.pi / 2.0)
    south = np.clip(lat_rad - half, -np.pi / 2.0, np.pi / 2.0)
    lon_width = np.deg2rad(resolution_deg)
    return earth_radius_m**2 * np.abs(np.sin(north) - np.sin(south)) * lon_width


# Month -> season, the inverse of the library's season definition. Used to label
# the monthly drivers with the same season and season-year that the seasonal
# water limit is aggregated onto, so that the later split of the wet capacity loss
# can group the months back onto their season and reproduce the mean-of-ratios
# water limit.
_SEASON_OF_MONTH: dict[int, str] = {
    month: season for season, months in CLIMATOLOGICAL_SEASONS.items() for month in months
}


def default_wet_synthetic_plants() -> tuple[SyntheticPlant, ...]:
    """Return the synthetic wet-cooling plants: coal 500 MW and gas 250 MW, each once-through and recirculating."""

    return tuple(
        _wet_archetype(fuel=fuel, mw=mw, cooling_physics=cooling_physics)
        for fuel, mw in (("coal", 500.0), ("gas", 250.0))
        for cooling_physics in ("once_through", "recirculating")
    )


def _wet_archetypes_named(
    names: Iterable[Any], synthetic_plants: Iterable[SyntheticPlant] | None = None
) -> dict[Any, SyntheticPlant]:
    """Return the synthetic wet plant of each name, keyed by name.

    Tables read back from a site evaluation or from the split of the wet capacity
    loss carry only the plant's name; its physical parameters (nameplate ``mw``,
    ``wet_fuel_type``, ``cooling_physics``) come from the plant definitions,
    ``synthetic_plants`` or :func:`default_wet_synthetic_plants` when ``None``, never from the
    name string.

    Raises:
        ValueError: naming the first name that is not one of the synthetic plants.
    """

    definitions = default_wet_synthetic_plants() if synthetic_plants is None else synthetic_plants
    known = {archetype.name: archetype for archetype in definitions}
    found: dict[Any, SyntheticPlant] = {}
    for name in names:
        archetype = known.get(name)
        if archetype is None:
            raise ValueError(f"unrecognized wet archetype: {name!r}")
        found[name] = archetype
    return found


def default_dry_synthetic_plants() -> tuple[SyntheticPlant, ...]:
    """Return one synthetic 500 MW dry-cooling plant for each steam-cycle family of Qin et al."""

    assumptions = DryCoolingAssumptions()
    definitions = (
        ("combustible_steam", "COAL", "steam", assumptions.combustible_steam_parasitic_load_fraction),
        ("combined_cycle", "GAS", "combined_cycle", assumptions.combined_cycle_parasitic_load_fraction),
        ("non_combustible_steam", "GEOTHERMAL", "steam", assumptions.non_combustible_steam_parasitic_load_fraction),
        ("nuclear_steam", "NUCLEAR", "steam", assumptions.nuclear_steam_parasitic_load_fraction),
    )
    return tuple(
        SyntheticPlant(
            name=f"qin_{engine_type}",
            kernel="dry_cooling",
            parameters={
                "mw": 500.0,
                "fuel": fuel,
                "utype": technology_family,
                "technology_family": technology_family,
                "cooling_category": "air",
                "cooling_physics": "dry",
                "dry_cooling_engine_type": engine_type,
                "dry_cooling_supported": True,
                "dry_cooling_support_reason": "supported_qin_family",
                "parasitic_load_fraction": parasitic_load,
            },
            provenance="Qin et al. (2023) supported engine family and default kernel assumptions",
        )
        for engine_type, fuel, technology_family, parasitic_load in definitions
    )


def build_synthetic_sites(
    sampled_cells: pd.DataFrame,
    synthetic_plants: Iterable[SyntheticPlant],
) -> pd.DataFrame:
    """Place every synthetic plant on every sampled cell: one row per cell and plant."""

    records: list[dict[str, object]] = []
    for cell in sampled_cells.to_dict(orient="records"):
        for archetype in synthetic_plants:
            entity_id = f"{cell['cell_id']}::{archetype.name}"
            records.append(
                {
                    **cell,
                    **archetype.parameters,
                    "UNITID": entity_id,
                    "LOCATIONID": str(cell["cell_id"]),
                    "archetype": archetype.name,
                    "archetype_provenance": archetype.provenance,
                }
            )
    sites = pd.DataFrame.from_records(records)
    if sites.empty:
        raise ValueError("at least one sampled cell and synthetic plant are required")
    if sites["UNITID"].duplicated().any():
        raise ValueError("synthetic plant identifiers are not unique")
    return sites


def _wet_monthly_with_ceiling(
    sites: pd.DataFrame,
    climate: xr.Dataset,
    assumptions: WetCoolingAssumptions | None,
) -> xr.Dataset:
    """Run the wet kernel at native monthly resolution and attach the monthly water limit.

    The water limit is the ratio of available discharge to withdrawal intensity;
    see :func:`evaluate_wet_potential_sites` for why it is formed monthly.
    """

    monthly = evaluate_mapped(
        WET_COOLING_KERNEL,
        sites,
        climate,
        duration="native",
        site_id_column="UNITID",
        assumptions=assumptions,
    )
    # A dry month has a water limit of zero whatever the intensity. A month with
    # unknown drivers (NaN discharge, or NaN intensity with water in the river)
    # has no water limit, so the seasonal mean skips it; the split of the wet
    # capacity loss leaves the same months out of its seasons.
    # The water limit is the cell's own, not capped at nameplate: a zero or
    # negative intensity leaves it unknown.
    ceiling = water_limited_capacity_mw(
        monthly["available_discharge_m3_per_s"],
        monthly["withdrawal_intensity_m3_per_mwh"],
        zero_intensity=np.nan,
        negative_intensity=np.nan,
        zero_discharge=0.0,
    )
    return monthly.assign(
        water_limited_capacity_ceiling_mw=ceiling.rename(
            "water_limited_capacity_ceiling_mw"
        )
    )


def evaluate_wet_potential_sites(
    sites: pd.DataFrame,
    climate: xr.Dataset,
    *,
    assumptions: WetCoolingAssumptions | None = None,
) -> xr.Dataset:
    """Evaluate the synthetic wet plants and return the cell's water limit, not capped at nameplate.

    The water limit is formed at native monthly resolution and aggregated to
    seasonal *after* the division. Taking the ratio of seasonal means instead
    would compute ratio-of-means, which differs from the mean-of-ratios the
    seasonal water limit is defined as whenever withdrawal intensity varies through
    the season (notably recirculating cooling, whose intensity is nonlinear in
    air temperature and humidity).
    """

    return aggregate_seasonal(_wet_monthly_with_ceiling(sites, climate, assumptions))


def evaluate_wet_potential_sites_seasonal_and_monthly(
    sites: pd.DataFrame,
    climate: xr.Dataset,
    *,
    assumptions: WetCoolingAssumptions | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Return the seasonal results (with the water limit) and the monthly drivers labelled by season.

    One kernel evaluation feeds both outputs. The seasonal dataset is exactly
    what :func:`evaluate_wet_potential_sites` returns (the mean-of-ratios water limit
    and the seasonal capacity factor). The monthly dataset carries the raw kernel
    drivers (available discharge, withdrawal intensity and the others) at their
    native monthly resolution, each month tagged with the season and season-year
    it belongs to (December rolled into the next year's DJF, matching
    :func:`aggregate_seasonal`) so that the later split of the wet capacity loss
    into thermal and water-availability parts (``split_wet_losses``) can
    group the months back onto the season the water limit uses.
    """

    monthly = _wet_monthly_with_ceiling(sites, climate, assumptions)
    return aggregate_seasonal(monthly), _label_monthly_seasons(monthly)


def _label_monthly_seasons(monthly: xr.Dataset, time_dim: str = "time") -> xr.Dataset:
    """Tag each month with its season name and rolled-forward season-year."""

    months = np.asarray(monthly[time_dim].dt.month.values, dtype=int)
    calendar_years = np.asarray(monthly[time_dim].dt.year.values, dtype=int)
    # Only December belongs to the following year's DJF; every other month keeps
    # its calendar year (the usual climatological convention that aggregate_seasonal
    # applies via its quarter-start year offset).
    season_years = calendar_years + (months == 12).astype(int)
    seasons = [_SEASON_OF_MONTH[int(month)] for month in months]
    return monthly.assign_coords(
        month=(time_dim, months),
        season=(time_dim, seasons),
        year=(time_dim, season_years),
    )


def evaluate_dry_potential_sites(sites: pd.DataFrame, climate: xr.Dataset) -> xr.Dataset:
    """Evaluate the capacity factor of the synthetic dry-cooling plants."""

    return evaluate_mapped(
        DRY_COOLING_KERNEL,
        sites,
        climate,
        duration="seasonal",
        site_id_column="UNITID",
    )


def _wet_archetype(*, fuel: str, mw: float, cooling_physics: str) -> SyntheticPlant:
    efficiency = fuel_efficiency(fuel, mw)
    cooling_category = "ot_fresh" if cooling_physics == "once_through" else "recirc_fresh"
    return SyntheticPlant(
        name=f"li_{fuel}_{int(mw)}mw_{cooling_physics}",
        kernel="wet_cooling",
        parameters={
            "mw": mw,
            "fuel": fuel.upper(),
            "wet_fuel_type": fuel,
            "cooling_category": cooling_category,
            "cooling_physics": cooling_physics,
            "thermal_efficiency": efficiency,
            "heat_rejection_coefficient": HEAT_REJECTION_COEFFICIENTS[fuel],
            "evaporation_coefficient": EVAPORATION_COEFFICIENTS[fuel],
            "cell_total_mw": mw,
            "unit_share_ratio": 1.0,
        },
        provenance="Li et al. (2025) default coefficients; efficiency derived from fuel and unit MW",
    )


@dataclass(frozen=True, slots=True)
class SyntheticPlant:
    """A synthetic plant: a named, serializable parameter set that an impact kernel reads."""

    name: str
    kernel: str
    parameters: dict[str, Any]
    provenance: str

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
