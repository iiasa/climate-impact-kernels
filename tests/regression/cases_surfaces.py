"""End-to-end tests of the warming-level summaries (buildings and thermoelectric).

The stored buildings output was not produced by the library. It holds the output
of an independent build script for the summary, run on the made-up inputs of
``fixtures_surfaces``. The test is marked ``frozen`` (hand-maintained), so the
update command (``IMPACTS_REGRESSION_REGENERATE``) skips it. The stored
thermoelectric output is written from the library, because the independent script
averages an incomplete first DJF from January and February where the library
leaves that season missing.

Only the attributes of the population-weighted buildings files are compared. The
land-area-weighted variant's numbers match the library given land-area weights,
but its attributes differ from the library's and are not compared.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from . import entrypoints as ep
from . import fixtures
from .fixtures_surfaces import SurfaceInputs, write_surface_inputs
from .registry import case

SCENARIOS = ("historical", "ssp126", "ssp370", "ssp585")

# Source text the buildings warming-level summaries carry in their attributes.
MATRIX_VERSION = (
    "seasonal_v2: impacts 0.1.15 CHILLED kernel, region-conditioned setpoints "
    "(ASHRAE revealed AC-mode, R12 setpoint table), cop=1, f_c=1, thermal-normalized"
)
WEIGHTING = (
    "population-weighted country->R12: weight = ssp2_2020 popdensity (persons/km2, "
    "0.5deg) x cos-lat cell area summed within each GAUL country mask; NaN-aware "
    "mean across member countries"
)

_INPUTS: dict[Path, SurfaceInputs] = {}


def surface_inputs(w) -> SurfaceInputs:
    if w.root not in _INPUTS:
        _INPUTS[w.root] = write_surface_inputs(w.root / "surfaces")
    return _INPUTS[w.root]


def open_reference(inputs: SurfaceInputs, name: str) -> xr.Dataset:
    with xr.open_dataset(inputs.annual_dir / name) as dataset:
        return dataset.load()


def library_assignments(inputs: SurfaceInputs) -> xr.Dataset:
    return ep.get("isimip.isimip_warming_level_assignments")(
        inputs.gmt_root, inputs.gcms, SCENARIOS
    )


def library_buildings(w, *, weights: str = "population") -> dict:
    """Library result for the buildings warming-level summaries; ``weights`` is population or land_area."""

    inputs = surface_inputs(w)
    with fixtures.common_definitions(w.definitions_root):
        population = ep.get("surfaces.country_population")(
            inputs.country_raster, inputs.popdensity
        )
        # The stored map is resolved against miniature definitions, so it holds
        # only the made-up countries; compare the packaged map on those.
        packaged = ep.get("surfaces.load_iso3_to_r12")()
        synthetic = sorted(set(pd.read_parquet(inputs.buildings_parquet, columns=["country"])["country"]))
        iso3_to_r12 = {iso3: packaged[iso3] for iso3 in synthetic if iso3 in packaged}
        land_area = ep.get("surfaces.country_land_area")(inputs.country_raster)
        surfaces = ep.get("surfaces.buildings")(
            inputs.buildings_parquet,
            open_reference(inputs, "region_EI_cool_gwl_binned.nc"),
            open_reference(inputs, "region_EI_heat_gwl_binned.nc"),
            iso3_to_r12,
            population if weights == "population" else land_area,
            library_assignments(inputs),
            matrix_version=MATRIX_VERSION,
            weighting=WEIGHTING,
        )
    return {
        "iso3_to_r12": iso3_to_r12,
        "population": population,
        "land_area": land_area,
        **{f"surface.{key}": value for key, value in surfaces.items()},
    }


def library_thermoelectric(w) -> dict:
    inputs = surface_inputs(w)
    countries = sorted(pd.read_parquet(inputs.thermo_parquet, columns=["country"])["country"].unique())
    with fixtures.common_definitions(w.definitions_root):
        name_to_r12 = ep.get("surfaces.build_country_name_to_r12")(countries)
        surface = ep.get("surfaces.thermoelectric")(
            inputs.thermo_parquet,
            open_reference(inputs, "r12_thermoelectric_gwl.nc"),
            library_assignments(inputs),
        )
    return {"name_to_r12": name_to_r12, "surface": surface}


@case("surfaces.buildings", frozen=True)
def surfaces_buildings(w):
    return library_buildings(w)


@case("surfaces.thermoelectric")
def surfaces_thermoelectric(w):
    return library_thermoelectric(w)
