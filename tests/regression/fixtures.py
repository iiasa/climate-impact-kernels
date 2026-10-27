"""Small made-up inputs for the end-to-end tests.

Everything here is built from fixed seeds and literal tables. Nothing calls the
library code under test, except where ``World`` derives a product (a cooling
mapping, a site evaluation) that later tests consume; those derivations are
themselves end-to-end tests.

The made-up input data is a 4 x 3 grid of 0.5 degree cells (lat 10.25-11.75,
lon 20.25-21.25) with a handful of countries, three R12 regions, a plant table
that exercises every fuel and cooling category, and a small ISIMIP3b-style tree
on disk. Edge regimes are injected at fixed cells and months so the physics
sees them: NaN river temperature, zero allowed temperature rise, relative humidity of
100 and above, zero and NaN discharge. Sea-surface temperature covers only the
south-east corner, so the saline plant on the land cell beside it draws from
the nearest ocean cell.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

LATS = [10.25, 10.75, 11.25, 11.75]
LONS = [20.25, 20.75, 21.25]
GCM = "GFDL-ESM4"
SCENARIOS = ("ssp126", "ssp585")
# The historical run of ISIMIP3b, which the present-day reference period (``REF-PD``)
# reads. That period is the season-years 1995-2014 and each year loads
# (year - 1, year); a 20-year site evaluation takes minutes, so the tests run it on
# the last three season-years (see ``World.short_reference_window``) and the tree
# holds one more year before them.
HISTORICAL = "historical"
REFERENCE_FIRST_YEAR = 2012
HISTORICAL_YEARS = (REFERENCE_FIRST_YEAR - 1, 2014)

# --- thermal plants -------------------------------------------------------------

PLANT_COLUMNS = [
    "UNITID", "LOCATIONID", "country", "lat", "lon", "mw", "has_coords",
    "fuel", "utype", "cool_imputed", "cool_source",
]  # fmt: skip

_PLANT_ROWS = [
    # UNITID, LOC, country, lat, lon, mw, has_coords, fuel, utype, cooling, source
    (1, 10, "Nigeria", 10.30, 20.30, 500.0, True, "COAL", "ST", "ot_fresh", "known"),
    (2, 10, "Nigeria", 10.30, 20.30, 250.0, True, "COAL", "ST", "ot_saline", "known"),
    (3, 11, "Nigeria", 10.60, 20.60, 600.0, True, "NUCLEAR", "ST", "cl_fresh", "known"),
    (4, 12, "France", 11.30, 20.30, 300.0, True, "LNG", "GT/C", "cl_fresh", "known"),
    (5, 12, "France", 11.30, 20.30, 120.0, True, "WSTH", "CCSS", "cl_fresh", "country_fuel"),
    (6, 13, "France", 11.70, 21.30, 200.0, True, "WSTH", "ST", "ot_fresh", "known"),
    (7, 14, "Nigeria", np.nan, np.nan, 90.0, False, "KERO", "ST", "cl_fresh", "country_fuel"),
    (8, 11, "Nigeria", np.nan, np.nan, 140.0, False, "UR", "ST", "cl_fresh", "known"),
    (9, 15, "Viet Nam", 10.70, 21.20, 400.0, True, "BIOMASS", "ST", "cl_fresh", "known"),
    (10, 16, "Viet Nam", 11.80, 20.80, 60.0, True, "GAS", "GT", "air", "engine_type"),
    (11, 17, "Nigeria", 10.30, 21.30, 350.0, True, "COAL", "ST", "air", "known"),
    (12, 18, "France", 11.30, 21.30, 280.0, True, "GEOTHERMAL", "ST", "air", "known"),
    (13, 19, "France", 11.80, 20.30, 450.0, True, "NUCLEAR", "ST", "air", "known"),
    (14, 20, "Nigeria", 10.60, 20.30, 150.0, True, "LNG", "CCGT", "air", "known"),
    (15, 21, "France", 11.70, 20.70, 40.0, True, "WASTE", "XYZ", "air", "engine_type"),
    (16, 22, "Nigeria", np.nan, np.nan, 75.0, None, "COAL", "ST", "air", "country_fuel"),
]  # fmt: skip


def plants() -> pd.DataFrame:
    frame = pd.DataFrame(_PLANT_ROWS, columns=PLANT_COLUMNS)
    frame["has_coords"] = pd.Series(
        [row[6] for row in _PLANT_ROWS], dtype=object
    )  # keeps the null for unit 16
    return frame


# --- climate on the grid --------------------------------------------------------


def _grid_noise(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    return rng.normal(0.0, scale, shape)


def monthly_climate() -> xr.Dataset:
    """Fifteen months, 2000-11 to 2002-01 (partial SON/DJF at both ends), with edge cases."""

    rng = np.random.default_rng(20260101)
    time = pd.date_range("2000-11-01", "2002-01-01", freq="MS")
    shape = (len(time), len(LATS), len(LONS))
    cycle = np.sin(2 * np.pi * (time.month.to_numpy() - 4) / 12.0)[:, None, None]
    cell = rng.uniform(-3.0, 3.0, (1, len(LATS), len(LONS)))
    tas = 298.0 + 8.0 * cycle + cell + _grid_noise(rng, shape, 0.5)
    tw = 293.0 + 5.0 * cycle + cell + _grid_noise(rng, shape, 0.4)
    dis = np.exp(rng.normal(1.0, 0.8, shape))
    hurs = np.clip(70.0 + 15.0 * rng.normal(size=shape), 20.0, 99.0)
    ps = 101325.0 - 900.0 * rng.uniform(0.0, 1.0, (1, len(LATS), len(LONS))) + _grid_noise(
        rng, shape, 40.0
    )
    # edge regimes
    tw[3, 0, 0] = np.nan  # NaN river temperature
    tw[5, 1, 1] = 305.15  # exactly the 32 C discharge cap: zero allowed rise
    tw[6, 1, 1] = 310.0  # above the cap: allowed rise clamped to zero
    tw[7, 1, 1] = 300.0  # allowed rise limited by the 10 K limit on the rise
    hurs[2, 2, 0] = 100.0  # saturated exactly
    hurs[4, 2, 0] = 105.0  # supersaturated
    hurs[6, 2, 0] = 0.5  # below the wet-bulb formula's floor
    hurs[8, 2, 0] = 1.0
    dis[4, 1, 0] = 0.0  # no flow
    dis[9, 3, 1] = np.nan  # missing flow
    # sea only at lat 10.25, lon 20.75-21.25: the saline plant's own cell is land
    tos = 26.0 + 3.0 * cycle + _grid_noise(rng, shape, 0.3)
    tos[:, 1:, :] = np.nan
    tos[:, 0, 0] = np.nan
    tos[5, 0, 1] = 32.0  # exactly the discharge cap: zero allowed rise for seawater
    tos[6, 0, 1] = 33.5  # above the cap
    tos[2, 0, 2] = np.nan  # a gap rules this cell out as an intake
    dataset = xr.Dataset(
        {
            "tw": (("time", "lat", "lon"), tw),
            "dis": (("time", "lat", "lon"), dis),
            "tas": (("time", "lat", "lon"), tas),
            "hurs": (("time", "lat", "lon"), hurs),
            "ps": (("time", "lat", "lon"), ps),
            "tos": (("time", "lat", "lon"), tos),
        },
        coords={"time": time, "lat": LATS, "lon": LONS},
    )
    for name in ("tw", "tas"):
        dataset[name].attrs["units"] = "K"
    dataset["ps"].attrs["units"] = "Pa"
    dataset["tos"].attrs["units"] = "degC"
    return dataset


def daily_air_temperature() -> xr.Dataset:
    """Daily ``tas`` from 2000-11-20 to 2002-01-10: partial edge seasons, leap-free."""

    rng = np.random.default_rng(20260102)
    time = pd.date_range("2000-11-20", "2002-01-10", freq="D")
    shape = (len(time), len(LATS), len(LONS))
    cycle = np.sin(2 * np.pi * (time.dayofyear.to_numpy() - 100) / 365.0)[:, None, None]
    cell = rng.uniform(-4.0, 4.0, (1, len(LATS), len(LONS)))
    tas = 296.0 + 12.0 * cycle + cell + rng.normal(0.0, 1.0, shape)
    tas[100, 0, 0] = 330.0  # very hot day: dry efficiency loss clipping
    tas[120, 1, 1] = np.nan
    dataset = xr.Dataset(
        {"tas": (("time", "lat", "lon"), tas)},
        coords={"time": time, "lat": LATS, "lon": LONS},
    )
    dataset["tas"].attrs["units"] = "K"
    return dataset


# --- CHILLED inputs -------------------------------------------------------------

_CELL_REGIONS = [
    ["AFR", "AFR", "AFR"],
    ["CPA", "CPA", "NAM"],
    ["NAM", "NAM", "WEU"],
    ["WEU", "PAS", "AFR"],  # PAS has no archetype row: its cell is dropped by the join
]
_ARCHETYPE_REGIONS = ["AFR", "CPA", "NAM", "WEU"]
_ARCHETYPE_COLUMNS = [
    "area_env", "gl_perc", "vol", "u_val", "ach_cl", "ach_op", "gn_int",
    "gl_g", "gl_sh", "roof_area", "roof_abs", "u_roof",
]  # fmt: skip


def chilled_spatial_units() -> pd.DataFrame:
    rows = []
    for i, lat in enumerate(LATS):
        for j, lon in enumerate(LONS):
            rows.append(
                {
                    "grid_lat": lat,
                    "grid_lon": lon,
                    "region": _CELL_REGIONS[i][j],
                    "grid_label": f"cell_{i}{j}",
                }
            )
    return pd.DataFrame(rows)


def chilled_archetypes() -> pd.DataFrame:
    rng = np.random.default_rng(20260103)
    rows = []
    for region in _ARCHETYPE_REGIONS:
        for building_type, urt, arch in (("sfh", "rur", "s1"), ("mfh", "urb", "s51")):
            params = dict(
                zip(
                    _ARCHETYPE_COLUMNS,
                    [
                        rng.uniform(3.0, 4.0),
                        rng.uniform(0.1, 0.2),
                        rng.uniform(2.5, 3.5),
                        rng.uniform(1.5, 3.5),
                        rng.uniform(0.3, 0.6),
                        rng.uniform(1.0, 2.0),
                        rng.uniform(2.0, 3.0),
                        rng.uniform(0.7, 0.9),
                        rng.uniform(0.6, 0.9),
                        rng.uniform(0.8, 1.2),
                        rng.uniform(0.5, 0.7),
                        rng.uniform(1.5, 3.5),
                    ],
                    strict=True,
                )
            )
            rows.append(
                {
                    "name": f"{region}_{building_type}_{arch}",
                    "building_type": building_type,
                    "urt": urt,
                    "arch": arch,
                    "region": region,
                    "permanent": 1,
                    **params,
                }
            )
    return pd.DataFrame(rows)


def chilled_solar() -> xr.Dataset:
    rng = np.random.default_rng(20260104)
    month = np.arange(1, 13)
    shape = (12, len(LATS), len(LONS))
    base = 120.0 + 40.0 * np.sin(2 * np.pi * (month - 3) / 12.0)[:, None, None]
    return xr.Dataset(
        {
            "solar_vertical": (("month", "lat", "lon"), base + rng.uniform(0, 20, shape)),
            "solar_horizontal": (("month", "lat", "lon"), 2 * base + rng.uniform(0, 30, shape)),
        },
        coords={"month": month, "lat": LATS, "lon": LONS},
    )


def chilled_setpoints() -> pd.DataFrame:
    """Region setpoints with CPA deliberately absent."""

    return pd.DataFrame(
        {
            "region": ["AFR", "NAM", "WEU"],
            "cooling_setpoint_c": [25.0, 23.0, 24.0],
            "cooling_max_c": [27.0, 25.0, 26.0],
        }
    )


def chilled_climate() -> xr.Dataset:
    """Sixteen days across a month boundary, with a sub-freezing and a very hot cell."""

    rng = np.random.default_rng(20260105)
    time = pd.date_range("2001-06-25", "2001-07-10", freq="D")
    shape = (len(time), len(LATS), len(LONS))
    tas = 292.0 + 9.0 * rng.uniform(-1, 1, (1, len(LATS), len(LONS))) + rng.normal(0, 2.0, shape)
    tas[3, 0, 0] = 255.0  # deep cold: heating branch
    tas[5, 3, 2] = 318.0  # extreme heat
    tas[7, 2, 1] = np.nan
    dataset = xr.Dataset(
        {"tas": (("time", "lat", "lon"), tas)},
        coords={"time": time, "lat": LATS, "lon": LONS},
    )
    dataset["tas"].attrs["units"] = "K"
    return dataset


def chilled_region_raster() -> xr.DataArray:
    values = np.asarray(_CELL_REGIONS, dtype="U8")
    values[3, 1] = ""  # a cell with no region
    return xr.DataArray(values, dims=("lat", "lon"), coords={"lat": LATS, "lon": LONS}, name="region")


# --- region rasters and definitions on disk ------------------------------------

_GAUL_IDS = np.array(
    [
        [1, 1, 2],
        [1, 2, 2],
        [3, 3, 5],
        [3, 4, 0],
    ]
)
_GAUL_ROWS = [
    # ID, ISO3, NAME
    (1, "NGA", "Nigeria"),
    (2, "CMR", "Cameroon"),
    (3, "FRA", "France"),
    (4, "VNM", "Viet Nam"),
    (5, "XXX", "Nowhere"),
]
_R12_YAML = {
    "World": {"child": ["R12_AFR", "R12_WEU", "R12_RCPA"]},
    "R12_AFR": {"child": ["NGA", "CMR"]},
    "R12_WEU": {"child": ["FRA"]},
    "R12_RCPA": {"child": ["VNM"]},
    "R12_EMPTY": {"description": "a node without children"},
}


def write_region_files(root: Path) -> dict[str, Path]:
    """GAUL raster (lat descending), its lookup CSV and an R12 node YAML."""

    root.mkdir(parents=True, exist_ok=True)
    raster = xr.DataArray(
        _GAUL_IDS[::-1].astype("float32"),
        dims=("lat", "lon"),
        coords={"lat": LATS[::-1], "lon": LONS},
        name="gaul",
    )
    paths = {
        "gaul_raster": root / "gaul.nc",
        "gaul_csv": root / "gaul.csv",
        "r12_yaml": root / "r12.yaml",
    }
    raster.to_netcdf(paths["gaul_raster"])
    pd.DataFrame(_GAUL_ROWS, columns=["ID", "ISO3", "NAME"]).to_csv(paths["gaul_csv"], index=False)
    paths["r12_yaml"].write_text(yaml.safe_dump(_R12_YAML), encoding="utf-8")
    return paths


def write_common_definitions(root: Path) -> Path:
    """A miniature common-definitions checkout with the two R12 files the loader reads."""

    codes = root / "mappings" / "MESSAGEix-GLOBIOM"
    countries = root / "definitions" / "region" / "native_regions" / "MESSAGEix-GLOBIOM"
    codes.mkdir(parents=True, exist_ok=True)
    countries.mkdir(parents=True, exist_ok=True)
    (codes / "MESSAGEix-GLOBIOM_2.1-R12.yaml").write_text(
        "model:\n  - MESSAGEix-GLOBIOM 2.1-M-R12\n\nnative_regions:\n"
        "  - R12_AFR: MESSAGEix-GLOBIOM 2.1-R12|Sub-Saharan Africa\n"
        "  - R12_RCPA: MESSAGEix-GLOBIOM 2.1-R12|Rest of Centrally Planned Asia\n"
        "  - R12_CHN: MESSAGEix-GLOBIOM 2.1-R12|China\n"
        "  - R12_NAM: MESSAGEix-GLOBIOM 2.1-R12|North America\n"
        "  - R12_PAO: MESSAGEix-GLOBIOM 2.1-R12|Pacific OECD\n"
        "  - R12_WEU: MESSAGEix-GLOBIOM 2.1-R12|Western Europe\n",
        encoding="utf-8",
    )
    (countries / "MESSAGEix-GLOBIOM_2.1-R12.yaml").write_text(
        "- MESSAGEix-GLOBIOM 2.1-R12:\n"
        "    - MESSAGEix-GLOBIOM 2.1-R12|Sub-Saharan Africa:\n"
        "        countries: [\n"
        "          Nigeria, Cameroon,\n"
        "          Democratic Republic of the Congo\n"
        "        ]\n"
        "    - MESSAGEix-GLOBIOM 2.1-R12|Rest of Centrally Planned Asia:\n"
        "        countries: [\n"
        "          Cambodia, Viet Nam\n"
        "        ]\n"
        "    - MESSAGEix-GLOBIOM 2.1-R12|China:\n"
        "        countries: [ China, Hong Kong ]\n"
        "    - MESSAGEix-GLOBIOM 2.1-R12|North America:\n"
        "        countries: [ Canada, United States ]\n"
        "    - MESSAGEix-GLOBIOM 2.1-R12|Pacific OECD:\n"
        "        countries: [ Japan, Australia ]\n"
        "    - MESSAGEix-GLOBIOM 2.1-R12|Western Europe:\n"
        "        countries: [\n"
        "          France, United Kingdom,\n"
        "          'Bonaire, Sint Eustatius and Saba'\n"
        "        ]\n",
        encoding="utf-8",
    )
    return root


@contextlib.contextmanager
def common_definitions(root: Path):
    """Point the region registry at the miniature definitions, then restore it."""

    from . import entrypoints

    previous = os.environ.get("COMMON_DEFINITIONS_ROOT")
    os.environ["COMMON_DEFINITIONS_ROOT"] = str(root)
    entrypoints.reset_region_caches()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("COMMON_DEFINITIONS_ROOT", None)
        else:
            os.environ["COMMON_DEFINITIONS_ROOT"] = previous
        entrypoints.reset_region_caches()


# --- ISIMIP-style tree ----------------------------------------------------------

_YEAR_BLOCKS = {
    "ssp126": ((2015, 2019), (2020, 2024)),
    "ssp585": ((2015, 2019), (2020, 2024)),
    HISTORICAL: ((2011, 2012), (2013, 2014)),
}
# Per scenario: random-stream offsets of the daily fields and of ``triver``, the
# warming per year of the daily air temperature and of ``triver`` (none under the
# historical run, whose reference period is fixed), and the first year the trends
# count from.
_SCENARIO_SEEDS = {"ssp126": (3, 5), "ssp585": (7, 11), HISTORICAL: (13, 17)}
_DAILY_TREND = {"ssp126": 0.1, "ssp585": 0.45, HISTORICAL: 0.0}
_TRIVER_TREND = {"ssp126": 0.08, "ssp585": 0.4, HISTORICAL: 0.0}
_FIRST_YEAR = {"ssp126": 2015, "ssp585": 2015, HISTORICAL: HISTORICAL_YEARS[0]}


def _daily_fields(gcm_seed: int, scenario: str) -> dict[str, xr.DataArray]:
    rng = np.random.default_rng(1000 * gcm_seed + _SCENARIO_SEEDS[scenario][0])
    first, last = _YEAR_BLOCKS[scenario][0][0], _YEAR_BLOCKS[scenario][-1][1]
    time = pd.date_range(f"{first}-01-01", f"{last}-12-31", freq="D")
    shape = (len(time), len(LATS), len(LONS))
    doy = time.dayofyear.to_numpy()[:, None, None]
    trend = ((time.year.to_numpy() - _FIRST_YEAR[scenario]) * _DAILY_TREND[scenario])[
        :, None, None
    ]
    cell = rng.uniform(-3.0, 3.0, (1, len(LATS), len(LONS)))
    tas = 297.0 + 9.0 * np.sin(2 * np.pi * (doy - 100) / 365.0) + cell + trend
    tas = tas + rng.normal(0.0, 1.2, shape)
    hurs = np.clip(65.0 + 20.0 * np.sin(2 * np.pi * (doy - 200) / 365.0) + rng.normal(0, 8, shape), 15.0, 100.0)
    ps = 101000.0 - 600.0 * rng.uniform(0, 1, (1, len(LATS), len(LONS))) + rng.normal(0, 30.0, shape)
    dis = np.exp(1.2 + 0.7 * np.sin(2 * np.pi * (doy - 150) / 365.0) + rng.normal(0, 0.5, shape))
    dis[:, 3, 1] = 0.0
    return {
        "tas": xr.DataArray(
            tas.astype("float32"), dims=("time", "lat", "lon"),
            coords={"time": time, "lat": LATS, "lon": LONS}, attrs={"units": "K"},
        ),
        "hurs": xr.DataArray(
            hurs.astype("float32"), dims=("time", "lat", "lon"),
            coords={"time": time, "lat": LATS, "lon": LONS}, attrs={"units": "%"},
        ),
        "ps": xr.DataArray(
            ps.astype("float32"), dims=("time", "lat", "lon"),
            coords={"time": time, "lat": LATS, "lon": LONS}, attrs={"units": "Pa"},
        ),
        "dis": xr.DataArray(
            dis.astype("float32"), dims=("time", "lat", "lon"),
            coords={"time": time, "lat": LATS, "lon": LONS}, attrs={"units": "m3 s-1"},
        ),
    }  # fmt: skip


def _monthly_triver(gcm_seed: int, scenario: str) -> xr.DataArray:
    rng = np.random.default_rng(1000 * gcm_seed + _SCENARIO_SEEDS[scenario][1])
    first, last = _YEAR_BLOCKS[scenario][0][0], _YEAR_BLOCKS[scenario][-1][1]
    time = pd.date_range(f"{first}-01-01", f"{last}-12-01", freq="MS")
    shape = (len(time), len(LATS), len(LONS))
    cycle = np.sin(2 * np.pi * (time.month.to_numpy() - 4) / 12.0)[:, None, None]
    trend = ((time.year.to_numpy() - _FIRST_YEAR[scenario]) * _TRIVER_TREND[scenario])[
        :, None, None
    ]
    cell = rng.uniform(-3.0, 3.0, (1, len(LATS), len(LONS)))
    tw = 292.0 + 6.0 * cycle + cell + trend + rng.normal(0.0, 0.5, shape)
    tw[:, 0, 0] += 6.0  # a warm river reaching the 32 C cap in late summers
    tw[:, 1, 2] = np.nan  # arid cell: no river temperature at all
    tw[:, 3, 1] = np.nan  # no hydrology under the Viet Nam cell
    if scenario == "ssp126":
        tw[:, 0, 2] = np.nan  # a cell that only has a river under ssp585
    tw[::2, 2, 0] = np.nan  # a river that only flows part of the year
    return xr.DataArray(
        tw.astype("float32"),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": LATS, "lon": LONS},
        attrs={"units": "K"},
        name="triver",
    )


def write_isimip_tree(root: Path, gcm: str = GCM, *, seed: int = 1) -> Path:
    """Write ``<root>/ISIMIP/ISIMIP3b/...`` for the made-up grid and return ``root``.

    Climate variables are daily, split in two files; ``dis`` is daily and
    ``triver`` monthly under the WaterGAP directory. Each scenario
    in ``SCENARIOS`` spans 2015-2024; the ``historical`` run spans 2011-2014,
    the window the present-day reference (``REF-PD``) reads here.
    """

    base = root / "ISIMIP" / "ISIMIP3b"
    gcm_lower = gcm.lower()
    for scenario in (*SCENARIOS, HISTORICAL):
        fields = _daily_fields(seed, scenario)
        first, last = _YEAR_BLOCKS[scenario][0][0], _YEAR_BLOCKS[scenario][-1][1]
        climate_dir = base / "InputData" / "climate_updated" / "bias-adjusted" / scenario / gcm
        water_dir = base / "OutputData" / "WaterGAP2-2e" / scenario / gcm
        climate_dir.mkdir(parents=True, exist_ok=True)
        water_dir.mkdir(parents=True, exist_ok=True)
        # ISIMIP3b historical WaterGAP runs sit under histsoc; SSPs under 2015soc here.
        soc = "histsoc" if scenario == "historical" else "2015soc"
        for start, end in _YEAR_BLOCKS[scenario]:
            window = slice(f"{start}-01-01", f"{end}-12-31")
            for name in ("tas", "hurs", "ps"):
                fields[name].sel(time=window).to_dataset(name=name).to_netcdf(
                    climate_dir / f"{gcm_lower}_r1i1p1f1_w5e5_{scenario}_{name}_global_daily_{start}_{end}.nc"
                )
            fields["dis"].sel(time=window).to_dataset(name="dis").to_netcdf(
                water_dir
                / f"watergap2-2e_{gcm_lower}_w5e5_{scenario}_{soc}_default_dis_global_daily_{start}_{end}.nc"
            )
        _monthly_triver(seed, scenario).to_dataset().to_netcdf(
            water_dir
            / f"watergap2-2e_{gcm_lower}_w5e5_{scenario}_{soc}_default_triver_global_monthly_{first}_{last}.nc"
        )
    return root


def write_gmt(root: Path, gcm: str = GCM) -> Path:
    """Monthly global-mean-temperature CSVs: historical to 2014, scenarios from mid-2014."""

    root.mkdir(parents=True, exist_ok=True)
    gcm_lower = gcm.lower()
    history = pd.date_range("1850-01-01", "2014-12-01", freq="MS")
    years = history.year.to_numpy() + (history.month.to_numpy() - 1) / 12.0
    anomaly = 0.9 * ((years - 1850.0) / 165.0) ** 2.2
    pd.DataFrame(
        {"date": history.strftime("%Y-%m-%d"), "mean_anomaly_1850_1900": anomaly}
    ).to_csv(root / f"{gcm_lower}_historical_monthly_temperature_gmt.csv", index=False)
    future = pd.date_range("2014-07-01", "2024-12-01", freq="MS")
    fyears = future.year.to_numpy() + (future.month.to_numpy() - 1) / 12.0
    for scenario, rate in (("ssp126", 0.05), ("ssp585", 0.30)):
        values = 0.9 + rate * (fyears - 2014.5)
        pd.DataFrame(
            {"date": future.strftime("%Y-%m-%d"), "mean_anomaly_1850_1900": values}
        ).to_csv(root / f"{gcm_lower}_{scenario}_monthly_temperature_gmt.csv", index=False)
    return root


def write_fleet_files(root: Path) -> tuple[Path, Path]:
    """Per-plant NetCDFs of the cells of today's fleet (wet and dry), as the audit reads them."""

    root.mkdir(parents=True, exist_ok=True)
    wet_cells = [(10.75, 20.75), (11.25, 20.25), (11.25, 20.25)]
    dry_cells = [(10.25, 21.25), (11.25, 20.25)]
    paths = []
    for label, cells in (("wet", wet_cells), ("dry", dry_cells)):
        path = root / f"fleet_{label}.nc"
        xr.Dataset(
            coords={
                "UNITID": ("entity", np.arange(len(cells))),
                "grid_lat": ("entity", [lat for lat, _ in cells]),
                "grid_lon": ("entity", [lon for _, lon in cells]),
            }
        ).to_netcdf(path)
        paths.append(path)
    return paths[0], paths[1]


def write_counterclim_tree(root: Path) -> Path:
    """A two-year ISIMIP3a counterclim layout (one daily tas file, one daily dis file)."""

    rng = np.random.default_rng(3)
    time = pd.date_range("1901-01-01", "1902-12-31", freq="D")
    shape = (len(time), len(LATS), len(LONS))
    coords = {"time": time, "lat": LATS, "lon": LONS}
    base = root / "ISIMIP" / "ISIMIP3a"
    climate = (
        base / "InputData" / "climate" / "counterclim" / "global" / "daily" / "historical" / "GSWP3-W5E5"
    )
    water = base / "OutputData" / "water_global" / "WaterGAP2-2e" / "gswp3-w5e5" / "historical"
    climate.mkdir(parents=True)
    water.mkdir(parents=True)
    tas = xr.DataArray(
        (290.0 + rng.normal(0, 3, shape)).astype("float32"), dims=("time", "lat", "lon"),
        coords=coords, attrs={"units": "K"},
    )  # fmt: skip
    dis = xr.DataArray(
        np.exp(rng.normal(1.0, 0.4, shape)).astype("float32"), dims=("time", "lat", "lon"),
        coords=coords, attrs={"units": "m3 s-1"},
    )  # fmt: skip
    tas.to_dataset(name="tas").to_netcdf(
        climate / "gswp3-w5e5_counterclim_tas_global_daily_1901_1902.nc"
    )
    for soc in ("2015soc", "histsoc"):
        dis.to_dataset(name="dis").to_netcdf(
            water / f"watergap2-2e_gswp3-w5e5_counterclim_{soc}_default_dis_global_daily_1901_1902.nc"
        )
    return root


# --- five-task export layout for the finalize step --------------------------------------------

FINALIZE_GCMS = ("GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL")
FINALIZE_REGIONS = tuple(f"R12_{name}" for name in (
    "AFR", "CHN", "EEU", "FSU", "LAM", "MEA", "NAM", "PAO", "PAS", "SAS", "WEU", "RCPA",
))  # fmt: skip
FINALIZE_PLANT_COOLING = (
    ("coal", "once_through"),
    ("coal", "recirculating"),
    ("gas", "once_through"),
    ("gas", "recirculating"),
)
FINALIZE_SEASONS = ("DJF", "MAM", "JJA", "SON")
FINALIZE_GWLS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
FINALIZE_DEPTH_RATIOS = (0.5, 1.0, 2.0)
FINALIZE_POINTS = 5
FINALIZE_CURVE_COLUMNS = [
    "population", "placement_rule", "gcm", "region_id", "plant_type", "cooling_type",
    "archetype", "season", "warming_level", "point_index", "capacity_axis_mw",
    "marginal_cell_id", "marginal_country", "marginal_capacity_mw", "marginal_usable_fraction",
    "thermal_loss_mw", "withdrawal_loss_mw", "usable_capacity_mw", "thermal_loss_fraction",
    "withdrawal_loss_fraction", "usable_fraction",
]  # fmt: skip


def _finalize_curves(gcm: str, levels: tuple, rng: np.random.Generator) -> pd.DataFrame:
    """One GCM task's potential-site capacity curves; marginal cells depend only on the curve point."""

    rows = []
    for region in FINALIZE_REGIONS:
        for plant, cooling in FINALIZE_PLANT_COOLING:
            archetype = f"{plant}_{cooling}"
            for season in FINALIZE_SEASONS:
                for level in levels:
                    for point in range(1, FINALIZE_POINTS + 1):
                        axis = 500.0 * (point - 1)
                        usable_fraction = float(rng.uniform(0.4, 1.0))
                        thermal = float(rng.uniform(0.0, 0.2)) * axis
                        usable = usable_fraction * axis
                        rows.append(
                            (
                                "candidate", "candidate_reach_aware", gcm, region, plant, cooling,
                                archetype, season, level, point, axis,
                                f"{gcm[:3]}_{region}_{archetype}_{point}", "AAA", 500.0 if point > 1 else 0.0,
                                usable_fraction, thermal, axis - thermal - usable, usable,
                                thermal / axis if axis else 0.0,
                                (axis - thermal - usable) / axis if axis else 0.0, usable_fraction,
                            )
                        )  # fmt: skip
    return pd.DataFrame(rows, columns=FINALIZE_CURVE_COLUMNS)


def _finalize_rime_samples(gcm: str, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for year in (2040, 2041):
        for region in FINALIZE_REGIONS:
            for plant, cooling in FINALIZE_PLANT_COOLING:
                for season in FINALIZE_SEASONS:
                    for level in FINALIZE_GWLS:
                        incumbent = float(rng.uniform(0.5, 1.0))
                        for ratio in FINALIZE_DEPTH_RATIOS:
                            candidate = float(rng.uniform(0.3, 1.0))
                            rows.append(
                                (gcm, "ssp585", year, region, plant, cooling, season, level,
                                 ratio, 1000.0, incumbent, candidate, incumbent + ratio * candidate)
                            )  # fmt: skip
    return pd.DataFrame(
        rows,
        columns=[
            "gcm", "scenario", "year", "region_id", "plant_type", "cooling_type", "season",
            "warming_level", "depth_ratio", "incumbent_capacity_mw", "incumbent_usable_fraction",
            "candidate_usable_fraction", "usable_capacity_ratio",
        ],
    )  # fmt: skip


def write_finalize_task_layout(root: Path) -> dict[str, Path]:
    """The five-GCM task directories of both calculations, as the finalize step finds them.

    ``root/gwl/tasks/<n>/`` holds a task's curves and RIME samples and
    ``root/reference/tasks/<n>/`` its present-day reference curves. Marginal cells depend on
    (GCM, region, synthetic plant, point) only, so the two calculations share them, and the RIME
    samples fill the packaged dataset's 12 x 2 x 2 x 4 x 6 x 3 grid.
    """

    rng = np.random.default_rng(906)
    roots = {"gwl": root / "gwl" / "tasks", "reference": root / "reference" / "tasks"}
    for index, gcm in enumerate(FINALIZE_GCMS):
        for arm, levels in (("gwl", FINALIZE_GWLS), ("reference", ("REF-PD",))):
            task = roots[arm] / str(index)
            task.mkdir(parents=True)
            _finalize_curves(gcm, levels, rng).to_parquet(
                task / "wet_ucr_curves_per_gcm.parquet", index=False
            )
        _finalize_rime_samples(gcm, rng).to_parquet(
            roots["gwl"] / str(index) / "wet_ucr_rime_samples.parquet", index=False
        )
    return roots


# --- flow-direction rasters for the river network --------------------------------------------

FLOWDIR_LATS = np.arange(9.25, 12.76, 0.5)
FLOWDIR_LONS = np.arange(19.25, 22.26, 0.5)


def _flow_codes(*, cyclic: bool) -> np.ndarray:
    """DDM30 codes on lat-ascending, lon-ascending axes for the made-up potential sites.

    Potential sites stand at (lat, lon) index (2,2) (2,3) (2,4) (3,2) (3,3) (4,2) (4,3)
    (5,2). The paths exercise: a step to a neighbouring cell, a path through an empty
    raster cell, an outlet (code 0), a missing code, and a path that leaves the raster.
    With ``cyclic`` a path from (4,3) enters a two-cell loop of empty raster cells.
    """

    codes = np.full((len(FLOWDIR_LATS), len(FLOWDIR_LONS)), 3.0)  # south everywhere
    codes[5, 2] = 3  # -> (4,2)
    codes[4, 2] = 1  # east -> (4,3)
    codes[4, 3] = 2  # south-east -> (3,4), empty
    codes[3, 4] = 1 if cyclic else 3  # east into the loop, or south -> (2,4)
    codes[3, 5] = 5  # west, back to (3,4)
    codes[3, 2] = 4  # south-west -> (2,1), empty
    codes[2, 1] = 9  # outlet
    codes[2, 2] = 0  # outlet
    codes[2, 3] = np.nan  # no code
    codes[2, 4] = 2  # south-east -> (1,5) -> south -> (0,5) -> off the raster
    return codes


def write_flow_directions(root: Path) -> dict[str, Path]:
    """Two flow-direction files (acyclic and with a loop), latitude stored descending."""

    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, cyclic in (("acyclic", False), ("cyclic", True)):
        raster = xr.DataArray(
            _flow_codes(cyclic=cyclic)[::-1],
            dims=("lat", "lon"),
            coords={"lat": FLOWDIR_LATS[::-1], "lon": FLOWDIR_LONS},
            name="flowdirection",
        )
        paths[name] = root / f"flowdir_{name}.nc"
        raster.to_dataset().to_netcdf(paths[name])
    return paths
