"""End-to-end tests of ISIMIP loading, the cells considered, the audit, and the wet-cooling output files."""

from __future__ import annotations

import contextlib
import os

import pandas as pd

from . import entrypoints as ep
from . import fixtures
from .registry import case, outcome, outcome_type

VARIABLES = {
    "tas": ("tas", "climate", "daily"),
    "hurs": ("hurs", "climate", "daily"),
    "ps": ("ps", "climate", "daily"),
    "dis": ("dis", "watergap", "daily"),
    "triver": ("triver", "watergap", "monthly"),
}


def _variable(name: str, **kwargs):
    return ep.get("isimip.ISIMIPVariable")(*VARIABLES[name], **kwargs)


@contextlib.contextmanager
def _environment(**values):
    """Set (or with ``None`` remove) environment variables for a block."""

    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _names(paths) -> list[str]:
    return [path.name for path in paths]


# --- ISIMIP loading --------------------------------------------------------------------------


@case("isimip.paths_and_discovery")
def isimip_paths_and_discovery(w):
    paths = w.isimip_paths
    discover = ep.get("isimip.discover_isimip_files")
    counterclim_root = w.counterclim_root
    counterclim = ep.get("isimip.isimip3a_counterclim_paths")(counterclim_root)
    histsoc = ep.get("isimip.isimip3a_counterclim_paths")(counterclim_root, soc="histsoc")
    gcm, scenario = fixtures.GCM, "ssp585"
    with _environment(WATXENE_DATA_PATH=None):
        no_env_3b = outcome(ep.get("isimip.isimip3b_paths"))
        no_env_3a = outcome(ep.get("isimip.isimip3a_counterclim_paths"))
    with _environment(WATXENE_DATA_PATH=str(w.watxene_root)):
        env_3b = ep.get("isimip.isimip3b_paths")()
        env_3a = ep.get("isimip.isimip3a_counterclim_paths")()
    return {
        "paths_3b": paths,
        "paths_3a": counterclim,
        "paths_3a_histsoc": histsoc,
        "paths_env_3b": env_3b,
        "paths_env_3a": env_3a,
        "paths_no_env_3b": no_env_3b,
        "paths_no_env_3a": no_env_3a,
        "tas_files": _names(discover(_variable("tas"), gcm, scenario, paths)),
        "hurs_files": _names(discover(_variable("hurs"), gcm, "ssp126", paths)),
        "dis_files": _names(discover(_variable("dis"), gcm, scenario, paths)),
        "triver_files": _names(discover(_variable("triver"), gcm, scenario, paths)),
        "year_range_first_block": _names(
            discover(_variable("tas"), gcm, scenario, paths, year_range=(2015, 2019))
        ),
        "year_range_straddle": _names(
            discover(_variable("tas"), gcm, scenario, paths, year_range=(2019, 2020))
        ),
        "year_range_outside": outcome(
            discover, _variable("tas"), gcm, scenario, paths, year_range=(1990, 1999)
        ),
        "unknown_gcm": outcome(discover, _variable("tas"), "NOPE-GCM", scenario, paths),
        "unknown_scenario": outcome(discover, _variable("tas"), gcm, "ssp370", paths),
        "counterclim_tas": _names(
            discover(_variable("tas"), "GSWP3-W5E5", "counterclim", counterclim)
        ),
        "counterclim_dis": _names(
            discover(_variable("dis"), "GSWP3-W5E5", "counterclim", counterclim)
        ),
        "counterclim_dis_histsoc": _names(
            discover(_variable("dis"), "GSWP3-W5E5", "counterclim", histsoc)
        ),
        "counterclim_missing_soc": outcome(
            discover,
            _variable("dis"),
            "GSWP3-W5E5",
            "counterclim",
            ep.get("isimip.isimip3a_counterclim_paths")(counterclim_root, soc="ssp2soc"),
        ),
    }


@case("isimip.load_variable")
def isimip_load_variable(w):
    load = ep.get("isimip.load_isimip_variable")
    paths = w.isimip_paths
    gcm = fixtures.GCM
    counterclim = ep.get("isimip.isimip3a_counterclim_paths")(w.counterclim_root)
    return {
        "tas_monthly": load(_variable("tas"), gcm, "ssp585", paths),
        "tas_native": load(_variable("tas"), gcm, "ssp585", paths, target_frequency="native"),
        "tas_year_range": load(_variable("tas"), gcm, "ssp126", paths, year_range=(2018, 2020)),
        "tas_native_year_range": load(
            _variable("tas"), gcm, "ssp126", paths, year_range=(2019, 2020), target_frequency="native"
        ),
        "tas_chunked": load(
            _variable("tas"), gcm, "ssp585", paths, year_range=(2016, 2017), chunks={"time": 100}
        ),
        "dis_monthly": load(_variable("dis"), gcm, "ssp585", paths, year_range=(2015, 2016)),
        "triver_monthly": load(_variable("triver"), gcm, "ssp126", paths),
        "triver_native": load(
            _variable("triver"), gcm, "ssp585", paths, target_frequency="native"
        ),
        "counterclim_tas": load(
            _variable("tas"), "GSWP3-W5E5", "counterclim", counterclim, target_frequency="native"
        ),
        "bad_frequency": outcome(
            load, _variable("tas"), gcm, "ssp585", paths, target_frequency="hourly"
        ),
        "missing_file": outcome(load, _variable("tas"), gcm, "ssp370", paths),
    }


@case("isimip.load_dataset")
def isimip_load_dataset(w):
    load = ep.get("isimip.load_isimip_dataset")
    paths = w.isimip_paths
    gcm = fixtures.GCM
    wet = [_variable("tas"), _variable("hurs"), _variable("ps"), _variable("dis")]
    renamed = ep.get("isimip.ISIMIPVariable")("triver", "watergap", "monthly", output_name="tw")
    return {
        "wet_inputs": load([*wet, renamed], gcm, "ssp585", paths, year_range=(2016, 2018)),
        "native_climate": load(
            wet[:3], gcm, "ssp126", paths, year_range=(2019, 2020), target_frequency="native"
        ),
        "native_mixed_frequency": outcome(
            load, [wet[0], renamed], gcm, "ssp585", paths, year_range=(2016, 2017),
            target_frequency="native",
        ),
        "chunked": load(wet[:2], gcm, "ssp585", paths, year_range=(2016, 2017), chunks={"time": 200}),
        "no_variables": outcome(load, [], gcm, "ssp585", paths),
        "missing_variable": outcome(load, [wet[0], _variable("hurs")], gcm, "ssp370", paths),
    }  # fmt: skip


@case("isimip.gmt")
def isimip_gmt(w):
    load = ep.get("isimip.load_isimip_gmt")
    root = w.gmt_root
    stripped = {}
    for scenario, kwargs in (
        ("historical", {}),
        ("ssp126", {}),
        ("ssp585", {"year_range": (2010, 2020)}),
        ("ssp585", {}),
    ):
        result = load(root, fixtures.GCM, scenario, **kwargs)
        stripped[f"{scenario}{'_range' if kwargs else ''}"] = result
    return {
        "series": stripped,
        "path_object": load(str(root), fixtures.GCM.lower(), "ssp126", year_range=(2014, 2016)),
        "missing_scenario": outcome(load, root, fixtures.GCM, "ssp370"),
        "missing_gcm": outcome(load, root, "NOPE-GCM", "ssp585"),
    }


# --- cells considered -------------------------------------------------------------------------


def _region_paths(w):
    paths = w.region_paths
    return ep.get("universe.RegionRasterPaths")(
        gaul_raster=paths["gaul_raster"], gaul_csv=paths["gaul_csv"], r12_yaml=paths["r12_yaml"]
    )


@case("universe.rasters_and_tagging")
def universe_rasters_and_tagging(w):
    region_paths = _region_paths(w)
    region, country = ep.get("universe.build_region_country_rasters")(region_paths)
    cells = pd.DataFrame(
        {
            "grid_lat": [10.25, 11.75, 10.75, 11.25, 11.75, 35.25, -89.75],
            "grid_lon": [20.25, 21.25, 20.75, 20.25, 20.25, 20.25, 20.75],
        }
    )
    tag = ep.get("universe.tag_cells")
    return {
        "region_raster": region,
        "country_raster": country,
        "tagged": outcome(tag, cells.iloc[:5], region_paths),
        "tagged_off_grid": outcome(tag, cells, region_paths),
        "tagged_without_region_requirement": outcome(
            tag, cells.iloc[:5], region_paths, require_region=False
        ),
        "tagged_one_degree": outcome(tag, cells.iloc[:3], region_paths, resolution_deg=1.0),
        "tagged_duplicates": outcome(tag, pd.concat([cells.iloc[:2], cells.iloc[:1]]), region_paths),
        "tagged_extra_columns": tag(cells.iloc[:3].assign(extra=1), region_paths),
        "tagged_missing_column": outcome_type(tag, cells.iloc[:3].drop(columns=["grid_lon"]), region_paths),
    }


@case("universe.valid_cells")
def universe_valid_cells(w):
    build = ep.get("universe.build_valid_cell_universe")
    region_paths = _region_paths(w)
    paths = w.isimip_paths
    gcm = fixtures.GCM
    return {
        "wet_ssp126": build(paths, gcm, "ssp126", region_paths),
        "wet_ssp585": build(paths, gcm, "ssp585", region_paths),
        "wet_both_scenarios": build(paths, gcm, ["ssp126", "ssp585"], region_paths),
        "wet_keep_unregioned": build(paths, gcm, "ssp585", region_paths, require_region=False),
        "dry": build(paths, gcm, "ssp126", region_paths, require_hydrology=False),
        "dry_keep_unregioned": build(
            paths, gcm, "ssp126", region_paths, require_hydrology=False, require_region=False
        ),
        "alternate_mask_variable": build(
            paths, gcm, "ssp585", region_paths, mask_variable=_variable("triver", output_name="tw")
        ),
        "empty_scenarios": outcome(build, paths, gcm, [], region_paths),
        "missing_scenario": outcome(build, paths, gcm, "ssp370", region_paths),
    }


@case("universe.fleet_cells")
def universe_fleet_cells(w):
    extract = ep.get("universe.extract_fleet_cells")
    region_paths = _region_paths(w)
    wet, dry = w.fleet_files
    wet_universe = ep.get("universe.build_valid_cell_universe")(
        w.isimip_paths, fixtures.GCM, "ssp585", region_paths
    )
    return {
        "all": extract(wet, dry, region_paths),
        "restricted": extract(wet, dry, region_paths, restrict_to=wet_universe),
        "restricted_to_nothing": outcome(
            extract, wet, dry, region_paths, restrict_to=wet_universe.iloc[0:0]
        ),
        "unregioned_kept": extract(wet, dry, region_paths, require_region=False),
        "missing_file": outcome(extract, wet.with_name("absent.nc"), dry, region_paths),
    }
