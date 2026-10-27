"""End-to-end tests of CHILLED, the country rasters, the RIME-X adapters and probability sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from . import entrypoints as ep
from . import fixtures
from .registry import case, outcome, outcome_type


def _kernel(setpoints: bool = False, **assumptions):
    config = ep.get("chilled.Assumptions")(**assumptions) if assumptions else None
    return ep.get("chilled.create_kernel")(config, region_setpoints=setpoints)


def _mapping(setpoints: bool):
    return ep.get("chilled.build_archetype_mapping")(
        fixtures.chilled_spatial_units(),
        fixtures.chilled_archetypes(),
        fixtures.chilled_solar(),
        setpoints=fixtures.chilled_setpoints() if setpoints else None,
    )


def _winter_climate() -> xr.Dataset:
    """Daily temperatures from late November to early March: a DJF with both edges partial."""

    rng = np.random.default_rng(20260106)
    time = pd.date_range("2000-11-25", "2001-03-05", freq="D")
    shape = (len(time), len(fixtures.LATS), len(fixtures.LONS))
    tas = 278.0 + 6.0 * rng.uniform(-1, 1, (1, len(fixtures.LATS), len(fixtures.LONS)))
    tas = tas + rng.normal(0.0, 3.0, shape)
    dataset = xr.Dataset(
        {"tas": (("time", "lat", "lon"), tas)},
        coords={"time": time, "lat": fixtures.LATS, "lon": fixtures.LONS},
    )
    dataset["tas"].attrs["units"] = "K"
    return dataset


# --- CHILLED kernel -------------------------------------------------------------------------------


@case("chilled.climate_inputs")
def chilled_climate_inputs(w):
    normalize = ep.get("chilled.normalize_climate_inputs")
    climate = fixtures.chilled_climate()
    celsius = climate.assign(tas=climate["tas"] - 273.15)
    celsius["tas"].attrs["units"] = "degC"
    return {
        "kelvin": normalize(climate),
        "celsius": normalize(celsius),
        "renamed": normalize(climate.rename(tas="t2m"), variable_names={"air_temperature_c": "t2m"}),
        "missing_variable": outcome(normalize, climate.drop_vars("tas")),
        "missing_units": outcome(normalize, climate.assign(tas=climate["tas"].drop_attrs())),
        "assumptions": ep.get("chilled.Assumptions")(),
        "assumptions_from_row": ep.get("chilled.Assumptions").from_parametric_row(
            {"t_sp_c": 24.0, "t_sp_c_max": 27.0, "t_sp_h": 19.0, "f_c": 0.4, "f_f": 0.1,
             "f_h": 0.9, "cop": 3.0, "eff": 0.8}
        ),
        "assumptions_from_incomplete_row": outcome(
            ep.get("chilled.Assumptions").from_parametric_row, {"t_sp_c": 24.0}
        ),
    }  # fmt: skip


@case("chilled.mapped")
def chilled_mapped(w):
    evaluate = ep.get("kernel.evaluate_mapped_kernel")
    climate = fixtures.chilled_climate()
    mapping, solar = _mapping(setpoints=False)
    regional, regional_solar = _mapping(setpoints=True)
    winter = _winter_climate()
    return {
        "mapping": mapping,
        "mapping_with_setpoints": regional,
        "solar": solar,
        "daily": evaluate(_kernel(), mapping, climate, duration="native", gridded_coords=solar),
        "native": evaluate(_kernel(), mapping, climate, duration="native", gridded_coords=solar),
        "monthly": evaluate(_kernel(), mapping, climate, duration="monthly", gridded_coords=solar),
        "seasonal_summer": evaluate(_kernel(), mapping, climate, duration="seasonal", gridded_coords=solar),
        "seasonal_winter_partial_djf": evaluate(
            _kernel(), mapping, winter, duration="seasonal", gridded_coords=solar
        ),
        "seasons_custom": evaluate(
            _kernel(), mapping, winter, duration="seasonal", gridded_coords=solar,
            seasons={"winter": (12, 1, 2), "spring": (3, 4, 5)},
        ),
        "region_setpoints_with_missing_region": evaluate(
            _kernel(setpoints=True), regional, climate, duration="native", gridded_coords=regional_solar
        ),
        "region_setpoints_flag_without_columns": outcome(
            evaluate, _kernel(setpoints=True), mapping, climate, duration="native", gridded_coords=solar
        ),
        "setpoints_ignored_without_flag": evaluate(
            _kernel(), regional, climate, duration="native", gridded_coords=regional_solar
        ),
        "assumptions": evaluate(
            _kernel(cooling_setpoint_c=24.0, cooling_max_c=26.0, ac_cop=2.5, heating_setpoint_c=18.0),
            mapping, climate, duration="monthly", gridded_coords=solar,
        ),
        "small_chunks": evaluate(
            ep.get("chilled.create_kernel")(None, entity_chunk=5, time_chunk=4),
            mapping, climate, duration="native", gridded_coords=solar,
        ),
        "no_solar": outcome_type(evaluate, _kernel(), mapping, climate, duration="native"),
        "bad_duration": outcome(
            evaluate, _kernel(), mapping, climate, duration="yearly", gridded_coords=solar
        ),
        "missing_air_temperature": outcome(
            evaluate, _kernel(), mapping, climate.drop_vars("tas"), duration="native", gridded_coords=solar
        ),
    }  # fmt: skip


@case("chilled.gridded")
def chilled_gridded(w):
    evaluate = ep.get("kernel.evaluate_gridded_kernel")
    grids = ep.get("chilled.build_parameter_grids")
    raster = fixtures.chilled_region_raster()
    archetypes, solar, setpoints = (
        fixtures.chilled_archetypes(),
        fixtures.chilled_solar(),
        fixtures.chilled_setpoints(),
    )
    climate = fixtures.chilled_climate()
    plain = grids(raster, archetypes, solar, building_type="sfh", urt="rur", vintage="s1")
    regional = grids(
        raster, archetypes, solar, building_type="mfh", urt="urb", vintage="s51", setpoints=setpoints
    )
    return {
        "parameters": plain,
        "parameters_with_setpoints": regional,
        "daily": evaluate(_kernel(), climate, plain, "native"),
        "monthly": evaluate(_kernel(), climate, plain, "monthly"),
        "seasonal": evaluate(_kernel(), climate, plain, "seasonal"),
        "seasonal_winter": evaluate(_kernel(), _winter_climate(), plain, "seasonal"),
        "region_setpoints_daily": evaluate(_kernel(), climate, regional, "native"),
        "unknown_archetype": outcome(
            grids, raster, archetypes, solar, building_type="sfh", urt="rur", vintage="s3"
        ),
        "setpoints_missing_columns": outcome(
            grids, raster, archetypes, solar, building_type="sfh", urt="rur", vintage="s1",
            setpoints=setpoints.drop(columns=["cooling_max_c"]),
        ),
        "solar_missing_cells": outcome_type(
            grids, raster, archetypes, solar.isel(lat=slice(0, 2)), building_type="sfh", urt="rur",
            vintage="s1",
        ),
        "bad_duration": outcome(evaluate, _kernel(), climate, plain, "decadal"),
        "wrong_spatial_dims": outcome_type(evaluate, _kernel(), climate, plain, "native", spatial_dims=("y", "x")),
    }  # fmt: skip


def chilled_graph_report(w):
    """Task-graph sizes and laziness of CHILLED output for chunked input."""

    chunked = fixtures.chilled_climate().chunk({"time": 4})
    mapping, solar = _mapping(setpoints=True)
    raster = fixtures.chilled_region_raster()
    parameters = ep.get("chilled.build_parameter_grids")(
        raster, fixtures.chilled_archetypes(), fixtures.chilled_solar(),
        building_type="sfh", urt="rur", vintage="s1",
    )  # fmt: skip
    mapped = ep.get("kernel.evaluate_mapped_kernel")
    gridded = ep.get("kernel.evaluate_gridded_kernel")
    results = {
        "mapped_daily": mapped(_kernel(True), mapping, chunked, duration="native", gridded_coords=solar),
        "mapped_seasonal": mapped(_kernel(True), mapping, chunked, duration="seasonal", gridded_coords=solar),
        "gridded_daily": gridded(_kernel(), chunked, parameters, "native"),
        "gridded_monthly": gridded(_kernel(), chunked, parameters, "monthly"),
    }  # fmt: skip
    sizes = {name: len(result.__dask_graph__()) for name, result in results.items()}
    lazy = {
        name: all(variable.chunks is not None for variable in result.data_vars.values())
        for name, result in results.items()
    }
    return sizes, lazy


# --- CHILLED input files and validation ----------------------------------------------------------------


@case("chilled.inputs_and_loaders")
def chilled_inputs_and_loaders(w):
    directory = w.work / "chilled_files"
    directory.mkdir(exist_ok=True)
    archetypes = fixtures.chilled_archetypes()
    setpoints = fixtures.chilled_setpoints()
    solar = fixtures.chilled_solar()
    validate = ep.get("chilled.validate_archetype_table")
    normalize_solar = ep.get("chilled.normalize_solar_climatology")
    archetype_csv = directory / "archetypes.csv"
    archetypes.to_csv(archetype_csv, index=False)
    setpoint_csv = directory / "setpoints.csv"
    setpoints.to_csv(setpoint_csv, index=False)
    duplicated_csv = directory / "setpoints_duplicated.csv"
    pd.concat([setpoints, setpoints.iloc[:1]]).to_csv(duplicated_csv, index=False)
    short_csv = directory / "setpoints_short.csv"
    setpoints.drop(columns=["cooling_max_c"]).to_csv(short_csv, index=False)
    extra_csv = directory / "setpoints_extra.csv"
    setpoints.assign(basis="ashrae", source_note="proxy").to_csv(extra_csv, index=False)
    parametric = pd.DataFrame(
        {
            "id_run": [1, 2], "name_run": ["base", "warm"], "cop": [3.2, 2.8], "eff": [0.9, 0.8],
            "t_sp_c": [26.0, 24.0], "t_sp_c_max": [28.0, 26.0], "t_sp_h": [20.0, 19.0],
            "f_c": [0.33, 0.4], "f_f": [0.2, 0.25], "f_h": [1.0, 0.9],
        }
    )  # fmt: skip
    parametric_csv = directory / "parametric.csv"
    parametric.to_csv(parametric_csv, index=False)
    incomplete_csv = directory / "parametric_incomplete.csv"
    parametric.drop(columns=["cop"]).to_csv(incomplete_csv, index=False)
    vertical, horizontal = directory / "vertical.nc", directory / "horizontal.nc"
    for path, name in ((vertical, "solar_vertical"), (horizontal, "solar_horizontal")):
        array = solar[name].transpose("lat", "lon", "month").isel(lat=slice(None, None, -1))
        array.rename("irrad").astype("float32").to_dataset().to_netcdf(path)
    solar_other = solar.rename(solar_vertical="v", solar_horizontal="h")
    return {
        "validate_ok": validate(archetypes),
        "validate_missing_column": outcome(validate, archetypes.drop(columns=["u_val"])),
        "validate_bad_building_type": outcome(validate, archetypes.assign(building_type="tower")),
        "validate_bad_urt": outcome(validate, archetypes.assign(urt="sub")),
        "validate_bad_arch": outcome(validate, archetypes.assign(arch="s9")),
        "validate_duplicates": outcome(validate, pd.concat([archetypes, archetypes.iloc[:2]])),
        "validate_with_nulls": validate(archetypes.assign(urt=archetypes["urt"].where(archetypes.index != 0))),
        "archetypes_file": ep.get("chilled.load_archetypes")(archetype_csv),
        "archetypes_file_missing": outcome(ep.get("chilled.load_archetypes"), directory / "absent.csv"),
        "parametric_file": ep.get("chilled.load_parametric_runs")(parametric_csv),
        "parametric_incomplete": outcome(ep.get("chilled.load_parametric_runs"), incomplete_csv),
        "parametric_no_path": outcome(ep.get("chilled.load_parametric_runs"), None),
        "setpoints_file": ep.get("chilled.load_region_cooling_setpoints")(setpoint_csv),
        "setpoints_extra_columns": ep.get("chilled.load_region_cooling_setpoints")(extra_csv),
        "setpoints_duplicated": outcome(ep.get("chilled.load_region_cooling_setpoints"), duplicated_csv),
        "setpoints_missing_column": outcome(ep.get("chilled.load_region_cooling_setpoints"), short_csv),
        "solar_normalized": normalize_solar(solar),
        "solar_unsorted_months": normalize_solar(solar.isel(month=slice(None, None, -1))),
        "solar_custom_names": normalize_solar(solar_other, vertical_name="v", horizontal_name="h"),
        "solar_missing_variable": outcome(normalize_solar, solar.drop_vars("solar_horizontal")),
        "solar_missing_month_coordinate": outcome(normalize_solar, solar.drop_vars("month")),
        "solar_incomplete_months": outcome(normalize_solar, solar.isel(month=slice(0, 11))),
        "solar_without_latitude": outcome(normalize_solar, solar.drop_vars("lat")),
        "solar_ewembi_files": ep.get("chilled.load_ewembi_solar_climatology")(
            vertical_path=vertical, horizontal_path=horizontal
        ),
        "solar_ewembi_missing_file": outcome(
            ep.get("chilled.load_ewembi_solar_climatology"),
            vertical_path=directory / "absent.nc", horizontal_path=horizontal,
        ),
        "region_raster_from_gaul": ep.get("chilled.load_r12_region_raster")(
            gaul_raster_path=w.region_paths["gaul_raster"],
            gaul_csv_path=w.region_paths["gaul_csv"],
            r12_yaml_path=w.region_paths["r12_yaml"],
        ),
        "mapping_missing_unit_columns": outcome(
            ep.get("chilled.build_archetype_mapping"),
            fixtures.chilled_spatial_units().drop(columns=["region"]), archetypes, solar,
        ),
        "mapping_no_matching_region": outcome(
            ep.get("chilled.build_archetype_mapping"),
            fixtures.chilled_spatial_units().assign(region="ZZZ"), archetypes, solar,
        ),
        "mapping_setpoints_missing_columns": outcome(
            ep.get("chilled.build_archetype_mapping"),
            fixtures.chilled_spatial_units(), archetypes, solar, setpoints=setpoints.drop(columns=["region"]),
        ),
        "mapping_without_permanent_column": ep.get("chilled.build_archetype_mapping")(
            fixtures.chilled_spatial_units().assign(spatial_unit_id=range(100, 112)),
            archetypes.drop(columns=["permanent"]), solar,
        )[0],
    }  # fmt: skip


# --- countries and the RIME-X adapters ---------------------------------------------------------------------


def _seasonal_entities() -> xr.Dataset:
    cells = [(lat, lon) for lat in fixtures.LATS for lon in fixtures.LONS]
    rng = np.random.default_rng(20260107)
    shape = (2, 3, len(cells))
    coords = {
        "season": ["DJF", "JJA"],
        "year": [2001, 2002, 2003],
        "entity": np.arange(len(cells)),
        "lat": ("entity", [lat for lat, _ in cells]),
        "lon": ("entity", [lon for _, lon in cells]),
    }
    variables = {
        name: (("season", "year", "entity"), rng.uniform(0.0, 50.0, shape))
        for name in ("cooling_energy_total", "heating_energy", "Nd", "Nf", "ignored")
    }
    variables["cooling_energy_total"][1][0, 2] = np.nan
    return xr.Dataset(variables, coords=coords)


@case("country.raster_and_aggregation")
def country_raster_and_aggregation(w):
    paths = w.region_paths
    load = ep.get("country.load_country_raster")
    aggregate = ep.get("country.aggregate_entity_to_country")
    raster = load(gaul_raster_path=paths["gaul_raster"], gaul_csv_path=paths["gaul_csv"])
    entities = _seasonal_entities()
    reductions = {
        "cooling_energy_total": "sum", "heating_energy": "sum", "Nd": "weighted_mean",
        "Nf": "weighted_mean",
    }  # fmt: skip
    lowercase_csv = w.work / "gaul_lowercase.csv"
    pd.read_csv(paths["gaul_csv"]).rename(columns={"ISO3": "iso3"}).to_csv(lowercase_csv, index=False)
    names_only = w.work / "gaul_names_only.csv"
    pd.read_csv(paths["gaul_csv"]).drop(columns=["ISO3"]).to_csv(names_only, index=False)
    no_label = w.work / "gaul_no_label.csv"
    pd.read_csv(paths["gaul_csv"])[["ID"]].to_csv(no_label, index=False)
    return {
        "raster": raster,
        "raster_by_name": load(
            gaul_raster_path=paths["gaul_raster"], gaul_csv_path=paths["gaul_csv"], country_column="NAME"
        ),
        "raster_lowercase_fallback": outcome(
            load, gaul_raster_path=paths["gaul_raster"], gaul_csv_path=lowercase_csv
        ),
        "raster_label_fallback_to_name": outcome(
            load, gaul_raster_path=paths["gaul_raster"], gaul_csv_path=names_only
        ),
        "raster_no_label_column": outcome(
            load, gaul_raster_path=paths["gaul_raster"], gaul_csv_path=no_label
        ),
        "sum_and_weighted_mean": aggregate(
            entities.drop_vars("ignored"), raster, variable_reductions=reductions
        ),
        "only_sums": aggregate(
            entities[["cooling_energy_total"]], raster, variable_reductions={"cooling_energy_total": "sum"}
        ),
        "no_entity_dimension": outcome(
            aggregate, entities.isel(entity=0, drop=True), raster, variable_reductions=reductions
        ),
        "no_coordinates": outcome(
            aggregate, entities.drop_vars("lat"), raster, variable_reductions=reductions
        ),
        "unsupported_reduction_name": outcome(
            aggregate, entities, raster, variable_reductions={**reductions, "ignored": "mean"}
        ),
        "unsupported_reduction": outcome(
            aggregate, entities, raster, variable_reductions={**reductions, "Nd": "median"}
        ),
        "reduction_missing_for_variable": outcome(
            aggregate, entities, raster, variable_reductions={"cooling_energy_total": "sum"}
        ),
    }  # fmt: skip


@case("rimex.buildings")
def rimex_buildings(w):
    parse = ep.get("rimex_buildings.parse_chilled_matrix_filename")
    aggregate = ep.get("rimex_buildings.aggregate_chilled_to_country")
    expand = ep.get("rimex_buildings.expand_chilled_metadata")
    paths = w.region_paths
    raster = ep.get("country.load_country_raster")(
        gaul_raster_path=paths["gaul_raster"], gaul_csv_path=paths["gaul_csv"]
    )
    entities = _seasonal_entities()
    aggregated = aggregate(entities, raster)
    metadata = parse("/data/seasonal_mpi_esm1_2_hr_ssp585_sfh_rur_s1_2015_2100.nc")
    return {
        "parse": metadata,
        "parse_historical": parse("seasonal_gfdl_esm4_historical_mfh_urb_s52_1850_2014.nc"),
        "parse_hyphenated": parse("seasonal_ukesm1-0-ll_ssp126_sfh_urb_s2_2015_2100.nc"),
        "parse_bad_name": outcome(parse, "daily_gfdl_esm4_ssp585_sfh_rur_s1_2015_2100.nc"),
        "parse_bad_scenario": outcome(parse, "seasonal_gfdl_ssp999_sfh_rur_s1_2015_2100.nc"),
        "parse_bad_vintage": outcome(parse, "seasonal_gfdl_ssp585_sfh_rur_s7_2015_2100.nc"),
        "aggregated": aggregated,
        "aggregated_only_one_variable": aggregate(entities[["Nd"]], raster),
        "aggregated_no_supported_variable": outcome(aggregate, entities[["ignored"]], raster),
        "expanded": expand(aggregated, metadata),
        "expanded_missing_key": outcome(expand, aggregated, {"gcm": "X"}),
    }


@case("rimex.thermoelectric")
def rimex_thermoelectric(w):
    directory = w.work / "rimex_thermo"
    directory.mkdir(exist_ok=True)
    load = ep.get("rimex_thermo.load_thermoelectric_ucr")
    seasonal = ep.get("rimex_thermo.aggregate_thermoelectric_seasonal")
    rng = np.random.default_rng(20260108)
    rows = []
    for cooling in ("wet", "dry"):
        for scenario, months in (
            ("historical", pd.date_range("1999-10-01", "2001-03-01", freq="MS")),
            ("ssp585", pd.date_range("2015-01-01", "2016-02-01", freq="MS")),
        ):
            for country in ("AAA", "BBB"):
                for time in months:
                    if cooling == "dry" and country == "BBB":
                        continue
                    total = 100.0 + 10.0 * rng.integers(0, 5)
                    rows.append((cooling, "GFDL-ESM4", scenario, time, country, total * rng.uniform(0.3, 1.0), total))
    frame = pd.DataFrame(
        rows, columns=["cooling", "gcm", "scenario", "time", "country", "usable_capacity_mw", "total_mw"]
    )
    gaps = frame.drop(index=frame.index[[5, 6, 30]])
    source = directory / "ucr.parquet"
    gappy = directory / "ucr_gaps.parquet"
    bad_total = directory / "ucr_bad_total.parquet"
    frame.to_parquet(source, index=False)
    gaps.to_parquet(gappy, index=False)
    frame.assign(total_mw=frame["total_mw"].where(frame.index != 3, 0.0)).to_parquet(bad_total, index=False)
    ucr = load(source)
    gap_ucr = load(gappy)
    return {
        "ucr": ucr,
        "ucr_with_gaps": gap_ucr,
        "ucr_zero_total": outcome(load, bad_total),
        "ucr_missing_file": outcome(load, directory / "absent.parquet"),
        "seasonal": seasonal(ucr),
        "seasonal_with_gaps": seasonal(gap_ucr),
        "seasonal_single_scenario": seasonal(ucr.sel(scenario=["ssp585"])),
        "seasonal_no_time": outcome(seasonal, ucr.isel(time=0, drop=True)),
        "seasonal_nan_month": seasonal(ucr.where(ucr["time"] != ucr["time"].values[3])),
    }  # fmt: skip


# --- probability sampling -----------------------------------------------------------------------------


