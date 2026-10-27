"""End-to-end tests of the thermoelectric fleet, wet and dry kernels, aggregation and reach routing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from . import entrypoints as ep
from . import fixtures
from .registry import case, outcome


def read_back(path) -> pd.DataFrame:
    path = str(path)
    return pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)


def _work(w, name: str):
    directory = w.work / name
    directory.mkdir(exist_ok=True)
    return directory


# --- fleet mapping ----------------------------------------------------------------


@case("fleet.snap_to_half_degree_cell")
def fleet_snap(w):
    values = pd.Series(
        [23.3979, 23.501, -6.001, -6.5001, 0.0, 0.5, -0.5, 10.25, 10.75, 179.99, -180.0, np.nan],
        index=list("abcdefghijkl"),
    )
    return ep.get("fleet.snap_to_half_degree_cell")(values)


@case("fleet.unit_cell_mappings")
def fleet_unit_cell_mappings(w):
    plants = fixtures.plants()
    exact = ep.get("fleet.build_unit_cell_mapping")(plants)
    recovered = ep.get("fleet.build_recovered_unit_cell_mapping")(plants)
    return {
        "exact": exact,
        "recovered": recovered,
        "exact_missing_columns": outcome(
            ep.get("fleet.build_unit_cell_mapping"), plants.drop(columns=["has_coords"])
        ),
        "recovered_missing_columns": outcome(
            ep.get("fleet.build_recovered_unit_cell_mapping"), plants.drop(columns=["mw"])
        ),
        "all_coordinates_present": ep.get("fleet.build_recovered_unit_cell_mapping")(
            plants[plants["has_coords"].eq(True)]
        ),
        "location_ambiguity": ep.get("fleet.summarize_location_ambiguity")(exact),
        "location_ambiguity_empty": ep.get("fleet.summarize_location_ambiguity")(exact.iloc[0:0]),
        "summary": ep.get("fleet.summarize_mapping")(exact),
        "summary_recovered": ep.get("fleet.summarize_mapping")(recovered),
        "summary_empty": ep.get("fleet.summarize_mapping")(exact.iloc[0:0]),
        "locations": ep.get("fleet.build_location_cell_mapping")(exact),
        "locations_empty": ep.get("fleet.build_location_cell_mapping")(exact.iloc[0:0]),
        "rollup_unit": ep.get("fleet.build_country_rollup")(exact, source_resolution="unit"),
        "rollup_location": ep.get("fleet.build_country_rollup")(
            ep.get("fleet.build_location_cell_mapping")(exact), source_resolution="location"
        ),
        "rollup_empty": ep.get("fleet.build_country_rollup")(exact.iloc[0:0], source_resolution="unit"),
        "rollup_no_country": outcome(
            ep.get("fleet.build_country_rollup"), exact.drop(columns=["country"]), source_resolution="unit"
        ),
    }


@case("fleet.imputed_mappings")
def fleet_imputed_mappings(w):
    plants = fixtures.plants()
    cooling = w.cooling
    wet = ep.get("fleet.build_imputed_wet_unit_cell_mapping")
    dry = ep.get("fleet.build_imputed_dry_unit_cell_mapping")
    return {
        "wet_seed0": wet(plants, cooling, random_seed=0),
        "wet_seed7": wet(plants, cooling, random_seed=7),
        "dry_seed0": dry(plants, cooling, random_seed=0),
        "dry_seed7": dry(plants, cooling, random_seed=7),
        "wet_all_resolved": wet(plants[plants["has_coords"].eq(True)], cooling, random_seed=0),
        "wet_missing_plant_columns": outcome(wet, plants.drop(columns=["lon"]), cooling),
        "wet_missing_cooling_columns": outcome(wet, plants, cooling.drop(columns=["requires_water"])),
        "dry_missing_cooling_columns": outcome(dry, plants, cooling.drop(columns=["fuel"])),
    }


@case("fleet.mapping_writers")
def fleet_mapping_writers(w):
    directory = _work(w, "fleet_writers")
    source = w.plants_parquet
    cooling_path = directory / "cooling.parquet"
    w.cooling.to_parquet(cooling_path, index=False)
    out = {}
    out["loaded"] = ep.get("fleet.load_and_map_thermal_plants")(source)
    writers = [
        ("unit_parquet", "fleet.write_unit_cell_mapping", (source,), "unit.parquet", {}),
        ("unit_csv", "fleet.write_unit_cell_mapping", (source,), "unit.csv", {}),
        (
            "unit_columns",
            "fleet.write_unit_cell_mapping",
            (source,),
            "unit_columns.parquet",
            {"columns": ["UNITID", "cell_id"]},
        ),
        ("recovered", "fleet.write_recovered_unit_cell_mapping", (source,), "recovered.parquet", {}),
        (
            "imputed_wet",
            "fleet.write_imputed_wet_unit_cell_mapping",
            (source, cooling_path),
            "imputed_wet.parquet",
            {"random_seed": 2},
        ),
        (
            "imputed_dry",
            "fleet.write_imputed_dry_unit_cell_mapping",
            (source, cooling_path),
            "imputed_dry.csv",
            {"random_seed": 2},
        ),
        ("locations", "fleet.write_location_cell_mapping", (source,), "locations.parquet", {}),
        ("country_location", "fleet.write_country_rollup", (source,), "country_location.parquet", {}),
        (
            "country_unit",
            "fleet.write_country_rollup",
            (source,),
            "country_unit.csv",
            {"from_resolution": "unit"},
        ),
    ]
    for label, name, args, filename, kwargs in writers:
        target = directory / filename
        summary = ep.get(name)(*args, target, **kwargs)
        out[f"{label}_summary"] = summary
        out[f"{label}_file"] = read_back(target)
    out["bad_suffix"] = outcome(
        ep.get("fleet.write_unit_cell_mapping"), source, directory / "unit.xlsx"
    )
    out["bad_suffix_recovered"] = outcome(
        ep.get("fleet.write_recovered_unit_cell_mapping"), source, directory / "unit.json"
    )
    return out


# --- cooling classification --------------------------------------------------------


@case("cooling.classification")
def cooling_classification(w):
    directory = _work(w, "cooling_writer")
    plants = fixtures.plants()
    classify = ep.get("cooling.classify_technology_family")
    build = ep.get("cooling.build_unit_cooling_mapping")
    out = {
        "families": {
            utype: classify(utype)
            for utype in ["GT/C", "GT/T", "CCSS", "CC", "ST", "ORC", "GT", "IC", "ECE", "FC", "PV", "", "st"]
        },
        "mapping": build(plants),
        "unknown_category": outcome(build, plants.assign(cool_imputed="cl_salt")),
        "missing_category": outcome(build, plants.assign(cool_imputed=None)),
        "missing_columns": outcome(build, plants.drop(columns=["cool_source"])),
    }
    plants.to_parquet(directory / "plants.parquet", index=False)
    ep.get("cooling.write_unit_cooling_mapping")(
        directory / "plants.parquet", directory / "cooling.csv", columns=["UNITID", "cooling_physics"]
    )
    ep.get("cooling.write_unit_cooling_mapping")(directory / "plants.parquet", directory / "cooling.parquet")
    out["written_csv"] = read_back(directory / "cooling.csv")
    out["written_parquet"] = read_back(directory / "cooling.parquet")
    out["bad_suffix"] = outcome(
        ep.get("cooling.write_unit_cooling_mapping"), directory / "plants.parquet", directory / "cooling.txt"
    )
    return out


# --- wet cooling ----------------------------------------------------------------------


@case("wet.climate_inputs")
def wet_climate_inputs(w):
    climate = fixtures.monthly_climate()
    normalize = ep.get("wet.normalize_climate_inputs")
    renamed = climate.rename({"tw": "river_temp", "hurs": "rh"})
    no_units = climate.copy()
    no_units["tw"].attrs.pop("units")
    return {
        "default": normalize(climate),
        "rename_and_units": normalize(
            renamed.assign(river_temp=renamed["river_temp"] - 273.15),
            variable_names={"water_temperature": "river_temp", "relative_humidity": "rh"},
            variable_units={"river_temp": "C"},
        ),
        "celsius_input": normalize(climate.assign(tas=climate["tas"] - 273.15).assign_attrs()),
        "missing_variable": outcome(normalize, climate.drop_vars("dis")),
        "missing_units": outcome(normalize, no_units),
        "bad_units": outcome(normalize, climate, variable_units={"tw": "F"}),
    }


@case("wet.fuel_efficiency")
def wet_fuel_efficiency(w):
    function = ep.get("wet.fuel_efficiency")
    sizes = [0.0, 1.0, 25.0, 100.0, 270.0, 500.0, 1500.0]
    return {
        fuel: [function(fuel, mw) for mw in sizes] for fuel in ("coal", "gas", "oil", "biomass", "nuclear")
    } | {"unknown_fuel": outcome(function, "hydrogen", 100.0)}


@case("wet.mapping")
def wet_mapping(w):
    plants = fixtures.plants()
    imputed = ep.get("fleet.build_imputed_wet_unit_cell_mapping")(plants, w.cooling, random_seed=0)
    build = ep.get("wet.build_mapping")
    cooling = w.cooling
    return {
        "mapping": w.wet_mapping,
        "without_country": build(imputed.drop(columns=["country"]), cooling),
        "country_nowhere": outcome(
            build, imputed.drop(columns=["country"]), cooling.drop(columns=["country"])
        ),
        "unsupported_fuel": outcome(build, imputed, cooling.assign(fuel="HYDROGEN")),
        "unsupported_physics": outcome(build, imputed, cooling.assign(cooling_physics="hybrid")),
        "duplicate_units": outcome(build, pd.concat([imputed, imputed.iloc[:1]]), cooling),
        "missing_mapping_columns": outcome(build, imputed.drop(columns=["grid_lat"]), cooling),
        "missing_cooling_columns": outcome(build, imputed, cooling.drop(columns=["technology_family"])),
        "no_wet_units": build(imputed.iloc[0:0], cooling),
        "wsth_by_family": build(
            imputed,
            cooling.assign(
                fuel="WSTH",
                technology_family=["steam", "other", "combined_cycle", "gas_turbine"] * 4,
            ),
        ),
    }


@case("wet.saline_intakes")
def wet_saline_intakes(w):
    assign = ep.get("wet.assign_saline_intakes")
    evaluate = ep.get("kernel.evaluate_mapped_kernel")
    climate = fixtures.monthly_climate()
    sea = climate["tos"]
    intake_columns = ["intake_lat", "intake_lon", "intake_distance_km"]
    mapping = w.wet_mapping.drop(columns=intake_columns)
    return {
        "intakes": w.wet_mapping[["UNITID", "cooling_category", "grid_lat", "grid_lon", *intake_columns]],
        "short_reach": assign(mapping, sea, max_distance_km=10.0)[intake_columns],
        "static_field": assign(mapping, sea.isel(time=0))[intake_columns],
        "no_saline_units": assign(mapping[mapping["cooling_category"] != "ot_saline"], sea),
        "no_sea": outcome(assign, mapping, sea.where(sea > 100.0)),
        "kernel_without_sea": outcome(
            evaluate, _wet_kernel(), w.wet_mapping, climate.drop_vars("tos"), duration="native"
        ),
        "kernel_without_intakes": outcome(evaluate, _wet_kernel(), mapping, climate, duration="native"),
        "river_fleet_without_sea": evaluate(
            _wet_kernel(),
            mapping[mapping["cooling_category"] != "ot_saline"],
            climate.drop_vars("tos"),
            duration="native",
        ),
    }


def _wet_kernel(**assumptions):
    kernel_factory = ep.get("wet.create_kernel")
    if assumptions:
        return kernel_factory(ep.get("wet.Assumptions")(**assumptions))
    return kernel_factory()


@case("wet.kernel.native")
def wet_kernel_native(w):
    evaluate = ep.get("kernel.evaluate_mapped_kernel")
    return evaluate(_wet_kernel(), w.wet_mapping, fixtures.monthly_climate(), duration="native")


@case("wet.kernel.durations")
def wet_kernel_durations(w):
    evaluate = ep.get("kernel.evaluate_mapped_kernel")
    climate = fixtures.monthly_climate()
    kernel = _wet_kernel()
    return {
        "monthly": evaluate(kernel, w.wet_mapping, climate, duration="monthly"),
        "seasonal": evaluate(kernel, w.wet_mapping, climate, duration="seasonal"),
        "seasonal_subset": evaluate(
            kernel, w.wet_mapping, climate, duration="seasonal", seasons={"JJA": (6, 7, 8), "DJF": (12, 1, 2)}
        ),
        "bad_duration": outcome(evaluate, kernel, w.wet_mapping, climate, duration="annual"),
    }


@case("wet.kernel.dtype_and_dask")
def wet_kernel_dtype_and_dask(w):
    evaluate = ep.get("kernel.evaluate_mapped_kernel")
    climate = fixtures.monthly_climate()
    kernel = _wet_kernel()
    single = climate.astype("float32")
    for name in ("tw", "tas", "ps", "tos"):
        single[name].attrs["units"] = climate[name].attrs["units"]
    return {
        "chunked_native": evaluate(kernel, w.wet_mapping, climate.chunk({"time": 4}), duration="native"),
        "chunked_seasonal": evaluate(
            kernel, w.wet_mapping, climate.chunk({"time": 4}), duration="seasonal"
        ),
        "float32_native": evaluate(kernel, w.wet_mapping, single, duration="native"),
    }


@case("wet.kernel.assumptions")
def wet_kernel_assumptions(w):
    evaluate = ep.get("kernel.evaluate_mapped_kernel")
    climate = fixtures.monthly_climate()
    return {
        "environmental_flow": evaluate(
            _wet_kernel(environmental_flow_fraction=0.5), w.wet_mapping, climate, duration="native"
        ),
        "cycles_and_approach": evaluate(
            _wet_kernel(concentration_cycles=3.0, tower_approach_c=4.0),
            w.wet_mapping, climate, duration="native",
        ),
        "tight_discharge_limit": evaluate(
            _wet_kernel(max_discharge_temp_c=28.0, max_temp_rise_c=5.0),
            w.wet_mapping, climate, duration="native",
        ),
    }


@case("wet.kernel.errors")
def wet_kernel_errors(w):
    evaluate = ep.get("kernel.evaluate_mapped_kernel")
    climate = fixtures.monthly_climate()
    kernel = _wet_kernel()
    return {
        "missing_variable": outcome(
            evaluate, kernel, w.wet_mapping, climate.drop_vars("hurs"), duration="native"
        ),
        "cell_outside_grid": outcome(
            evaluate,
            kernel,
            w.wet_mapping.assign(grid_lat=w.wet_mapping["grid_lat"] + 40.0),
            climate,
            duration="native",
        ),
        "missing_mapping_columns": outcome(
            evaluate, kernel, w.wet_mapping.drop(columns=["thermal_efficiency"]), climate, duration="native"
        ),
        "no_entities": outcome(evaluate, kernel, w.wet_mapping.iloc[0:0], climate, duration="native"),
    }


# --- dry cooling -------------------------------------------------------------------------


@case("dry.classification")
def dry_classification(w):
    classify = ep.get("dry.classify_engine_type")
    families = ["combined_cycle", "steam", "gas_turbine", "internal_combustion", "other", ""]
    fuels = ["COAL", "NUCLEAR", "UR", "URANIUM", "GEOTHERMAL", "SOLAR", "WASTE HEAT", "WASTE", "LNG", "HYDRO", "mystery"]
    return {
        f"{family or 'blank'}|{fuel}": outcome(classify, technology_family=family, fuel=fuel)
        for family in families
        for fuel in fuels
    }


@case("dry.mapping")
def dry_mapping(w):
    plants = fixtures.plants()
    imputed = ep.get("fleet.build_imputed_dry_unit_cell_mapping")(plants, w.cooling, random_seed=0)
    build = ep.get("dry.build_mapping")
    all_dry = w.cooling.assign(cooling_physics="dry", cooling_category="air", requires_water=False)
    all_units = ep.get("fleet.build_recovered_unit_cell_mapping")(plants)
    return {
        "mapping": w.dry_mapping,
        "all_families": build(all_units, all_dry),
        "without_country": build(imputed.drop(columns=["country"]), w.cooling),
        "country_nowhere": outcome(
            build, imputed.drop(columns=["country"]), w.cooling.drop(columns=["country"])
        ),
        "missing_mapping_columns": outcome(build, imputed.drop(columns=["mw"]), w.cooling),
        "missing_cooling_columns": outcome(build, imputed, w.cooling.drop(columns=["utype"])),
        "duplicate_cooling_units": outcome(build, imputed, pd.concat([w.cooling, w.cooling.iloc[:1]])),
    }


def _dry_mapping_all_families(w) -> pd.DataFrame:
    plants = fixtures.plants()
    all_dry = w.cooling.assign(cooling_physics="dry", cooling_category="air", requires_water=False)
    units = ep.get("fleet.build_recovered_unit_cell_mapping")(plants)
    return ep.get("dry.build_mapping")(units, all_dry)


@case("dry.climate_inputs")
def dry_climate_inputs(w):
    normalize = ep.get("dry.normalize_climate_inputs")
    climate = fixtures.daily_air_temperature()
    celsius = climate.assign(tas=climate["tas"] - 273.15)
    celsius["tas"].attrs["units"] = "degC"
    return {
        "kelvin": normalize(climate),
        "celsius": normalize(celsius),
        "renamed": normalize(climate.rename(tas="t2m"), variable_names={"air_temperature_k": "t2m"}),
        "missing_variable": outcome(normalize, climate.drop_vars("tas")),
        "missing_units": outcome(normalize, climate.assign(tas=climate["tas"].drop_attrs())),
        "units_override": normalize(
            climate.assign(tas=climate["tas"].drop_attrs()),
            variable_units={"air_temperature_k": "K"},
        ),
        "units_override_wrong_key": outcome(
            normalize, climate.assign(tas=climate["tas"].drop_attrs()), variable_units={"tas": "K"}
        ),
    }


@case("dry.kernel.native")
def dry_kernel_native(w):
    evaluate = ep.get("kernel.evaluate_mapped_kernel")
    return {
        "fleet": evaluate(
            ep.get("dry.create_kernel")(), w.dry_mapping, fixtures.daily_air_temperature(), duration="native"
        ),
        "all_families": evaluate(
            ep.get("dry.create_kernel")(),
            _dry_mapping_all_families(w),
            fixtures.daily_air_temperature(),
            duration="native",
        ),
    }


@case("dry.kernel.durations")
def dry_kernel_durations(w):
    evaluate = ep.get("kernel.evaluate_mapped_kernel")
    climate = fixtures.daily_air_temperature()
    kernel = ep.get("dry.create_kernel")()
    mapping = _dry_mapping_all_families(w)
    return {
        "monthly": evaluate(kernel, mapping, climate, duration="monthly"),
        "seasonal": evaluate(kernel, mapping, climate, duration="seasonal"),
        "seasonal_chunked": evaluate(kernel, mapping, climate.chunk({"time": 90}), duration="seasonal"),
        "bad_duration": outcome(evaluate, kernel, mapping, climate, duration="weekly"),
    }


@case("dry.kernel.assumptions")
def dry_kernel_assumptions(w):
    evaluate = ep.get("kernel.evaluate_mapped_kernel")
    assumptions = ep.get("dry.Assumptions")(
        combustible_steam_parasitic_load_fraction=0.03,
        nuclear_steam_parasitic_load_fraction=0.05,
        unsupported_capacity_factor=0.5,
    )
    return evaluate(
        ep.get("dry.create_kernel")(assumptions),
        _dry_mapping_all_families(w),
        fixtures.daily_air_temperature(),
        duration="monthly",
    )


# --- regional aggregation ------------------------------------------------------------------


@case("aggregation.regions")
def aggregation_regions(w):
    evaluate = ep.get("kernel.evaluate_mapped_kernel")
    aggregate = ep.get("aggregation.aggregate_regions")
    wet = evaluate(_wet_kernel(), w.wet_mapping, fixtures.monthly_climate(), duration="seasonal")
    dry = evaluate(
        ep.get("dry.create_kernel")(),
        w.dry_mapping,
        fixtures.daily_air_temperature(),
        duration="seasonal",
    )
    unmapped = wet.copy(deep=True)
    countries = unmapped.coords["country"].values.astype(object).copy()
    countries[0] = "Atlantis"
    unmapped = unmapped.assign_coords(country=("entity", countries))
    with w.region_definitions():
        return {
            "wet_country": aggregate(wet, region_set="country"),
            "wet_r12": aggregate(wet, region_set="message_r12"),
            "dry_country": aggregate(dry, region_set="country"),
            "dry_r12": aggregate(dry, region_set="message_r12"),
            "unmapped_strict": outcome(aggregate, unmapped, region_set="message_r12"),
            "no_mw": outcome(aggregate, wet.drop_vars("mw"), region_set="country"),
            "no_country": outcome(aggregate, wet.drop_vars("country"), region_set="country"),
            "wrong_entity_dim": outcome(aggregate, wet, region_set="country", site_dim="unit"),
            "unknown_set": outcome(aggregate, wet, region_set="continents"),
        }


@case("aggregation.files")
def aggregation_files(w):
    directory = _work(w, "aggregation_files")
    evaluate = ep.get("kernel.evaluate_mapped_kernel")
    wet = evaluate(_wet_kernel(), w.wet_mapping, fixtures.monthly_climate(), duration="seasonal").compute()
    source = directory / "wet_seasonal.nc"
    wet.to_netcdf(source)
    with w.region_definitions():
        written = ep.get("aggregation.write_aggregated_file")(
            source, directory / "nested" / "r12.nc", region_set="message_r12", chunks={"entity": 3}
        )
        with xr.open_dataset(written) as reopened:
            written_content = reopened.load()
        return {
            "country": ep.get("aggregation.aggregate_file")(source, region_set="country"),
            "r12_chunked": ep.get("aggregation.aggregate_file")(
                source, region_set="message_r12", chunks={"entity": 3}
            ),
            "written_name": written.name,
            "written_content": written_content,
            "missing_file": outcome(
                ep.get("aggregation.aggregate_file"), directory / "absent.nc", region_set="country"
            ),
        }


# --- synthetic-plant evaluation -------------------------------------------------------------------


def _cells() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": ["+010.250_+020.250", "+010.750_+020.750", "+011.250_+020.250"],
            "region_id": ["AFR", "AFR", "WEU"],
            "country": ["NGA", "CMR", "FRA"],
            "grid_lat": [10.25, 10.75, 11.25],
            "grid_lon": [20.25, 20.75, 20.25],
            "cell_area_m2": [3.0e9, 2.9e9, 2.8e9],
            "population": ["candidate", "candidate", "incumbent"],
        }
    )


@case("archetypes.definitions")
def archetypes_definitions(w):
    wet = ep.get("archetypes.default_wet")()
    dry = ep.get("archetypes.default_dry")()
    build = ep.get("archetypes.build_synthetic_entities")
    cells = _cells()
    return {
        "wet": [archetype.to_record() for archetype in wet],
        "dry": [archetype.to_record() for archetype in dry],
        "wet_entities": build(cells, wet),
        "dry_entities": build(cells, dry),
        "no_cells": outcome(build, cells.iloc[0:0], wet),
        "no_archetypes": outcome(build, cells, []),
        "duplicate_cells": outcome(build, pd.concat([cells, cells.iloc[:1]]), wet),
    }


@case("archetypes.evaluation")
def archetypes_evaluation(w):
    build = ep.get("archetypes.build_synthetic_entities")
    wet_entities = build(_cells(), ep.get("archetypes.default_wet")())
    dry_entities = build(_cells(), ep.get("archetypes.default_dry")())
    climate = fixtures.monthly_climate()
    seasonal, monthly = ep.get("archetypes.evaluate_wet_seasonal_and_monthly")(wet_entities, climate)
    assumptions = ep.get("wet.Assumptions")(environmental_flow_fraction=0.6)
    return {
        "wet_seasonal": ep.get("archetypes.evaluate_wet")(wet_entities, climate),
        "wet_seasonal_of_pair": seasonal,
        "wet_monthly_of_pair": monthly,
        "wet_assumptions": ep.get("archetypes.evaluate_wet")(wet_entities, climate, assumptions=assumptions),
        "wet_chunked": ep.get("archetypes.evaluate_wet")(wet_entities, climate.chunk({"time": 4})),
        "dry": ep.get("archetypes.evaluate_dry")(dry_entities, fixtures.daily_air_temperature()),
        "dry_missing_variable": outcome(
            ep.get("archetypes.evaluate_dry"), dry_entities, fixtures.daily_air_temperature().drop_vars("tas")
        ),
    }


# --- water shared along the river -----------------------------------------------------------------------------


@case("reach.allocation")
def reach_allocation(w):
    order = ep.get("reach.topological_order")
    allocate = ep.get("reach.reach_capped_allocation")
    cumulative = ep.get("reach.cumulative_upstream_usage")
    residual = ep.get("reach.residual_ceilings")
    # a tributary tree: 0->2, 1->2, 2->4, 3->4, 4 is the outlet, 5 is isolated
    downstream = np.array([2, 2, 4, 4, -1, -1])
    ceiling = np.array([30.0, np.nan, 50.0, 10.0, 70.0, 0.0])
    placed_1d = np.array([40.0, 20.0, 15.0, 25.0, 5.0, 8.0])
    placed_2d = np.stack([placed_1d * scale for scale in (0.0, 0.5, 1.0, 2.0)], axis=1)
    return {
        "order": order(downstream),
        "order_chain": order(np.array([1, 2, 3, -1])),
        "order_cycle": outcome(order, np.array([1, 0, -1])),
        "order_self_loop": outcome(order, np.array([0])),
        "allocation_1d": allocate(ceiling, placed_1d, downstream),
        "allocation_2d": allocate(ceiling, placed_2d, downstream),
        "allocation_no_ceiling": allocate(np.full(6, np.nan), placed_1d, downstream),
        "allocation_zero_ceiling": allocate(np.zeros(6), placed_1d, downstream),
        "allocation_zero_placement": allocate(ceiling, np.zeros(6), downstream),
        "allocation_infinite": allocate(np.array([np.inf] * 6), placed_1d, downstream),
        "allocation_cycle": outcome(allocate, ceiling, placed_1d, np.array([1, 0, 4, 4, -1, -1])),
        "cumulative": cumulative(placed_1d, downstream),
        "residual": residual(ceiling, placed_1d, downstream),
        "residual_overdrawn": residual(np.array([5.0, 5.0, 5.0, 5.0, 5.0, np.nan]), placed_1d, downstream),
    }
