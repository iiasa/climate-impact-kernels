"""End-to-end tests of the shared core: units, time and space transforms, warming levels, regions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from . import entrypoints as ep
from . import fixtures
from .registry import case, outcome, outcome_type

# --- units -----------------------------------------------------------------------


@case("units.resolve_and_normalize")
def units_resolve_and_normalize(w):
    temperature = xr.DataArray([0.0, 20.0, np.nan, -40.0], dims="x", name="t", attrs={"units": "C"})
    kelvin = xr.DataArray([273.15, 300.0, np.nan], dims="x", name="t", attrs={"units": " K "})
    pressure = xr.DataArray([101325.0, 0.0, np.nan], dims="x", name="p", attrs={"units": "Pa"})
    kilo = xr.DataArray([101.325, 90.0], dims="x", name="p", attrs={"units": "kPa", "note": "kept"})
    resolve = ep.get("units.resolve_unit")
    to_temperature = ep.get("units.normalize_temperature")
    to_pressure = ep.get("units.normalize_pressure")
    return {
        "unit_attr": resolve(kelvin),
        "unit_override": resolve(kelvin, variable_units={"t": "C"}),
        "unit_missing": outcome(resolve, xr.DataArray([1.0], dims="x", name="q")),
        "c_to_k": to_temperature(temperature, target_unit="K", unit="C"),
        "k_to_c": to_temperature(kelvin, target_unit="degC", unit="Kelvin"),
        "c_to_c": to_temperature(temperature, target_unit="celsius", unit="degrees_celsius"),
        "k_to_k": to_temperature(kelvin, target_unit="kelvin", unit="k"),
        "temperature_unsupported": outcome(
            to_temperature, temperature, target_unit="F", unit="C"
        ),
        "pa_to_kpa": to_pressure(pressure, target_unit="kPa", unit="Pa"),
        "kpa_to_pa": to_pressure(kilo, target_unit="pascal", unit="kilopascal"),
        "kpa_to_kpa": to_pressure(kilo, target_unit="kPa", unit="kPa"),
        "pressure_unsupported": outcome(to_pressure, pressure, target_unit="bar", unit="Pa"),
    }


# --- temporal ----------------------------------------------------------------------


def _daily_series() -> xr.Dataset:
    time = pd.date_range("2000-11-20", "2002-01-10", freq="D")
    rng = np.random.default_rng(5)
    values = rng.normal(10.0, 3.0, (len(time), 2))
    values[40, 0] = np.nan
    return xr.Dataset(
        {
            "a": (("time", "x"), values),
            "b": (("time",), np.arange(len(time), dtype=float)),
            "c": (("x",), [1.0, 2.0]),
        },
        coords={"time": time, "x": [0, 1]},
    )


@case("temporal.aggregate")
def temporal_aggregate(w):
    daily = _daily_series()
    monthly = ep.get("temporal.aggregate_monthly")(daily)
    seasonal = ep.get("temporal.aggregate_seasonal")
    return {
        "monthly": monthly,
        "seasonal_daily": seasonal(daily),
        "seasonal_monthly": seasonal(monthly),
        "seasonal_dataarray": seasonal(monthly["a"]),
        "seasonal_subset": seasonal(monthly, seasons={"JJA": (6, 7, 8), "DJF": (12, 1, 2)}),
        "seasonal_shifted": seasonal(
            monthly, seasons={"NDJ": (11, 12, 1), "FMA": (2, 3, 4), "MJJ": (5, 6, 7), "ASO": (8, 9, 10)}
        ),
        "seasonal_chunked": seasonal(monthly.chunk({"time": 4})),
        "no_time": outcome(seasonal, daily[["c"]]),
        "bad_season": outcome(seasonal, monthly, seasons={"X": (1, 2, 4)}),
        "mixed_phase": outcome(seasonal, monthly, seasons={"DJF": (12, 1, 2), "AMJ": (4, 5, 6)}),
        "empty_seasons": outcome(seasonal, monthly, seasons={}),
    }


@case("temporal.aggregate_cftime")
def temporal_aggregate_cftime(w):
    time = xr.date_range("2001-01-01", periods=720, freq="D", calendar="360_day", use_cftime=True)
    data = xr.DataArray(np.arange(720.0), dims="time", coords={"time": time}, name="v")
    return {
        "monthly": ep.get("temporal.aggregate_monthly")(data),
        "seasonal": ep.get("temporal.aggregate_seasonal")(data),
    }


# --- spatial -------------------------------------------------------------------------


@case("spatial.transforms")
def spatial_transforms(w):
    rng = np.random.default_rng(6)
    lat = xr.DataArray(fixtures.LATS, dims="lat", coords={"lat": fixtures.LATS})
    lon = xr.DataArray(fixtures.LONS, dims="lon", coords={"lon": fixtures.LONS})
    values = xr.DataArray(
        rng.normal(size=(2, 4, 3)),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": fixtures.LATS, "lon": fixtures.LONS},
        name="v",
    )
    values[0, 0, 0] = np.nan
    weights = ep.get("spatial.grid_cell_area")(lat, lon)
    return {
        "cell_area": weights,
        "cell_area_single": ep.get("spatial.grid_cell_area")(
            lat.isel(lat=[0]), lon.isel(lon=[0]), resolution_deg=1.0
        ),
        "cell_area_2d": outcome(ep.get("spatial.grid_cell_area"), values, lon),
    }


@case("spatial.attach_gridded")
def spatial_attach_gridded(w):
    grid = xr.Dataset(
        {
            "v": (("time", "lat", "lon"), np.arange(24.0).reshape(2, 4, 3)),
            "static": (("lat", "lon"), np.arange(12.0).reshape(4, 3)),
        },
        coords={"time": [0, 1], "lat": fixtures.LATS, "lon": fixtures.LONS},
    )
    mapping = pd.DataFrame(
        {
            "UNITID": [7, 8, 9],
            "LOCATIONID": [1, 1, 2],
            "grid_lat": [10.25, 11.75, 10.75],
            "grid_lon": [20.25, 21.25, 20.75],
            "mw": [10.0, 20.0, 30.0],
            "tag": ["a", "b", "c"],
        }
    )
    attach = ep.get("spatial.attach_gridded_dataset")
    shifted = grid.assign_coords(lon=[350.25, 350.75, 351.25])
    mapping_west = mapping.assign(grid_lon=[-9.75, -8.75, -9.25])
    return {
        "all_metadata": attach(mapping, grid, site_id_column="UNITID"),
        "selected_metadata": attach(mapping, grid, site_id_column="UNITID", metadata_columns=("UNITID", "mw")),
        "no_metadata": attach(mapping, grid, site_id_column="UNITID", metadata_columns=()),
        "entity_id_column": attach(mapping, grid, site_id_column="tag", metadata_columns=("tag",)),
        "dataarray": ep.get("spatial.attach_gridded_dataarray")(mapping, grid["v"], site_id_column="UNITID"),
        "longitude_360": attach(mapping_west, shifted, site_id_column="UNITID", metadata_columns=("UNITID",)),
        "missing_cell": outcome(attach, mapping.assign(grid_lat=[10.25, 11.75, 99.25]), grid, site_id_column="UNITID"),
        "missing_columns": outcome(attach, mapping.drop(columns=["grid_lat"]), grid, site_id_column="UNITID"),
        "missing_coordinate": outcome(attach, mapping, grid.rename(lat="latitude"), site_id_column="UNITID"),
        "bad_metadata_column": outcome(attach, mapping, grid, site_id_column="UNITID", metadata_columns=("nope",)),
    }


# --- warming levels -----------------------------------------------------------------------


def _gmt(values, years=None) -> xr.DataArray:
    years = years if years is not None else list(range(2000, 2000 + len(values)))
    return xr.DataArray(values, dims="year", coords={"year": years}, name="gmt")


@case("warming.levels")
def warming_levels(w):
    gmt = _gmt([0.7, 0.95, 1.1, 1.49, 1.5, 1.75, np.nan, 2.05, 2.5, 2.52])
    assign = ep.get("warming.assign_years_to_levels")
    values = xr.DataArray(np.arange(10.0), dims="year", coords={"year": gmt["year"]}, name="v")
    return {
        "assign": assign(gmt, [1.0, 1.5, 2.0, 2.5]),
        "assign_two_levels": assign(gmt, [1.0, 2.0]),
        "assign_empty": outcome(assign, gmt, []),
        "bin_mean": ep.get("warming.bin_by_warming_level")(values, gmt, [1.0, 2.0, 3.0]),
        "bin_sum": ep.get("warming.bin_by_warming_level")(values, gmt, [1.0, 2.0], reducer="sum"),
        "bin_bad_reducer": outcome(
            ep.get("warming.bin_by_warming_level"), values, gmt, [1.0, 2.0], reducer="max"
        ),
    }


@case("warming.assignments")
def warming_assignments(w):
    later = _gmt([0.8, 1.35, 1.6, 1.9], years=[2015, 2016, 2017, 2018])
    ssp = _gmt([1.0, 1.2, np.nan, 2.4], years=[2015, 2016, 2017, 2018])
    other = _gmt([0.7, 0.9, 1.6, 2.2], years=[2015, 2016, 2017, 2018])
    ragged = _gmt([0.9, 1.6, 2.2], years=[2016, 2017, 2018])
    gmt_by_path = {
        ("GFDL-ESM4", "ssp585"): later,
        ("GFDL-ESM4", "ssp126"): ssp,
        ("IPSL-CM6A-LR", "ssp126"): other,
    }
    build = ep.get("warming.build_warming_level_assignments")
    assignments = build(gmt_by_path, path_columns=("gcm", "scenario"), bin_width=0.5)
    return {
        "assignments": assignments,
        "assignments_clamped": build(
            gmt_by_path, path_columns=("gcm", "scenario"), bin_width=0.25,
            min_warming_level=0.5, max_warming_level=2.0,
        ),
        "assignments_ragged_years": outcome_type(
            build, {**gmt_by_path, ("IPSL-CM6A-LR", "ssp126"): ragged}, path_columns=("gcm", "scenario")
        ),
        "assignments_bad_width": outcome(build, gmt_by_path, path_columns=("gcm", "scenario"), bin_width=0),
        "assignments_bad_key": outcome(build, gmt_by_path, path_columns=("gcm",)),
        "assignments_empty": outcome(build, {}, path_columns=("gcm",)),
        "assignments_negative_min": outcome(
            build, gmt_by_path, path_columns=("gcm", "scenario"), min_warming_level=-1.0
        ),
    }


# --- regions ----------------------------------------------------------------------------------


@case("regions.registry")
def regions_registry(w):
    resolve = ep.get("regions.resolve_region_labels")
    members = [
        "USA", "VIETNAM", "Nigeria", "FRANCE", "CHANNEL ISLANDS", "DUTCH CARIBBEAN",
        "Cameroon", "R12_NAM", "scotland", "  Japan  ",
    ]  # fmt: skip
    with w.region_definitions():
        return {
            "labels": ep.get("regions.message_r12_region_labels")(),
            "country_to_region": ep.get("regions.message_r12_country_to_region")(),
            "resolve_country": resolve(["a", "b", 3], region_set="country"),
            "resolve_r12_strict": outcome(resolve, members, region_set="message_r12"),
            "resolve_r12_lenient": resolve(members, region_set="message_r12", strict=False),
            "resolve_r12_mapped": resolve(
                [m for m in members if m != "DUTCH CARIBBEAN"], region_set="message_r12"
            ),
            "resolve_strict_unmapped": outcome(
                resolve, ["Nigeria", "Atlantis", "Mu"], region_set="message_r12"
            ),
            "resolve_lenient": outcome(
                resolve, ["Nigeria", "Atlantis"], region_set="message_r12", strict=False
            ),
            "resolve_unknown_set": outcome(resolve, ["Nigeria"], region_set="un_regions"),
        }


@case("regions.aggregate_entities")
def regions_aggregate_entities(w):
    entity = [101, 202, 303, 404, 505]
    dataset = xr.Dataset(
        {
            "capacity_factor": (("time", "entity"), [[0.5, 1.0, 0.25, 0.75, np.nan], [0.1, 0.2, 0.3, 0.4, 0.5]]),
            "usable_capacity_mw": (("time", "entity"), [[50.0, 300.0, 50.0, 300.0, 10.0], [1.0, 2.0, 3.0, 4.0, 5.0]]),
            "static": (("entity",), [1.0, 2.0, 3.0, 4.0, 5.0]),
            "scalar": ((), 3.0),
        },
        coords={"time": [0, 1], "entity": entity},
        attrs={"origin": "fixture"},
    )
    labels = ["R12_NAM", "R12_NAM", "R12_WEU", "R12_PAO", "R12_WEU"]
    weights = xr.DataArray([100.0, 300.0, 200.0, 400.0, 0.0], dims="entity", coords={"entity": entity})
    dataarray = ep.get("regions.aggregate_entity_dataarray")
    dataset_fn = ep.get("regions.aggregate_entity_dataset")
    return {
        "sum": dataarray(dataset["usable_capacity_mw"], labels, reduction="sum"),
        "weighted_mean": dataarray(
            dataset["capacity_factor"], labels, reduction="weighted_mean", weights=weights
        ),
        "weighted_mean_list_weights": dataarray(
            dataset["static"], labels, reduction="weighted_mean", weights=[1.0, 2.0, 3.0, 4.0, 5.0],
            region_dim="zone",
        ),
        "weights_required": outcome(dataarray, dataset["static"], labels, reduction="weighted_mean"),
        "zero_weight_region": outcome(
            dataarray, dataset["static"], labels, reduction="weighted_mean",
            weights=[1.0, 1.0, 0.0, 1.0, 0.0],
        ),
        "bad_reduction": outcome(dataarray, dataset["static"], labels, reduction="median"),
        "label_length": outcome(dataarray, dataset["static"], labels[:3], reduction="sum"),
        "weights_bad_dims": outcome(
            dataarray, dataset["static"], labels, reduction="weighted_mean",
            weights=xr.DataArray([[1.0]], dims=("a", "b")),
        ),
        "dataset": dataset_fn(
            dataset.drop_vars("static"), labels,
            variable_reductions={"capacity_factor": "weighted_mean", "usable_capacity_mw": "sum", "scalar": "sum"},
            weights=weights,
        ),
        "dataset_not_entity": outcome(
            dataset_fn, dataset.rename(entity="unit"), labels, variable_reductions={"static": "sum"}
        ),
    }


# --- public functions and constants ----------------------------------------------------------------


@case("public.contracts_and_constants")
def public_contracts_and_constants(w):
    constants = sorted(name for name in (*ep.TABLE, *ep.DERIVED) if name.startswith("constants."))
    return {
        "constants": {name.split(".", 1)[1]: ep.get(name) for name in constants},
        "cell_mapping_summary_fields": [field for field in ep.get("types.CellMappingSummary").__dataclass_fields__],
        "location_ambiguity_fields": [field for field in ep.get("types.LocationAmbiguity").__dataclass_fields__],
    }
