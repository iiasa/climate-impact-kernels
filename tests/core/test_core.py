import unittest
from datetime import datetime
import os

import cftime
import numpy as np
import pandas as pd
import xarray as xr

try:
    import dask.array as da  # noqa: F401
except ImportError:
    da = None

from climateresponse import (
    aggregate_by_region,
    aggregate_annual,
    aggregate_monthly,
    aggregate_seasonal,
    attach_gridded_dataarray,
    attach_gridded_dataset,
    attach_warming_levels,
    assign_years_to_levels,
    bin_by_warming_level,
    build_warming_level_assignments,
    build_warming_level_quantile_surface,
    compute_anomaly,
    compute_running_mean,
    deterministic_resampling,
    evaluate_mapped_dataset,
    grid_cell_area,
    interpolate_along_warming_level,
    match_warming_levels,
    normalize_pressure,
    normalize_temperature,
    resolve_unit,
    weighted_average,
    weighted_quantiles,
)
from climateresponse.regions import (
    aggregate_entity_dataarray,
    aggregate_entity_dataset,
    common_definitions_root,
    list_country_geometry_sources,
    message_r12_country_to_region,
    message_r12_region_labels,
    resolve_region_labels,
)


class BaselineTest(unittest.TestCase):
    def test_compute_anomaly_uses_baseline_mean(self) -> None:
        anomalies = compute_anomaly(
            data=xr.DataArray([1.0, 3.0, 5.0], dims=["year"], coords={"year": [2000, 2001, 2002]}),
            baseline_period=(2000, 2001),
        )

        self.assertTrue(np.allclose(anomalies.values, [-1.0, 1.0, 3.0]))


class GMTTest(unittest.TestCase):
    def test_centered_running_mean(self) -> None:
        running = compute_running_mean(
            data=xr.DataArray(
                [0.0, 1.0, 2.0, 3.0, 4.0],
                dims=["year"],
                coords={"year": [2000, 2001, 2002, 2003, 2004]},
            ),
            window=3,
        )

        self.assertTrue(np.isnan(running.sel(year=2000)))
        self.assertEqual(running.sel(year=2002).item(), 2.0)


class WarmingLevelTest(unittest.TestCase):
    def test_match_warming_levels_uses_buckets_and_diagnostics(self) -> None:
        matches = match_warming_levels(
            gmt=xr.DataArray(
                [0.7, 1.1, 1.6],
                dims=["year"],
                coords={"year": [2000, 2001, 2002]},
            ),
            target_levels=[1.0, 1.5],
            bucket_width=0.5,
        )

        self.assertEqual(matches["year"].values.tolist(), [2001, 2002])
        self.assertEqual(matches["warming_level"].values.tolist(), [1.0, 1.5])
        self.assertTrue("accumulated_warming" in matches.data_vars)

    def test_assign_years_to_levels(self) -> None:
        assignments = assign_years_to_levels(
            gmt=xr.DataArray(
                [0.9, 1.2, 1.8],
                dims=["year"],
                coords={"year": [2000, 2001, 2002]},
            ),
            target_levels=[1.0, 2.0],
        )

        self.assertEqual(assignments["warming_level"].values.tolist(), [1.0, 1.0, 2.0])

    def test_bin_by_warming_level_groups_values(self) -> None:
        bins = bin_by_warming_level(
            values=xr.DataArray(
                [10.0, 11.0, 12.0],
                dims=["year"],
                coords={"year": [2000, 2001, 2002]},
            ),
            gmt=xr.DataArray(
                [0.9, 1.2, 1.8],
                dims=["year"],
                coords={"year": [2000, 2001, 2002]},
            ),
            target_levels=[1.0, 2.0],
        )

        self.assertTrue(np.allclose(bins.values, [10.5, 12.0]))


class TemporalTest(unittest.TestCase):
    def test_aggregate_monthly_preserves_pandas_month_starts(self) -> None:
        monthly = aggregate_monthly(
            xr.DataArray(
                [1.0, 3.0, 5.0],
                dims=["time"],
                coords={
                    "time": np.array(
                        ["2000-01-01", "2000-01-15", "2000-02-01"],
                        dtype="datetime64[ns]",
                    )
                },
            )
        )

        self.assertEqual(
            monthly.time.values.tolist(),
            [datetime(2000, 1, 1), datetime(2000, 2, 1)],
        )
        self.assertEqual(monthly.sel(time="2000-01-01").item(), 2.0)
        self.assertEqual(monthly.sel(time="2000-02-01").item(), 5.0)

    def test_aggregate_monthly_preserves_360_day_calendar(self) -> None:
        time = [
            cftime.Datetime360Day(2001, 2, 1),
            cftime.Datetime360Day(2001, 2, 30),
            cftime.Datetime360Day(2001, 3, 1),
        ]
        monthly = aggregate_monthly(
            xr.DataArray(
                [10.0, 14.0, 20.0],
                dims=["time"],
                coords={"time": time},
            )
        )

        self.assertEqual(
            monthly.time.values.tolist(),
            [
                cftime.Datetime360Day(2001, 2, 1),
                cftime.Datetime360Day(2001, 3, 1),
            ],
        )
        self.assertEqual(monthly.sel(time=cftime.Datetime360Day(2001, 2, 1)).item(), 12.0)
        self.assertEqual(monthly.sel(time=cftime.Datetime360Day(2001, 3, 1)).item(), 20.0)

    def test_aggregate_monthly_preserves_monthly_constants_across_lengths(self) -> None:
        day_counts = [31, 28, 31, 30]
        values = []
        time = []

        for month, day_count in enumerate(day_counts, start=1):
            values.extend([float(month)] * day_count)
            time.extend(cftime.DatetimeNoLeap(2001, month, day) for day in range(1, day_count + 1))

        monthly = aggregate_monthly(
            xr.DataArray(
                values,
                dims=["time"],
                coords={"time": time},
            )
        )

        self.assertEqual(monthly.values.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_aggregate_annual(self) -> None:
        annual = aggregate_annual(
            xr.DataArray(
                [1.0, 3.0, 5.0],
                dims=["time"],
                coords={
                    "time": np.array(
                        ["2000-01-01", "2000-02-01", "2001-01-01"],
                        dtype="datetime64[ns]",
                    )
                },
            )
        )

        self.assertEqual(annual.sel(year=2000).item(), 2.0)
        self.assertEqual(annual.sel(year=2001).item(), 5.0)

    def test_aggregate_seasonal_rolls_december_forward(self) -> None:
        seasonal = aggregate_seasonal(
            xr.DataArray(
                [6.0, 3.0, 0.0],
                dims=["time"],
                coords={
                    "time": np.array(
                        ["2000-12-01", "2001-01-01", "2001-02-01"],
                        dtype="datetime64[ns]",
                    )
                },
            )
        )

        self.assertEqual(seasonal.sel(season="DJF", year=2001).item(), 3.0)

    def test_aggregate_seasonal_preserves_standard_subset_order(self) -> None:
        seasonal = aggregate_seasonal(
            xr.DataArray(
                [12.0, 1.0, 2.0, 6.0, 7.0, 8.0],
                dims=["time"],
                coords={
                    "time": np.array(
                        [
                            "2000-12-01",
                            "2001-01-01",
                            "2001-02-01",
                            "2001-06-01",
                            "2001-07-01",
                            "2001-08-01",
                        ],
                        dtype="datetime64[ns]",
                    )
                },
            ),
            seasons={"JJA": (6, 7, 8), "DJF": (12, 1, 2)},
        )

        self.assertEqual(seasonal.coords["season"].values.tolist(), ["JJA", "DJF"])
        self.assertEqual(seasonal.sel(season="JJA", year=2001).item(), 7.0)
        self.assertEqual(seasonal.sel(season="DJF", year=2001).item(), 5.0)

    def test_aggregate_seasonal_supports_phase_shifted_quarters(self) -> None:
        seasonal = aggregate_seasonal(
            xr.DataArray(
                np.arange(1.0, 13.0),
                dims=["time"],
                coords={
                    "time": np.array(
                        [
                            "2001-01-01",
                            "2001-02-01",
                            "2001-03-01",
                            "2001-04-01",
                            "2001-05-01",
                            "2001-06-01",
                            "2001-07-01",
                            "2001-08-01",
                            "2001-09-01",
                            "2001-10-01",
                            "2001-11-01",
                            "2001-12-01",
                        ],
                        dtype="datetime64[ns]",
                    )
                },
            ),
            seasons={
                "JFM": (1, 2, 3),
                "AMJ": (4, 5, 6),
                "JAS": (7, 8, 9),
                "OND": (10, 11, 12),
            },
        )

        self.assertEqual(seasonal.coords["season"].values.tolist(), ["JFM", "AMJ", "JAS", "OND"])
        self.assertEqual(seasonal.sel(season="JFM", year=2001).item(), 2.0)
        self.assertEqual(seasonal.sel(season="OND", year=2001).item(), 11.0)

    def test_aggregate_seasonal_preserves_360_day_calendar(self) -> None:
        seasonal = aggregate_seasonal(
            xr.DataArray(
                [6.0, 3.0, 0.0],
                dims=["time"],
                coords={
                    "time": [
                        cftime.Datetime360Day(2000, 12, 1),
                        cftime.Datetime360Day(2001, 1, 1),
                        cftime.Datetime360Day(2001, 2, 1),
                    ]
                },
            )
        )

        self.assertEqual(seasonal.sel(season="DJF", year=2001).item(), 3.0)

    def test_aggregate_seasonal_rejects_unsupported_seasons(self) -> None:
        data = xr.DataArray(
            [4.0, 5.0, 6.0],
            dims=["time"],
            coords={
                "time": np.array(
                    ["2001-04-01", "2001-05-01", "2001-06-01"],
                    dtype="datetime64[ns]",
                )
            },
        )

        with self.assertRaisesRegex(ValueError, "quarterly phase"):
            aggregate_seasonal(
                data,
                seasons={
                    "DJF": (12, 1, 2),
                    "JFM": (1, 2, 3),
                },
            )

    def test_aggregate_seasonal_supports_datasets_with_heterogeneous_dims(self) -> None:
        seasonal = aggregate_seasonal(
            xr.Dataset(
                data_vars={
                    "capacity_factor": (
                        ("time", "entity"),
                        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    ),
                    "mean_temperature": (
                        ("time",),
                        [10.0, 12.0, 14.0],
                    ),
                },
                coords={
                    "time": np.array(
                        ["2001-01-01", "2001-02-01", "2001-03-01"],
                        dtype="datetime64[ns]",
                    ),
                    "entity": [101, 202],
                },
            ),
            seasons={"JFM": (1, 2, 3)},
        )

        self.assertEqual(seasonal["capacity_factor"].dims, ("season", "year", "entity"))
        self.assertEqual(seasonal["mean_temperature"].dims, ("season", "year"))
        self.assertEqual(seasonal["mean_temperature"].sel(season="JFM", year=2001).item(), 12.0)

    @unittest.skipIf(da is None, "dask is not installed")
    def test_aggregate_seasonal_consolidates_standard_dask_chunks(self) -> None:
        time = pd.date_range("2000-01-01", "2001-12-31", freq="D")
        data = xr.DataArray(
            np.arange(len(time) * 4, dtype=float).reshape(len(time), 4),
            dims=["time", "entity"],
            coords={"time": time, "entity": [101, 202, 303, 404]},
        ).chunk({"time": 90, "entity": 2})

        seasonal = aggregate_seasonal(data)

        self.assertEqual(seasonal.dims, ("season", "year", "entity"))
        self.assertIsInstance(seasonal.data, da.Array)
        self.assertEqual(seasonal.chunks[0], (4,))
        self.assertEqual(seasonal.chunks[1], (3,))
        self.assertEqual(seasonal.chunks[2], (2, 2))

    @unittest.skipIf(da is None, "dask is not installed")
    def test_aggregate_seasonal_keeps_datasets_lazy(self) -> None:
        seasonal = aggregate_seasonal(
            xr.DataArray(
                np.arange(12.0),
                dims=["time"],
                coords={"time": pd.date_range("2001-01-01", periods=12, freq="MS")},
            ).chunk({"time": 3}).to_dataset(name="tas")
        )

        self.assertTrue(hasattr(seasonal["tas"].data, "__dask_graph__"))


class SpatialTest(unittest.TestCase):
    def test_weighted_average_respects_mask(self) -> None:
        value = weighted_average(
            values=xr.DataArray(
                [[10.0, 20.0], [30.0, 40.0]],
                dims=["lat", "lon"],
                coords={"lat": [0, 1], "lon": [0, 1]},
            ),
            weights=xr.DataArray(
                [[1.0, 3.0], [1.0, 1.0]],
                dims=["lat", "lon"],
                coords={"lat": [0, 1], "lon": [0, 1]},
            ),
            mask=xr.DataArray(
                [[True, False], [False, False]],
                dims=["lat", "lon"],
                coords={"lat": [0, 1], "lon": [0, 1]},
            ),
            dims=("lat", "lon"),
        )

        self.assertEqual(value.item(), 10.0)

    def test_attach_gridded_dataset_preserves_requested_metadata(self) -> None:
        attached = attach_gridded_dataset(
            mapped_entities=pd.DataFrame(
                {
                    "UNITID": [101, 202],
                    "country": ["A", "B"],
                    "grid_lat": [0.5, 1.5],
                    "grid_lon": [-0.5, 0.5],
                }
            ),
            dataset=xr.Dataset(
                data_vars={
                    "tas": (
                        ("time", "lat", "lon"),
                        [[[1.0, 2.0], [3.0, 4.0]]],
                    )
                },
                coords={"time": [0], "lat": [0.5, 1.5], "lon": [-0.5, 0.5]},
            ),
            metadata_columns=["UNITID", "country"],
        )

        self.assertEqual(attached["tas"].dims, ("time", "entity"))
        self.assertEqual(attached.coords["UNITID"].values.tolist(), [101, 202])
        self.assertEqual(attached.coords["country"].values.tolist(), ["A", "B"])

    def test_attach_gridded_dataarray_normalizes_longitude_to_360_grid(self) -> None:
        attached = attach_gridded_dataarray(
            mapped_entities=pd.DataFrame(
                {
                    "UNITID": [101],
                    "grid_lat": [0.5],
                    "grid_lon": [-0.5],
                }
            ),
            data=xr.DataArray(
                [[[9.0]]],
                dims=["time", "lat", "lon"],
                coords={"time": [0], "lat": [0.5], "lon": [359.5]},
                name="tas",
            ),
        )

        self.assertEqual(attached.sel(entity=101).item(), 9.0)

    def test_grid_cell_area_returns_positive_weights(self) -> None:
        weights = grid_cell_area(
            xr.DataArray([0.0, 1.0], dims=["lat"], coords={"lat": [0.0, 1.0]}),
            xr.DataArray([10.0, 11.0], dims=["lon"], coords={"lon": [10.0, 11.0]}),
        )

        self.assertEqual(weights.dims, ("lat", "lon"))
        self.assertTrue((weights > 0).all().item())

    def test_aggregate_by_region_supports_weighted_mean_and_sum(self) -> None:
        values = xr.DataArray(
            [[[1.0, 2.0], [3.0, 4.0]]],
            dims=["time", "lat", "lon"],
            coords={"time": [0], "lat": [0.0, 1.0], "lon": [10.0, 11.0]},
            name="value",
        )
        regions = xr.DataArray(
            [["A", "A"], ["B", "B"]],
            dims=["lat", "lon"],
            coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]},
            name="country",
        )
        weights = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0]],
            dims=["lat", "lon"],
            coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]},
        )

        regional_mean = aggregate_by_region(
            values,
            regions,
            weights=weights,
            reducer="weighted_mean",
            region_dim="country",
        )
        regional_sum = aggregate_by_region(
            values,
            regions,
            weights=weights,
            reducer="weighted_sum",
            region_dim="country",
        )

        self.assertAlmostEqual(regional_mean.sel(country="A").item(), 5.0 / 3.0)
        self.assertAlmostEqual(regional_mean.sel(country="B").item(), 25.0 / 7.0)
        self.assertAlmostEqual(regional_sum.sel(country="A").item(), 5.0)
        self.assertAlmostEqual(regional_sum.sel(country="B").item(), 25.0)

    def test_aggregate_entity_dataarray_weighted_mean(self) -> None:
        values = xr.DataArray(
            [[0.5, 1.0, 0.25]],
            dims=["time", "entity"],
            coords={"time": [0], "entity": [101, 202, 303]},
        )

        reduced = aggregate_entity_dataarray(
            values,
            region_labels=["A", "A", "B"],
            reduction="weighted_mean",
            weights=xr.DataArray(
                [100.0, 300.0, 200.0],
                dims=["entity"],
                coords={"entity": [101, 202, 303]},
            ),
        )

        self.assertEqual(reduced.dims, ("time", "region"))
        self.assertAlmostEqual(reduced.sel(region="A").item(), 0.875)
        self.assertAlmostEqual(reduced.sel(region="B").item(), 0.25)

    def test_aggregate_entity_dataset_mixes_sum_and_weighted_mean(self) -> None:
        dataset = xr.Dataset(
            data_vars={
                "capacity_factor": (("time", "entity"), [[0.5, 1.0, 0.25]]),
                "usable_capacity_mw": (("time", "entity"), [[50.0, 300.0, 50.0]]),
            },
            coords={
                "time": [0],
                "entity": [101, 202, 303],
                "mw": ("entity", [100.0, 300.0, 200.0]),
            },
        )

        reduced = aggregate_entity_dataset(
            dataset,
            region_labels=["A", "A", "B"],
            variable_reductions={
                "capacity_factor": "weighted_mean",
                "usable_capacity_mw": "sum",
            },
            weights=dataset.coords["mw"],
        )

        self.assertAlmostEqual(reduced["capacity_factor"].sel(region="A").item(), 0.875)
        self.assertAlmostEqual(reduced["usable_capacity_mw"].sel(region="A").item(), 350.0)


class RegionRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        if os.environ.get("COMMON_DEFINITIONS_ROOT") is None:
            self.skipTest("COMMON_DEFINITIONS_ROOT is required for region registry tests")

    def test_message_r12_resolution_uses_upstream_mapping_and_aliases(self) -> None:
        resolved = resolve_region_labels(
            ["USA", "VIETNAM", "BOSNIA-HERZEGOVINA", "CHANNEL ISLANDS"],
            region_set="message_r12",
        )

        self.assertEqual(resolved, ("R12_NAM", "R12_RCPA", "R12_EEU", "R12_WEU"))
        self.assertEqual(message_r12_region_labels()["R12_NAM"], "North America")
        self.assertEqual(message_r12_country_to_region()["United States"], "R12_NAM")
        self.assertEqual(common_definitions_root().name, "common-definitions")

    def test_country_geometry_source_registry_contains_natural_earth(self) -> None:
        source = list_country_geometry_sources()[0]

        self.assertEqual(source.source, "Natural Earth")
        self.assertEqual(source.dataset_slug, "ne_110m_admin_0_countries")


class UnitsTest(unittest.TestCase):
    def test_resolve_and_normalize_temperature(self) -> None:
        data = xr.DataArray([300.15], dims=["time"], name="tas", attrs={"units": "K"})

        unit = resolve_unit(data)
        normalized = normalize_temperature(data, target_unit="C", unit=unit)

        self.assertEqual(unit, "K")
        self.assertAlmostEqual(normalized.item(), 27.0)

    def test_normalize_pressure(self) -> None:
        normalized = normalize_pressure(
            xr.DataArray([101325.0], dims=["time"], name="ps", attrs={"units": "Pa"}),
            target_unit="kPa",
            unit="Pa",
        )

        self.assertAlmostEqual(normalized.item(), 101.325)


class EvaluationTest(unittest.TestCase):
    def test_evaluate_mapped_dataset_runs_full_pipeline(self) -> None:
        evaluated = evaluate_mapped_dataset(
            mapped_entities=pd.DataFrame(
                {
                    "UNITID": [101],
                    "grid_lat": [0.5],
                    "grid_lon": [1.5],
                    "mw": [100.0],
                }
            ),
            climate_inputs=xr.Dataset(
                data_vars={
                    "tas": (
                        ("time", "lat", "lon"),
                        [[[300.15, 301.15]]],
                    )
                },
                coords={"time": [0], "lat": [0.5], "lon": [0.5, 1.5]},
            ),
            normalize_inputs=lambda dataset: dataset.rename({"tas": "air_temperature"}),
            physics_kernel=lambda attached: xr.Dataset(
                data_vars={
                    "capacity_factor": ((1.0 - (attached["air_temperature"] - 300.0))).rename("capacity_factor"),
                }
            ),
            metadata_columns=["UNITID", "mw"],
            clips={"capacity_factor": (0.0, 1.0)},
        )

        self.assertEqual(evaluated["capacity_factor"].dims, ("time", "entity"))
        self.assertEqual(evaluated.coords["UNITID"].values.tolist(), [101])
        self.assertAlmostEqual(evaluated["capacity_factor"].sel(entity=101).item(), 0.0)


class ResponseTest(unittest.TestCase):
    def test_linear_interpolation(self) -> None:
        interpolated = interpolate_along_warming_level(
            values=xr.DataArray(
                [10.0, 30.0],
                dims=["warming_level"],
                coords={"warming_level": [1.0, 2.0]},
            ),
            target_levels=1.5,
        )

        self.assertEqual(interpolated.item(), 20.0)


class StatsTest(unittest.TestCase):
    def test_weighted_quantiles(self) -> None:
        quantiles = weighted_quantiles(
            values=[1.0, 2.0, 3.0],
            weights=[1.0, 1.0, 2.0],
            quantiles=[0.25, 0.75],
        )

        self.assertEqual(quantiles, (1.0, 3.0))

    def test_deterministic_resampling(self) -> None:
        samples = deterministic_resampling(values=[1.0, 2.0, 3.0, 4.0], n_samples=2)

        self.assertEqual(samples, (1.0, 3.0))


class WarmingLevelLookupTest(unittest.TestCase):
    def test_assignments_attach_and_quantile_surface_round_trip(self) -> None:
        gmt_by_path = {
            ("GFDL-ESM4", "historical"): xr.DataArray(
                [0.1, 0.6],
                dims=["year"],
                coords={"year": [1850, 1851]},
            ),
            ("GFDL-ESM4", "ssp126"): xr.DataArray(
                [1.0, 1.0],
                dims=["year"],
                coords={"year": [2015, 2016]},
            ),
        }
        values = xr.DataArray(
            [[
                [
                    [[0.4], [0.6], [np.nan], [np.nan]],
                    [[np.nan], [np.nan], [0.7], [0.9]],
                ]
            ]],
            dims=["impact_type", "gcm", "scenario", "time", "region"],
            coords={
                "impact_type": ["wet"],
                "gcm": ["GFDL-ESM4"],
                "scenario": ["historical", "ssp126"],
                "time": pd.to_datetime(["1850-01-01", "1851-01-01", "2015-01-01", "2016-01-01"]),
                "region": ["A"],
            },
            name="value",
        )

        assignments = build_warming_level_assignments(
            gmt_by_path,
            path_columns=("gcm", "scenario"),
            bin_width=0.5,
        )
        enriched = attach_warming_levels(values, assignments)
        quantile_surface = build_warming_level_quantile_surface(
            enriched,
            sample_dims=("gcm", "scenario", "time"),
            quantiles=(0.5,),
        )

        self.assertEqual(assignments["target_warming_level"].values.tolist(), [0.0, 0.5, 1.0])
        historical_levels = enriched["warming_level"].sel(
            gcm="GFDL-ESM4",
            scenario="historical",
        ).values
        self.assertEqual(historical_levels[:2].tolist(), [0.0, 0.5])
        self.assertTrue(np.isnan(historical_levels[2:]).all())
        self.assertEqual(
            quantile_surface["sample_size"]
            .sel(impact_type="wet", region="A", warming_level=1.0)
            .item(),
            2,
        )
        self.assertAlmostEqual(
            quantile_surface["value"]
            .sel(impact_type="wet", region="A", warming_level=1.0, quantile=0.5)
            .item(),
            0.8,
        )


if __name__ == "__main__":
    unittest.main()
