from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
import xarray as xr

from impacts.isimip import isimip3b_paths
from impacts.thermoelectric.siting.site_evaluation import (
    GWLSampling,
    ReferenceSampling,
    SampledYear,
    SiteCells,
    SiteEvaluationSummary,
    _load_or_build_universe,
    build_site_cells,
    reference_climate_paths,
    run_site_evaluation,
    sample_years_within_gwl,
    sample_years_within_reference,
    write_site_evaluation,
)

from .fixtures import one_cell


class GWLSamplingTest(unittest.TestCase):
    def test_explicit_target_levels_are_not_expanded_to_bin_width_grid(self) -> None:
        gmt = xr.DataArray(
            [0.96, 1.03, 1.08, 1.13, 1.48],
            dims="year",
            coords={"year": [2016, 2017, 2018, 2019, 2020]},
        )

        with mock.patch(
            "impacts.thermoelectric.siting.site_evaluation.load_isimip_gmt",
            return_value=gmt,
        ):
            sampled = sample_years_within_gwl(
                "gmt-root",
                "GFDL-ESM4",
                ["ssp585"],
                GWLSampling(target_levels=(1.0, 1.1), years_per_bin=10, seed=1),
            )

        levels = sorted({round(sample.warming_level, 1) for sample in sampled})
        years = {sample.year for sample in sampled}
        self.assertEqual(levels, [1.0, 1.1])
        self.assertEqual(years, {2016, 2017, 2018, 2019})

    def test_years_beyond_the_level_range_are_not_clamped_into_the_end_bins(self) -> None:
        gmt = xr.DataArray(
            [0.2, 1.0, 1.1, 4.0, 5.5],
            dims="year",
            coords={"year": [2016, 2017, 2018, 2019, 2020]},
        )

        with mock.patch(
            "impacts.thermoelectric.siting.site_evaluation.load_isimip_gmt",
            return_value=gmt,
        ):
            sampled = sample_years_within_gwl(
                "gmt-root",
                "GFDL-ESM4",
                ["ssp585"],
                GWLSampling(bin_width=0.5, min_level=1.0, max_level=4.0, years_per_bin=10, seed=1),
            )

        # 0.2 K is 0.8 K below the lowest bin and 5.5 K is 1.5 K above the top one.
        self.assertEqual({sample.year for sample in sampled}, {2017, 2018, 2019})
        self.assertEqual(
            {(sample.year, sample.warming_level) for sample in sampled},
            {(2017, 1.0), (2018, 1.0), (2019, 4.0)},
        )

    def test_year_floor_is_scenario_aware(self) -> None:
        """historical is floored at 1851 (prior-Dec at 1850); ssp stays at 2016."""

        historical_gmt = xr.DataArray(
            [1.0, 1.0, 1.0],
            dims="year",
            coords={"year": [1849, 1850, 1851]},
        )
        ssp_gmt = xr.DataArray(
            [1.0, 1.0, 1.0],
            dims="year",
            coords={"year": [2014, 2015, 2016]},
        )

        def _load_isimip_gmt(gmt_root, gcm, scenario, **kwargs):
            return historical_gmt if scenario == "historical" else ssp_gmt

        with mock.patch(
            "impacts.thermoelectric.siting.site_evaluation.load_isimip_gmt",
            side_effect=_load_isimip_gmt,
        ):
            sampled = sample_years_within_gwl(
                "gmt-root",
                "GFDL-ESM4",
                ["ssp585", "historical"],
                GWLSampling(target_levels=(1.0,), years_per_bin=10, seed=1),
            )

        years_by_scenario: dict[str, set[int]] = {}
        for sample in sampled:
            years_by_scenario.setdefault(sample.scenario, set()).add(sample.year)

        self.assertEqual(years_by_scenario["historical"], {1851})
        self.assertEqual(years_by_scenario["ssp585"], {2016})

    def test_level_tolerance_widens_only_the_matching_gcm_and_level(self) -> None:
        """A per-(gcm, level) override must not change the years of any other bin.

        Nearest-level assignment hard-partitions years, so at the interior
        target 1.1 (spacing 0.1 to both neighbours) the shared half_width of
        0.05 already bounds every naturally-assigned year's error; widening
        that shared half_width could never add years there. The override
        instead borrows a year whose *nearest* level is 1.0 (2017, error .03)
        into the 1.1 bin because it also lies within the override's wider
        0.075 tolerance of 1.1 -- without removing it from the 1.0 bin, and
        without touching the 1.2 bin at all.
        """

        gmt = xr.DataArray(
            [1.10, 1.03, 1.18],
            dims="year",
            coords={"year": [2016, 2017, 2018]},
        )

        with mock.patch(
            "impacts.thermoelectric.siting.site_evaluation.load_isimip_gmt",
            return_value=gmt,
        ):
            unwidened = sample_years_within_gwl(
                "gmt-root",
                "IPSL-CM6A-LR",
                ["ssp585"],
                GWLSampling(target_levels=(1.0, 1.1, 1.2), years_per_bin=10, seed=1),
            )
            widened = sample_years_within_gwl(
                "gmt-root",
                "IPSL-CM6A-LR",
                ["ssp585"],
                GWLSampling(
                    target_levels=(1.0, 1.1, 1.2),
                    years_per_bin=10,
                    seed=1,
                    level_tolerances=(
                        {"gcm": "IPSL-CM6A-LR", "level": 1.1, "half_width": 0.075},
                    ),
                ),
            )
            other_gcm_widened = sample_years_within_gwl(
                "gmt-root",
                "UKESM1-0-LL",
                ["ssp585"],
                GWLSampling(
                    target_levels=(1.0, 1.1, 1.2),
                    years_per_bin=10,
                    seed=1,
                    level_tolerances=(
                        {"gcm": "IPSL-CM6A-LR", "level": 1.1, "half_width": 0.075},
                    ),
                ),
            )

        def _years_at(sampled: list[SampledYear], level: float) -> set[int]:
            return {sample.year for sample in sampled if sample.warming_level == level}

        self.assertEqual(_years_at(unwidened, 1.1), {2016})
        self.assertEqual(_years_at(widened, 1.1), {2016, 2017})
        # The neighbouring bins are untouched by the override.
        self.assertEqual(_years_at(widened, 1.0), {2017})
        self.assertEqual(_years_at(widened, 1.2), {2018})
        # The override is scoped to IPSL-CM6A-LR; a different gcm sees the
        # unwidened pool even with the same override list in its config.
        self.assertEqual(_years_at(other_gcm_widened, 1.1), {2016})


class ReferenceSamplingTest(unittest.TestCase):
    def test_ref_pd_covers_historical_window_with_reference_label(self) -> None:
        sampled = sample_years_within_reference(ReferenceSampling(mode="REF-PD"))

        self.assertEqual(len(sampled), 20)
        self.assertEqual(sampled[0], SampledYear("historical", 1995, "REF-PD"))
        self.assertEqual(sampled[-1], SampledYear("historical", 2014, "REF-PD"))

    def test_ref_pd_window_is_1995_to_2014_without_any_patch(self) -> None:
        sampled = sample_years_within_reference(ReferenceSampling("REF-PD"))

        self.assertEqual([item.year for item in sampled], list(range(1995, 2015)))
        self.assertEqual({item.scenario for item in sampled}, {"historical"})

    def test_ref_pi_uses_complete_season_years_with_single_member_label(self) -> None:
        sampled = sample_years_within_reference(ReferenceSampling(mode="REF-PI"))

        self.assertEqual(len(sampled), 118)
        self.assertEqual(sampled[0], SampledYear("counterclim", 1902, "REF-PI"))
        self.assertEqual(sampled[-1], SampledYear("counterclim", 2019, "REF-PI"))

    def test_ref_pi_requires_the_counterclim_forcing_member(self) -> None:
        with self.assertRaisesRegex(ValueError, "GSWP3-W5E5"):
            reference_climate_paths(ReferenceSampling("REF-PI"), "GFDL-ESM4")

    def test_ref_pi_water_use_forcing_is_a_choice_and_defaults_to_2015soc(self) -> None:
        """The water-use forcing of REF-PI is a choice, so the caller states it on
        the sampling rather than inheriting a fixed value."""

        default_paths = reference_climate_paths(
            ReferenceSampling("REF-PI"), "GSWP3-W5E5", "/tmp/watxene"
        )
        chosen_paths = reference_climate_paths(
            ReferenceSampling("REF-PI", soc="histsoc"), "GSWP3-W5E5", "/tmp/watxene"
        )

        self.assertEqual(default_paths.soc, "2015soc")
        self.assertEqual(chosen_paths.soc, "histsoc")

    def test_ref_pi_rejects_an_empty_water_use_forcing(self) -> None:
        with self.assertRaisesRegex(ValueError, "soc"):
            reference_climate_paths(ReferenceSampling("REF-PI", soc=""), "GSWP3-W5E5")

    def test_ref_pd_reads_the_isimip3b_run_of_its_gcm(self) -> None:
        paths = reference_climate_paths(ReferenceSampling("REF-PD"), "GFDL-ESM4", "/tmp/watxene")

        self.assertEqual(paths, isimip3b_paths(Path("/tmp/watxene")))


class CensusUniverseTest(unittest.TestCase):
    """Cells considered: for dry cooling every land cell in an R12 region, for wet
    cooling only the cells with river temperature data."""

    def test_universe_cache_reads_and_writes_expected_columns(self) -> None:
        expected = pd.DataFrame(
            {
                "cell_id": ["+000.250_+000.250"],
                "region_id": ["AFR"],
                "country": ["NGA"],
                "grid_lat": [0.25],
                "grid_lon": [0.25],
                "cell_area_m2": [3.0e9],
                "ignored": ["drop"],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "GFDL-ESM4_wet_universe.parquet"
            with mock.patch(
                "impacts.thermoelectric.siting.site_evaluation.build_valid_cells",
                return_value=expected,
            ) as build:
                first = _load_or_build_universe(
                    paths=None,
                    gcm="GFDL-ESM4",
                    scenarios=["ssp126", "ssp370"],
                    region_paths=None,
                    require_hydrology=True,
                    cache_path=str(cache),
                )
                second = _load_or_build_universe(
                    paths=None,
                    gcm="GFDL-ESM4",
                    scenarios=["ssp126", "ssp370"],
                    region_paths=None,
                    require_hydrology=True,
                    cache_path=str(cache),
                )

        build.assert_called_once()
        self.assertEqual(list(second.columns), list(first.columns))
        self.assertEqual(
            second.to_dict(orient="records"), first.to_dict(orient="records")
        )

    def test_census_cells_accept_a_tuple_of_universe_scenarios(self) -> None:
        frame = pd.DataFrame({"cell_id": ["a"]})
        with (
            mock.patch(
                "impacts.thermoelectric.siting.site_evaluation._load_or_build_universe",
                side_effect=lambda *args, **kwargs: frame.copy(),
            ) as load,
            mock.patch(
                "impacts.thermoelectric.siting.site_evaluation._fleet_cells",
                side_effect=lambda *args: frame.copy(),
            ),
        ):
            build_site_cells(
                None, "GFDL-ESM4", ["ssp126"], None,
                fleet_wet="w", fleet_dry="d", cells_considered_scenarios=("ssp126", "historical"),
            )
            build_site_cells(
                None, "GFDL-ESM4", ["ssp126"], None,
                fleet_wet="w", fleet_dry="d", cells_considered_scenarios="historical",
            )

        wet_calls = [c for c in load.call_args_list if c.kwargs.get("require_hydrology")]
        self.assertEqual(wet_calls[0].args[2], ["ssp126", "historical"])
        self.assertEqual(wet_calls[1].args[2], ["historical"])

    def test_universe_scenarios_reject_empty_lists(self) -> None:
        with self.assertRaisesRegex(ValueError, "cells_considered_scenarios"):
            build_site_cells(
                None, "GFDL-ESM4", ["ssp126"], None,
                fleet_wet="w", fleet_dry="d", cells_considered_scenarios=[],
            )


def _windows(*values: int):
    for value in values:
        yield pd.DataFrame({"cell_id": [f"cell{value}"], "value": [float(value)]})


def test_census_windows_are_written_under_the_final_name_only_when_complete(tmp_path):
    output = tmp_path / "nested" / "census.parquet"

    def windows():
        yield from _windows(1, 2)
        # Mid-write the final name is not taken and the work is in a temporary file.
        assert not output.exists()
        assert len(list(output.parent.iterdir())) == 1
        yield from _windows(3)

    assert write_site_evaluation(windows(), output) == output
    assert pd.read_parquet(output)["value"].tolist() == [1.0, 2.0, 3.0]
    assert list(output.parent.iterdir()) == [output]


def test_a_census_is_not_written_over_an_earlier_census_unless_asked(tmp_path):
    output = tmp_path / "census.parquet"
    output.write_text("earlier")
    drawn = []

    def windows():
        drawn.append(True)
        yield from _windows(1)

    with pytest.raises(FileExistsError, match=r"census\.parquet"):
        write_site_evaluation(windows(), output)
    assert output.read_text() == "earlier"
    assert not drawn  # refused before the expensive evaluation started

    write_site_evaluation(windows(), output, overwrite=True)
    assert pd.read_parquet(output)["value"].tolist() == [1.0]


def test_a_census_with_no_windows_raises_and_leaves_an_earlier_census_alone(tmp_path):
    output = tmp_path / "census.parquet"
    with pytest.raises(ValueError, match="no blocks of site-evaluation rows"):
        write_site_evaluation(iter(()), output)
    assert list(tmp_path.iterdir()) == []

    output.write_text("earlier")
    with pytest.raises(ValueError, match="no blocks of site-evaluation rows"):
        write_site_evaluation(iter(()), output, overwrite=True)
    assert output.read_text() == "earlier"


def test_a_census_that_fails_midway_leaves_no_partial_file(tmp_path):
    output = tmp_path / "census.parquet"
    output.write_text("earlier")

    def windows():
        yield from _windows(1)
        raise RuntimeError("kernel failed")

    with pytest.raises(RuntimeError, match="kernel failed"):
        write_site_evaluation(windows(), output, overwrite=True)
    assert [path.name for path in tmp_path.iterdir()] == ["census.parquet"]
    assert output.read_text() == "earlier"


def test_a_census_refuses_existing_files_before_evaluating(tmp_path):
    output = tmp_path / "census.parquet"
    (tmp_path / "census_verdict.parquet").write_bytes(b"earlier")
    cells = SiteCells(wet_cells={}, dry_cells={})
    with mock.patch(
        "impacts.thermoelectric.siting.site_evaluation.evaluate_sites"
    ) as evaluate:
        # By default the per-region summary and the comparison of potential with
        # existing sites are written too, so they are checked.
        with pytest.raises(FileExistsError, match="census_verdict"):
            run_site_evaluation(cells, [], paths=None, gcm="GFDL-ESM4", output=output)
        with pytest.raises(FileExistsError, match="census_verdict"):
            run_site_evaluation(cells, [], paths=None, gcm="GFDL-ESM4", output=output, summary=SiteEvaluationSummary())
        # Without a summary only the site evaluation itself is checked.
        evaluate.return_value = _windows(1, 2)
        written = run_site_evaluation(cells, [], paths=None, gcm="GFDL-ESM4", output=output, summary=None)
    assert written == [output]
    assert pd.read_parquet(output)["value"].tolist() == [1.0, 2.0]


def test_the_census_module_is_not_hidden_by_a_package_attribute():
    import impacts.thermoelectric.siting.site_evaluation as module
    from impacts.thermoelectric import siting

    assert module.__name__ == "impacts.thermoelectric.siting.site_evaluation"
    assert siting.run_site_evaluation is module.run_site_evaluation


@pytest.mark.parametrize("cache", ["{gcm}_wet_universe.parquet", Path("wet_{gcm}.parquet")])
def test_a_templated_universe_cache_path_is_refused(tmp_path, cache):
    with mock.patch(
        "impacts.thermoelectric.siting.site_evaluation.build_valid_cells"
    ) as build:
        with pytest.raises(ValueError, match="one concrete path per GCM"):
            _load_or_build_universe(
                None, "GFDL-ESM4", ["ssp126"], None,
                require_hydrology=True, cache_path=tmp_path / cache,
            )
        with pytest.raises(ValueError, match="one concrete path per GCM"):
            _load_or_build_universe(
                None, "GFDL-ESM4", ["ssp126"], None,
                require_hydrology=True, cache_path=str(tmp_path / cache),
            )
    build.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_a_templated_wet_cache_is_refused_before_the_dry_universe_is_built(tmp_path):
    with mock.patch(
        "impacts.thermoelectric.siting.site_evaluation.build_valid_cells"
    ) as build, pytest.raises(ValueError, match="one concrete path per GCM"):
        build_site_cells(
            None, "GFDL-ESM4", ["ssp126"], None,
            fleet_wet=tmp_path / "wet.nc", fleet_dry=tmp_path / "dry.nc",
            dry_cache=tmp_path / "dry_universe.parquet",
            wet_cache=tmp_path / "{gcm}_wet_universe.parquet",
        )
    build.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_a_reference_sampling_needs_a_known_mode():
    with pytest.raises(ValueError, match="REF-PD, REF-PI; got 'REF-XX'"):
        ReferenceSampling("REF-XX")


def test_a_one_population_census_with_a_summary_is_refused_before_evaluating(tmp_path):
    cells = SiteCells(wet_cells={"candidate": one_cell()}, dry_cells={})
    with mock.patch(
        "impacts.thermoelectric.siting.site_evaluation.evaluate_sites"
    ) as evaluate:
        evaluate.return_value = _windows(1)
        with pytest.raises(ValueError, match=r"no incumbent population.*pass summary=None"):
            run_site_evaluation(cells, [], paths=None, gcm="GFDL-ESM4", output=tmp_path / "census.parquet")
    assert list(tmp_path.iterdir()) == []


def test_a_census_of_no_sampled_years_is_refused_and_writes_nothing(tmp_path):
    cells = SiteCells(wet_cells={"candidate": one_cell()}, dry_cells={})
    with pytest.raises(ValueError, match="climate sampling selected no years"):
        run_site_evaluation(cells, [], paths=None, gcm="GFDL-ESM4", output=tmp_path / "census.parquet")
    assert list(tmp_path.iterdir()) == []
