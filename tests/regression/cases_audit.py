"""End-to-end tests of site evaluation (``audit``) and of splitting the wet capacity loss."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import entrypoints as ep
from . import fixtures
from .registry import case, outcome, outcome_type


def read_back(path) -> pd.DataFrame:
    return pd.read_parquet(path)


def read_back_or_absent(path):
    """The written frame, or the marker "not written" when the call wrote no file."""
    return read_back(path) if Path(path).exists() else "not written"


def _gmt_sampling(w, scenarios, **sampling):
    return ep.get("audit.sample_years_within_gwl")(
        w.gmt_root, fixtures.GCM, list(scenarios), ep.get("audit.GWLSampling")(**sampling)
    )


@case("audit.year_sampling")
def audit_year_sampling(w):
    both = ("ssp126", "ssp585")
    reference = ep.get("audit.ReferenceSampling")
    reference_years = ep.get("audit.sample_years_within_reference")
    sample = ep.get("audit.sample_years_within_gwl")
    sampling = ep.get("audit.GWLSampling")
    return {
        "defaults": sampling(),
        "bins": _gmt_sampling(w, both, bin_width=0.5, min_level=1.0, max_level=2.5, years_per_bin=2),
        "bins_seed": _gmt_sampling(w, both, bin_width=0.5, min_level=1.0, max_level=2.5, years_per_bin=2, seed=5),
        "single_scenario": _gmt_sampling(w, ("ssp585",), bin_width=0.5, min_level=1.0, max_level=3.0, years_per_bin=3),
        "more_years_than_available": _gmt_sampling(w, both, bin_width=1.0, min_level=1.0, max_level=2.0, years_per_bin=50),
        "levels_out_of_range": _gmt_sampling(w, both, bin_width=0.5, min_level=8.0, max_level=9.0, years_per_bin=2),
        "target_levels": _gmt_sampling(w, both, target_levels=(1.0, 1.5, 2.0), years_per_bin=2, bin_width=0.5),
        "single_target_level": _gmt_sampling(w, both, target_levels=(1.2,), years_per_bin=2, bin_width=0.4),
        "tolerance_override": _gmt_sampling(
            w, both, target_levels=(1.0, 2.0), years_per_bin=3, bin_width=0.5,
            level_tolerances=({"gcm": fixtures.GCM, "level": 1.5, "half_width": 0.3},),
        ),
        "tolerance_other_gcm": _gmt_sampling(
            w, both, target_levels=(1.0, 2.0), years_per_bin=3, bin_width=0.5,
            level_tolerances=({"gcm": "IPSL-CM6A-LR", "level": 1.5, "half_width": 0.3},),
        ),
        "no_target_levels": outcome(_gmt_sampling, w, both, target_levels=()),
        "historical_missing": outcome(sample, w.gmt_root, fixtures.GCM, ["ssp370"], sampling()),
        "reference_pd": reference_years(reference("REF-PD")),
        "reference_pi": reference_years(reference("REF-PI", soc="histsoc")),
    }  # fmt: skip


@case("audit.run_summary")
def audit_run_summary(w):
    run = w.audit_summary_run
    return {name: read_back(path) for name, path in run.items()}


@case("audit.run_monthly_and_baseline")
def audit_run_monthly_and_baseline(w):
    return {
        "monthly": read_back(w.audit_monthly_run),
        "baseline": read_back(w.audit_baseline_run),
        "no_summary_written": not w.audit_monthly_run.with_name(
            f"{w.audit_monthly_run.stem}_summary.parquet"
        ).exists(),
    }


@case("audit.universe_cache")
def audit_universe_cache(w):
    load = ep.get("audit.load_or_build_universe")
    region_paths = ep.get("universe.RegionRasterPaths")(**w.region_paths)
    cache = str(w.work / "cache" / "universe_{gcm}.parquet")
    arguments = (w.isimip_paths, fixtures.GCM, "ssp585", region_paths)
    first = load(*arguments, require_hydrology=True, cache_path=cache)
    written = pd.read_parquet(w.work / "cache" / f"universe_{fixtures.GCM}.parquet")
    second = load(*arguments, require_hydrology=True, cache_path=cache)
    stale = w.work / "cache" / "stale_{gcm}.parquet"
    stale.parent.mkdir(exist_ok=True)
    first.drop(columns=["country"]).to_parquet(
        w.work / "cache" / f"stale_{fixtures.GCM}.parquet", index=False
    )
    return {
        "first": first,
        "written": written,
        "second": second,
        "uncached": load(*arguments, require_hydrology=False, cache_path=None),
        "stale_cache": outcome(load, *arguments, require_hydrology=True, cache_path=str(stale)),
        "bad_cache_type": outcome(load, *arguments, require_hydrology=True, cache_path=3),
        "empty_cache_path": outcome(load, *arguments, require_hydrology=True, cache_path=""),
    }


def _census_frame() -> pd.DataFrame:
    rng = np.random.default_rng(404)
    rows = []
    for population in ("candidate", "incumbent"):
        for region in ("AFR", "WEU"):
            for cell in range(6):
                for scenario, year in (("ssp126", 2016), ("ssp126", 2018), ("ssp585", 2017), ("ssp585", 2020)):
                    for season in ("DJF", "JJA"):
                        value = rng.uniform(0.2, 1.0)
                        rows.append(
                            {
                                "population": population,
                                "region_id": region,
                                "archetype": "coal_once_through",
                                "gcm": fixtures.GCM,
                                "scenario": scenario,
                                "year": year,
                                "season": season,
                                "warming_level": 1.5,
                                "response": "capacity_factor",
                                "unit": "1",
                                "cell_id": f"{region}{cell:02d}",
                                "cell_area_m2": 2.0e9 + 1.0e8 * cell,
                                "value": value,
                            }
                        )
    frame = pd.DataFrame(rows)
    frame.loc[3, "value"] = np.nan
    frame.loc[40, "value"] = np.inf
    frame.loc[(frame["region_id"] == "WEU") & (frame["population"] == "incumbent"), "value"] = np.nan
    return frame


@case("audit.summary_and_verdict")
def audit_summary_and_verdict(w):
    summarize = ep.get("audit.summarize_census")
    compare = ep.get("audit.compare_grid_to_fleet")
    frame = _census_frame()
    summary = summarize(frame, bootstrap_replicates=7, seed=3)
    return {
        "summary": summary,
        "summary_thresholds": summarize(
            frame, low_value_thresholds={"capacity_factor": 0.6}, bootstrap_replicates=4, seed=9
        ),
        "summary_one_replicate": summarize(frame, bootstrap_replicates=1),
        "summary_no_replicates": summarize(frame, bootstrap_replicates=0),
        "summary_single_population": summarize(
            frame[frame["population"] == "candidate"], bootstrap_replicates=3, seed=1
        ),
        "summary_empty": outcome(summarize, frame.iloc[0:0], bootstrap_replicates=3),
        "summary_missing_column": outcome(summarize, frame.drop(columns=["cell_area_m2"]), bootstrap_replicates=3),
        "verdict": compare(summary),
        "verdict_single_population": outcome(
            compare, summarize(frame[frame["population"] == "candidate"], bootstrap_replicates=3, seed=1)
        ),
    }  # fmt: skip


# --- splitting the wet capacity loss ----------------------------------------------------------------------


@case("decomposition.reference_and_decomposition")
def decomposition_reference_and_decomposition(w):
    build = ep.get("decomposition.build_low_gwl_reference")
    decompose = ep.get("decomposition.decompose_censuses")
    directory = w.work / "decomposition_cases"
    directory.mkdir(exist_ok=True)
    reference = w.low_gwl_reference
    first_cell = sorted(reference["cell_id"].unique())[0]
    thin = reference[reference["cell_id"] != first_cell].reset_index(drop=True)
    sparse = reference.iloc[::2].reset_index(drop=True)
    labelled = decompose(
        [w.audit_monthly_run], reference, directory / "labelled.parquet", reference="custom"
    )
    fallback = decompose([w.audit_monthly_run], thin, directory / "thin.parquet")
    two_censuses = decompose(
        [w.audit_monthly_run, w.audit_baseline_run], reference, directory / "two.parquet"
    )
    return {
        "reference": reference,
        "reference_from_both": build([w.audit_baseline_run, w.audit_monthly_run]),
        "reference_no_paths": outcome_type(build, []),
        "reference_summary_census_has_no_monthly_rows": outcome(build, [w.audit_summary_run["long"]]),
        "decomposition": read_back(w.decomposition_path),
        "decomposition_labelled": read_back(labelled),
        "decomposition_fallback": read_back(fallback),
        "decomposition_two_censuses": read_back(two_censuses),
        "decomposition_sparse_reference": outcome(
            decompose, [w.audit_monthly_run], sparse, directory / "sparse.parquet"
        ),
        "decomposition_no_censuses": read_back_or_absent(
            decompose([], reference, directory / "none.parquet")
        ),
        "decomposition_missing_reference_columns": read_back_or_absent(
            decompose(
                [w.audit_monthly_run],
                reference.drop(columns=["fallback_thermal_conductance"]),
                directory / "bad.parquet",
            )
        ),
    }  # fmt: skip
