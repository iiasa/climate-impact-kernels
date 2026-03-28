"""Run reduced-matrix wet validation for the imputed full-fleet mapping on UniCC.

The script assumes:

- exact-coordinate wet outputs already exist under `JOB_ROOT/output/wet/`
- the canonical imputed mapping artifact exists at
  `JOB_ROOT/data/imputed_wet_unit_cell_mapping.parquet`

It recomputes wet-kernel results only for the non-exact rows of the imputed
mapping, combines them with the existing exact outputs, and writes headline
metrics for exact vs imputed fleets.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impacts.kernel import evaluate_kernel
from impacts.thermoelectric import build_wet_cooling_mapping, create_wet_cooling_kernel
from impacts.thermoelectric import WET_COOLING_VARIABLES
from impacts.isimip import isimip3b_paths, load_isimip_dataset
from _staging.thermoelectric.validation_uncertainty import add_size_class, summarise_metrics

GCMS = ["GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL"]
REDUCED_SCENARIOS = {
    "historical": {"year_range": (1995, 2014), "benchmark": None},
    "ssp370": {"year_range": (2045, 2054), "benchmark": "future_decline_share"},
}
HISTORICAL_BENCHMARKS = {
    "large_ucr": 0.729,
    "small_ucr": 0.823,
    "high_risk_share": 0.126,
}
FUTURE_BENCHMARKS = {"future_decline_share": 0.605}
JOB_ROOT = Path(os.environ.get("COOLING_VALIDATION_JOB_ROOT", "/home/raghunathan/job/cooling_validation"))
WATXENE_ROOT = Path(os.environ.get("WATXENE_ROOT", "/pdrive/projects/watxene"))
OUTPUT_ROOT = Path(
    os.environ.get(
        "IMPUTED_WET_VALIDATION_OUTPUT",
        JOB_ROOT / "results" / "imputed_wet_validation",
    )
)


def exact_output_path(gcm: str, scenario: str) -> Path:
    return JOB_ROOT / "output" / "wet" / f"{gcm}_{scenario}.nc"


def load_exact_headlines(gcm: str) -> pd.DataFrame:
    historical = xr.open_dataset(exact_output_path(gcm, "historical"))
    future = xr.open_dataset(exact_output_path(gcm, "ssp370"))

    frame = pd.DataFrame(
        {
            "UNITID": historical.coords["UNITID"].values.astype(int),
            "LOCATIONID": historical.coords["LOCATIONID"].values.astype(int),
            "country": historical.coords["country"].values.astype(str),
            "cooling_physics": historical.coords["cooling_physics"].values.astype(str),
            "mw": historical.coords["mw"].values.astype(float),
            "historical_ucr": historical["capacity_factor"].mean("time").values.astype(float),
            "future_ucr_2050s": future["capacity_factor"]
            .sel(time=slice("2045", "2054"))
            .mean("time")
            .values.astype(float),
        }
    )
    historical.close()
    future.close()

    frame["future_decline"] = frame["future_ucr_2050s"] < frame["historical_ucr"]
    return add_size_class(frame)


def load_imputed_mapping() -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_parquet(JOB_ROOT / "data" / "imputed_wet_unit_cell_mapping.parquet")
    exact = mapping.loc[mapping["coordinate_source"].eq("source")].copy()
    added = mapping.loc[~mapping["coordinate_source"].eq("source")].copy()
    return exact, added


def compute_added_outcomes(
    added_mapping: pd.DataFrame,
    cooling_mapping: pd.DataFrame,
    gcm: str,
    *,
    scenario: str,
    year_range: tuple[int, int],
) -> pd.DataFrame:
    climate = load_isimip_dataset(
        WET_COOLING_VARIABLES,
        gcm,
        scenario,
        isimip3b_paths(WATXENE_ROOT),
        year_range=year_range,
    )
    wet_mapping = build_wet_cooling_mapping(added_mapping, cooling_mapping)
    monthly = evaluate_kernel(create_wet_cooling_kernel(), wet_mapping, climate, duration="monthly")
    annual = monthly["capacity_factor"].mean("time").values.astype(float)
    climate.close()

    frame = wet_mapping.loc[:, ["UNITID", "LOCATIONID", "country", "cooling_physics", "mw"]].copy()
    frame[f"{scenario}_ucr"] = annual
    return frame


def evaluate_gcm(
    gcm: str,
    exact_headlines: pd.DataFrame,
    added_mapping: pd.DataFrame,
    cooling_mapping: pd.DataFrame,
) -> pd.DataFrame:
    historical_added = compute_added_outcomes(
        added_mapping,
        cooling_mapping,
        gcm,
        scenario="historical",
        year_range=REDUCED_SCENARIOS["historical"]["year_range"],
    )
    future_added = compute_added_outcomes(
        added_mapping,
        cooling_mapping,
        gcm,
        scenario="ssp370",
        year_range=REDUCED_SCENARIOS["ssp370"]["year_range"],
    )
    added = historical_added.merge(
        future_added.loc[:, ["UNITID", "ssp370_ucr"]],
        on="UNITID",
        how="inner",
        validate="one_to_one",
    )
    added["future_decline"] = added["ssp370_ucr"] < added["historical_ucr"]
    added = add_size_class(added.rename(columns={"historical_ucr": "historical_ucr"}))

    full = pd.concat(
        [
            exact_headlines.loc[
                :,
                [
                    "UNITID",
                    "LOCATIONID",
                    "country",
                    "cooling_physics",
                    "mw",
                    "historical_ucr",
                    "future_ucr_2050s",
                    "future_decline",
                    "size_class",
                ],
            ],
            added.rename(columns={"ssp370_ucr": "future_ucr_2050s"}),
        ],
        ignore_index=True,
        sort=False,
    )
    exact_metrics = summarise_metrics(exact_headlines)
    imputed_metrics = summarise_metrics(full)

    rows = []
    for baseline, metrics in [("exact", exact_metrics), ("imputed", imputed_metrics)]:
        rows.append({"gcm": gcm, "baseline": baseline, **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    exact_mapping, added_mapping = load_imputed_mapping()
    cooling_mapping = pd.read_parquet(JOB_ROOT / "data" / "unit_cooling_mapping.parquet")

    per_gcm = pd.concat(
        [
            evaluate_gcm(gcm, load_exact_headlines(gcm), added_mapping, cooling_mapping)
            for gcm in GCMS
        ],
        ignore_index=True,
    )
    ensemble = (
        per_gcm.groupby("baseline", as_index=False)[
            ["fleet_ucr", "large_ucr", "small_ucr", "high_risk_share", "future_decline_share"]
        ]
        .mean()
    )
    ensemble.to_csv(OUTPUT_ROOT / "reduced_validation_ensemble.csv", index=False)
    per_gcm.to_csv(OUTPUT_ROOT / "reduced_validation_per_gcm.csv", index=False)

    benchmark_report = {
        "historical": HISTORICAL_BENCHMARKS,
        "future": FUTURE_BENCHMARKS,
        "ensemble": ensemble.to_dict(orient="records"),
        "added_units": int(len(added_mapping)),
        "added_mw_gw": float(added_mapping["mw"].sum() / 1000),
    }
    (OUTPUT_ROOT / "benchmark_report.json").write_text(json.dumps(benchmark_report, indent=2))
    print(ensemble.to_string(index=False))


if __name__ == "__main__":
    main()
