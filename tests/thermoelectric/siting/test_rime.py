from __future__ import annotations

import pandas as pd
import pytest

from impacts.thermoelectric.siting.curves import build_fixed_cell_ordering
from impacts.thermoelectric.siting.rime import (
    build_warming_level_bin_percentiles,
    build_wet_rime_samples,
    percentiles_to_dataset,
)


def _decomposition_row(
    population: str,
    year: int,
    *,
    nameplate_mw: float,
    usable_capacity_mw: float,
) -> dict:
    return {
        "population": population,
        "region_id": "MEA",
        "country": "AAA",
        "cell_id": f"{population}-cell",
        "grid_lat": 0.25,
        "grid_lon": 0.25,
        "archetype": "li_coal_500mw_once_through",
        "gcm": "GFDL-ESM4",
        "scenario": "ssp585",
        "year": year,
        "season": "JJA",
        "warming_level": 3.0,
        "nameplate_mw": nameplate_mw,
        "thermal_loss_mw": nameplate_mw - usable_capacity_mw,
        "withdrawal_loss_mw": 0.0,
        "usable_capacity_mw": usable_capacity_mw,
    }


def _sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    incumbent = pd.DataFrame(
        [
            _decomposition_row(
                "incumbent", 2050, nameplate_mw=100.0, usable_capacity_mw=80.0
            ),
            _decomposition_row(
                "incumbent", 2051, nameplate_mw=100.0, usable_capacity_mw=40.0
            ),
        ]
    )
    candidate = pd.DataFrame(
        [
            _decomposition_row(
                "candidate", 2050, nameplate_mw=200.0, usable_capacity_mw=200.0
            ),
            _decomposition_row(
                "candidate", 2051, nameplate_mw=400.0, usable_capacity_mw=400.0
            ),
        ]
    )
    ceilings = candidate[
        [
            "population",
            "region_id",
            "country",
            "cell_id",
            "grid_lat",
            "grid_lon",
            "archetype",
            "gcm",
            "scenario",
            "year",
            "season",
            "warming_level",
        ]
    ].assign(water_ceiling_mw=[200.0, 200.0])
    river_network = pd.DataFrame(
        {
            "population": ["candidate"],
            "scenario": ["ssp585"],
            "cell_id": ["candidate-cell"],
            "downstream_cell_id": [None],
        }
    )
    return incumbent, candidate, ceilings, river_network


def test_percentiles_use_year_resolved_gwl_window_samples() -> None:
    incumbent, candidate, ceilings, river_network = _sample_inputs()

    samples = build_wet_rime_samples(
        incumbent,
        candidate,
        ceilings,
        river_network,
        fixed_ordering=build_fixed_cell_ordering(candidate),
        new_build_ratios=[1.0],
        capacity_grid_points=3,
    ).sort_values("year")

    # 2050: existing site 0.8 + potential site 1.0. 2051's potential-site curve
    # has a different capacity axis; interpolation at 100 MW gives fraction 1.0.
    assert samples["usable_capacity_ratio"].tolist() == pytest.approx([1.8, 1.4])

    percentiles = build_warming_level_bin_percentiles(samples)
    median = percentiles.loc[percentiles["statistic"] == "p50"].iloc[0]
    assert median["usable_capacity_ratio"] == pytest.approx(1.6)
    assert median["sample_count"] == 2
    assert median["gcm_count"] == 1


def test_candidate_fraction_nets_out_incumbent_water_on_a_shared_cell() -> None:
    incumbent, candidate, ceilings, river_network = _sample_inputs()
    incumbent["cell_id"] = "candidate-cell"

    samples = build_wet_rime_samples(
        incumbent,
        candidate,
        ceilings,
        river_network,
        fixed_ordering=build_fixed_cell_ordering(candidate),
        new_build_ratios=[2.0],
        capacity_grid_points=3,
    ).sort_values("year")

    # The existing site's post-thermal draw (80 MW in 2050, 40 in 2051) leaves 120
    # and 160 of the 200 MW water limit. At new-build ratio 2.0 (200 MW placed)
    # that is a potential-site fraction of 0.6 and 0.8; full-flow water would
    # give 1.0 in both.
    assert samples["candidate_usable_fraction"].tolist() == pytest.approx([0.6, 0.8])


def test_rime_samples_refuse_band_mean_water_ceilings() -> None:
    incumbent, candidate, ceilings, river_network = _sample_inputs()

    with pytest.raises(ValueError, match="year-resolved water limits"):
        build_wet_rime_samples(
            incumbent,
            candidate,
            ceilings.drop(columns="year"),
            river_network,
            fixed_ordering=build_fixed_cell_ordering(candidate),
        )


def test_percentile_dataset_uses_rime_variable_suffixes() -> None:
    incumbent, candidate, ceilings, river_network = _sample_inputs()
    samples = build_wet_rime_samples(
        incumbent,
        candidate,
        ceilings,
        river_network,
        fixed_ordering=build_fixed_cell_ordering(candidate),
        new_build_ratios=[1.0],
        capacity_grid_points=3,
    )
    samples = pd.concat(
        [samples.assign(season=season) for season in ("SON", "JJA", "DJF", "MAM")],
        ignore_index=True,
    )

    dataset = percentiles_to_dataset(build_warming_level_bin_percentiles(samples))

    assert "usable_capacity_ratio_p10" in dataset
    assert "usable_capacity_ratio_p50" in dataset
    assert "usable_capacity_ratio_p90" in dataset
    assert "gwl" in dataset.coords
    assert "warming_level" not in dataset.coords
    assert dataset.coords["season"].values.tolist() == ["DJF", "MAM", "JJA", "SON"]
    assert (dataset["sample_count"] == 2).all()
