from __future__ import annotations

import pandas as pd
import pytest

from impacts.thermoelectric.siting import loss_split
from impacts.thermoelectric.siting.loss_split import (
    DEFAULT_REFERENCE,
    PRESENT_DAY_REFERENCE,
    _decompose_monthly_to_sums,
    _decompose_path,
    _seasonal_components,
    build_low_warming_reference,
    split_site_evaluations,
    split_wet_losses,
)


def _decompose_monthly(monthly, low_warming_reference, reference):
    return _seasonal_components(_decompose_monthly_to_sums(monthly, low_warming_reference), reference)


def _monthly_rows(
    *,
    warming_level: float,
    intensity: float,
    discharge: float,
    year: int = 2030,
    month: int = 6,
    season: str = "JJA",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "population": ["candidate"],
            "region_id": ["MEA"],
            "country": ["SAU"],
            "cell_id": ["cell-a"],
            "grid_lat": [10.25],
            "grid_lon": [20.25],
            "cell_area_m2": [1.0],
            "archetype": ["li_coal_500mw_once_through"],
            "gcm": ["GFDL-ESM4"],
            "scenario": ["ssp585"],
            "year": [year],
            "season": [season],
            "month": [month],
            "warming_level": [warming_level],
            "available_discharge_m3_per_s": [discharge],
            "withdrawal_intensity_m3_per_mwh": [intensity],
        }
    )


def _write_long_monthly(path, monthly: pd.DataFrame) -> None:
    rows = []
    for response in (
        "available_discharge_m3_per_s",
        "withdrawal_intensity_m3_per_mwh",
    ):
        row = monthly.drop(
            columns=["available_discharge_m3_per_s", "withdrawal_intensity_m3_per_mwh"]
        ).copy()
        row["response"] = response
        row["value"] = monthly[response]
        rows.append(row)
    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)


def test_decomposition_splits_thermal_then_withdrawal_loss() -> None:
    reference = pd.DataFrame(
        {
            "population": ["candidate"],
            "cell_id": ["cell-a"],
            "archetype": ["li_coal_500mw_once_through"],
            "month": [6],
            "baseline_thermal_conductance": [100.0],
        }
    )
    # Capacity per unit flow = 80 MW per m3/s. With nameplate=500, f_ref=5.
    # Thermal-adjusted capacity = 400. Water limit (flow times capacity per
    # unit flow) = 240.
    monthly = _monthly_rows(warming_level=2.0, intensity=45.0, discharge=3.0)

    decomposed = _decompose_monthly(monthly, reference, "test-reference")
    row = decomposed.iloc[0]

    assert row["nameplate_mw"] == pytest.approx(500.0)
    assert row["thermal_loss_mw"] == pytest.approx(100.0)
    assert row["withdrawal_loss_mw"] == pytest.approx(160.0)
    assert row["usable_capacity_mw"] == pytest.approx(240.0)
    assert row["capacity_factor"] == pytest.approx(0.48)
    assert (
        row["thermal_loss_mw"] + row["withdrawal_loss_mw"] + row["usable_capacity_mw"]
        == pytest.approx(row["nameplate_mw"])
    )


def _reference_for_months(*months: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "population": ["candidate"] * len(months),
            "cell_id": ["cell-a"] * len(months),
            "archetype": ["li_coal_500mw_once_through"] * len(months),
            "month": list(months),
            "baseline_thermal_conductance": [100.0] * len(months),
        }
    )


def test_a_dry_month_with_unknown_intensity_loses_its_capacity_to_the_water_channel() -> None:
    monthly = _monthly_rows(warming_level=2.0, intensity=float("nan"), discharge=0.0)

    row = _decompose_monthly(monthly, _reference_for_months(6), "test-reference").iloc[0]

    assert row["thermal_loss_mw"] == pytest.approx(0.0)
    assert row["withdrawal_loss_mw"] == pytest.approx(500.0)
    assert row["usable_capacity_mw"] == pytest.approx(0.0)
    assert row["capacity_factor"] == pytest.approx(0.0)


def test_a_dry_month_with_zero_intensity_is_not_unknown() -> None:
    # Zero intensity gives infinite capacity per unit flow, and 0 x infinity is NaN
    # if taken literally.
    monthly = _monthly_rows(warming_level=2.0, intensity=0.0, discharge=0.0)

    row = _decompose_monthly(monthly, _reference_for_months(6), "test-reference").iloc[0]

    assert row["thermal_loss_mw"] == pytest.approx(0.0)
    assert row["withdrawal_loss_mw"] == pytest.approx(500.0)
    assert row["usable_capacity_mw"] == pytest.approx(0.0)


def test_months_with_unknown_drivers_drop_out_of_the_season() -> None:
    known = _monthly_rows(warming_level=2.0, intensity=45.0, discharge=3.0, month=6)
    unknown_intensity = _monthly_rows(
        warming_level=2.0, intensity=float("nan"), discharge=3.0, month=7
    )
    unknown_discharge = _monthly_rows(
        warming_level=2.0, intensity=45.0, discharge=float("nan"), month=8
    )
    monthly = pd.concat([known, unknown_intensity, unknown_discharge], ignore_index=True)

    row = _decompose_monthly(monthly, _reference_for_months(6, 7, 8), "test-reference").iloc[0]

    # Same as the known month alone: the others are neither thermal losses nor zeros.
    assert row["thermal_loss_mw"] == pytest.approx(100.0)
    assert row["withdrawal_loss_mw"] == pytest.approx(160.0)
    assert row["usable_capacity_mw"] == pytest.approx(240.0)
    assert row["capacity_factor"] == pytest.approx(0.48)


def test_a_season_with_no_known_month_is_left_out(tmp_path) -> None:
    monthly = pd.concat(
        [
            _monthly_rows(warming_level=2.0, intensity=float("nan"), discharge=3.0, month=6),
            _monthly_rows(warming_level=2.0, intensity=45.0, discharge=float("nan"), month=7),
        ],
        ignore_index=True,
    )
    path = tmp_path / "monthly.parquet"
    _write_long_monthly(path, monthly)

    decomposed = _decompose_path(path, _reference_for_months(6, 7), "test-reference")

    assert decomposed.empty
    assert _decompose_monthly(monthly, _reference_for_months(6, 7), "test-reference").empty


def test_channels_add_to_nameplate_whatever_the_month_is_like() -> None:
    months = [
        (45.0, 3.0),  # thermal and water loss
        (45.0, 0.0),  # dry river
        (float("nan"), 0.0),  # dry river, unknown intensity
        (0.0, 5.0),  # zero intensity
        (float("inf"), 5.0),  # no allowed temperature rise
        (float("nan"), 5.0),  # unknown, excluded
    ]
    decomposed = 0
    for index, (intensity, discharge) in enumerate(months):
        monthly = _monthly_rows(warming_level=2.0, intensity=intensity, discharge=discharge)

        rows = _decompose_monthly(monthly, _reference_for_months(6), "test-reference")
        if rows.empty:
            continue
        row = rows.iloc[0]
        decomposed += 1
        total = row["thermal_loss_mw"] + row["withdrawal_loss_mw"] + row["usable_capacity_mw"]
        assert total == pytest.approx(row["nameplate_mw"]), months[index]
    assert decomposed == 5


def test_decomposition_path_accumulates_month_chunks(tmp_path) -> None:
    monthly = pd.concat(
        [
            _monthly_rows(warming_level=2.0, intensity=45.0, discharge=3.0, month=6),
            _monthly_rows(warming_level=2.0, intensity=60.0, discharge=6.0, month=7),
        ],
        ignore_index=True,
    )
    path = tmp_path / "monthly.parquet"
    _write_long_monthly(path, monthly)
    reference = pd.DataFrame(
        {
            "population": ["candidate", "candidate"],
            "cell_id": ["cell-a", "cell-a"],
            "archetype": ["li_coal_500mw_once_through", "li_coal_500mw_once_through"],
            "month": [6, 7],
            "baseline_thermal_conductance": [100.0, 100.0],
        }
    )

    decomposed = _decompose_path(path, reference, "test-reference")
    row = decomposed.iloc[0]

    assert row["thermal_loss_mw"] == pytest.approx((100.0 + 200.0) / 2.0)
    assert row["withdrawal_loss_mw"] == pytest.approx((160.0 + 0.0) / 2.0)
    assert row["usable_capacity_mw"] == pytest.approx((240.0 + 300.0) / 2.0)
    assert (
        row["thermal_loss_mw"] + row["withdrawal_loss_mw"] + row["usable_capacity_mw"]
        == pytest.approx(row["nameplate_mw"])
    )


def test_low_gwl_reference_uses_max_thermal_conductance(tmp_path) -> None:
    rows = []
    for year, intensity in ((2020, 60.0), (2021, 36.0)):
        monthly = _monthly_rows(
            warming_level=1.0,
            intensity=intensity,
            discharge=3.0,
            year=year,
        )
        for response in (
            "available_discharge_m3_per_s",
            "withdrawal_intensity_m3_per_mwh",
        ):
            row = monthly.drop(
                columns=["available_discharge_m3_per_s", "withdrawal_intensity_m3_per_mwh"]
            ).copy()
            row["response"] = response
            row["value"] = monthly[response]
            rows.append(row)
    path = tmp_path / "low.parquet"
    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)

    reference = build_low_warming_reference([path])

    assert len(reference) == 1
    assert reference["baseline_thermal_conductance"].iloc[0] == pytest.approx(100.0)
    assert reference["fallback_thermal_conductance"].iloc[0] == pytest.approx(100.0)


def test_low_gwl_reference_fallback_ignores_invalid_cell_baseline(tmp_path) -> None:
    valid = _monthly_rows(
        warming_level=1.0,
        intensity=36.0,
        discharge=3.0,
        month=7,
    )
    invalid = _monthly_rows(
        warming_level=1.0,
        intensity=float("inf"),
        discharge=3.0,
        month=7,
    ).assign(cell_id="cell-missing")
    path = tmp_path / "low.parquet"
    _write_long_monthly(path, pd.concat([valid, invalid], ignore_index=True))

    reference = build_low_warming_reference([path])

    assert set(reference["cell_id"]) == {"cell-a"}
    assert reference["baseline_thermal_conductance"].iloc[0] == pytest.approx(100.0)
    assert reference["fallback_thermal_conductance"].iloc[0] == pytest.approx(100.0)


def test_decomposition_uses_archetype_month_fallback_for_missing_cell_reference() -> None:
    monthly = _monthly_rows(
        warming_level=2.0,
        intensity=45.0,
        discharge=3.0,
        month=7,
    ).assign(cell_id="cell-missing")
    reference = pd.DataFrame(
        {
            "population": ["candidate"],
            "cell_id": ["cell-a"],
            "archetype": ["li_coal_500mw_once_through"],
            "month": [7],
            "baseline_thermal_conductance": [100.0],
        }
    )

    decomposed = _decompose_monthly(monthly, reference, "test-reference")
    row = decomposed.iloc[0]

    assert row["thermal_loss_mw"] == pytest.approx(100.0)
    assert row["withdrawal_loss_mw"] == pytest.approx(160.0)
    assert row["usable_capacity_mw"] == pytest.approx(240.0)


def _census(tmp_path, name="census.parquet"):
    path = tmp_path / name
    _write_long_monthly(path, _monthly_rows(warming_level=1.0, intensity=60.0, discharge=3.0))
    return path


def test_the_two_baseline_labels_differ_and_one_must_be_chosen(tmp_path) -> None:
    assert DEFAULT_REFERENCE != PRESENT_DAY_REFERENCE
    census = _census(tmp_path)
    with pytest.raises(TypeError, match="reference"):
        split_wet_losses([census], [census], tmp_path / "out.parquet")

    result = split_wet_losses(
        [census], [census], tmp_path / "out.parquet", reference=PRESENT_DAY_REFERENCE
    )
    assert set(pd.read_parquet(result.output)["baseline_reference"]) == {PRESENT_DAY_REFERENCE}


def test_a_decomposition_is_not_written_over_an_earlier_one(tmp_path) -> None:
    census = _census(tmp_path)
    output = tmp_path / "out.parquet"
    output.write_text("earlier")

    with pytest.raises(FileExistsError, match=r"out\.parquet"):
        split_wet_losses(
            [census], [census], output, reference=DEFAULT_REFERENCE
        )
    assert output.read_text() == "earlier"

    split_wet_losses(
        [census], [census], output, reference=DEFAULT_REFERENCE, overwrite=True
    )
    assert len(pd.read_parquet(output)) == 1


def _hidden_temporaries(directory) -> list[str]:
    return sorted(path.name for path in directory.iterdir() if path.name.startswith("."))


def test_a_decomposition_that_fails_midway_leaves_the_earlier_output_and_no_temporary(
    tmp_path, monkeypatch
) -> None:
    census = _census(tmp_path)
    output = tmp_path / "out.parquet"
    split_wet_losses(
        [census], [census], output, reference=DEFAULT_REFERENCE
    )
    earlier = output.read_bytes()

    real = loss_split._decompose_path
    calls = []

    def fail_on_second(path, low_warming_reference, reference, **kwargs):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return real(path, low_warming_reference, reference, **kwargs)

    monkeypatch.setattr(loss_split, "_decompose_path", fail_on_second)
    with pytest.raises(RuntimeError, match="boom"):
        split_wet_losses(
            [census], [census, census], output,
            reference=DEFAULT_REFERENCE, overwrite=True,
        )

    assert output.read_bytes() == earlier
    assert _hidden_temporaries(tmp_path) == []

    fresh = tmp_path / "fresh.parquet"
    calls.clear()
    with pytest.raises(RuntimeError, match="boom"):
        split_wet_losses(
            [census], [census, census], fresh, reference=DEFAULT_REFERENCE
        )
    assert not fresh.exists()
    assert _hidden_temporaries(tmp_path) == []


def test_a_census_of_custom_archetypes_is_decomposed_and_curved(tmp_path) -> None:
    from impacts.thermoelectric.siting.cells import SyntheticPlant
    from impacts.thermoelectric.siting.curves import export_wet_usable_capacity_curves

    custom = SyntheticPlant(
        name="custom_gas_100mw_tower",
        kernel="wet_cooling",
        parameters={"mw": 100.0, "wet_fuel_type": "gas", "cooling_physics": "recirculating"},
        provenance="test",
    )
    census = tmp_path / "census.parquet"
    monthly = _monthly_rows(warming_level=1.0, intensity=60.0, discharge=3.0).assign(
        archetype=custom.name
    )
    _write_long_monthly(census, monthly)
    reference = build_low_warming_reference([census])

    with pytest.raises(ValueError, match="unrecognized wet archetype: 'custom_gas_100mw_tower'"):
        split_site_evaluations([census], reference, tmp_path / "default.parquet")
    output = split_site_evaluations(
        [census], reference, tmp_path / "decomposition.parquet", wet_synthetic_plants=[custom]
    )
    decomposed = pd.read_parquet(output)
    assert decomposed["nameplate_mw"].tolist() == [100.0]

    with pytest.raises(ValueError, match="unrecognized wet archetype"):
        export_wet_usable_capacity_curves([output], tmp_path / "default_curves", population="candidate")
    per_gcm, _pooled, _oracle = export_wet_usable_capacity_curves(
        [output], tmp_path / "curves", population="candidate", wet_synthetic_plants=[custom]
    )
    curves = pd.read_parquet(per_gcm)
    assert set(zip(curves["plant_type"], curves["cooling_type"], strict=True)) == {
        ("gas", "recirculating")
    }
    assert curves["capacity_axis_mw"].max() == 100.0
