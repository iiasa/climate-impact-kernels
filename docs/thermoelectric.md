# Thermoelectric

`impacts.thermoelectric` computes the usable capacity of thermal power units
under climate. Usable capacity is the share of a unit's nameplate capacity (its
rated maximum output) that it can run at, in MW. The page covers wet cooling
after Li et al. (2025) and dry cooling after Qin et al. (2023). Wet cooling is
once-through (river or sea water passes through the condenser once and returns
warmer) or recirculating (water is reused through a cooling tower, so the plant
withdraws only the water lost to evaporation and blowdown). Dry cooling uses
air-cooled condensers and no cooling water. A kernel is one impact calculation:
the climate inputs it reads, its constants and its equations.

The package also maps a unit-level fleet table onto the 0.5-degree grid and
aggregates unit outputs to countries or MESSAGE R12 regions (MESSAGE's twelve
world regions). The siting methods for new plants, which use these kernels, are
in [siting.md](siting.md); units, constants and modelling assumptions are
collected in [assumptions.md](assumptions.md).

## Wet cooling: equations and code

Equation numbers are those of the Methods of Li et al. (2025),
<https://doi.org/10.1038/s41893-025-01692-9>. The physics is
`wet_cooling_physics` in `impacts/thermoelectric/wet.py`; the constants are the
fields of `WetCoolingAssumptions` in the same file.

| Li et al. | Quantity | Code |
| --- | --- | --- |
| eq. 1 | Once-through withdrawal intensity `WI_ot` (water withdrawn per MWh generated), m3 MWh-1 | `once_through` in `wet_cooling_physics` |
| eq. 1, denominator | Allowed temperature rise (how much the cooling water may warm) `max(min(T_lmax - T_w, dT_lmax), 0)` | `headroom` in `wet_cooling_physics` |
| eq. 2 | Recirculating withdrawal intensity `WI_rc`, m3 MWh-1 | `recirculating` in `wet_cooling_physics` |
| eqs. 1-2 | Rejected heat per MWh, `(1 - eta_net - k_os) / eta_net` | `rejected_heat` in `wet_cooling_physics` |
| eq. 3 with eq. 4 substituted | Usable capacity `UC = min(3600 Q_unit / WI, NC)` | `water_limited_capacity_mw` |
| eq. 4 | Water available to a unit, `(Q - Q_env) NC / NC_total`, per second (`Q_env` is the environmental flow, the discharge reserved for ecosystems) | `available_discharge`, `unit_available_discharge` |
| eq. 5 | Usable capacity ratio (UCR in Li et al.) `UCR = UC / NC`, clipped to [0, 1] | `capacity_factor` |

Eq. 6 of the paper, the change in usable capacity ratio between periods, is an
analysis step and is not in the kernel. The paper does not print the psychrometric steps that feed
eq. 2 (humidity ratios, moist-air enthalpies, wet-bulb temperature); the kernel
takes them from the authors' released code (Li & Qin,
<https://doi.org/10.5281/zenodo.17369544>, CC BY 4.0):

| Step | Code |
| --- | --- |
| Saturation vapour pressure, Magnus form | `_saturation_vapour_pressure_pa` |
| Humidity ratio `W = 621.9907 e / (p - e)`, g per kg dry air | `_humidity_ratio_g_per_kg` |
| Moist-air enthalpy `h = T (1.01 + 0.00189 W) + 2.5 W`, kJ per kg dry air | `_moist_air_enthalpy_kj_per_kg` |
| Wet-bulb temperature (the lowest temperature evaporation can cool air to), Stull (2011) | `_stull_wet_bulb_c` |
| Thermal efficiency from fuel class and unit size | `fuel_efficiency`, `FUEL_EFFICIENCY_CURVE_PARAMETERS` |
| Heat-loss coefficient `k_os` | `HEAT_REJECTION_COEFFICIENTS` |

Moist-air enthalpy is evaluated with air temperature in degrees Celsius, as its
constants require (they are referenced to 0 °C); the authors' released code
evaluates it with air temperature in kelvin.

Where the kernel goes beyond the printed equations:

- Eq. 1 is undefined at zero allowed temperature rise. The kernel sets the
  once-through intensity to infinity there (zero capacity) and to NaN where the
  allowed temperature rise is unknown.
- Eq. 2 is 0/0 at relative humidity 100%, where the humidity-ratio difference
  is zero. The enthalpy difference is proportional to that difference
  (`dh/dW = c_pv T + L` at fixed `T`), so the kernel cancels it and returns the
  finite limit (`recirculating_saturated`).
- The paper names `pr` as the pressure input of the outlet humidity ratio;
  the kernel uses surface pressure `ps`.
- `NC_total` in eq. 4 counts only units that draw on the river; saline
  once-through units are described below.

## Dry cooling: equations and code

Equation numbers are those of the Methods of Qin et al. (2023),
<https://doi.org/10.1038/s44221-023-00120-6>. The physics is
`dry_cooling_physics` in `impacts/thermoelectric/dry.py`.

| Qin et al. | Quantity | Code |
| --- | --- | --- |
| eqs. 1-4 | Turbine efficiency loss per steam-cycle family, a polynomial in condenser backpressure (steam pressure at the turbine exhaust; it rises as cooling air warms and lowers efficiency) | `EFFICIENCY_LOSS_COEFFICIENTS`, `_efficiency_loss` |
| eq. 5 | Dry-cooling backpressure, exponential in air temperature | substituted into eqs. 1-4; `_backpressure_inhg` reports it |
| table of fan and pumping energy | Dry-tower parasitic load (power the cooling system's own fans and pumps use) per family | `DryCoolingAssumptions` |

Substituting eq. 5 into eqs. 1-4 turns each loss into a sum of exponentials in
air temperature `T` (K):
`loss(T) = c3 e^(0.1026 T) + c2 e^(0.0684 T) + c1 e^(0.0342 T) + c0`. The
capacity factor is `1 - loss - parasitic load`, clipped to [0, 1]. The family of
a unit (the four plant classes of Qin et al.) comes from `classify_dry_cooling_engine_type`: combined cycle,
combustible steam, nuclear steam and non-combustible steam. Gas turbines,
internal-combustion units and other technology families have no Qin family and
keep `unsupported_capacity_factor`. The nuclear fan load is taken from EPA
(2001), Table 3-19, rather than Qin's table.

## Kernel inputs and the one-call helpers

`WET_COOLING_KERNEL` reads river temperature (`triver`, renamed `tw`), discharge
(`dis`), air temperature (`tas`), relative humidity (`hurs`), surface pressure
(`ps`) and, for saline units, sea-surface temperature (`tos`).
`WET_COOLING_VARIABLES` and `SALINE_INTAKE_VARIABLES` list them for
`impacts.isimip` (loaders for ISIMIP, the Inter-Sectoral Impact Model Intercomparison
Project's bias-adjusted climate data). `DRY_COOLING_KERNEL` reads `tas` only (`DRY_COOLING_VARIABLES`).
The engine converts each input to the unit the physics declares (see
[assumptions.md](assumptions.md#units)).

`wet_cooling(unit_cell_mapping, cooling_mapping, climate, duration)` builds the
wet mapping (`build_wet_cooling_mapping`), assigns saline intakes when the
climate holds `tos`, and evaluates the kernel. `dry_cooling(...)` does the same
with `build_dry_cooling_mapping`. To run the steps one at a time, call those
functions and `impacts.engine.evaluate_mapped(..., site_id_column="UNITID")`.

## Saline once-through intakes

Units with `cooling_category == "ot_saline"` draw seawater. They take no share
of the river discharge, and their allowed temperature rise is eq. 1's rule applied
to the sea-surface temperature at an intake point instead of the river
temperature. The kernel reports the temperature used as `intake_temperature`.

`tos` is the GCM's (global climate model's) own monthly ocean output on a 1-degree grid, not
bias-adjusted. `load_isimip_dataset` copies it onto the 0.5-degree land grid.
`assign_saline_intakes(mapping, climate["tos"])` gives each saline unit the
nearest cell, by great-circle distance from the unit's cell centre, where the
sea-surface temperature is finite at every time step, and records it as
`intake_lat`, `intake_lon` and `intake_distance_km`. A unit with no such cell
within `max_distance_km` (150 km by default) gets no intake and NaN capacity.

A wet fleet with saline units needs both `tos` and the intake columns; without
either, evaluation raises. A fleet without saline units runs without them. The
ISIMIP3a counterclim climate (observed weather with the warming trend removed)
has no ocean field, so a counterclim evaluation of
a fleet with saline units has no sea-surface temperature to read.

## Edge cases of the wet kernel

- A unit that draws on the river has zero usable capacity wherever the
  available discharge is zero, whatever the river temperature.
- A missing river or sea temperature is propagated: the allowed temperature rise, withdrawal
  intensity, usable capacity and capacity factor are NaN, not zero. Seasonal
  means skip the unknown months.
- A cell's available discharge is divided among its river-drawing units by MW
  share. Saline units are left out of `cell_total_mw` and have
  `unit_share_ratio` 0.

`water_limited_capacity_mw` is shared by the kernel, the site evaluation and the
splitting of the wet capacity loss, which treat the edges of the quotient `3600 Q / I` differently.
Each passes its rule as an argument:

| Caller | Zero `I` | Negative `I` | Zero `Q` | Other | Cap |
| --- | --- | --- | --- | --- | --- |
| Kernel usable capacity (per unit) | quotient (inf) | quotient | quotient | | nameplate |
| Site evaluation `water_limited_capacity_ceiling_mw` (per cell) | NaN | NaN | 0, even for unknown `I` | | none |
| Reference capacity per unit flow in the split of the wet capacity loss (`Q = 1`) | NaN | NaN | | NaN unless finite and positive | none |
| Capacity per unit flow in the split of the wet capacity loss (`Q = 1`) | inf | 0 | | | none |

An unknown `Q` or `I` gives NaN except where a zero-`Q` rule applies. The
kernel applies its dry-river rule after the function.

## Fuel codes

Both kernels classify the fleet's `fuel` column through `FUEL_CLASSES` in
`impacts/thermoelectric/fuels.py`. A code is matched whole after upper-casing,
never as a substring, and a code not in the table raises `ValueError`. The dry
kernel reads the fuel only for steam units, so an unknown code on a gas-turbine
or combined-cycle unit does not raise there.

The wet kernel has coefficients for coal, gas, oil, biomass and nuclear.
`WSTH` is the steam unit split off a combined cycle. The wet kernel classes it
as `gas` for combined-cycle, gas-turbine and internal-combustion rows and as
`steam_bottoming` for steam rows; `steam_bottoming` takes the coal efficiency
curve and coefficients, under its own name so the choice shows in
`wet_fuel_type`. The dry kernel treats `WSTH` and the other non-combustible
codes as non-combustible steam.

## Fleet mapping

`fleet.py` builds the fleet tables the kernels read:

- `build_unit_cell_mapping`: exact unit-to-cell mapping from coordinates.
- `build_recovered_unit_cell_mapping`: also places a unit without coordinates
  when its `LOCATIONID` resolves to one exact point.
- `build_imputed_wet_unit_cell_mapping`, `build_imputed_dry_unit_cell_mapping`:
  map every wet or dry unit, imputing a cell from donor units where neither of
  the above applies, and record how each unit's cell was found (exact,
  inherited or imputed).
- `build_location_cell_mapping`: location-level mapping that keeps ambiguous
  sites as explicit location-cell splits.
- `build_country_totals`: country totals of distinct locations and MW.
- `build_unit_cooling_mapping`: per-unit technology family and cooling assignment.
  `build_wet_cooling_mapping` raises for a unit whose `requires_water` is null.

Each has a `write_*` counterpart. The fleet table these read is licensed (WEPP)
and is not distributed, so neither are the tables derived from it.

## Aggregation

`regions.py` aggregates unit outputs to countries or R12 regions with a
reduction per variable: usable capacity and the two available discharges
(`available_discharge_m3_per_s`, `unit_available_discharge_m3_per_s`) are summed,
and the other diagnostics, including the raw `discharge`, are MW-weighted
means. A variable indexed by `entity` that is in neither list raises.

- `available_discharge_m3_per_s` belongs to the grid cell, so its regional sum
  counts each cell once; it needs `grid_lat`/`grid_lon` on the `entity`
  dimension. `unit_available_discharge_m3_per_s` sums over units.
- A unit with no value carries no weight in a weighted mean, and a region whose
  units are all missing is NaN, for sums and means. `total_mw` is the MW of all
  units and `valid_mw` the MW of units with a finite `usable_capacity_mw`.

`warming_levels.py` turns monthly country usable-capacity ratios into a
warming-level summary: per cooling type, R12 region and season, the
mean, standard deviation, 10th/50th/90th percentile and sample count over every
(GCM, scenario, year) sample in each 0.1 K warming-level bin. Country ratios are
aggregated to R12 by a nameplate-weighted mean over the member countries with
data.

## Scope

In scope: fleet mappings, cooling assignments, the wet and dry kernels, and
aggregation to countries and R12 regions. Out of scope: heuristic geocoding of
unresolved plants, fuzzy registry linkage, and file discovery beyond
`impacts.isimip`.
