# Units, constants and assumptions

## Units

A kernel (one impact calculation, `impacts.engine.Kernel`) declares the unit
each climate input must have (`ClimateInput.unit`), and the engine converts to
it before the physics runs. Conversion covers
temperature (K and °C) and pressure (Pa and kPa) only
(`impacts/climatecore/units.py`); an input declared without a unit is passed
through unconverted and unchecked, so it must already be in the unit below.
`variable_units=` overrides a unit missing from a file's metadata.

| Kernel | Input | Unit the physics receives |
| --- | --- | --- |
| wet | air temperature `tas`, river temperature `tw`, sea-surface temperature `tos` | °C (converted) |
| wet | surface pressure `ps` | kPa (converted) |
| wet | relative humidity `hurs` | % (not converted) |
| wet | discharge `dis` | m3 s-1 (not converted) |
| dry | air temperature `tas` | K (converted) |
| CHILLED | air temperature `tas` | °C (converted) |

Outputs:

- Thermoelectric capacities are MW; capacity factors and usable capacity ratios
  are fractions of nameplate in [0, 1]; withdrawal intensity is m3 MWh-1;
  capacity per unit flow (`thermal_conductance`) is MW per m3 s-1. Inside the
  wet kernel humidity ratios
  are g per kg dry air and enthalpies kJ per kg dry air.
- CHILLED energy intensities are MJ per m2 of floor area per day. `Nd` and `Nf`
  are 0 or 1 per day, so their monthly and seasonal values are shares of days.
  Annual warming-level summaries are MJ m-2 yr-1.
- Monthly and seasonal durations are means over the period, never sums.

Every evaluation writes each assumption to the output attributes as its value,
`<name>_unit` and `<name>_source`.

## Named constants

| Constant | Value | Code | Source |
| --- | --- | --- | --- |
| Saturation vapour pressure, Magnus form | `e_s = 6.11 hPa x 10^(7.5 T / (237.7 + T))`, `T` in °C | `MAGNUS_E0_HPA`, `MAGNUS_A`, `MAGNUS_B_C`, `wet.py` | Li & Qin released code |
| Humidity ratio factor | 621.9907 g kg-1 (1000 times the ratio of the molar masses of water vapour and dry air) | `WATER_TO_DRY_AIR_G_PER_KG`, `wet.py` | Li & Qin released code |
| Moist-air enthalpy constants | `c_pa` 1.01 kJ kg-1 K-1, `c_pv` 0.00189 kJ g-1 K-1, `L` 2.5 kJ g-1, referenced to 0 °C | `DRY_AIR_HEAT_CAPACITY_KJ_PER_KG_K`, `WATER_VAPOUR_HEAT_CAPACITY_KJ_PER_G_K`, `LATENT_HEAT_KJ_PER_G`, `wet.py` | Li & Qin released code |
| Wet-bulb temperature (the lowest temperature evaporation can cool air to) | Stull (2011) fit, stated for relative humidity 5–99%, air temperature −20 to 50 °C and sea-level pressure (101.325 kPa); the kernel clips relative humidity to [1, 100]% and does not limit temperature | `_stull_wet_bulb_c`, `STULL_RELATIVE_HUMIDITY_CLIP_PERCENT`, `wet.py` | Stull (2011), <https://doi.org/10.1175/JAMC-D-11-0143.1> |
| Water density, specific heat | 1000 kg m-3, 4.184 J g-1 K-1 | `WetCoolingAssumptions` | Li et al. (2025), eqs. 1-2 |
| Thermal efficiency curve | logistic in unit MW per fuel class; nuclear 0.35 | `FUEL_EFFICIENCY_CURVE_PARAMETERS`, `fuel_efficiency`, `wet.py` | Li & Qin released code |
| Heat-loss coefficient `k_os` | 0.12 (nuclear 0) | `HEAT_REJECTION_COEFFICIENTS`, `wet.py` | Li & Qin released code |
| Dry efficiency-loss coefficients | per Qin steam-cycle family (combined cycle, combustible steam, nuclear steam, non-combustible steam) | `EFFICIENCY_LOSS_COEFFICIENTS`, `EFFICIENCY_LOSS_EXPONENTS_PER_K`, `dry.py` | Qin et al. (2023), eqs. 1-5 |
| Dry fan loads (power the cooling system's own fans and pumps use) | 2.43% combustible steam, 0.81% combined cycle, 0.56% non-combustible steam, 3.04% nuclear steam | `DryCoolingAssumptions`, `dry.py` | Qin et al. (2023) table of fan and pumping energy; nuclear EPA (2001) Table 3-19 |
| Dry classification of `WSTH` steam rows | Non-nuclear, non-combustible steam: the non-combustible steam efficiency-loss fit and its 0.56% fan load | `_classify_fuel_group`, `non_combustible_steam` entries of `EFFICIENCY_LOSS_COEFFICIENTS` and the parasitic-load map, `dry.py`; `WSTH` is `waste_heat` in `fuels.py` | Qin et al. (2023) |
| Condenser backpressure (steam pressure at the turbine exhaust) | `1.031 inHg x e^(0.019 T_F)` | `BACKPRESSURE_INHG`, `BACKPRESSURE_EXPONENT_PER_F`, `dry.py` | Qin et al. (2023) eq. 5; reported, not used in the capacity factor |
| External surface resistance `R_se` | 0.04 m2 K W-1 | `EXTERNAL_SURFACE_RESISTANCE_M2_K_PER_W`, `chilled.py` | standard outer-surface value (EN ISO 6946), as in message-ix-buildings |
| Volumetric heat capacity of air | 1200 J m-3 K-1 | `AIR_VOLUMETRIC_HEAT_CAPACITY_J_PER_M3_K`, `chilled.py` | message-ix-buildings |
| CHILLED equipment and setpoint defaults | see the fields | `ChilledEnergyAssumptions`, `chilled.py` | message-ix-buildings CHILLED constants, cited per field |

## Assumptions register

| Assumption | Value | Where set | Source or rationale |
| --- | --- | --- | --- |
| Environmental flow | 80% of discharge reserved for ecosystems; 20% available to plants | `WetCoolingAssumptions.environmental_flow_fraction` | Li et al. (2025), eq. 4 |
| Discharge temperature limit | 32 °C | `WetCoolingAssumptions.max_discharge_temp_c` | Li et al. (2025), eq. 1 |
| Temperature rise limit | 10 K | `WetCoolingAssumptions.max_temp_rise_c` | Li et al. (2025), eq. 1 |
| Concentration cycles (how many times tower water is reused before it is drained) | 5 | `WetCoolingAssumptions.concentration_cycles` | Li et al. (2025), eq. 2 |
| Tower approach (how close the tower cools its water to the wet-bulb temperature) | 6 K | `WetCoolingAssumptions.tower_approach_c` | Li et al. (2025), eq. 2 |
| Water-use forcing of the hydrology | `2015soc` for SSP (Shared Socioeconomic Pathway) and counterclim runs, `histsoc` for historical runs | `ISIMIPPaths.soc`, `ISIMIPPaths.historical_soc` (`isimip3b_paths`, `isimip3a_counterclim_paths`, `ReferenceSampling.soc`) | SSP runs read the WaterGAP `2015soc` experiment by default, which holds human water use at present-day levels, and historical runs read `histsoc`. |
| Saline once-through units (once-through cooling, where water passes through the condenser once and returns warmer, here with seawater) | allowed temperature rise from sea-surface temperature at the nearest ocean cell within 150 km, under the same 32 °C and 10 K limits; no river share | `assign_saline_intakes`, `DEFAULT_SALINE_INTAKE_MAX_DISTANCE_KM`, `wet_cooling_physics` | In-house extension of Li et al.: their river limits applied unchanged to seawater |
| Saturated air | finite limit of eq. 2 at relative humidity 100% | `recirculating_saturated`, `wet_cooling_physics` | Eq. 2 is 0/0 there |
| Combined-cycle steam units (`WSTH`, steam rows) | coal efficiency curve and coefficients, class `steam_bottoming` | `FUEL_EFFICIENCY_CURVE_PARAMETERS`, `_classify_wet_fuel_type`, `wet.py` | Rankine steam cycle without its own coefficients |
| Evaporation coefficients | 0.0042 coal to 0.0088 gas and oil | `EVAPORATION_COEFFICIENTS`, `wet.py` | Carried from the released code and written to `evaporation_coefficient_fraction`; no equation uses them |
| Dry units without a Qin family | capacity factor 1 | `DryCoolingAssumptions.unsupported_capacity_factor` | Gas turbines and internal-combustion units have no steam cycle to cool |
| Synthetic plants | wet: coal 500 MW and gas 250 MW, each once-through and recirculating (water reused through a cooling tower), each taking the whole cell discharge; dry: one 500 MW unit per Qin family | `default_wet_synthetic_plants`, `default_dry_synthetic_plants`, `siting/cells.py` | Holds plant type and size fixed across cells so cells can be compared |
| Existing sites | the synthetic plants placed on the fleet's distinct cells; no MW, fuel or cooling of real units | `extract_existing_site_cells`, `siting/cells.py`; `build_existing_site_totals`, `siting/rime.py` | Existing-site capacity is a cell count times the synthetic plant's nameplate |
| Low-warming reference | `c_ref`, the reference capacity per unit flow, is the largest finite, positive capacity per unit flow per set of cells, cell, synthetic plant and calendar month over a baseline site evaluation: the low-warming site evaluation for the warming-level calculation, the `REF-PD` site evaluation (present-day reference period) for the present-day reference calculation | `build_low_warming_reference`, `split_wet_losses`, `siting/loss_split.py` | Defines thermal loss against a fixed design flow (see [siting.md](siting.md#splitting-the-wet-capacity-loss)) |
| Fallback reference when splitting the wet capacity loss | a cell with no reference month takes the largest `c_ref` over all cells of its set of cells, synthetic plant and month | `REFERENCE_FALLBACK_INDEX`, `_reference_with_fallback`, `siting/loss_split.py` | Gives every row a reference; a row with neither raises |
| Fixed cell ordering | cells ranked once per region and synthetic plant; shared by every GCM (global climate model), season, warming level and both calculations | `build_fixed_cell_ordering`, `build_residual_water_ordering` | Climate changes the usable share at each position in the order; it does not re-site capacity |
| Capacity steps of potential-site curves | 61 steps from 0 to the region's potential-site nameplate | `capacity_grid_points` | |
| New-build ratios | 0.5, 1, 2 times existing-site capacity | `DEFAULT_NEW_BUILD_RATIOS`, `siting/rime.py` | |
| Percentiles of the results | 10th, 50th, 90th within each warming-level bin | `DEFAULT_QUANTILES`, `siting/rime.py` | |
| Warming-level sampling | bins of 0.5 K from 1.0 to 4.0 K, 3 years per bin, seed 730 | `GWLSampling`, `siting/site_evaluation.py` | Years more than half a bin from their level are not sampled |
| Reference periods | `REF-PD` (present-day reference period) 1995–2014 historical; `REF-PI` (pre-industrial reference climate) 1901–2019 counterclim (observed weather with the warming trend removed; season-years 1902–2019) | `REFERENCE_WINDOWS`, `REFERENCE_SAMPLE_STARTS`, `siting/site_evaluation.py` | DJF (December to February) needs the preceding December |
| Dry kernel time step in the site evaluation | daily `tas`, aggregated to seasons after the nonlinearity | `evaluate_dry_potential_sites`, `siting/cells.py` | The efficiency loss is exponential in temperature |
| Region cooling setpoints | none by default; opt-in table | `load_region_cooling_setpoints`, `CHILLED_SETPOINT_PATH` | See [buildings.md](buildings.md#region-cooling-setpoints) |
