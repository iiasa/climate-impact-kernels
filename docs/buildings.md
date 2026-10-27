# Buildings

`impacts.buildings` computes daily cooling and heating energy intensity of
building archetypes with the CHILLED kernel. A kernel is one impact calculation:
the climate inputs it reads, its constants and its equations. A building
archetype is a typical building, by type, location class and age, used to
represent the stock. The method is the variable degree-day (VDD) method of
Mastrucci et al. (2019),
<https://doi.org/10.1016/j.enbuild.2019.01.015>, and the code is adapted from
[message-ix-buildings](https://github.com/iiasa/message-ix-buildings)
(`message_ix_buildings/chilled/functions/buildings_funcs_grid.py`, Apache
License 2.0). Heating applies the same balance to heating degree-days; it is
message-ix-buildings' extension of the method, not part of the 2019 paper.

| Module | Owns |
| --- | --- |
| `chilled.py` | The physics, `ChilledEnergyAssumptions`, `CHILLED_KERNEL`, `CHILLED_REGION_SETPOINT_KERNEL` and `chilled_energy`. Reads no files. |
| `archetypes.py` | Loaders and validators for the archetype, parametric-run, setpoint, solar and region inputs; the gridded-parameter and archetype-mapping builders. |
| `_config.py` | Input paths from environment variables; the only module here that reads `os.environ`. |
| `regions.py` | Country-label raster, area-weighted site-to-country reduction, and the reshaping of CHILLED outputs for warming-level summaries. |
| `warming_levels.py` | Energy-intensity summaries by warming level. |

## Equations and code

The physics is `chilled_physics` in `impacts/buildings/chilled.py`. The source
paper is not cited by equation number here; each step is named by the
message-ix-buildings function it follows. Variable degree-days are degree-days
computed against a balance temperature that depends on the building. The
balance temperature is the outdoor temperature at which a building needs neither
heating nor cooling. COP is the coefficient of performance of an air
conditioner: cooling delivered per unit electricity.

| Step | message-ix-buildings | `chilled_physics` |
| --- | --- | --- |
| Transmission heat-loss coefficient `H_tr = U A_env`, W K-1 | `calc_H_tr` | `h_transmission` |
| Ventilation coefficients `H_v = 1200 V ach / 3600`, windows closed and open, W K-1 | `calc_H_v_cl`, `calc_H_v_op` | `h_ventilation_closed`, `h_ventilation_open` |
| Solar gain through glazing and roof, W | `calc_gn_sol_tot` | `solar_gain` |
| Heating balance temperature `T_sp,h - gains / (H_tr + H_v,cl)` | `calc_t_bal_h` | `t_balance_heating` |
| Cooling balance temperature `T_sp,c - gains / (H_tr + H_v,cl)` | `calc_t_bal_c` | `t_balance_cooling` |
| Maximum comfortable outdoor temperature `T_max,c - gains / (H_tr + H_v,op)` | `calc_t_max_c` | `t_max_cooling` |
| Heating degree-days `max(T_bal,h - T_out, 0)` | `calc_vdd_h` | `vdd_h` |
| Cooling degree-days above `T_max,c` | `calc_vdd_tmax_c` | `vdd_tmax_c` |
| AC days `Nd` (`T_out > T_max,c`) and fan days `Nf` (`T_out > T_bal,c`) | `calc_Nd`, `calc_Nf` | `Nd`, `Nf` |
| Cooling need `(H_tr + H_v,cl) f_c (VDD + (T_max,c - T_bal,c) Nd)` | `Q_c_tmax` | `cooling_need` |
| Heating need `(H_tr + H_v,cl) f_h VDD_h` | `Q_h` | `heating_need` |
| AC energy `Q_c / COP` | `calc_E_c_ac` | `cooling_energy_ac` |
| Fan energy `f_f P_fan Nf / A_fan` | `calc_E_c_fan` | `cooling_energy_fan` |
| Heating energy `Q_h / eta` | `calc_E_h` | `heating_energy` |

The roof term of the solar gain carries the external surface resistance
`R_se = 0.04 m2 K W-1` (`EXTERNAL_SURFACE_RESISTANCE_M2_K_PER_W`). Balance
temperatures are monthly, from the monthly solar climatology; degree-days and
day counts are evaluated per day against them. Outputs are MJ m-2 day-1, so
monthly and seasonal outputs are means of daily values rather than monthly
sums as in message-ix-buildings. `cooling_energy_total` is AC plus fan energy.

The default equipment and setpoint values are the fields of
`ChilledEnergyAssumptions`, each cited to the message-ix-buildings constant it
takes. `ChilledEnergyAssumptions.from_parametric_row` builds them from a row of
the message-ix-buildings parametric-run table (`par_var.csv`).

## Inputs

- Climate: daily near-surface air temperature `tas`, converted to °C by the
  engine. Load it from `impacts.isimip` with `target_frequency="native"`;
  monthly means would bias the degree-days.
- Archetypes: the message-ix-buildings ALPS2025 table `arch_input_reg.csv`,
  read and validated by `load_message_ix_buildings_archetypes`.
  `building_type` (`sfh`, `mfh`), `urt` (`rur`, `urb`), `arch` (`s1`, `s2`, `s3`,
  `s51`, `s52`) and `region` are dimensions; every row of the table is taken as
  given, including its cross combinations of building type and urban/rural class.
- Solar: the EWEMBI monthly vertical and horizontal irradiance climatologies
  (`load_ewembi_solar_climatology`), carried to the kernel as a
  `(month, entity)` site value.
- Regions: an R12 (MESSAGE's twelve world regions) region raster from the GAUL country raster, its lookup CSV and
  the MESSAGE R12 node definitions (`load_r12_region_raster`).
- Parametric runs, optional: `load_message_ix_buildings_parametric_runs`, which
  raises when neither a path nor `CHILLED_PARAMETRIC_PATH` is given.

Every loader takes an explicit path; the environment variables that stand in
for those paths are listed in the [README](../README.md#data-paths).

## Region cooling setpoints

By default the kernel uses the scalar `cooling_setpoint_c` and `cooling_max_c`
of `ChilledEnergyAssumptions` everywhere. Region-specific setpoints are opt-in,
from a table the caller supplies; none ships with the package.

The table is a CSV with one row per region and the columns `region` (matching
the archetype table's `region`), `cooling_setpoint_c` and `cooling_max_c`, in
°C. `load_region_cooling_setpoints(path)` reads it, or reads the file named by
`CHILLED_SETPOINT_PATH` when no path is given, and raises for a missing column
or a duplicated region. Pass the result as `setpoints=` to:

- `build_chilled_parameter_grids`, which rasterises the two columns by region
  for `evaluate_gridded`; or
- `build_chilled_archetype_mapping` or `chilled_energy`, which join them onto
  the mapping on `region`. The mapped path then evaluates
  `CHILLED_REGION_SETPOINT_KERNEL`, which carries the two columns as site values.

A region missing from the table, or a missing value, falls back to the scalar
on `ChilledEnergyAssumptions`. The setpoint is fixed per region, not adaptive to
outdoor temperature.

The output's `cooling_setpoint_c` and `cooling_max_c` attributes then describe
per-region values, and the matching `_source` attributes name the table file
and the scalar fallback. The loader binds that source information to the region-to-value
pairs it read: sorting the table or evaluating a subset of the mapping keeps it,
while a table, mapping column or grid edited afterwards is reported as of
unrecorded source.

## Entry points

From `impacts.buildings.chilled`:

- `CHILLED_KERNEL`: reads daily `tas` in °C.
- `CHILLED_REGION_SETPOINT_KERNEL`: the same kernel, whose mapped path also
  carries `cooling_setpoint_c`/`cooling_max_c`. The gridded path reads them from
  the parameter grid whenever present, so the distinction is mapped-path only.
- `chilled_energy(spatial_units, archetypes, solar_climatology, climate, duration, *, setpoints=None, assumptions=None, chunks=None, ...)`:
  builds the archetype mapping and evaluates it in one call.
- `CHILLED_CHUNKS = {"time": 365, "entity": 10_000}`: the kernels' chunk
  default. `evaluate_mapped`, `evaluate_gridded` and `chilled_energy` rechunk to
  it when `chunks=None`; pass other widths, or `{}` to keep the loader's chunks.
- `CHILLED_SITE_ID_COLUMN`: a mapping row has no identifier of its own, so
  `chilled_energy` numbers the rows under this column and passes it as
  `site_id_column`. A direct caller of `evaluate_mapped` does the same.

From `impacts.buildings.archetypes`:

- `build_chilled_parameter_grids(region_raster, archetypes, solar, *, building_type, urt, vintage, setpoints=None)`:
  the `(lat, lon)` parameter Dataset for one archetype combination, consumed by
  `evaluate_gridded`.
- `build_chilled_archetype_mapping(spatial_units, archetypes, solar, *, setpoints=None)`:
  spatial units crossed with the archetypes of their region; returns the
  mapping and the solar climatology to pass as `gridded_coords`.
- `validate_archetype_table`: the archetype-table checks, for tables built
  outside the loader.

`evaluate_gridded` returns `(time, entity)` outputs (`(season, year, entity)`
when seasonal) with `lat`/`lon` as coordinates on `entity`;
`reset_index("entity")` gives a `(time, lat, lon)` layout.

## Country aggregation and warming-level summaries

`aggregate_chilled_to_country` (`regions.py`) labels each grid cell (site) with its
country from the GAUL raster (`load_country_raster`, column `ISO3` by default)
and reduces all four outputs it carries (`cooling_energy_total`,
`heating_energy`, `Nd`, `Nf`) as cell-area-weighted means; none is summed.

`buildings_warming_level_summaries` (`warming_levels.py`) pools the country
seasonal intensities over every sample (one year of one GCM, a global climate
model, under one scenario) in each 0.1 K warming-level bin (the range of years whose
warming is closest to one level of global warming above pre-industrial) and reports, per R12 region, archetype, urban/rural class and
season, the mean, standard deviation, 10th/50th/90th percentile and sample
count. Countries are aggregated to R12 by a weighted mean over the members with
data; the caller supplies the weights. The `urt="total"` class is the
unweighted mean of the rural and urban classes. Annual summaries sum the seasonal rate
times the season length (MJ m-2 yr-1).

## Scope

In scope: archetype-table validation, mapping archetypes onto spatial units, the
kernel and its chunking. Out of scope: scenario assembly, running the steps of
a run in order, raster generation beyond the input files above, and inference of
parametric-run tables.
