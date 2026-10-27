# Siting of new plants

`impacts.thermoelectric.siting` estimates how much new wet- or dry-cooled
capacity a region could build and what share of it climate would leave usable.
It runs the kernels (impact calculations) of [thermoelectric.md](thermoelectric.md) for synthetic
plants (made-up plants of fixed type and size) at every potential site, a grid
cell where a new plant could be built. It splits the wet capacity loss into a
thermal part and a water-availability part, shares river water along the river
network, and writes capacity curves and warming-level percentiles for
[RIME-X](https://github.com/iiasa/rimeX), IIASA's emulator of regional climate
impacts by warming level. These are in-house methods, not part of Li et al.
(2025) or Qin et al. (2023); this page states them. Constants and assumptions
are listed in [assumptions.md](assumptions.md).

Terms used on this page:

- Nameplate capacity: a plant's rated maximum output, in MW.
- Usable capacity: the share of nameplate a plant can run at, given its
  cooling-water or air limits, in MW. The usable fraction is usable capacity
  divided by nameplate.
- R12 regions: MESSAGE's twelve world regions. MESSAGE is IIASA's energy system
  model, which reads these outputs.
- Warming level: a level of global warming, such as 1.5 or 2 K above
  pre-industrial. Years are grouped by the level they reach, and a warming-level
  bin is the range of years whose warming is closest to one level.
- Season-year: a season and the year it belongs to. DJF (December to February)
  takes the December of the year before.

## Potential sites, existing sites and synthetic plants

A site evaluation covers two sets of 0.5-degree cells. Each cell is tagged with
its R12 region, ISO3 country and spherical cell area (`cells.py`). The
`population` column names the set:

- Potential sites (`candidate`): every cell a kernel could be sited on. For dry
  cooling this is every land cell in an R12 region
  (`require_hydrology=False`), because dry cooling reads only `tas`. For wet
  cooling it is the cells where river temperature is finite in any month of any
  scenario passed as `cells_considered_scenarios`.
- Existing sites (`incumbent`): the distinct cells of a fleet's per-plant files
  (`extract_existing_site_cells`), restricted to the cells considered for that kernel.

Both sets hold synthetic plants, not real units. Every cell receives each
synthetic plant of `default_wet_synthetic_plants()` (coal 500 MW and gas 250 MW, each
once-through and recirculating, each taking the whole cell discharge) or of
`default_dry_synthetic_plants()` (one 500 MW unit per Qin steam-cycle family:
combined cycle, combustible steam, nuclear steam and non-combustible steam).
An existing site therefore holds the same synthetic plants as a potential site.
Existing-site capacity is a cell count times the synthetic plant's nameplate,
and no MW, fuel or cooling information of the real units is kept. Tables written
by the site evaluation and by the splitting of the wet capacity loss carry the
synthetic plant by name (the `archetype` column). Its nameplate and types are
looked up in the plant definitions (`wet_synthetic_plants=`), and an unknown name
raises.

## Site evaluation

`run_site_evaluation` evaluates both kernels at both sets of cells for sampled whole
years. It writes one long parquet table whose key columns are `cell_id`,
`archetype`, `scenario`, `year`, `season`, `warming_level`, `response` and
`population`. The dry kernel runs on daily `tas` and is aggregated to seasons
after its nonlinearity; the wet kernel runs monthly.

Years come from one of two samplers:

- `sample_years_within_gwl(gmt_root, gcm, scenarios, GWLSampling(...))` snaps
  each (scenario, year) to the nearest level on the `bin_width` grid between
  `min_level` and `max_level` and draws `years_per_bin` years per level, pooled
  across the scenarios. A year more than half a bin width from its level is not
  sampled. (GCM is a global climate model; `gmt_root` holds global mean
  temperature by year.)
- `sample_years_within_reference(ReferenceSampling(mode))` takes a fixed window
  and writes `REF-PD` or `REF-PI` in the `warming_level` column. `REF-PD` is the
  present-day reference period: each ISIMIP3b GCM's historical run for
  1995–2014. `REF-PI` is the pre-industrial reference climate: the ISIMIP3a
  GSWP3-W5E5 `counterclim` data for 1901–2019 (observed weather with the
  warming trend removed), written as season-years 1902–2019 because DJF needs
  the preceding December. Its climate paths come from `reference_climate_paths`.

The outputs for MESSAGE come from two calculations. The warming-level
calculation uses years sampled by warming level. The present-day reference
calculation uses the `REF-PD` window. A `REF-PI` site evaluation can be run, but
the river-network step does not support it, so it feeds no output for MESSAGE.

The `responses` argument selects what is written: seasonal `wet` and `dry`
responses, and `wet_monthly`, the per-month wet drivers that the splitting of
the wet capacity loss reads (available discharge, withdrawal intensity and the
others). Each `wet_monthly` row carries its `month` and season-year. The wet
evaluation also writes `water_limited_capacity_ceiling_mw`, the water limit: the
capacity a cell's available discharge can cool, formed per month and averaged
over the season.

Unless `summary=None`, `run_site_evaluation` also writes a per-region summary and a
comparison of potential sites with existing sites
(`compare_potential_to_existing_sites`). The summary holds area-weighted quantiles and mean,
the share of cell area at or below a threshold (`low_value_thresholds`), and a
cluster bootstrap over sampled years (an uncertainty estimate that resamples
whole years together). A site evaluation with only one of the two sets of cells
is refused unless `summary=None`.

## Splitting the wet capacity loss

`split_wet_losses` (`loss_split.py`) splits each wet row's capacity into
three parts that add up to nameplate: thermal loss, water-availability loss and
usable capacity. For one row (set of cells, cell, synthetic plant, month), with
nameplate `N`, available discharge `Q` (m3 s-1) and withdrawal intensity `I`
(m3 MWh-1, the water withdrawn per MWh generated):

1. Capacity per unit flow `c = 3600 / I`, the capacity one m3 s-1 can cool
   (MW per m3 s-1; `thermal_conductance` in the code). `c` is infinite at
   `I = 0` and 0 for `I < 0`.
2. Reference capacity per unit flow `c_ref`: the largest finite, positive `c` of
   that set of cells, cell, synthetic plant and calendar month over a baseline
   site evaluation (`build_low_warming_reference`). Where a cell has none, the
   largest over all cells of the same set, synthetic plant and month is used.
3. Thermally adjusted capacity `T = min(N, N c / c_ref)`.
4. Water limit `R = min(N, max(Q c, 0))`, the capacity the available river
   water can cool, capped at nameplate. `R = 0` when `Q = 0`.
5. Thermal loss `L_th = N - T`; water-availability loss
   `L_w = max(T - R, 0)`; usable capacity `U = max(T - L_w, 0)`.

Either `R >= T` (then `L_w = 0`, `U = T`) or `R < T` (then `L_w = T - R`,
`U = R`), so `L_th + L_w + U = N` for every row.

Thermal loss is capacity lost because each MWh needs more water than in the
reference month. Water-availability loss is capacity lost because the river
carries less water than the plant needs. `T` is the capacity the plant could
run at with its intake fixed at `N / c_ref`, the flow it needs at full output
in its best reference month. A rise in withdrawal intensity above that month
cuts capacity as thermal loss, whatever the river carries. Water-availability
loss is what the actual flow removes on top. The kernel's own usable capacity,
`min(N, Q c)`, cuts capacity only when water is limiting, so the split's `U` is
at or below it.

Months enter the season as follows. A month with `Q = 0` counts: an unknown
intensity is taken at the reference (no thermal loss) and the whole capacity is
lost as water-availability loss. A month with an unknown driver (NaN `Q`, or
NaN `I` with water in the river) is left out. The seasonal value is the mean
over the counted months (`month_count`); a season with none is omitted.

The `baseline_reference` column records which site evaluation `c_ref` was taken
from; the `reference` argument sets it. The code does not check it against the
files. The warming-level calculation passes
`DEFAULT_REFERENCE` with a low-warming site evaluation as the baseline.
The present-day reference calculation passes `PRESENT_DAY_REFERENCE` with the
`REF-PD` site evaluation as both the baseline and the evaluation that is split.

## Capacity curves of existing sites

A capacity curve is the cumulative usable capacity as cells are added in a fixed
order. `export_wet_usable_capacity_curves` reads the parquet files that hold the split loss
and writes three tables:

- `wet_ucr_curves_per_gcm.parquet`: per GCM, region, synthetic plant, season and
  warming level, the cumulative thermal loss, water-availability loss and usable
  capacity as cells are added in a fixed order. Each point records its capacity,
  the marginal cell (the cell added at that point) and that cell's usable
  fraction.
- `wet_ucr_curves_pooled_median.parquet`: the median of those curves across
  GCMs, with `gcm_count`. Pooling raises when a curve point is missing for some
  GCMs or its capacity differs between them. The medians of the three
  quantities need not add up to the capacity at that point, because each is a
  median taken separately.
- `wet_ucr_curve_additivity_oracle.parquet`: the additivity check. Per GCM,
  region, season and warming level, it compares the sum of the final points of
  all per-synthetic-plant curves with the direct sum of the split-loss rows. The
  export raises if they differ by more than 1e-6 MW.

The order in which cells are added is an argument (`fixed_ordering_path`). The
default, `"own_mean_usable_fraction"`, ranks each region's and synthetic plant's
cells by their own mean usable fraction over every row of the table, highest
first, ties broken by `cell_id`.

## Sharing water along the river network

Potential sites share river water along the river network (`river_network.py`).
`build_river_networks` follows each cell down a DDM30 flow-direction raster (a
0.5-degree river flow-direction map) until it meets another potential site of
the same scenario, which becomes its downstream cell. Outlets and paths leaving
the raster have none. The links form a forest: each cell has at most one
downstream cell, and the links never loop. The scenario of each river network is
read from its file name, which must name an SSP (`ssp` and three digits; SSP is
the Shared Socioeconomic Pathway, the scenario family of the climate runs) or
`historical`. A `REF-PI` river network is not supported.

`river_network_capped_allocation(ceiling, placed, downstream)` allocates placed capacity
`p_i` under water limits `C_i` (MW of capacity the cell's water can cool). A
cell's water limit caps the total cooled by water withdrawn at that cell and at
every cell upstream of it. In topological order (every cell before its
downstream cell):

- inflow `F_i = p_i + sum of s_u over the cells u directly upstream of i`;
- passed-on capacity `s_i = min(C_i, F_i)`;
- scaling factor `r_i = s_i / F_i` (1 when `F_i = 0`).

Each cell's allocation is `a_i = p_i` times the product of `r` over the cell
and every cell downstream of it. The most limiting cell therefore cuts every
upstream contribution by the same factor. A NaN water limit marks a cell that
passes flow on with no cap.

`remaining_water_limits(ceiling, usage, downstream)` subtracts the usage summed over
each cell's upstream subtree (itself included) from its water limit, floored at
0. The result is the remaining water limit: the water limit left after existing
sites upstream take their water. NaN water limits stay NaN.

## Capacity curves of potential sites and shared water

`build_wet_siting_outputs` builds the potential-site curves from the files
that hold the split loss, the year-resolved water limits and the river network,
and from a fixed ordering. For each climate sample (one year of one GCM under
one scenario, with synthetic plant, season and warming level), over the whole
domain at once:

1. Existing-site water demand at each cell is the existing sites' post-thermal
   capacity, `N - L_th` (at least 0), matched on sample, synthetic plant and
   cell (`attach_existing_site_water_demand`). It is allocated with
   `river_network_capped_allocation` and subtracted with `remaining_water_limits`, giving the
   potential sites' water limits. An upstream existing site in one region thus
   reduces the water left downstream in another. Existing-site demand on cells
   outside the potential-site river network is not subtracted.
2. For each region, capacity steps (`capacity_grid_points`; 61 points) run from 0
   to the region's total potential-site nameplate. At each step, potential-site
   cells are filled in the fixed order. Cells of other regions stay in the
   river network as limits, with nothing placed on them.
3. Thermal loss scales with the placed nameplate: `placed * L_th / N` per cell.
   The post-thermal capacity `placed * (1 - L_th / N)` is allocated against the
   potential sites' water limits. Usable capacity is the allocation, and
   water-availability loss is the post-thermal capacity minus it.

The sample curves are averaged over the sampled scenarios and years of each
warming level to give the per-GCM curves. These are pooled to a cross-GCM
median with the same check as above. The additivity check table for potential
sites is written but not checked, and it would not pass: water sharing makes the
curve's usable capacity smaller than the sum of each cell's own usable capacity,
by the water given up to upstream cells. As above, the pooled medians of the
three quantities need not add up to the capacity at that point.
`build_wet_siting_outputs` also returns, per region, the share of
existing-site nameplate that sits on potential-site cells
(`incumbent_overlap`). It raises when that share is zero everywhere, since
sharing water with existing sites would then change nothing.

The order is fixed before any climate sample is evaluated, and every GCM,
season, warming level and both calculations use the same table. Climate changes
the usable share at each position in the order. It does not move capacity to
the cells that rank best in a given season or warming-level bin.

## Cell ordering

`build_fixed_cell_ordering(reference_samples)` ranks the cells of each region
and synthetic plant by their mean usable fraction `U / N` over the reference
samples, highest first, ties broken by `cell_id`. It numbers them `fixed_order`
from 0. Built from the split loss of the `REF-PD` site evaluation pooled over
GCMs, this is the full-flow order: cells ranked as if no existing plant took any
water.

The residual-water order ranks cells on the water left after existing plants
upstream take theirs. It replaces each reference sample's usable capacity before
ranking (`residual_water_reference_samples`). With the full-flow water limit `C`
and the remaining water limit `C'` from step 1 above, a cell is charged

    loss = min(U, C) - min(U, C')

and ranked on `U - loss`. The charge is local to the cell; sharing among
potential sites and downstream limits stay in the curve build. With no
existing-site water upstream `C' = C`, the loss is exactly 0, and the order
equals the full-flow order. `build_residual_water_ordering` builds it from one
file of split loss, one file of water limits and one river network per GCM,
charging each GCM's samples and pooling them before ranking. Given a baseline
ordering, it reports, per region and synthetic plant, the number of cells and
the number whose rank moved, and raises if none did.

## New-build ratios and the RIME-X percentiles

The samples for RIME-X (`rime.py`) pair, for each climate sample (GCM, scenario,
year) and each region, plant type, cooling type, season and warming level, the
existing-site total with the potential-site curve of the same sample:

- existing-site capacity `K`, the sum of existing-site synthetic-plant nameplate,
  and existing-site usable fraction `f_inc`, their usable capacity over `K`;
- for each new-build ratio `r` (`depth_ratio` in the code; default 0.5, 1 and
  2), new capacity `D = r K` and potential-site usable fraction
  `f_cand = U_cand(D) / D`, with `U_cand` interpolated linearly on the sample's
  potential-site curve. A `D` beyond the end of the curve raises;
- `usable_capacity_ratio = f_inc + r f_cand = (f_inc K + U_cand(D)) / K`, the
  usable capacity of existing plants plus new build as a fraction of
  existing-site nameplate.

`build_warming_level_bin_percentiles` pools the samples within each warming-level bin
(combining samples from several GCMs, scenarios and years before taking
statistics) and takes the 10th, 50th and 90th percentiles, with `sample_count`
and `gcm_count`. `percentiles_to_dataset` writes them as a NetCDF file indexed
by `region_id`, `plant_type`, `cooling_type`, `season`, `gwl` and `depth_ratio`.

## Pooling across GCMs

The pooling step reads the per-GCM outputs of both calculations, one output
directory per GCM (`read_gcm_outputs`, called by
`finalize_wet_siting_outputs`). It checks that every curve point (GCM,
region, synthetic plant, point index) has the same marginal cell in every
season and warming level and in both calculations. It takes the RIME-X
percentiles over the warming-level calculation's samples, checks the dataset's
dimension sizes against `expected_rime_dimensions` (or skips the check when
`None`), refuses missing values, and writes `gwl/`, `reference_pd/` and
`validation.json` (`write_pooled_wet_outputs`). Nothing is written unless every
check passes.
