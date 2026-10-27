# End-to-end tests

These tests run the library on small made-up inputs and compare each result with
a stored copy of an earlier result. A change that alters any result fails a test.
`world.py` writes the inputs (an ISIMIP3b-style tree, GMT tables, region rasters,
the cells of today's fleet, a common-definitions checkout) into a temporary
directory. Each end-to-end test (`cases_*.py`, registered with `@case`) runs
library functions on those inputs. It reaches them through `entrypoints.py`, the
only module that names where the functions live. Its result is compared with the
stored reference output under `golden/`.

Run the tests with the rest of the suite:

    uv run --frozen pytest tests/regression -q

## Comparison modes

`canon.py` turns each result into a form that can be compared: a structure
listing (`manifest.json.gz`) of names, dimensions, data types, attributes and
order, plus the number arrays. It compares in one of two modes:

- `exact` (default): structure and every value identical; NaN and inf in the
  same places.
- `tolerance`: values within a relative 1e-12 (with an absolute floor scaled
  to the array) and the same NaN and inf masks; structure stays exact.

Choose the mode per test with `@case(name, tier="tolerance")`, or for a whole run
with `IMPACTS_REGRESSION_TIER=tolerance`.

## Updating the stored outputs

When a change of output is intended, write the stored outputs again from the
current code and review the diff:

    IMPACTS_REGRESSION_REGENERATE=1 uv run --frozen pytest tests/regression -q

Add `-k` with a test name to update selected tests only, for example
`-k decomposition`. The update runs each test twice and refuses a test whose
result differs between the two runs. Commit the result under `golden/`.

Tests marked `frozen` (`surfaces.buildings`) have stored outputs that were not
produced by the code under test. They are maintained by hand, and the update
skips them.

When a function moves or is renamed, edit its row in `entrypoints.py`. The tests
and the stored outputs stay as they are. `test_surface.py` fails when a public
name of the package is missing from that table.

## Not covered

Real data distributions, climate fields at full size, and reading the real
country raster and population files. The pre-industrial reference climate
(`REF-PI`) is covered for its year sampling only. The present-day reference
period (`REF-PD`) runs end to end on a historical window of three season-years
(`World.short_reference_window`). The `library.reference_pd` test checks its site
evaluation, its split of the wet capacity loss, its river network, its
residual-water order and its output files.
