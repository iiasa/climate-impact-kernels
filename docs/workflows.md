# Workflows

This repository is a library, but a few recurring workflows cut across modules.

## Thermoelectric

The thermoelectric workflow is:

1. Build unit-cell mappings in `mapping.py`.
2. Build cooling assignments in `cooling.py`.
3. Normalize climate inputs and construct wet or dry kernels.
4. Evaluate entity-level outputs.
5. Aggregate to countries or MESSAGE R12 regions in `aggregation.py`.

The mapping and cooling artifacts are durable contracts. If their logic changes,
regenerate the checked-in outputs deliberately.

## Buildings / CHILLED

The CHILLED workflow is:

1. Load and validate the ALPS2025 archetype table.
2. Attach monthly solar climatology to the spatial mapping.
3. Load daily climate inputs, typically through `impacts.isimip`.
4. Normalize climate inputs in `buildings/chilled.py`.
5. Evaluate the kernel at `native`, `daily`, `monthly`, or `seasonal`
   duration.

The important boundary is that `impacts` owns the kernel and its contracts, but
not scenario assembly or data-discovery logic above it.

## ISIMIP Ingress

`impacts.isimip` is the repository's concrete ISIMIP3b ingress surface.

Key points:

- `load_isimip_variable` can return either `target_frequency="native"` or
  `"monthly"`
- daily-sensitive kernels such as CHILLED should use `"native"`
- `load_isimip_dataset` merges variables after frequency handling and time-index
  standardization
- `load_isimip_gmt` handles historical + scenario stitching for annual GMT

Do not silently downsample daily sources to monthly when the downstream kernel
depends on daily event structure.
