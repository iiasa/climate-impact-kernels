# Regions

`climateresponse.regions` provides shared helpers for reading region
definitions and aggregating entity-indexed outputs to named regions.

## Shared Source Of Truth

Country and MESSAGE-region metadata are read from the local
`common-definitions` repository.

Set `COMMON_DEFINITIONS_ROOT` before using these helpers.

The goal is to keep region semantics anchored in the upstream source of truth
rather than duplicating those definitions in each downstream repo.

## Main Entry Points

- `common_definitions_root`
  Resolves the configured `common-definitions` checkout.
- `message_r12_region_labels`
  Returns MESSAGEix-GLOBIOM 2.1 R12 code-to-label mappings.
- `message_r12_country_to_region`
  Returns canonical country-to-R12 membership.
- `resolve_region_labels`
  Resolves member labels for supported region sets.
- `aggregate_entity_dataarray`
  Aggregates one entity-indexed `DataArray` by region.
- `aggregate_entity_dataset`
  Aggregates an entity-indexed `Dataset` using variable-specific reductions.

## Supported Region Sets

The current shared surface supports:

- `country`
- `message_r12`

## Boundary

This module owns shared region-loading and aggregation helpers.

It does not own:

- geometry files themselves
- downstream mapping artifacts
- domain-specific choices about which variables should be summed or averaged

Those remain the responsibility of the consuming package.
