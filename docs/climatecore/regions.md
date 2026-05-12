# Regions

`impacts.climatecore.regions` provides shared helpers for reading region
definitions and aggregating entity-indexed outputs to named regions.

## Source of truth

Country and MESSAGE-region metadata are read from the local
`common-definitions` repository. Set `COMMON_DEFINITIONS_ROOT` before
using these helpers. The intent is to keep region semantics anchored in
the upstream source rather than duplicating definitions in each
downstream repo.

## Entry points

- `common_definitions_root` — resolves the configured
  `common-definitions` checkout.
- `message_r12_region_labels` — returns MESSAGEix-GLOBIOM 2.1 R12
  code-to-label mappings.
- `message_r12_country_to_region` — returns canonical country-to-R12
  membership.
- `resolve_region_labels` — resolves member labels for supported region
  sets.
- `aggregate_entity_dataarray` — aggregates one entity-indexed
  `DataArray` by region.
- `aggregate_entity_dataset` — aggregates an entity-indexed `Dataset`
  using variable-specific reductions.

## Supported region sets

`country` and `message_r12`.

## Out of scope

- Geometry files themselves.
- Downstream mapping artifacts.
- Domain-specific choices about which variables to sum or average.
