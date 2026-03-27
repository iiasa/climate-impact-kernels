from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
import csv
import io
import os
from pathlib import Path
import re
from typing import Literal, TypeAlias

import xarray as xr

Reduction: TypeAlias = Literal["sum", "weighted_mean"]


@dataclass(frozen=True, slots=True)
class CountryGeometrySource:
    """Registered external country-boundary source."""

    name: str
    source: str
    format: str
    source_url: str
    dataset_slug: str
    notes: str


COUNTRY_GEOMETRY_SOURCES: tuple[CountryGeometrySource, ...] = (
    CountryGeometrySource(
        name="natural_earth_admin_0_countries_110m",
        source="Natural Earth",
        format="shapefile",
        source_url="https://www.naturalearthdata.com/downloads/110m-cultural-vectors/",
        dataset_slug="ne_110m_admin_0_countries",
        notes=(
            "Registered as the default lightweight global country-boundary source for "
            "future grid-to-country reductions. Geometry files are not vendored in this "
            "repository; the registry stores the canonical upstream source metadata."
        ),
    ),
)

_MESSAGE_R12_ALIASES = {
    "ANTIGUA & BARBUDA": "Antigua and Barbuda",
    "ASCENSION ISLAND": "Saint Helena, Ascension and Tristan da Cunha",
    "BOSNIA-HERZEGOVINA": "Bosnia and Herzegovina",
    "BRUNEI": "Brunei Darussalam",
    "CAPE VERDE": "Cabo Verde",
    "CHANNEL ISLANDS": "R12_WEU",
    "CONGO DEM REP": "Democratic Republic of the Congo",
    "COTE D'IVOIRE": "Côte d'Ivoire",
    "CZECH REPUBLIC": "Czechia",
    "DIEGO GARCIA": "British Indian Ocean Territory",
    "DUTCH CARIBBEAN": "R12_LAM",
    "ENGLAND & WALES": "United Kingdom",
    "FALKLAND ISLANDS": "Falkland Islands (Malvinas)",
    "NORTHERN IRELAND": "United Kingdom",
    "NORTHERN MARIANAS": "Northern Mariana Islands",
    "REUNION": "Réunion",
    "RUSSIA": "Russian Federation",
    "SAINT-MARTIN": "Saint Martin (French part)",
    "SAO TOME & PRINCIPE": "Sao Tome and Principe",
    "SCOTLAND": "United Kingdom",
    "ST HELENA": "Saint Helena, Ascension and Tristan da Cunha",
    "ST KITTS & NEVIS": "Saint Kitts and Nevis",
    "ST LUCIA": "Saint Lucia",
    "ST PIERRE & MIQUELON": "Saint Pierre and Miquelon",
    "ST VINCENT & GRENADINES": "Saint Vincent and the Grenadines",
    "TIMOR LESTE": "Timor-Leste",
    "TRINIDAD & TOBAGO": "Trinidad and Tobago",
    "TRISTAN DA CUNHA": "Saint Helena, Ascension and Tristan da Cunha",
    "TURKS & CAICOS": "Turks and Caicos Islands",
    "USA": "United States",
    "VIETNAM": "Viet Nam",
    "WALLIS & FUTUNA": "Wallis and Futuna",
}

_MESSAGE_R12_CODE_PATH = Path(
    "mappings/MESSAGEix-GLOBIOM/MESSAGEix-GLOBIOM_2.1-R12.yaml"
)
_MESSAGE_R12_COUNTRY_PATH = Path(
    "definitions/region/native_regions/MESSAGEix-GLOBIOM/MESSAGEix-GLOBIOM_2.1-R12.yaml"
)


def list_country_geometry_sources() -> tuple[CountryGeometrySource, ...]:
    """Return registered country-boundary sources."""

    return COUNTRY_GEOMETRY_SOURCES


def common_definitions_root() -> Path:
    """Return the configured common-definitions repository root."""

    configured = os.environ.get("COMMON_DEFINITIONS_ROOT")
    if not configured:
        raise FileNotFoundError(
            "COMMON_DEFINITIONS_ROOT is not set; point it to the common-definitions repository"
        )

    path = Path(configured).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            "common-definitions repository not found; set COMMON_DEFINITIONS_ROOT"
        )
    return path


@cache
def message_r12_region_labels() -> dict[str, str]:
    """Return MESSAGEix-GLOBIOM 2.1 R12 code -> label mapping."""

    code_yaml = (common_definitions_root() / _MESSAGE_R12_CODE_PATH).read_text()
    return dict(
        re.findall(
            r"-\s+(R12_[A-Z]+):\s+MESSAGEix-GLOBIOM 2\.1-R12\|(.+)",
            code_yaml,
        )
    )


@cache
def message_r12_country_to_region() -> dict[str, str]:
    """Return canonical country -> R12 region mapping."""

    label_to_code = {label: code for code, label in message_r12_region_labels().items()}
    mapping: dict[str, str] = {}
    current_label: str | None = None
    collecting = False
    buffer: list[str] = []

    country_yaml = (common_definitions_root() / _MESSAGE_R12_COUNTRY_PATH).read_text()

    for line in country_yaml.splitlines():
        stripped = line.strip()
        if stripped.startswith("- MESSAGEix-GLOBIOM 2.1-R12|") and stripped.endswith(":"):
            current_label = stripped.removeprefix("- MESSAGEix-GLOBIOM 2.1-R12|")[:-1]
            continue
        if current_label and "countries:" in stripped:
            collecting = True
            buffer = [stripped.split("countries:", maxsplit=1)[1].strip()]
        elif collecting:
            buffer.append(stripped)
        else:
            continue

        if buffer[-1].endswith("]"):
            collecting = False
            country_text = " ".join(buffer).strip()[1:-1]
            row = next(
                csv.reader(
                    io.StringIO(country_text),
                    delimiter=",",
                    quotechar='"',
                    skipinitialspace=True,
                )
            )
            code = label_to_code[current_label]
            for country in row:
                member = country.strip().strip("'")
                if member:
                    mapping[member] = code
            current_label = None

    return mapping


def resolve_region_labels(
    members: Sequence[object],
    *,
    region_set: str,
    strict: bool = True,
) -> tuple[str, ...]:
    """Resolve member labels to a registered region set."""

    match region_set:
        case "country":
            return tuple(str(member) for member in members)
        case "message_r12":
            return _resolve_message_r12_labels(members, strict=strict)
        case _:
            raise ValueError(f"unsupported region set: {region_set!r}")


def aggregate_entity_dataarray(
    values: xr.DataArray,
    region_labels: Sequence[object],
    *,
    reduction: Reduction,
    entity_dim: str = "entity",
    region_dim: str = "region",
    weights: xr.DataArray | Sequence[float] | None = None,
) -> xr.DataArray:
    """Aggregate an entity-indexed array to named regions."""

    if entity_dim not in values.dims:
        raise ValueError(f"{entity_dim!r} is not a dimension of the input values")

    labels = _labels_dataarray(
        region_labels=region_labels,
        entity_coord=values.coords[entity_dim],
        entity_dim=entity_dim,
        region_dim=region_dim,
    )
    labeled = values.assign_coords({region_dim: labels})

    match reduction:
        case "sum":
            return labeled.groupby(region_dim).sum(dim=entity_dim)
        case "weighted_mean":
            if weights is None:
                raise ValueError("weights are required for weighted_mean aggregation")
            weight_values = _weights_dataarray(
                weights=weights,
                entity_coord=values.coords[entity_dim],
                entity_dim=entity_dim,
                region_dim=region_dim,
                region_labels=region_labels,
            )
            numerator = (values * weight_values).assign_coords({region_dim: labels}).groupby(region_dim).sum(
                dim=entity_dim
            )
            denominator = weight_values.groupby(region_dim).sum(dim=entity_dim)
            if bool((denominator <= 0).any()):
                raise ValueError("total weight must be positive for every region")
            return numerator / denominator
        case _:
            raise ValueError(f"unsupported reduction: {reduction!r}")


def aggregate_entity_dataset(
    dataset: xr.Dataset,
    region_labels: Sequence[object],
    *,
    variable_reductions: Mapping[str, Reduction],
    entity_dim: str = "entity",
    region_dim: str = "region",
    weights: xr.DataArray | Sequence[float] | None = None,
) -> xr.Dataset:
    """Aggregate entity-indexed dataset variables to named regions."""

    if entity_dim not in dataset.dims:
        raise ValueError(f"{entity_dim!r} is not a dimension of the input dataset")

    aggregated: dict[str, xr.DataArray] = {}
    for name, variable in dataset.data_vars.items():
        if entity_dim not in variable.dims:
            aggregated[name] = variable
            continue

        reduction = variable_reductions.get(name)
        if reduction is None:
            continue

        variable_weights = weights if reduction == "weighted_mean" else None
        aggregated[name] = aggregate_entity_dataarray(
            variable,
            region_labels,
            reduction=reduction,
            entity_dim=entity_dim,
            region_dim=region_dim,
            weights=variable_weights,
        )

    return xr.Dataset(data_vars=aggregated, attrs=dict(dataset.attrs))


def _resolve_message_r12_labels(members: Sequence[object], *, strict: bool) -> tuple[str, ...]:
    lookup = _normalized_message_r12_lookup()
    resolved: list[str] = []
    missing: list[str] = []

    for member in members:
        text = str(member)
        key = _normalize_member_name(text)
        alias = _MESSAGE_R12_ALIASES.get(key, text)
        if alias in message_r12_region_labels():
            resolved.append(alias)
            continue

        region = lookup.get(_normalize_member_name(alias))
        if region is None:
            missing.append(text)
            continue
        resolved.append(region)

    if missing and strict:
        sample = ", ".join(sorted(set(missing))[:8])
        raise ValueError(f"unmapped members for region_set='message_r12': {sample}")

    return tuple(resolved)


@cache
def _normalized_message_r12_lookup() -> dict[str, str]:
    return {
        _normalize_member_name(country): region
        for country, region in message_r12_country_to_region().items()
    }


def _labels_dataarray(
    *,
    region_labels: Sequence[object],
    entity_coord: xr.DataArray,
    entity_dim: str,
    region_dim: str,
) -> xr.DataArray:
    if len(region_labels) != entity_coord.size:
        raise ValueError("region_labels must have the same length as the entity dimension")
    return xr.DataArray(
        [str(label) for label in region_labels],
        dims=(entity_dim,),
        coords={entity_dim: entity_coord.values},
        name=region_dim,
    )


def _weights_dataarray(
    *,
    weights: xr.DataArray | Sequence[float],
    entity_coord: xr.DataArray,
    entity_dim: str,
    region_dim: str,
    region_labels: Sequence[object],
) -> xr.DataArray:
    if isinstance(weights, xr.DataArray):
        if weights.dims != (entity_dim,):
            raise ValueError(f"weights must have dims ({entity_dim!r},)")
        data = weights.astype(float)
    else:
        if len(weights) != entity_coord.size:
            raise ValueError("weights must have the same length as the entity dimension")
        data = xr.DataArray(
            [float(weight) for weight in weights],
            dims=(entity_dim,),
            coords={entity_dim: entity_coord.values},
        )
    return data.assign_coords(
        {
            region_dim: _labels_dataarray(
                region_labels=region_labels,
                entity_coord=entity_coord,
                entity_dim=entity_dim,
                region_dim=region_dim,
            )
        }
    )


def _normalize_member_name(value: str) -> str:
    return " ".join(value.strip().upper().split())
