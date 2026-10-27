"""Fuel codes of the thermal fleet and the classes the cooling kernels use.

Both the wet and the dry kernel read the fleet's ``fuel`` column through this one
table. A code is matched exactly (after upper-casing), never as a substring of a
longer string, and an unknown code raises rather than falling into a default class.
"""

from __future__ import annotations

# Fuel code -> class. The wet kernel supports coal, oil, gas, biomass and nuclear
# (plus WSTH, which it resolves from the technology family); the dry kernel reads
# nuclear, non_combustible and waste_heat apart from the combustible classes.
FUEL_CLASSES: dict[str, str] = {
    # coal
    "COAL": "coal",
    "COKE": "coal",
    "LIGNITE": "coal",
    "PEAT": "coal",
    "PETCOKE": "coal",
    "SHALE": "coal",
    # oil
    "OIL": "oil",
    "DIESEL": "oil",
    "FO": "oil",
    "HFO": "oil",
    "LFO": "oil",
    "KERO": "oil",
    "NAPHTHA": "oil",
    # gas
    "GAS": "gas",
    "NG": "gas",
    "LNG": "gas",
    "LGAS": "gas",
    "BGAS": "gas",
    # biomass
    "BIO": "biomass",
    "BIOMASS": "biomass",
    "WOOD": "biomass",
    "BAGASSE": "biomass",
    # nuclear
    "UR": "nuclear",
    "URANIUM": "nuclear",
    "NUC": "nuclear",
    "NUCLEAR": "nuclear",
    # combustible, with no wet-cooling coefficients
    "WASTE": "waste",
    # no combustion
    "GEOTHERMAL": "non_combustible",
    "SOLAR": "non_combustible",
    # heat recovered from another process: WSTH is the steam unit of a combined
    # cycle or similar, and carries no fuel of its own
    "WSTH": "waste_heat",
    "WASTE HEAT": "waste_heat",
}


def classify_fuel(fuel: str) -> str:
    """Return the class of a fuel code; raise ``ValueError`` for an unknown code."""

    try:
        return FUEL_CLASSES[fuel.upper()]
    except KeyError:
        raise ValueError(f"unknown fuel code: {fuel!r}") from None
