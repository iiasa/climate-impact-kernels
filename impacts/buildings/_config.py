"""Input locations for the CHILLED workflow, taken from the environment.

Each loader in ``impacts.buildings`` accepts an explicit path. When none is
given, the path comes from the environment variable named here; when that is
unset too, the loader raises. The kernel itself never touches the filesystem.

- ``CHILLED_ARCHETYPE_PATH``: ALPS2025 building archetype table (CSV).
- ``CHILLED_PARAMETRIC_PATH``: parametric-run table (CSV).
- ``CHILLED_SETPOINT_PATH``: region cooling setpoint table (CSV).
- ``MESSAGE_R12_YAML_PATH``: MESSAGE R12 node definitions (YAML).
- ``ALPS_DATA_PATH``: directory holding the GAUL country raster and lookup CSV
  and the EWEMBI solar irradiance climatologies.
"""

from __future__ import annotations

import os
from pathlib import Path

ARCHETYPE_PATH_VARIABLE = "CHILLED_ARCHETYPE_PATH"
PARAMETRIC_PATH_VARIABLE = "CHILLED_PARAMETRIC_PATH"
SETPOINT_PATH_VARIABLE = "CHILLED_SETPOINT_PATH"
R12_YAML_PATH_VARIABLE = "MESSAGE_R12_YAML_PATH"
ALPS_DATA_VARIABLE = "ALPS_DATA_PATH"

GAUL_RASTER_FILE = "gaul_lvl0_hybrid_05_3.nc"
GAUL_CSV_FILE = "GAUL_lvl0_raster0.5.csv"
SOLAR_VERTICAL_FILE = "EWEMBI_vert_irrad_1980-2009_avg.nc"
SOLAR_HORIZONTAL_FILE = "EWEMBI_horiz_irrad_1980-2009_avg.nc"


def resolve_input_path(path: Path | str | None, variable: str, filename: str | None = None) -> Path:
    """Return ``path``, else the location named by environment ``variable``.

    ``variable`` names a file, or a directory when ``filename`` is given. Raises
    ``FileNotFoundError`` naming the variable when neither is set.
    """

    if path is not None:
        return Path(path)
    value = os.environ.get(variable)
    if not value:
        target = f"the directory holding {filename}" if filename else "the file"
        raise FileNotFoundError(f"no path was given and {variable} is not set; pass the path or set {variable} to {target}")
    return Path(value) / filename if filename else Path(value)
