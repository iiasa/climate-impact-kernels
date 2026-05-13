"""Operational path defaults for the CHILLED workflow.

These defaults are operational conveniences for the CHILLED matrix runner
and diagnostic scripts. They are not part of the pure kernel contract;
callers are free to pass explicit paths to every loader in
``impacts.domain.buildings.chilled_io``.

This is the only module in ``impacts.domain.buildings`` that reads from
``os.environ`` at import time. The kernel itself never touches the
filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path

WATXENE_PATH = Path(os.environ.get("WATXENE_PATH", "/mnt/p/watxene"))
ALPS_DATA_PATH = WATXENE_PATH / "ISIMIP_postprocessed" / "data_for_vignesh" / "ALPS_2024"

DEFAULT_ARCHETYPE_INPUT_PATH = Path(
    os.environ.get(
        "CHILLED_ARCHETYPE_PATH",
        str(Path.home() / "message-ix-buildings/message_ix_buildings/data/chilled/version/ALPS2025/arch_input_reg.csv"),
    )
)

# Sentinel: the parametric-run table has no defensible default. The loader
# raises FileNotFoundError when this is None.
DEFAULT_PARAMETRIC_INPUT_PATH = (
    Path(os.environ["CHILLED_PARAMETRIC_PATH"])
    if os.environ.get("CHILLED_PARAMETRIC_PATH")
    else None
)

DEFAULT_R12_YAML_PATH = Path(
    os.environ.get(
        "MESSAGE_R12_YAML_PATH",
        str(Path.home() / "message-ix-models/message_ix_models/data/node/R12.yaml"),
    )
)

DEFAULT_GAUL_RASTER_PATH = ALPS_DATA_PATH / "gaul_lvl0_hybrid_05_3.nc"
DEFAULT_GAUL_CSV_PATH = ALPS_DATA_PATH / "GAUL_lvl0_raster0.5.csv"
DEFAULT_SOLAR_VERTICAL_PATH = ALPS_DATA_PATH / "EWEMBI_vert_irrad_1980-2009_avg.nc"
DEFAULT_SOLAR_HORIZONTAL_PATH = ALPS_DATA_PATH / "EWEMBI_horiz_irrad_1980-2009_avg.nc"
