"""Small synthetic CHILLED inputs shared by the buildings tests: a 2 x 2 climate
grid of four months, a monthly solar climatology, two spatial units and two
archetypes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from impacts.buildings.archetypes import load_region_cooling_setpoints


def climate() -> xr.Dataset:
    climate = xr.Dataset(
        data_vars={
            "tas": (
                ("time", "lat", "lon"),
                [
                    [[273.15, 274.15], [275.15, 276.15]],
                    [[283.15, 284.15], [285.15, 286.15]],
                    [[293.15, 294.15], [295.15, 296.15]],
                    [[303.15, 304.15], [305.15, 306.15]],
                ],
            )
        },
        coords={
            "time": pd.to_datetime(["2001-01-01", "2001-02-01", "2001-07-01", "2001-08-01"]),
            "lat": [10.0, 20.0],
            "lon": [100.0, 110.0],
        },
    )
    climate["tas"].attrs["units"] = "K"
    return climate


def solar() -> xr.Dataset:
    month = np.arange(1, 13)
    solar = xr.Dataset(
        data_vars={
            "solar_vertical": (
                ("month", "lat", "lon"),
                np.stack(
                    [
                        [[100.0 + month_id, 110.0 + month_id], [120.0 + month_id, 130.0 + month_id]]
                        for month_id in month
                    ]
                ),
            ),
            "solar_horizontal": (
                ("month", "lat", "lon"),
                np.stack(
                    [
                        [[200.0 + month_id, 210.0 + month_id], [220.0 + month_id, 230.0 + month_id]]
                        for month_id in month
                    ]
                ),
            ),
        },
        coords={"month": month, "lat": [10.0, 20.0], "lon": [100.0, 110.0]},
    )
    return solar


def spatial_units() -> pd.DataFrame:
    spatial_units = pd.DataFrame(
        {
            "grid_lat": [10.0, 20.0],
            "grid_lon": [100.0, 110.0],
            "region": ["AFR", "CPA"],
            "grid_label": ["cell_a", "cell_b"],
        }
    )
    return spatial_units


def archetypes() -> pd.DataFrame:
    archetypes = pd.DataFrame(
        {
            "name": ["AFR_sfh_s1", "CPA_sfh_s1"],
            "building_type": ["sfh", "sfh"],
            "urt": ["rur", "urb"],
            "arch": ["s1", "s1"],
            "region": ["AFR", "CPA"],
            "permanent": [1, 1],
            "area_env": [3.7, 3.6],
            "gl_perc": [0.125, 0.125],
            "vol": [3.0, 3.0],
            "fl_cnd": [1.0, 1.0],
            "u_val": [3.475, 3.150],
            "ach_cl": [0.5, 0.5],
            "ach_op": [1.5, 1.5],
            "gn_int": [2.14, 2.14],
            "gl_g": [0.85, 0.85],
            "gl_sh": [0.8, 0.8],
            "roof_area": [1.0, 1.0],
            "roof_abs": [0.6, 0.6],
            "u_roof": [3.3, 2.0],
        }
    )
    return archetypes


def setpoint_table(case: unittest.TestCase) -> pd.DataFrame:
    """A two-region setpoint table read from a CSV, as a caller supplies one."""

    directory = tempfile.TemporaryDirectory()
    case.addCleanup(directory.cleanup)
    path = Path(directory.name) / "setpoints.csv"
    pd.DataFrame(
        {
            "region": ["AFR", "CPA"],
            "cooling_setpoint_c": [25.0, 23.0],
            "cooling_max_c": [27.0, 25.0],
        }
    ).to_csv(path, index=False)
    return load_region_cooling_setpoints(path)
