"""Small made-up inputs for one potential site, shared by the siting tests."""

from __future__ import annotations

import pandas as pd
import xarray as xr

# Once-through withdrawal intensity blows up as water temperature approaches the
# discharge cap (the allowed temperature rise collapses), so its seasonal water
# limit is where mean-of-ratios and ratio-of-means diverge sharply.
COAL_ONCE_THROUGH = "li_coal_500mw_once_through"


def one_cell() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": ["+000.250_+000.250"],
            "region_id": ["AFR"],
            "country": ["NGA"],
            "grid_lat": [0.25],
            "grid_lon": [0.25],
            "cell_area_m2": [3.0e9],
        }
    )


def djf_climate() -> xr.Dataset:
    # Dec 2000, Jan and Feb 2001 make the complete DJF 2001. A cool month and a
    # near-cap hot month (10 C -> 30 C water) swing the once-through withdrawal
    # intensity ~5x while discharge falls, so the instantaneous water limit
    # 3600*dis/W differs sharply between the months and mean(ratio) separates
    # from ratio(means).
    climate = xr.Dataset(
        data_vars={
            "tw": (("time", "lat", "lon"), [[[288.15]], [[283.15]], [[303.15]]]),
            "dis": (("time", "lat", "lon"), [[[0.6]], [[1.0]], [[0.3]]]),
            "tas": (("time", "lat", "lon"), [[[295.15]], [[290.15]], [[300.15]]]),
            "hurs": (("time", "lat", "lon"), [[[65.0]], [[70.0]], [[60.0]]]),
            "ps": (("time", "lat", "lon"), [[[101200.0]], [[101325.0]], [[101000.0]]]),
        },
        coords={
            "time": pd.to_datetime(["2000-12-01", "2001-01-01", "2001-02-01"]),
            "lat": [0.25],
            "lon": [0.25],
        },
    )
    for name in ("tw", "tas"):
        climate[name].attrs["units"] = "K"
    climate["ps"].attrs["units"] = "Pa"
    return climate
