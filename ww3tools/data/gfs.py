from __future__ import annotations

import os
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr


def read_gfs_data(
    data_path: str,
    pattern: str = "*.nc",
    start_date: str | datetime | pd.Timestamp | None = None,
    end_date: str | datetime | pd.Timestamp | None = None,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """
    Read GFS model outputs from NetCDF or GRIB2 files for any specified date range.

    Parameters
    ----------
    data_path : str
        Directory path or glob pattern containing GFS output files.
    pattern : str, optional
        File search pattern if data_path is a directory (default "*.nc").
    start_date : str or datetime, optional
        Start date/time for filtering model data.
    end_date : str or datetime, optional
        End date/time for filtering model data.
    variables : list of str, optional
        Variables to extract (e.g. ['hs', 'wnd']).

    Returns
    -------
    xr.Dataset
    """
    if os.path.isdir(data_path):
        files = sorted(glob.glob(os.path.join(data_path, pattern)))
    else:
        files = sorted(glob.glob(data_path))

    if not files:
        raise FileNotFoundError(f"No GFS model files found matching {data_path}")

    datasets = [xr.open_dataset(f) for f in files]
    combined = datasets[0] if len(datasets) == 1 else xr.concat(datasets, dim='time')

    # Standardize longitudes to [0, 360] and sort monotonically
    for lon_var in ['longitude', 'lon']:
        if lon_var in combined.coords:
            lons = combined[lon_var].values
            if np.any(lons < 0):
                lons = np.where(lons < 0, lons + 360, lons)
                combined = combined.assign_coords({lon_var: lons}).sortby(lon_var)

    # Standardize time coordinate
    if 'time' in combined.coords and (start_date is not None or end_date is not None):
        start_dt = pd.to_datetime(start_date) if start_date is not None else None
        end_dt = pd.to_datetime(end_date) if end_date is not None else None

        times = pd.to_datetime(combined['time'].values)
        mask = np.ones(len(times), dtype=bool)
        if start_dt is not None:
            mask &= (times >= start_dt)
        if end_dt is not None:
            mask &= (times <= end_dt)
        combined = combined.isel(time=mask)

    if variables:
        available_vars = [v for v in variables if v in combined.data_vars]
        if available_vars:
            combined = combined[available_vars]

    return combined
