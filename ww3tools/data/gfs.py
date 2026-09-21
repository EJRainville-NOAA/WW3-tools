"""
Data utilities for reading, loading, and managing GFS wave/wind model data.
"""

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
    Read GFS model outputs from NetCDF/GRIB2 files for any specified date range.

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
        Variables to extract (e.g. ['hs', 'wnd'] or ['HTSGW_surface', 'WIND_surface']).

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

    datasets = []
    for f in files:
        try:
            ds = xr.open_dataset(f)
            datasets.append(ds)
        except Exception:
            continue

    if not datasets:
        raise ValueError(f"Could not open any GFS dataset from {data_path}")

    if len(datasets) == 1:
        combined = datasets[0]
    else:
        combined = xr.concat(datasets, dim='time')

    # Standardize longitudes to [0, 360] if needed
    if 'longitude' in combined.coords:
        lons = combined['longitude'].values
        if np.any(lons < 0):
            lons[lons < 0] += 360
            combined = combined.assign_coords(longitude=lons)
    elif 'lon' in combined.coords:
        lons = combined['lon'].values
        if np.any(lons < 0):
            lons[lons < 0] += 360
            combined = combined.assign_coords(lon=lons)

    # Standardize time coordinate
    time_coord = 'time' if 'time' in combined.coords else None
    if time_coord and (start_date is not None or end_date is not None):
        start_dt = pd.to_datetime(start_date) if start_date is not None else None
        end_dt = pd.to_datetime(end_date) if end_date is not None else None

        times = pd.to_datetime(combined[time_coord].values)
        mask = np.ones(len(times), dtype=bool)
        if start_dt is not None:
            mask &= (times >= start_dt)
        if end_dt is not None:
            mask &= (times <= end_dt)
        combined = combined.isel({time_coord: mask})

    if variables:
        available_vars = [v for v in variables if v in combined.data_vars]
        if available_vars:
            combined = combined[available_vars]

    return combined


def create_sample_gfs_dataset(
    start_date: str = "2026-01-01",
    end_date: str = "2026-01-03",
    freq: str = "3h",
    lat_range: tuple[float, float] = (10.0, 50.0),
    lon_range: tuple[float, float] = (180.0, 260.0),
    spatial_res: float = 1.0,
) -> xr.Dataset:
    """
    Generate a synthetic GFS wave/wind dataset for testing and validation workflows.

    Parameters
    ----------
    start_date : str
    end_date : str
    freq : str
    lat_range : tuple of float
    lon_range : tuple of float
    spatial_res : float

    Returns
    -------
    xr.Dataset
    """
    times = pd.date_range(start=start_date, end=end_date, freq=freq)
    lats = np.arange(lat_range[0], lat_range[1] + spatial_res, spatial_res)
    lons = np.arange(lon_range[0], lon_range[1] + spatial_res, spatial_res)

    nt = len(times)
    nlat = len(lats)
    nlon = len(lons)

    np.random.seed(42)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

    base_hs = 2.0 + 0.5 * np.sin(np.radians(lat_grid)) + 0.3 * np.cos(np.radians(lon_grid))
    time_factor = 1.0 + 0.2 * np.sin(np.linspace(0, 4 * np.pi, nt))[:, None, None]

    hs_data = base_hs[None, :, :] * time_factor + np.random.normal(0, 0.1, (nt, nlat, nlon))
    hs_data = np.clip(hs_data, 0.1, 15.0)

    wnd_data = hs_data * 4.0 + np.random.normal(0, 0.5, (nt, nlat, nlon))
    wnd_data = np.clip(wnd_data, 0.5, 40.0)

    ds = xr.Dataset(
        data_vars={
            'hs': (('time', 'latitude', 'longitude'), hs_data, {'units': 'm', 'long_name': 'significant wave height'}),
            'wnd': (('time', 'latitude', 'longitude'), wnd_data, {'units': 'm/s', 'long_name': 'wind speed'}),
            'HTSGW_surface': (('time', 'latitude', 'longitude'), hs_data, {'units': 'm'}),
            'WIND_surface': (('time', 'latitude', 'longitude'), wnd_data, {'units': 'm/s'}),
        },
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons,
        },
        attrs={
            'title': 'Synthetic GFS Wave Model Output',
            'model': 'GFS-Wave',
        }
    )
    return ds
