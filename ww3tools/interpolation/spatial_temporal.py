"""
Spatial and temporal interpolation routines for model outputs and observation datasets.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


def setup_spatial_temporal_interpolator(
    model_ds: xr.Dataset,
    var_name: str,
    lat_coord: str = "latitude",
    lon_coord: str = "longitude",
    time_coord: str = "time",
) -> tuple[RegularGridInterpolator, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a SciPy RegularGridInterpolator for a model variable on (latitude, longitude, time).

    Parameters
    ----------
    model_ds : xr.Dataset
    var_name : str
    lat_coord : str
    lon_coord : str
    time_coord : str

    Returns
    -------
    interpolator : RegularGridInterpolator
    lats : np.ndarray
    lons : np.ndarray
    time_unix : np.ndarray
    """
    if var_name not in model_ds:
        raise KeyError(f"Variable '{var_name}' not in dataset.")

    lats = model_ds[lat_coord].values if lat_coord in model_ds else model_ds['lat'].values
    lons = model_ds[lon_coord].values if lon_coord in model_ds else model_ds['lon'].values
    times = model_ds[time_coord].values if time_coord in model_ds else model_ds['time'].values

    lons = np.where(lons < 0, lons + 360, lons)
    times_unix = (pd.to_datetime(times) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

    data = model_ds[var_name]
    if 'time' in data.dims and data.dims.index('time') != 2:
        dims_other = [d for d in data.dims if d != 'time']
        data = data.transpose(dims_other[0], dims_other[1], 'time')

    interpolator = RegularGridInterpolator(
        (lats, lons, times_unix.values),
        data.values,
        bounds_error=False,
        fill_value=np.nan
    )
    return interpolator, lats, lons, times_unix.values


def interpolate_model_to_points(
    model_ds: xr.Dataset,
    target_lats: np.ndarray | list[float],
    target_lons: np.ndarray | list[float],
    target_times: np.ndarray | list,
    variables: list[str] | None = None,
    lat_coord: str = "latitude",
    lon_coord: str = "longitude",
    time_coord: str = "time",
) -> xr.Dataset:
    """
    Interpolate gridded model dataset to target point locations and times.

    Parameters
    ----------
    model_ds : xr.Dataset
    target_lats : array-like
        Target latitudes.
    target_lons : array-like
        Target longitudes.
    target_times : array-like
        Target times (datetime, Timestamp, or string).
    variables : list of str, optional
        Variables to interpolate. If None, interpolates all data variables.
    lat_coord, lon_coord, time_coord : str

    Returns
    -------
    xr.Dataset
        Dataset of interpolated variables along dimension 'point' or 'time'.
    """
    target_lats = np.asarray(target_lats, dtype=float)
    target_lons = np.asarray(target_lons, dtype=float)
    target_lons = np.where(target_lons < 0, target_lons + 360, target_lons)
    target_times_dt = pd.to_datetime(target_times)
    target_times_unix = (target_times_dt - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

    target_points = np.column_stack([target_lats, target_lons, target_times_unix.values])

    if variables is None:
        variables = list(model_ds.data_vars.keys())

    interp_vars = {}
    for v in variables:
        if v not in model_ds:
            continue
        try:
            interp, _, _, _ = setup_spatial_temporal_interpolator(
                model_ds, v, lat_coord=lat_coord, lon_coord=lon_coord, time_coord=time_coord
            )
            interp_vals = interp(target_points)
            interp_vars[f"model_{v}"] = (('time',), interp_vals)
        except Exception:
            continue

    coords = {
        'time': target_times_dt,
        'latitude': (('time',), target_lats),
        'longitude': (('time',), target_lons),
    }

    return xr.Dataset(data_vars=interp_vars, coords=coords)


def interpolate_model_to_buoy(
    model_ds: xr.Dataset,
    buoy_ds: xr.Dataset,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """
    Interpolate a gridded model dataset to the locations and times of a buoy dataset.

    Parameters
    ----------
    model_ds : xr.Dataset
        Gridded model dataset (e.g. GFS model).
    buoy_ds : xr.Dataset
        Buoy observation dataset containing latitude, longitude, and time coordinates.
    variables : list of str, optional
        Model variables to interpolate.

    Returns
    -------
    xr.Dataset
        Combined dataset containing buoy observations and collocated/interpolated model outputs.
    """
    buoy_time = buoy_ds['time'].values

    if 'latitude' in buoy_ds:
        buoy_lats = buoy_ds['latitude'].values
    elif 'lat' in buoy_ds:
        buoy_lats = buoy_ds['lat'].values
    else:
        raise KeyError("Buoy dataset does not contain latitude/lat variable.")

    if 'longitude' in buoy_ds:
        buoy_lons = buoy_ds['longitude'].values
    elif 'lon' in buoy_ds:
        buoy_lons = buoy_ds['lon'].values
    else:
        raise KeyError("Buoy dataset does not contain longitude/lon variable.")

    if buoy_lats.ndim == 0:
        buoy_lats = np.full(len(buoy_time), buoy_lats.item())
    if buoy_lons.ndim == 0:
        buoy_lons = np.full(len(buoy_time), buoy_lons.item())

    model_interp = interpolate_model_to_points(
        model_ds,
        target_lats=buoy_lats,
        target_lons=buoy_lons,
        target_times=buoy_time,
        variables=variables,
    )

    merged = model_interp.copy(deep=True)
    for bvar in buoy_ds.data_vars:
        if bvar not in merged:
            merged[f"obs_{bvar}"] = buoy_ds[bvar]

    return merged
