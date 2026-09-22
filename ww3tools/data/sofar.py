from __future__ import annotations

from datetime import datetime
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr


def convert_sofar_bufr_to_nc(bufr_file_path: str, nc_output_path: str) -> str:
    """
    Convert SoFar buoy data from BUFR format to NetCDF file.

    Parameters
    ----------
    bufr_file_path : str
        Path to input BUFR file.
    nc_output_path : str
        Path where output NetCDF file will be saved.

    Returns
    -------
    str
        Path to created NetCDF file.
    """
    try:
        import bufr
    except ImportError:
        raise ImportError(
            "The 'bufr' Python library is required to convert BUFR files. "
            "Please ensure NCEP BUFR library/bindings are installed in your environment."
        )

    import netCDF4 as nc

    q = bufr.QuerySet()
    q.add('year', '*/YEAR')
    q.add('month', '*/MNTH')
    q.add('day', '*/DAYS')
    q.add('hour', '*/HOUR')
    q.add('minute', '*/MINU')
    q.add('station_name', '*/LSTN')
    q.add('buoy_type', '*/BUYT')
    q.add('latitude', '*/CLATH')
    q.add('longitude', '*/CLONH')
    q.add('hs', '*/BASICWAV/SGWH')
    q.add('dsdw', '*/BASICWAV/DSDW')
    q.add('ddwc', '*/BASICWAV/DDWC')
    q.add('avwp', '*/BASICWAV/AVWP')
    q.add('spwp', '*/BASICWAV/SPWP')
    q.add('wspd', '*/WSPD')
    q.add('wdir', '*/WDIR')

    with bufr.File(bufr_file_path) as f:
        r = f.execute(q)

    os.makedirs(os.path.dirname(os.path.abspath(nc_output_path)), exist_ok=True)
    rootgrp = nc.Dataset(nc_output_path, 'w', clobber=True)
    rootgrp.description = "SoFar buoy data converted from BUFR"
    rootgrp.source = "Converted from BUFR data"

    year = r.get('year')
    month = r.get('month')
    day = r.get('day')
    hour = r.get('hour')
    minute = r.get('minute')

    time_dim = rootgrp.createDimension('time', year.size)
    time_nc = rootgrp.createVariable('time', 'f8', ('time',))
    time_nc.units = "seconds since 1970-01-01 00:00:00"
    time_nc.calendar = "gregorian"

    datetime_array = np.empty(shape=year.shape)
    for n in range(year.size):
        dt = datetime(year[n], month[n], day[n], hour[n], minute[n])
        datetime_array[n] = nc.date2num(dt, units=time_nc.units, calendar=time_nc.calendar)

    time_nc[:] = datetime_array

    lats = rootgrp.createVariable('latitude', 'f4', ('time',))
    lats.long_name = 'latitude'
    lats.units = 'degrees_north'
    lats[:] = r.get('latitude')

    lons = rootgrp.createVariable('longitude', 'f4', ('time',))
    lons.long_name = 'longitude'
    lons.units = 'degrees_east'
    lons[:] = r.get('longitude')

    lstn = rootgrp.createVariable('station_name', str, ('time',))
    lstn.long_name = 'station name'
    lstn[:] = r.get('station_name')

    buoy_type = rootgrp.createVariable('buoy_type', 'i4', ('time',))
    buoy_type.units = 'int'
    buoy_type[:] = r.get('buoy_type')

    hs = rootgrp.createVariable('hs', 'f4', ('time',))
    hs.long_name = 'significant wave height'
    hs.units = 'meters'
    hs[:] = r.get('hs')

    dsdw = rootgrp.createVariable('dsdw', 'i4', ('time',))
    dsdw.long_name = 'directional spread of dominant wave'
    dsdw.units = 'degree'
    dsdw[:] = r.get('dsdw')

    ddwc = rootgrp.createVariable('ddwc', 'i4', ('time',))
    ddwc.long_name = 'direction from which dominant waves are coming'
    ddwc.units = 'degree true'
    ddwc[:] = r.get('ddwc')

    avwp = rootgrp.createVariable('avwp', 'f4', ('time',))
    avwp.long_name = 'average wave period'
    avwp.units = 'seconds'
    avwp[:] = r.get('avwp')

    spwp = rootgrp.createVariable('spwp', 'f4', ('time',))
    spwp.long_name = 'spectral peak wave period'
    spwp.units = 'seconds'
    spwp[:] = r.get('spwp')

    wspd = rootgrp.createVariable('wspd', 'f4', ('time',))
    wspd.long_name = 'wind speed'
    wspd.units = 'meters per second'
    wspd[:] = r.get('wspd')

    wdir = rootgrp.createVariable('wdir', 'f4', ('time',))
    wdir.long_name = 'wind direction'
    wdir.units = 'degree true'
    wdir[:] = r.get('wdir')

    rootgrp.close()
    return nc_output_path


def read_sofar_nc(
    nc_path: str,
    start_date: str | datetime | pd.Timestamp | None = None,
    end_date: str | datetime | pd.Timestamp | None = None,
) -> xr.Dataset:
    """
    Read a SoFar buoy NetCDF file into an xarray Dataset, optionally filtering by date range.

    Parameters
    ----------
    nc_path : str
        File path to NetCDF file.
    start_date : str, datetime, or Timestamp, optional
        Start time filter (inclusive).
    end_date : str, datetime, or Timestamp, optional
        End time filter (inclusive).

    Returns
    -------
    xr.Dataset
    """
    ds = xr.open_dataset(nc_path)

    if np.issubdtype(ds['time'].dtype, np.number):
        time_vals = pd.to_datetime(ds['time'].values, unit='s', origin='1970-01-01')
        ds = ds.assign_coords(time=time_vals)

    if start_date is not None or end_date is not None:
        start_dt = pd.to_datetime(start_date) if start_date is not None else None
        end_dt = pd.to_datetime(end_date) if end_date is not None else None

        times = pd.to_datetime(ds['time'].values)
        mask = np.ones(len(times), dtype=bool)
        if start_dt is not None:
            mask &= (times >= start_dt)
        if end_dt is not None:
            mask &= (times <= end_dt)

        ds = ds.isel(time=mask)

    return ds


def load_sofar_buoys(
    data_path: str,
    pattern: str = "*.nc",
    start_date: str | datetime | pd.Timestamp | None = None,
    end_date: str | datetime | pd.Timestamp | None = None,
    buoy_ids: list[str] | None = None,
) -> xr.Dataset:
    """
    Load and merge SoFar buoy datasets from a directory or file pattern for a specified date range.

    Parameters
    ----------
    data_path : str
        Directory path or glob pattern.
    pattern : str, optional
        File pattern if data_path is a directory (default "*.nc").
    start_date : str or datetime, optional
        Start date filter.
    end_date : str or datetime, optional
        End date filter.
    buoy_ids : list of str, optional
        List of station names / buoy IDs to filter.

    Returns
    -------
    xr.Dataset
    """
    if os.path.isdir(data_path):
        files = sorted(glob.glob(os.path.join(data_path, pattern)))
    else:
        files = sorted(glob.glob(data_path))

    if not files:
        raise FileNotFoundError(f"No SoFar buoy files found matching {data_path}")

    datasets = []
    for file_path in files:
        try:
            ds = read_sofar_nc(file_path, start_date=start_date, end_date=end_date)
            if ds.sizes.get('time', 0) > 0:
                if buoy_ids is not None and 'station_name' in ds:
                    st_name = str(ds['station_name'].values)
                    if not any(bid in st_name for bid in buoy_ids):
                        continue
                datasets.append(ds)
        except Exception:
            continue

    if not datasets:
        raise ValueError("No valid SoFar buoy records found for the specified criteria/date range.")

    if len(datasets) == 1:
        return datasets[0]

    return xr.concat(datasets, dim='time')
