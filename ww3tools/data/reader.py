"""
Unified data reading and date range filtering helper functions.
"""

from datetime import datetime
import pandas as pd
import xarray as xr


def filter_dataset_by_date(
    ds: xr.Dataset,
    start_date: str | datetime | pd.Timestamp | None = None,
    end_date: str | datetime | pd.Timestamp | None = None,
    time_coord: str = "time"
) -> xr.Dataset:
    """
    Filter an xarray Dataset by start and end dates.

    Parameters
    ----------
    ds : xr.Dataset
    start_date : str, datetime, or Timestamp, optional
    end_date : str, datetime, or Timestamp, optional
    time_coord : str, optional (default 'time')

    Returns
    -------
    xr.Dataset
    """
    if time_coord not in ds.coords:
        return ds

    if start_date is None and end_date is None:
        return ds

    times = pd.to_datetime(ds[time_coord].values)
    mask = pd.Series(True, index=range(len(times)))

    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        mask &= (times >= start_dt)

    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        mask &= (times <= end_dt)

    return ds.isel({time_coord: mask.values})
