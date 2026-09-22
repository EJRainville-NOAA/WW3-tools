"""
High-level validation and verification routines comparing GFS model predictions with SoFar buoys.
"""

from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr

from ..data.gfs import read_gfs_data
from ..data.sofar import load_sofar_buoys, read_sofar_nc
from ..interpolation.spatial_temporal import interpolate_model_to_buoy
from .metrics import calculate_metrics


def validate_gfs_with_sofar_buoys(
    gfs_data: str | xr.Dataset,
    sofar_data: str | xr.Dataset,
    start_date: str | datetime | pd.Timestamp | None = None,
    end_date: str | datetime | pd.Timestamp | None = None,
    variable_mapping: dict[str, str] | None = None,
) -> dict:
    """
    Perform validation and verification of GFS model predictions using SoFar buoy data for any date range.

    Parameters
    ----------
    gfs_data : str or xr.Dataset
        Path to GFS file/directory or an opened GFS xarray.Dataset.
    sofar_data : str or xr.Dataset
        Path to SoFar buoy file/directory or an opened SoFar buoy xarray.Dataset.
    start_date : str or datetime, optional
        Start date filter for validation.
    end_date : str or datetime, optional
        End date filter for validation.
    variable_mapping : dict, optional
        Mapping from GFS model variable names to SoFar buoy variable names.
        Default mapping: {'hs': 'hs', 'wnd': 'wspd', 'HTSGW_surface': 'hs', 'WIND_surface': 'wspd'}

    Returns
    -------
    dict
        Dictionary containing:
        - 'collocated_ds': xr.Dataset with interpolated model and buoy observations
        - 'metrics': dict of validation error metrics per variable
        - 'summary': pandas.DataFrame summary table of metrics
    """
    if isinstance(gfs_data, str):
        gfs_ds = read_gfs_data(gfs_data, start_date=start_date, end_date=end_date)
    else:
        gfs_ds = gfs_data

    if isinstance(sofar_data, str):
        try:
            sofar_ds = load_sofar_buoys(sofar_data, start_date=start_date, end_date=end_date)
        except Exception:
            sofar_ds = read_sofar_nc(sofar_data, start_date=start_date, end_date=end_date)
    else:
        sofar_ds = sofar_data

    if variable_mapping is None:
        variable_mapping = {
            'hs': 'hs',
            'wnd': 'wspd',
            'HTSGW_surface': 'hs',
            'WIND_surface': 'wspd',
        }

    model_vars_to_interp = [mvar for mvar in variable_mapping.keys() if mvar in gfs_ds.data_vars]

    collocated_ds = interpolate_model_to_buoy(
        model_ds=gfs_ds,
        buoy_ds=sofar_ds,
        variables=model_vars_to_interp,
    )

    metrics_by_var = {}
    summary_rows = []

    for mvar in model_vars_to_interp:
        bvar = variable_mapping[mvar]
        model_key = f"model_{mvar}"
        obs_key = f"obs_{bvar}" if f"obs_{bvar}" in collocated_ds else (bvar if bvar in collocated_ds else None)

        if obs_key and model_key in collocated_ds:
            m_vals = collocated_ds[model_key].values
            o_vals = collocated_ds[obs_key].values

            m_metrics = calculate_metrics(m_vals, o_vals)
            metrics_by_var[mvar] = m_metrics

            row = {'variable': mvar, 'obs_variable': bvar}
            row.update(m_metrics)
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    return {
        'collocated_ds': collocated_ds,
        'metrics': metrics_by_var,
        'summary': summary_df,
    }
