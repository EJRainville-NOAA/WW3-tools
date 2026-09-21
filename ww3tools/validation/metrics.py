"""
Statistical verification metrics for comparing model outputs with observation data.
"""

import numpy as np
import pandas as pd


def calculate_metrics(
    model: np.ndarray | list,
    obs: np.ndarray | list,
    vmin: float = -np.inf,
    vmax: float = np.inf,
    maxdiff: float = np.inf,
) -> dict[str, float]:
    """
    Calculate verification error metrics comparing model predictions against observations.

    Parameters
    ----------
    model : array-like
    obs : array-like
    vmin, vmax : float, optional
        Quality control limits.
    maxdiff : float, optional
        Maximum allowed difference between model and obs.

    Returns
    -------
    dict
        Dictionary of metrics: bias, rmse, mae, normalized_bias, scatter_index, correlation, count.
    """
    model = np.asarray(model, dtype=float).ravel()
    obs = np.asarray(obs, dtype=float).ravel()

    if len(model) != len(obs):
        raise ValueError("Model and observations arrays must have equal length.")

    valid = (
        ~np.isnan(model)
        & ~np.isnan(obs)
        & (model >= vmin)
        & (model <= vmax)
        & (obs >= vmin)
        & (obs <= vmax)
        & (np.abs(model - obs) <= maxdiff)
    )

    m = model[valid]
    o = obs[valid]
    n = len(o)

    if n == 0:
        return {
            "bias": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "normalized_bias": np.nan,
            "normalized_rmse": np.nan,
            "scatter_index": np.nan,
            "correlation": np.nan,
            "count": 0,
        }

    diff = m - o
    bias = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    mean_obs = float(np.mean(o))

    norm_bias = bias / abs(mean_obs) if mean_obs != 0 else np.nan
    norm_rmse = float(np.sqrt(np.sum(diff**2) / np.sum(o**2))) if np.sum(o**2) != 0 else np.nan

    if np.sum(o**2) != 0:
        scatter_index = float(np.sqrt(np.sum(((m - np.mean(m)) - (o - mean_obs)) ** 2) / np.sum(o**2)))
    else:
        scatter_index = np.nan

    if n > 1 and np.std(m) > 0 and np.std(o) > 0:
        corr = float(np.corrcoef(m, o)[0, 1])
    else:
        corr = np.nan

    return {
        "bias": bias,
        "rmse": rmse,
        "mae": mae,
        "normalized_bias": norm_bias,
        "normalized_rmse": norm_rmse,
        "scatter_index": scatter_index,
        "correlation": corr,
        "count": n,
    }
