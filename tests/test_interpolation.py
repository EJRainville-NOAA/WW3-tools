import numpy as np
import pandas as pd
import xarray as xr
import ww3tools.data as wdata
import ww3tools.interpolation as winterp


def test_interpolate_model_to_points():
    ds_gfs = wdata.create_sample_gfs_dataset("2026-01-01", "2026-01-03", freq="1h")

    target_lats = [20.0, 25.0]
    target_lons = [200.0, 210.0]
    target_times = ["2026-01-01 05:00", "2026-01-02 12:00"]

    res = winterp.interpolate_model_to_points(
        ds_gfs,
        target_lats=target_lats,
        target_lons=target_lons,
        target_times=target_times,
        variables=["hs", "wnd"],
    )

    assert "model_hs" in res
    assert "model_wnd" in res
    assert len(res.time) == 2


def test_interpolate_model_to_buoy():
    ds_gfs = wdata.create_sample_gfs_dataset("2026-01-01", "2026-01-03", freq="1h")

    times = pd.date_range("2026-01-01 02:00", "2026-01-02 20:00", freq="2h")
    buoy_ds = xr.Dataset(
        data_vars={
            "hs": (("time",), np.ones(len(times))),
        },
        coords={
            "time": times,
            "latitude": (("time",), np.full(len(times), 30.0)),
            "longitude": (("time",), np.full(len(times), 220.0)),
        },
    )

    collocated = winterp.interpolate_model_to_buoy(ds_gfs, buoy_ds, variables=["hs"])
    assert "model_hs" in collocated
    assert "obs_hs" in collocated
