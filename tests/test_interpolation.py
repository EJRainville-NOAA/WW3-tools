import numpy as np
import pandas as pd
import xarray as xr
import ww3tools.interpolation as winterp


def _create_dummy_gfs():
    times = pd.date_range("2026-01-01", "2026-01-03", freq="1h")
    lats = np.arange(10, 35, 1.0)
    lons = np.arange(180, 230, 1.0)
    hs = np.random.uniform(1, 5, (len(times), len(lats), len(lons)))
    return xr.Dataset(
        data_vars={"hs": (("time", "latitude", "longitude"), hs)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )


def test_interpolate_model_to_points():
    ds_gfs = _create_dummy_gfs()

    target_lats = [20.0, 25.0]
    target_lons = [200.0, 210.0]
    target_times = ["2026-01-01 05:00", "2026-01-02 12:00"]

    res = winterp.interpolate_model_to_points(
        ds_gfs,
        target_lats=target_lats,
        target_lons=target_lons,
        target_times=target_times,
        variables=["hs"],
    )

    assert "model_hs" in res
    assert len(res.time) == 2


def test_interpolate_model_to_buoy():
    ds_gfs = _create_dummy_gfs()

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
