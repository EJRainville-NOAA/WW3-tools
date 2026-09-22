import numpy as np
import pandas as pd
import xarray as xr
import ww3tools.validation as wval


def _create_dummy_gfs():
    times = pd.date_range("2026-01-01", "2026-01-05", freq="3h")
    lats = np.arange(10, 35, 1.0)
    lons = np.arange(180, 230, 1.0)
    hs = np.random.uniform(1, 5, (len(times), len(lats), len(lons)))
    return xr.Dataset(
        data_vars={"hs": (("time", "latitude", "longitude"), hs)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )


def test_calculate_metrics():
    model = [2.0, 3.0, 4.0, 5.0]
    obs = [2.1, 2.9, 4.2, 4.8]

    m = wval.calculate_metrics(model, obs)
    assert "bias" in m
    assert "rmse" in m
    assert "mae" in m
    assert "correlation" in m
    assert m["count"] == 4
    assert abs(m["bias"]) < 0.2


def test_validate_gfs_with_sofar_buoys():
    ds_gfs = _create_dummy_gfs()

    times = pd.date_range("2026-01-01 06:00", "2026-01-04 18:00", freq="3h")
    sofar_ds = xr.Dataset(
        data_vars={
            "hs": (("time",), 2.0 + np.sin(np.linspace(0, 5, len(times)))),
        },
        coords={
            "time": times,
            "latitude": (("time",), np.full(len(times), 30.0)),
            "longitude": (("time",), np.full(len(times), 200.0)),
        },
    )

    res = wval.validate_gfs_with_sofar_buoys(
        gfs_data=ds_gfs,
        sofar_data=sofar_ds,
        start_date="2026-01-02",
        end_date="2026-01-04",
    )

    assert "collocated_ds" in res
    assert "metrics" in res
    assert "summary" in res
    assert not res["summary"].empty
