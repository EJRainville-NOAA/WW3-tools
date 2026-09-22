import os
import pytest
import numpy as np
import pandas as pd
import xarray as xr
import ww3tools.data as wdata


def test_convert_sofar_bufr_import_error():
    with pytest.raises(ImportError):
        wdata.convert_sofar_bufr_to_nc("nonexistent.bufr", "output.nc")


def test_read_and_load_sofar_nc(tmp_path):
    nc_file = os.path.join(tmp_path, "sofar_test.nc")
    times = pd.date_range("2026-01-01", "2026-01-02", freq="1h")
    ds_out = xr.Dataset(
        data_vars={
            "hs": (("time",), np.ones(len(times))),
            "station_name": "SOFAR_101",
        },
        coords={
            "time": times,
            "latitude": 20.0,
            "longitude": 200.0,
        },
    )
    ds_out.to_netcdf(nc_file)

    ds_read = wdata.read_sofar_nc(nc_file, start_date="2026-01-01 06:00", end_date="2026-01-01 18:00")
    assert len(ds_read.time) == 13

    ds_loaded = wdata.load_sofar_buoys(str(tmp_path), pattern="*.nc")
    assert "hs" in ds_loaded
