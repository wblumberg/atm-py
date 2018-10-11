import xarray as xr
import atmPy.precipitation.radar as _radar
import pandas as _pd
import atmPy.general.timeseries as _timeseries
# def read_ceilometer_nc(fname):
#     ds = xr.open_dataset(fname)
#     ceil = _ceilometry.Ceilometer()
#     ceil.backscatter = _ceilometry.Backscatter(ds.backscatter.to_pandas())
#     return ceil

def read_kazr_nc(fname, keep_xr_dataset = False):
    kazr = _radar.Kazr()
    if type(fname) == str:
        fname = [fname]

    reflects = []
    stds = []
    for fn in fname:
        ds = xr.open_dataset(fn)
        stds.append(ds.snr_copol.to_pandas())
        reflects.append(ds.reflectivity.to_pandas())


    kazr.signal2noise_ratio = _pd.concat(stds).sort_index()
    kazr.reflectivity = _pd.concat(reflects).sort_index()

    if keep_xr_dataset:
        kazr.xr_dataset = ds
    return kazr