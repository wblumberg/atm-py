import xarray as xr
import atmPy.clouds.ceilometry as _ceilometry
import pandas as _pd
# def read_ceilometer_nc(fname):
#     ds = xr.open_dataset(fname)
#     ceil = _ceilometry.Ceilometer()
#     ceil.backscatter = _ceilometry.Backscatter(ds.backscatter.to_pandas())
#     return ceil

def read_ceilometer_nc(fname, keep_xr_dataset = False):

    ceil = _ceilometry.Ceilometer()
    if type(fname) == str:
        fname = [fname]

    bs = []
    cloudbases = []
    for fn in fname:
        ds = xr.open_dataset(fn)
        bs.append(ds.backscatter.to_pandas())

        fcb = ds.first_cbh.to_pandas()
        scb = ds.second_cbh.to_pandas()
        tcb = ds.third_cbh.to_pandas()
        cloudbases.append(_pd.DataFrame({'First_cloud_base': fcb,
                      'Second_cloud_base': scb,
                      'Third_cloud_base': tcb}))


    bsdf = _pd.concat(bs).sort_index()
    ceil.backscatter = bsdf

    cbdf = _pd.concat(cloudbases).sort_index()
    ceil.cloudbase = cbdf



    if keep_xr_dataset:
        ceil.xr_dataset = ds
    return ceil
