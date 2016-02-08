import pandas as pd

#todo: this can be simplified with an internal function (from netCDF4 import num2date)
def _get_time(file_obj):
    bt = file_obj.variables['base_time']
    toff = file_obj.variables['time_offset']
    time = pd.to_datetime(0) + pd.to_timedelta(bt[:].flatten()[0], unit = 's') + pd.to_timedelta(toff[:], unit = 's')
    return time

