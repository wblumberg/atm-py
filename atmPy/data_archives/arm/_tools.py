import pandas as _pd
import os as _os

#todo: this can be simplified with an internal function (from netCDF4 import num2date)
def _get_time(file_obj):
    bt = file_obj.variables['base_time']
    toff = file_obj.variables['time_offset']
    time = _pd.to_datetime(0) + _pd.to_timedelta(bt[:].flatten()[0], unit = 's') + _pd.to_timedelta(toff[:], unit = 's')
    return time


def is_in_time_window(f,time_window, verbose = False):
    out = True
    if time_window:
        fnt = _os.path.split(f)[-1].split('.')
        ts = fnt[-3]
        file_start_data = _pd.to_datetime(ts)
        start_time = _pd.to_datetime(time_window[0])
        end_time = _pd.to_datetime(time_window[1])
        dt_start = file_start_data - start_time
        dt_end = file_start_data - end_time
        out = file_start_data

        if dt_start.total_seconds() < -86399:
            if verbose:
                print('outside (before) the time window ... skip')
            out = False
        elif dt_end.total_seconds() > 86399:
            if verbose:
                print('outside (after) the time window ... skip')
            out = False
    return out