import pandas as pd
import pylab as plt
from copy import deepcopy as _deepcopy

from atmPy.general import timeseries
from atmPy.tools import plt_tools


class VerticalProfile(object):
    def __init__(self, data):
        self.data = data
        self._x_label = None
        self._y_label = 'Altitude'

    def algin_to(self, ts_other):
        return align_to(self, ts_other)

    def merge(self, ts):
        return merge(self,ts)

    def plot(self, ax=False, color=False):
        if not ax:
            f, a = plt.subplots()
        else:
            a = ax
            # f = a.get_figure()
        if type(color) == bool:
            if not color:
                color = plt_tools.color_cycle[0]
        a.plot(self.data.values, self.data.index, color=color, linewidth=2)
        # print(plt_tools.color_cycle[0])
        a.set_ylabel(self._y_label)
        a.set_xlabel(self._x_label)
        a.set_ylim((self.data.index.min(), self.data.index.max()))
        return a

    def save(self, fname):
        self.data.to_csv(fname)

    def copy(self):
        return _deepcopy(self)

    def convert2timeseries(self, ts):
        """merges a vertical profile with a timeseries that contains height data
        and returns the a time series where the data of the vertical profile is interpolated
        along the time of the timeseries. ...

        Arguments
        ---------
        ts: timeseries"""
        hk_tmp = ts.convert2verticalprofile()
        data = hk_tmp.data[['TimeUTC']]
        cat_sort_int = pd.concat([data, self.data]).sort_index().interpolate()
        cat_sort_int = cat_sort_int.dropna()
        cat_sort_int.index = cat_sort_int.TimeUTC
        cat_sort_int = cat_sort_int.drop('TimeUTC', axis=1)
        return timeseries.TimeSeries(cat_sort_int)

#### Tools
def merge(ts, ts_other):
    """ Merges current with other timeseries. The returned timeseries has the same time-axes as the current
    one (as opposed to the one merged into it). Missing or offset data points are linearly interpolated.

    Argument
    --------
    ts_orig: the other time series will be merged to this, therefore this timeseries
    will define the time stamps.
    ts: timeseries or one of its subclasses.
        List of TimeSeries objects.

    Returns
    -------
    TimeSeries object or one of its subclasses

    """
    ts_this = ts.copy()
    ts_data_list = [ts_this.data, ts_other.data]
    catsortinterp = pd.concat(ts_data_list).sort_index().interpolate()
    merged = catsortinterp.groupby(catsortinterp.index).mean().reindex(ts_data_list[0].index)
    ts_this.data = merged
    return ts_this

def align_to(ts, ts_other):
    """
    Align the TimeSeries ts to another time_series by interpolating (linearly).

    Parameters
    ----------
    ts: original time series
    ts_other: timeseries to align to

    Returns
    -------
    timeseries eqivalent to the original but with an index aligned to the other
    """
    ts = ts.copy()
    ts_other = ts_other.copy()
    ts_other.data = ts_other.data.loc[:,[]]
    ts_t =  merge(ts_other, ts)
    ts.data = ts_t.data
    return ts