import pandas as pd
import pylab as plt

from atmPy.general import timeseries
from atmPy.tools import plt_tools


class VerticalProfile(object):
    def __init__(self, data):
        self.data = data

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
        a.set_ylabel('Altitude (m)')
        a.set_ylim((self.data.index.min(), self.data.index.max()))
        return a

    def save(self, fname):
        self.data.to_csv(fname)

    def convert2timeseries(self, ts):
        """merges a vertical profile with a timeseries that contains height data
        and returns the a time series where the data of the vertical profile is interpolated
        along the time of the timeseries.

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
